"""Fine-tune Ishaara's landmark transformer on prepared, signer-separated ISL data.

Run only after ``prepare_isl_training_data.py`` has completed successfully:

    INCLUDE\\.venv\\Scripts\\python.exe scripts\\train_isl_refinement.py \
      --data-dir data\\isl-training-keypoints --epochs 35

This trains a new scoped ISL classifier.  It intentionally does not replace the
review server's checkpoint: promote a model only after reviewing the held-out,
unseen-signer metrics printed at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.core.multiarray import scalar
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_ROOT = ROOT / "INCLUDE"
sys.path.insert(0, str(INCLUDE_ROOT))

from configs import TransformerConfig  # noqa: E402
from dataset import KeypointsDataset  # noqa: E402
from models import Transformer  # noqa: E402


def load_base_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Safely load the trusted INCLUDE tensor state under PyTorch 2.6+."""

    torch.serialization.add_safe_globals(
        [
            (scalar, "numpy.core.multiarray.scalar"),
            (np.dtype, "numpy.dtype"),
            (np.dtypes.Float64DType, "numpy.dtypes.Float64DType"),
        ]
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    return checkpoint["model"]


def evaluate(model: Transformer, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["data"].to(device))
            predicted.extend(torch.argmax(logits, dim=1).cpu().tolist())
            expected.extend(batch["label"].tolist())
    return accuracy_score(expected, predicted), f1_score(expected, predicted, average="macro", zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a scoped ISL transformer.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Output directory from prepare_isl_training_data.py")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "isl-refinement")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = args.data_dir.resolve()
    label_path = data_dir / "label_map_ishaara.json"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing {label_path}; run the preparation script first.")
    label_map: dict[str, int] = json.loads(label_path.read_text(encoding="utf-8"))
    if len(label_map) < 2:
        raise ValueError("A recognizer needs at least two sign labels.")

    train_set = KeypointsDataset(str(data_dir / "ishaara_train_keypoints"), False, label_map, mode="train", max_frame_len=169)
    val_set = KeypointsDataset(str(data_dir / "ishaara_val_keypoints"), False, label_map, mode="val", max_frame_len=169)
    test_set = KeypointsDataset(str(data_dir / "ishaara_test_keypoints"), False, label_map, mode="test", max_frame_len=169)
    if not len(train_set) or not len(val_set) or not len(test_set):
        raise ValueError("Every signer-separated split needs at least one recording. Collect more examples from each signer.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(TransformerConfig(size="small"), n_classes=len(label_map)).to(device)
    base_checkpoint = INCLUDE_ROOT / "checkpoints" / "include_no_cnn_transformer_small.pth"
    checkpoint_state = load_base_checkpoint_state(base_checkpoint)
    base_state = {name: value for name, value in checkpoint_state.items() if not name.startswith("l2.")}
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if unexpected or set(missing) != {"l2.weight", "l2.bias"}:
        raise RuntimeError("The base checkpoint does not match the expected transformer architecture.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best_score = float("-inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["data"].to(device))
            loss = functional.cross_entropy(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_accuracy, val_f1 = evaluate(model, val_loader, device)
        print(f"epoch={epoch:02d} train_loss={np.mean(losses):.4f} val_accuracy={val_accuracy:.3f} val_macro_f1={val_f1:.3f}")
        if val_f1 > best_score:
            best_score = val_f1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model.")
    model.load_state_dict(best_state)
    test_accuracy, test_f1 = evaluate(model, test_loader, device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "labels": label_map, "validation_macro_f1": best_score, "test_accuracy": test_accuracy, "test_macro_f1": test_f1}, args.output / "ishaara_isl_refinement.pth")
    summary = {"labels": list(label_map), "validation_macro_f1": best_score, "held_out_signer_accuracy": test_accuracy, "held_out_signer_macro_f1": test_f1}
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nHeld-out signer metrics (do not promote without reviewing these):")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
