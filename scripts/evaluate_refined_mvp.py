"""Evaluate a scoped refinement checkpoint on signer-separated landmark data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_ROOT = ROOT / "INCLUDE"
sys.path.insert(0, str(INCLUDE_ROOT))

from configs import TransformerConfig  # noqa: E402
from dataset import KeypointsDataset  # noqa: E402
from models import Transformer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the scoped Ishaara MVP model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    checkpoint = torch.load(args.model.resolve(), map_location="cpu", weights_only=True)
    label_to_index = checkpoint["labels"]
    label_names = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    model = Transformer(TransformerConfig(size="small"), n_classes=len(label_names))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    for split in ("val", "test"):
        dataset = KeypointsDataset(
            str(args.data_dir.resolve() / f"ishaara_{split}_keypoints"),
            False,
            label_to_index,
            mode=split,
            max_frame_len=169,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        expected: list[int] = []
        predicted: list[int] = []
        confidence: list[float] = []
        with torch.inference_mode():
            for batch in loader:
                probabilities = torch.softmax(model(batch["data"]), dim=-1)
                scores, indices = torch.max(probabilities, dim=-1)
                expected.extend(batch["label"].tolist())
                predicted.extend(indices.tolist())
                confidence.extend(scores.tolist())

        expected_array = np.asarray(expected)
        predicted_array = np.asarray(predicted)
        confidence_array = np.asarray(confidence)
        accepted = confidence_array >= args.threshold
        accepted_accuracy = float(np.mean(predicted_array[accepted] == expected_array[accepted])) if accepted.any() else 0.0
        print(f"\n{split.upper()} · samples={len(dataset)}")
        print(classification_report(expected_array, predicted_array, target_names=label_names, digits=3, zero_division=0))
        print("confusion_matrix=")
        print(confusion_matrix(expected_array, predicted_array).tolist())
        print(
            json.dumps(
                {
                    "threshold": args.threshold,
                    "accepted_samples": int(accepted.sum()),
                    "coverage": float(accepted.mean()),
                    "accepted_accuracy": accepted_accuracy,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
