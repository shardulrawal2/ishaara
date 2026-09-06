"""Export a trained Ishaara checkpoint to a deployment-sized ONNX model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ishaara_model import Transformer, TransformerConfig  # noqa: E402
from ishaara_runtime import FEATURE_DIM, WINDOW_SIZE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a refined Ishaara model for deployment.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    model = Transformer(TransformerConfig(size="small"), n_classes=len(labels))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.float.onnx")
    example = torch.zeros((1, WINDOW_SIZE, FEATURE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        temporary,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(temporary, output, weight_type=QuantType.QInt8)
    temporary.unlink(missing_ok=True)
    print(f"Exported {len(labels)} labels to {output} ({output.stat().st_size / 1024 / 1024:.2f} MB).")


if __name__ == "__main__":
    main()
