"""Export the verified INCLUDE Transformer checkpoint to a fixed-shape ONNX graph.

This script intentionally uses the values verified from the local INCLUDE source and
checkpoint, rather than the placeholder values in the integration blueprint:

* model: INCLUDE small no-CNN Transformer
* input: [1, 169, 134] float32
* output: [1, 263] float32 probabilities

Run from the Ishaara repository root:
    INCLUDE\\.venv\\Scripts\\python.exe export_to_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import transformers
from numpy.core.multiarray import scalar
from transformers.models.bert.modeling_bert import BertLayer


ROOT = Path(__file__).resolve().parent
INCLUDE_ROOT = ROOT / "INCLUDE"
CHECKPOINT_PATH = INCLUDE_ROOT / "checkpoints" / "include_no_cnn_transformer_small.pth"
OUTPUT_PATH = ROOT / "exported" / "ishaara_include_transformer_small.onnx"

WINDOW_SIZE = 169
FEATURE_DIM = 134
NUM_CLASSES = 263


def install_transformers_compatibility_alias() -> None:
    """Support INCLUDE's legacy ``transformers.BertLayer`` import expectation."""

    if not hasattr(transformers, "BertLayer"):
        transformers.BertLayer = BertLayer


def load_checkpoint_state() -> dict[str, torch.Tensor]:
    """Read tensors without enabling arbitrary pickle code execution."""

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Download it before exporting."
        )

    # The official checkpoint stores NumPy metadata alongside tensor weights.
    # These explicit type allowlists preserve PyTorch's safe weights-only loading.
    torch.serialization.add_safe_globals(
        [
            (scalar, "numpy.core.multiarray.scalar"),
            (np.dtype, "numpy.dtype"),
            (np.dtypes.Float64DType, "numpy.dtypes.Float64DType"),
        ]
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    return checkpoint["model"]


class ExportWrapper(nn.Module):
    """Return probabilities so mobile code does not need to reproduce softmax."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, keypoint_sequence: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(keypoint_sequence), dim=-1)


def build_model() -> ExportWrapper:
    install_transformers_compatibility_alias()
    sys.path.insert(0, str(INCLUDE_ROOT))

    from configs import TransformerConfig
    from models import Transformer

    model = Transformer(config=TransformerConfig(size="small"), n_classes=NUM_CLASSES)
    model.load_state_dict(load_checkpoint_state(), strict=True)
    return ExportWrapper(model).eval()


def verify_onnx_model(sample_input: torch.Tensor, reference: np.ndarray) -> None:
    exported_model = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(exported_model)

    graph_input = exported_model.graph.input[0]
    graph_output = exported_model.graph.output[0]
    input_shape = tuple(dim.dim_value for dim in graph_input.type.tensor_type.shape.dim)
    output_shape = tuple(dim.dim_value for dim in graph_output.type.tensor_type.shape.dim)
    expected_input_shape = (1, WINDOW_SIZE, FEATURE_DIM)
    expected_output_shape = (1, NUM_CLASSES)

    if input_shape != expected_input_shape:
        raise ValueError(f"Unexpected ONNX input shape {input_shape}; expected {expected_input_shape}")
    if output_shape != expected_output_shape:
        raise ValueError(
            f"Unexpected ONNX output shape {output_shape}; expected {expected_output_shape}"
        )

    session = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        ["class_probabilities"],
        {"keypoint_sequence": sample_input.numpy()},
    )[0]
    np.testing.assert_allclose(reference, onnx_output, rtol=1e-4, atol=1e-5)

    print(f"ONNX input shape:  {input_shape}")
    print(f"ONNX output shape: {output_shape}")
    print(f"Maximum PyTorch/ONNX difference: {np.max(np.abs(reference - onnx_output)):.8f}")


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)

    model = build_model()
    sample_input = torch.randn(1, WINDOW_SIZE, FEATURE_DIM, dtype=torch.float32)
    with torch.inference_mode():
        reference = model(sample_input).numpy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["keypoint_sequence"],
        output_names=["class_probabilities"],
        dynamic_axes=None,
        dynamo=False,
    )
    verify_onnx_model(sample_input, reference)
    print(f"Exported: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
