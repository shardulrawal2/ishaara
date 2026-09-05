"""Quantize Ishaara's verified ONNX model and convert it to ORT format.

Run from the repository root after ``export_to_onnx.py``:
    INCLUDE\\.venv\\Scripts\\python.exe quantize_and_convert.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT = Path(__file__).resolve().parent
EXPORTED_DIR = ROOT / "exported"
ONNX_PATH = EXPORTED_DIR / "ishaara_include_transformer_small.onnx"
QUANTIZED_ONNX_PATH = EXPORTED_DIR / "ishaara_include_transformer_small.quant.onnx"
ORT_PATH = EXPORTED_DIR / "ishaara_include_transformer_small.quant.ort"

WINDOW_SIZE = 169
FEATURE_DIM = 134


def file_size(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.2f} MB"


def run_session(path: Path, sample_input: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(
        ["class_probabilities"],
        {"keypoint_sequence": sample_input},
    )[0]


def main() -> None:
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(f"Exported model not found: {ONNX_PATH}")

    onnx.checker.check_model(onnx.load(ONNX_PATH))
    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=QUANTIZED_ONNX_PATH,
        weight_type=QuantType.QInt8,
    )
    onnx.checker.check_model(onnx.load(QUANTIZED_ONNX_PATH))

    rng = np.random.default_rng(7)
    sample_input = rng.standard_normal((1, WINDOW_SIZE, FEATURE_DIM)).astype(np.float32)
    reference = run_session(ONNX_PATH, sample_input)
    quantized = run_session(QUANTIZED_ONNX_PATH, sample_input)
    if int(reference.argmax()) != int(quantized.argmax()):
        raise RuntimeError("Quantization changed the top prediction on the validation input.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnxruntime.tools.convert_onnx_models_to_ort",
            str(QUANTIZED_ONNX_PATH),
            "--output_dir",
            str(EXPORTED_DIR),
            "--optimization_style",
            "Fixed",
            "--target_platform",
            "arm",
        ],
        check=True,
    )
    if not ORT_PATH.is_file():
        raise FileNotFoundError(f"ORT conversion did not create {ORT_PATH}")

    print(f"ONNX:           {file_size(ONNX_PATH)}")
    print(f"Quantized ONNX: {file_size(QUANTIZED_ONNX_PATH)}")
    print(f"ORT:            {file_size(ORT_PATH)}")
    print(f"Top class:      {int(quantized.argmax())}")
    print(f"Max difference: {np.max(np.abs(reference - quantized)):.8f}")


if __name__ == "__main__":
    main()
