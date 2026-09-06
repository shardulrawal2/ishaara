"""Inference for the locally fine-tuned, scoped Ishaara ISL model.

The refined checkpoint is deliberately separate from the 263-class INCLUDE
baseline. It is trained and evaluated on Ishaara's own camera recordings and
must exist locally before the mobile recognition endpoint starts.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ishaara_runtime import extract_feature_tensor


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "hello-thankyou-v1" / "ishaara_isl_refinement.quant.onnx"
DEFAULT_METADATA_PATH = ROOT / "artifacts" / "hello-thankyou-v1" / "metrics.json"
MINIMUM_CONFIDENCE = 0.75


def model_path() -> Path:
    configured = os.environ.get("ISHAARA_REFINED_MODEL")
    return Path(configured).resolve() if configured else DEFAULT_MODEL_PATH


def metadata_path() -> Path:
    configured = os.environ.get("ISHAARA_REFINED_METADATA")
    return Path(configured).resolve() if configured else model_path().with_name("metrics.json")


@lru_cache(maxsize=1)
def load_model() -> tuple[ort.InferenceSession, dict[int, str], dict[str, float]]:
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Deployment model not found at {path}. Export and validate the scoped ISL model first."
        )
    metadata_file = metadata_path()
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Model metadata not found at {metadata_file}.")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    index_to_label = {index: str(label) for index, label in enumerate(metadata["labels"])}
    if len(index_to_label) < 2:
        raise ValueError("The refined recognizer requires at least two labels.")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    model = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    metrics = {
        "validation_macro_f1": float(metadata.get("validation_macro_f1", 0.0)),
        "held_out_signer_accuracy": float(metadata.get("held_out_signer_accuracy", 0.0)),
        "held_out_signer_macro_f1": float(metadata.get("held_out_signer_macro_f1", 0.0)),
    }
    return model, index_to_label, metrics


def labels() -> list[str]:
    _, index_to_label, _ = load_model()
    return [index_to_label[index] for index in sorted(index_to_label)]


def model_metrics() -> dict[str, float]:
    _, _, metrics = load_model()
    return metrics


def recognize_feature_tensor(input_tensor: np.ndarray, top_k: int = 2) -> list[tuple[str, float]]:
    model, index_to_label, _ = load_model()
    input_name = model.get_inputs()[0].name
    logits = model.run(None, {input_name: input_tensor.astype(np.float32, copy=False)})[0][0]
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    ranked = np.argsort(probabilities)[::-1][: min(top_k, len(index_to_label))]
    return [(index_to_label[int(index)], float(probabilities[int(index)])) for index in ranked]


def recognize(video_path: Path, top_k: int = 2) -> list[tuple[str, float]]:
    return recognize_feature_tensor(extract_feature_tensor(video_path), top_k)
