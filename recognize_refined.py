"""Inference for the locally fine-tuned, scoped Ishaara ISL model.

The refined checkpoint is deliberately separate from the 263-class INCLUDE
baseline. It is trained and evaluated on Ishaara's own camera recordings and
must exist locally before the mobile recognition endpoint starts.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from recognize_video import load_feature_tensor


ROOT = Path(__file__).resolve().parent
INCLUDE_ROOT = ROOT / "INCLUDE"
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "hello-thankyou-v1" / "ishaara_isl_refinement.pth"
MINIMUM_CONFIDENCE = 0.75


def model_path() -> Path:
    configured = os.environ.get("ISHAARA_REFINED_MODEL")
    return Path(configured).resolve() if configured else DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def load_model() -> tuple[torch.nn.Module, dict[int, str], dict[str, float]]:
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Refined model not found at {path}. Train and validate the scoped ISL model first."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    label_to_index = checkpoint["labels"]
    index_to_label = {int(index): str(label) for label, index in label_to_index.items()}
    if len(index_to_label) < 2:
        raise ValueError("The refined recognizer requires at least two labels.")

    sys.path.insert(0, str(INCLUDE_ROOT))
    from configs import TransformerConfig
    from models import Transformer

    model = Transformer(TransformerConfig(size="small"), n_classes=len(index_to_label))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    metrics = {
        "validation_macro_f1": float(checkpoint.get("validation_macro_f1", 0.0)),
        "held_out_signer_accuracy": float(checkpoint.get("test_accuracy", 0.0)),
        "held_out_signer_macro_f1": float(checkpoint.get("test_macro_f1", 0.0)),
    }
    return model, index_to_label, metrics


def labels() -> list[str]:
    _, index_to_label, _ = load_model()
    return [index_to_label[index] for index in sorted(index_to_label)]


def model_metrics() -> dict[str, float]:
    _, _, metrics = load_model()
    return metrics


def recognize(video_path: Path, top_k: int = 2) -> list[tuple[str, float]]:
    input_tensor = load_feature_tensor(video_path)
    model, index_to_label, _ = load_model()
    with torch.inference_mode():
        logits = model(torch.from_numpy(input_tensor))
        probabilities = torch.softmax(logits, dim=-1)[0].numpy()
    ranked = np.argsort(probabilities)[::-1][: min(top_k, len(index_to_label))]
    return [(index_to_label[int(index)], float(probabilities[int(index)])) for index in ranked]
