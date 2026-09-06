"""Few-shot landmark templates for device-owned personal gestures.

These templates do not modify the trained classifier. They normalize body and
hand motion so a browser can enroll a private name or phrase with a few clips
and send only landmark templates with later recognition requests.
"""

from __future__ import annotations

import numpy as np


RESAMPLED_FRAMES = 24
FEATURE_PAIRS = (13, 14, 15, 16)
MAX_PERSONAL_SIGNS = 12
EXAMPLES_PER_SIGN = 3
MATCH_THRESHOLD = 0.38


def _active_sequence(tensor: np.ndarray) -> np.ndarray:
    sequence = np.asarray(tensor, dtype=np.float32)
    if sequence.shape == (1, 169, 134):
        sequence = sequence[0]
    if sequence.shape != (169, 134):
        raise ValueError(f"Unexpected personal-sign tensor shape: {sequence.shape}")
    active = np.flatnonzero(np.any(sequence != 0, axis=1))
    if not len(active):
        raise ValueError("No tracked landmark sequence was found.")
    return sequence[: active[-1] + 1].copy()


def build_embedding(tensor: np.ndarray) -> np.ndarray:
    """Create a translation/scale-normalized temporal landmark signature."""

    sequence = _active_sequence(tensor)
    pose = sequence[:, :50].reshape(-1, 25, 2)
    first_hand = sequence[:, 50:92].reshape(-1, 21, 2)
    second_hand = sequence[:, 92:134].reshape(-1, 21, 2)

    pose[:, :, 0] /= 1920.0
    pose[:, :, 1] /= 1080.0
    first_hand[:, :, 0] /= 1920.0
    first_hand[:, :, 1] /= 1080.0
    second_hand[:, :, 0] /= 1920.0
    second_hand[:, :, 1] /= 1080.0

    shoulders = pose[:, [11, 12]]
    center = shoulders.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    valid_scale = scale[scale > 0.03]
    fallback = float(np.median(valid_scale)) if len(valid_scale) else 0.2
    scale = np.where(scale > 0.03, scale, fallback).reshape(-1, 1, 1)

    selected_pose = pose[:, FEATURE_PAIRS]
    groups = []
    for group in (selected_pose, first_hand, second_hand):
        missing = np.all(group == 0, axis=(1, 2))
        normalized = (group - center) / scale
        normalized[missing] = 0
        groups.append(normalized.reshape(len(sequence), -1))
    normalized_sequence = np.concatenate(groups, axis=1)

    source = np.linspace(0.0, 1.0, len(normalized_sequence))
    target = np.linspace(0.0, 1.0, RESAMPLED_FRAMES)
    resampled = np.stack(
        [np.interp(target, source, normalized_sequence[:, column]) for column in range(normalized_sequence.shape[1])],
        axis=1,
    ).astype(np.float32)
    velocity = np.diff(resampled, axis=0, prepend=resampled[:1])
    embedding = np.concatenate((resampled, velocity * 0.65), axis=1).reshape(-1)
    return np.clip(embedding, -8.0, 8.0).astype(np.float32)


def match_personal_sign(
    embedding: np.ndarray,
    templates: dict[str, list[list[float]]],
) -> tuple[str, float] | None:
    """Return a label and bounded similarity only for a strong template match."""

    best: tuple[str, float] | None = None
    for label, examples in list(templates.items())[:MAX_PERSONAL_SIGNS]:
        if not isinstance(label, str) or not isinstance(examples, list) or len(examples) < EXAMPLES_PER_SIGN:
            continue
        distances = []
        for raw_example in examples[:EXAMPLES_PER_SIGN]:
            example = np.asarray(raw_example, dtype=np.float32)
            if example.shape != embedding.shape or not np.isfinite(example).all():
                continue
            distances.append(float(np.sqrt(np.mean(np.square(embedding - example)))))
        if len(distances) != EXAMPLES_PER_SIGN:
            continue
        distance = float(np.mean(sorted(distances)[:2]))
        confidence = max(0.0, min(1.0, 1.0 - distance / (2.0 * MATCH_THRESHOLD)))
        if distance <= MATCH_THRESHOLD and (best is None or confidence > best[1]):
            best = (label, confidence)
    return best
