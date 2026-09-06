"""Measure personal-template separation on available held-out captures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ishaara_runtime import extract_feature_tensor  # noqa: E402
from personal_signs import build_embedding, match_personal_sign  # noqa: E402


def main() -> None:
    captures = ROOT / "data" / "isl-field-captures"
    entries = [json.loads(line) for line in (captures / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    samples: dict[str, list[Path]] = {}
    for entry in entries:
        path = captures / entry["file"]
        if path.is_file() and len(samples.setdefault(entry["label"], [])) < 4:
            samples[entry["label"]].append(path)
    embeddings = {
        label: [build_embedding(extract_feature_tensor(path)) for path in paths]
        for label, paths in samples.items()
        if len(paths) >= 4
    }
    for first_label, first_samples in embeddings.items():
        for second_label, second_samples in embeddings.items():
            distances = [
                float(np.sqrt(np.mean(np.square(first - second))))
                for first in first_samples
                for second in second_samples
                if first is not second
            ]
            if distances:
                print(f"{first_label:12s} -> {second_label:12s}: min={min(distances):.3f} mean={np.mean(distances):.3f}")

    for label, label_embeddings in embeddings.items():
        templates = {label: [example.tolist() for example in label_embeddings[:3]]}
        same_match = match_personal_sign(label_embeddings[3], templates)
        if same_match is None or same_match[0] != label:
            raise AssertionError(f"A held-out {label} example did not match its three templates: {same_match}")
        other_label = next(candidate for candidate in embeddings if candidate != label)
        other_match = match_personal_sign(embeddings[other_label][3], templates)
        if other_match is not None:
            raise AssertionError(f"{other_label} incorrectly matched the personal {label} templates: {other_match}")
        print(f"personal={label:12s} held_out_match={same_match[1]:.1%} different_gesture=rejected")


if __name__ == "__main__":
    main()
