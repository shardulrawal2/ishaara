"""Run local, end-to-end recognition on a short signing video.

The initial checkpoint is trained on INCLUDE's 263-label vocabulary. This tool
uses the upstream extraction and preprocessing path so its input matches the
model contract exactly: 169 frames x 134 features.

Usage:
    INCLUDE\\.venv\\Scripts\\python.exe recognize_video.py path\\to\\signing-clip.mp4

Use a short, single-sign clip (at most 169 frames) with the signer visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


ROOT = Path(__file__).resolve().parent
INCLUDE_ROOT = ROOT / "INCLUDE"
MODEL_PATH = ROOT / "exported" / "ishaara_include_transformer_small.quant.ort"
LABEL_MAP_PATH = INCLUDE_ROOT / "label_maps" / "label_map_include.json"
WINDOW_SIZE = 169
FEATURE_DIM = 134


def load_feature_tensor(video_path: Path) -> np.ndarray:
    """Extract and preprocess a single video through INCLUDE's exact dataset path."""

    sys.path.insert(0, str(INCLUDE_ROOT))
    from dataset import KeypointsDataset
    from generate_keypoints import process_video

    with tempfile.TemporaryDirectory(prefix="ishaara-keypoints-") as temporary_root:
        keypoints_dir = Path(temporary_root) / "keypoints"
        keypoints_dir.mkdir()
        process_video(video_path.resolve().as_posix(), str(keypoints_dir))

        keypoint_files = sorted(keypoints_dir.glob("*.json"))
        if len(keypoint_files) != 1:
            raise RuntimeError("Expected exactly one extracted keypoint file.")

        raw = pd.read_json(keypoint_files[0], typ="series")
        landmark_arrays = (raw.pose_x, raw.pose_y, raw.hand1_x, raw.hand1_y, raw.hand2_x, raw.hand2_y)
        detected_landmark = any(
            np.isfinite(np.asarray(values, dtype=np.float32)).any() for values in landmark_arrays
        )
        if not detected_landmark:
            raise ValueError(
                "No hands or upper-body pose were detected. Use a brighter, closer clip with the signer in frame."
            )
        if int(raw.n_frames) > WINDOW_SIZE:
            raise ValueError(
                f"This clip has {int(raw.n_frames)} frames; use a clip of {WINDOW_SIZE} frames or fewer."
            )

        source_label = "".join(character for character in str(raw.label) if character.isalpha()).lower()
        dataset = KeypointsDataset(
            keypoints_dir=str(keypoints_dir),
            use_augs=False,
            label_map={source_label: 0},
            mode="test",
            max_frame_len=WINDOW_SIZE,
        )
        tensor = dataset[0]["data"].numpy()

    if tensor.shape != (WINDOW_SIZE, FEATURE_DIM):
        raise RuntimeError(f"Unexpected feature tensor shape: {tensor.shape}")
    if not np.isfinite(tensor).all():
        raise RuntimeError("Preprocessing produced non-finite values.")
    return np.expand_dims(tensor.astype(np.float32, copy=False), axis=0)


def load_labels() -> dict[int, str]:
    with LABEL_MAP_PATH.open(encoding="utf-8") as file:
        label_to_index = json.load(file)
    return {int(index): label for label, index in label_to_index.items()}


def recognize(video_path: Path, top_k: int) -> list[tuple[str, float]]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model package not found: {MODEL_PATH}")

    input_tensor = load_feature_tensor(video_path)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    probabilities = session.run(
        ["class_probabilities"],
        {"keypoint_sequence": input_tensor},
    )[0][0]

    labels = load_labels()
    ranked_indices = np.argsort(probabilities)[::-1][:top_k]
    return [(labels[int(index)], float(probabilities[int(index)])) for index in ranked_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize one short signing video locally.")
    parser.add_argument("video", type=Path, help="Path to a signing video (169 frames or fewer).")
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 11))
    args = parser.parse_args()

    results = recognize(args.video, args.top_k)
    print("Top recognition candidates:")
    for rank, (label, probability) in enumerate(results, start=1):
        print(f"{rank}. {label} — {probability:.1%}")


if __name__ == "__main__":
    main()
