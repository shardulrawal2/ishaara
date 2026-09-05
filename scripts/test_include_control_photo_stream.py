"""Validate the ESP32-style JPEG snapshot path on official INCLUDE test clips."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from photo_stream_recognition import SnapshotSequence, prepare_feature_tensor
from recognize_video import WINDOW_SIZE, load_feature_tensor, recognize


CONTROL_ROOT = ROOT / "data" / "include-control"


def expected_label(video_path: Path) -> str:
    return video_path.parent.name.split(". ", 1)[1].lower()


def recognize_as_snapshots(video_path: Path) -> tuple[int, int, list[tuple[str, float]], np.ndarray]:
    sequence = SnapshotSequence()
    capture = cv2.VideoCapture(str(video_path))
    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            if len(sequence.frames) >= WINDOW_SIZE:
                raise ValueError(f"{video_path.name} exceeds the {WINDOW_SIZE}-frame model limit.")
            encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not encoded:
                raise RuntimeError(f"Could not encode {video_path.name} as a JPEG snapshot.")
            sequence.add_jpeg(jpeg.tobytes())
        tensor = prepare_feature_tensor(sequence.frames)
        candidates = sequence.recognize(top_k=3)
        return len(sequence.frames), sequence.signer_hand_frames, candidates, tensor
    finally:
        capture.release()
        sequence.close()


def main() -> None:
    clips = sorted(CONTROL_ROOT.rglob("*.MOV"))
    if not clips:
        raise FileNotFoundError("Run fetch_include_control_clips.py first.")
    for clip in clips:
        frame_count, hand_frames, candidates, snapshot_tensor = recognize_as_snapshots(clip)
        label, confidence = candidates[0]
        direct_label, direct_confidence = recognize(clip, top_k=1)[0]
        direct_tensor = load_feature_tensor(clip)
        mean_feature_delta = float(np.abs(direct_tensor - snapshot_tensor).mean())
        print(
            f"expected={expected_label(clip):<7} snapshot={label:<12} {confidence:.1%} "
            f"direct={direct_label:<12} {direct_confidence:.1%} frames={frame_count} "
            f"signer_hand_frames={hand_frames} mean_feature_delta={mean_feature_delta:.2f}"
        )


if __name__ == "__main__":
    main()
