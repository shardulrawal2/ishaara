"""ESP32-friendly sign recognition from an ordered stream of JPEG snapshots.

The device sends individual photos instead of a video file. Each photo is
converted to the same 134 landmark values used at training time, while the
server retains only those values in memory. A client marks the final photo of a
single sign and receives the model's candidate labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from ishaara_runtime import FEATURE_DIM, WINDOW_SIZE
from recognize_refined import recognize_feature_tensor


MINIMUM_FRAMES = 16
MINIMUM_CONFIDENCE = 0.75


def _landmarks_to_array(landmarks, count: int) -> np.ndarray:
    values = np.full((count, 2), np.nan, dtype=np.float32)
    if landmarks is not None:
        for index, landmark in enumerate(landmarks.landmark[:count]):
            values[index] = (landmark.x, landmark.y)
    return values


def _swap_hands_if_needed(pose: np.ndarray, first_hand: np.ndarray, second_hand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match INCLUDE's wrist-based first/second hand convention."""

    if not np.isfinite(pose[[15, 16]]).all():
        return first_hand, second_hand

    first_detected = np.isfinite(first_hand).any()
    second_detected = np.isfinite(second_hand).any()
    if first_detected == second_detected:
        return first_hand, second_hand

    hand = first_hand if first_detected else second_hand
    hand_origin = hand[0]
    left_distance = float(np.sum((pose[15] - hand_origin) ** 2))
    right_distance = float(np.sum((pose[16] - hand_origin) ** 2))
    should_be_second = right_distance < left_distance
    if (first_detected and should_be_second) or (second_detected and not should_be_second):
        return second_hand, first_hand
    return first_hand, second_hand


def _belongs_to_signer(hand: np.ndarray, pose: np.ndarray) -> bool:
    """Keep only a hand plausibly attached to the single tracked upper body."""

    if not np.isfinite(hand).any():
        return False
    shoulders = pose[[11, 12]]
    wrists = pose[[15, 16]]
    if not (np.isfinite(shoulders).all() and np.isfinite(wrists).all()):
        return True

    shoulder_width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
    nearest_wrist = min(float(np.linalg.norm(hand[0] - wrist)) for wrist in wrists)
    # The wrist and hand detectors are independent; allow normal detector drift
    # while excluding an unrelated hand elsewhere in the frame.
    return nearest_wrist <= max(0.08, shoulder_width * 1.5)


def prepare_feature_tensor(frames: list[np.ndarray]) -> np.ndarray:
    """Apply INCLUDE's interpolation, scaling, concatenation, and zero padding."""

    if not frames:
        raise ValueError("No snapshots were received.")
    if len(frames) > WINDOW_SIZE:
        raise ValueError(f"A sign can contain at most {WINDOW_SIZE} snapshots.")

    raw = np.stack(frames).astype(np.float32, copy=False)
    if raw.shape[1] != FEATURE_DIM:
        raise RuntimeError("Snapshot landmark schema does not match the model.")

    groups = ((0, 50), (50, 92), (92, 134))
    processed_groups: list[np.ndarray] = []
    for start, end in groups:
        landmarks = raw[:, start:end].reshape(len(raw), -1, 2)
        x_values = pd.DataFrame(landmarks[:, :, 0]).interpolate(method="linear", limit_direction="both").to_numpy()
        y_values = pd.DataFrame(landmarks[:, :, 1]).interpolate(method="linear", limit_direction="both").to_numpy()
        if not np.isfinite(x_values).any():
            x_values = np.zeros_like(x_values)
        if not np.isfinite(y_values).any():
            y_values = np.zeros_like(y_values)
        processed_groups.append(np.stack((x_values * 1920, y_values * 1080), axis=-1).reshape(len(raw), -1))

    sequence = np.concatenate(processed_groups, axis=1).astype(np.float32, copy=False)
    padded = np.pad(sequence, ((0, WINDOW_SIZE - len(sequence)), (0, 0)), mode="constant")
    return np.expand_dims(padded, axis=0)


@dataclass
class SnapshotSequence:
    hands: object = field(default_factory=lambda: mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ))
    pose: object = field(default_factory=lambda: mp.solutions.pose.Pose(
        static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ))
    frames: list[np.ndarray] = field(default_factory=list)
    detected_frames: int = 0
    pose_frames: int = 0
    signer_hand_frames: int = 0
    updated_at: float = field(default_factory=monotonic)

    def add_jpeg(self, image_bytes: bytes) -> bool:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("The snapshot is not a readable JPEG image.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb)
        pose_results = self.pose.process(rgb)
        pose = _landmarks_to_array(pose_results.pose_landmarks, 25)

        found_hands = hand_results.multi_hand_landmarks or []
        first_hand = _landmarks_to_array(found_hands[0] if len(found_hands) > 0 else None, 21)
        second_hand = _landmarks_to_array(found_hands[1] if len(found_hands) > 1 else None, 21)
        if not _belongs_to_signer(first_hand, pose):
            first_hand = np.full((21, 2), np.nan, dtype=np.float32)
        if not _belongs_to_signer(second_hand, pose):
            second_hand = np.full((21, 2), np.nan, dtype=np.float32)
        first_hand, second_hand = _swap_hands_if_needed(pose, first_hand, second_hand)
        frame = np.concatenate((pose.reshape(-1), first_hand.reshape(-1), second_hand.reshape(-1)))

        detected = bool(np.isfinite(frame).any())
        self.frames.append(frame)
        self.detected_frames += int(detected)
        self.pose_frames += int(np.isfinite(pose).any())
        self.signer_hand_frames += int(np.isfinite(first_hand).any() or np.isfinite(second_hand).any())
        self.updated_at = monotonic()
        return detected

    def recognize(self, top_k: int = 3) -> list[tuple[str, float]]:
        if len(self.frames) < MINIMUM_FRAMES:
            raise ValueError(f"Send at least {MINIMUM_FRAMES} snapshots for one sign before finalizing.")
        if self.detected_frames == 0:
            raise ValueError("No hands or upper-body pose were found in the snapshots.")
        return recognize_feature_tensor(prepare_feature_tensor(self.frames), top_k)

    def close(self) -> None:
        self.hands.close()
        self.pose.close()
