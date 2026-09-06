"""Self-contained video preprocessing for deployment.

This module reproduces the 169 x 134 INCLUDE inference contract without
depending on the ignored upstream research checkout. Training utilities may
still use that checkout, but the deployed Flask service does not.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


WINDOW_SIZE = 169
FEATURE_DIM = 134
POSE_POINTS = 25
HAND_POINTS = 21
TRAINING_ASPECT_RATIO = 4 / 3


def _crop_to_training_aspect_ratio(frame: np.ndarray) -> np.ndarray:
    """Match portrait/mobile input to the 4:3 geometry used by all training clips.

    The camera preview uses a cover-style crop. Applying the same centered crop
    before landmark extraction keeps normalized coordinates comparable with the
    640 x 480 recordings used to train the scoped model.
    """

    height, width = frame.shape[:2]
    if not height or not width:
        return frame
    aspect_ratio = width / height
    if abs(aspect_ratio - TRAINING_ASPECT_RATIO) < 0.02:
        return frame
    if aspect_ratio < TRAINING_ASPECT_RATIO:
        crop_height = max(1, min(height, round(width / TRAINING_ASPECT_RATIO)))
        top = (height - crop_height) // 2
        return frame[top : top + crop_height, :]
    crop_width = max(1, min(width, round(height * TRAINING_ASPECT_RATIO)))
    left = (width - crop_width) // 2
    return frame[:, left : left + crop_width]


def _landmarks_xy(landmarks, count: int) -> tuple[list[float], list[float]]:
    if landmarks is None:
        return [np.nan] * count, [np.nan] * count
    selected = landmarks.landmark[:count]
    return [point.x for point in selected], [point.y for point in selected]


def _swap_single_hand(
    pose_x: list[float],
    pose_y: list[float],
    first_x: list[float],
    first_y: list[float],
    second_x: list[float],
    second_y: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    if len(pose_x) <= 16:
        return first_x, first_y, second_x, second_y
    first_detected = bool(first_x)
    second_detected = bool(second_x)
    if first_detected == second_detected:
        return first_x, first_y, second_x, second_y
    hand_x, hand_y = (first_x[0], first_y[0]) if first_detected else (second_x[0], second_y[0])
    left_distance = (pose_x[15] - hand_x) ** 2 + (pose_y[15] - hand_y) ** 2
    right_distance = (pose_x[16] - hand_x) ** 2 + (pose_y[16] - hand_y) ** 2
    should_be_second = right_distance < left_distance
    if (first_detected and should_be_second) or (second_detected and not should_be_second):
        return second_x, second_y, first_x, first_y
    return first_x, first_y, second_x, second_y


def _interpolate_group(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    x_values = pd.DataFrame(x_values).interpolate(method="linear", limit_direction="both").to_numpy()
    y_values = pd.DataFrame(y_values).interpolate(method="linear", limit_direction="both").to_numpy()
    if not np.isfinite(x_values).any():
        x_values = np.zeros_like(x_values)
    if not np.isfinite(y_values).any():
        y_values = np.zeros_like(y_values)
    return np.stack((x_values * 1920, y_values * 1080), axis=-1).reshape(len(x_values), -1)


def extract_feature_tensor(video_path: Path) -> np.ndarray:
    """Convert one isolated-sign video into a padded ``(1, 169, 134)`` tensor."""

    pose_x_frames: list[list[float]] = []
    pose_y_frames: list[list[float]] = []
    first_x_frames: list[list[float]] = []
    first_y_frames: list[list[float]] = []
    second_x_frames: list[list[float]] = []
    second_y_frames: list[list[float]] = []
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    capture = cv2.VideoCapture(str(video_path.resolve()))
    try:
        while capture.isOpened():
            success, bgr = capture.read()
            if not success:
                break
            bgr = _crop_to_training_aspect_ratio(bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(rgb)
            pose_result = pose.process(rgb)
            pose_x, pose_y = _landmarks_xy(pose_result.pose_landmarks, POSE_POINTS)
            detected_hands = hand_result.multi_hand_landmarks or []
            first_x, first_y = _landmarks_xy(detected_hands[0] if detected_hands else None, HAND_POINTS)
            second_x, second_y = _landmarks_xy(detected_hands[1] if len(detected_hands) > 1 else None, HAND_POINTS)

            raw_first_x = [] if not detected_hands else first_x
            raw_first_y = [] if not detected_hands else first_y
            raw_second_x = [] if len(detected_hands) < 2 else second_x
            raw_second_y = [] if len(detected_hands) < 2 else second_y
            raw_first_x, raw_first_y, raw_second_x, raw_second_y = _swap_single_hand(
                pose_x,
                pose_y,
                raw_first_x,
                raw_first_y,
                raw_second_x,
                raw_second_y,
            )
            if not raw_first_x:
                raw_first_x, raw_first_y = [np.nan] * HAND_POINTS, [np.nan] * HAND_POINTS
            if not raw_second_x:
                raw_second_x, raw_second_y = [np.nan] * HAND_POINTS, [np.nan] * HAND_POINTS

            pose_x_frames.append(pose_x)
            pose_y_frames.append(pose_y)
            first_x_frames.append(raw_first_x)
            first_y_frames.append(raw_first_y)
            second_x_frames.append(raw_second_x)
            second_y_frames.append(raw_second_y)
            if len(pose_x_frames) > WINDOW_SIZE:
                raise ValueError(f"This clip has more than {WINDOW_SIZE} frames; use a shorter clip.")
    finally:
        capture.release()
        hands.close()
        pose.close()

    frame_count = len(pose_x_frames)
    if not frame_count:
        raise ValueError("The video contains no readable frames.")
    hand_values = np.concatenate(
        (
            np.asarray(first_x_frames, dtype=np.float32),
            np.asarray(first_y_frames, dtype=np.float32),
            np.asarray(second_x_frames, dtype=np.float32),
            np.asarray(second_y_frames, dtype=np.float32),
        ),
        axis=1,
    )
    hand_frames = int(np.isfinite(hand_values).any(axis=1).sum())
    required_hand_frames = max(3, int(np.ceil(frame_count * 0.2)))
    if hand_frames < required_hand_frames:
        raise ValueError(
            f"Hands were tracked in only {hand_frames} frames; at least {required_hand_frames} are required. "
            "Move closer, improve lighting, and keep both hands visible."
        )

    pose_group = _interpolate_group(np.asarray(pose_x_frames), np.asarray(pose_y_frames))
    first_group = _interpolate_group(np.asarray(first_x_frames), np.asarray(first_y_frames))
    second_group = _interpolate_group(np.asarray(second_x_frames), np.asarray(second_y_frames))
    sequence = np.concatenate((pose_group, first_group, second_group), axis=1).astype(np.float32, copy=False)
    if sequence.shape != (frame_count, FEATURE_DIM):
        raise RuntimeError(f"Unexpected extracted feature shape: {sequence.shape}")
    sequence = np.pad(sequence, ((0, WINDOW_SIZE - frame_count), (0, 0)), mode="constant")
    if not np.isfinite(sequence).all():
        raise RuntimeError("Preprocessing produced non-finite landmark values.")
    return np.expand_dims(sequence, axis=0)
