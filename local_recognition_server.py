"""Web service for Ishaara's video-recognition pipeline.

Uploaded clips are stored in a temporary file just long enough to extract
landmarks and run the ONNX model, then deleted. The same service runs locally or
as the Render API used by the Vercel frontend.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from time import monotonic
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from ishaara_runtime import extract_feature_tensor
from personal_signs import EXAMPLES_PER_SIGN, MAX_PERSONAL_SIGNS, build_embedding, match_personal_sign
from photo_stream_recognition import MINIMUM_CONFIDENCE, MINIMUM_FRAMES, SnapshotSequence
from recognize_refined import (
    MINIMUM_CONFIDENCE as REFINED_MINIMUM_CONFIDENCE,
    labels as refined_labels,
    model_metrics as refined_model_metrics,
    recognize as recognize_refined,
    recognize_feature_tensor,
)


ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "review-demo"
FIELD_CAPTURE_DIR = ROOT / "data" / "isl-field-captures"
ALLOWED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".webm"}
MAX_ACTIVE_SEQUENCES = 8
MAX_PAIRED_DEVICES = 128
SEQUENCE_IDLE_SECONDS = 120
sequences: dict[str, SnapshotSequence] = {}
device_results: dict[str, dict] = {}
landmark_extraction_lock = Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def configured_origins() -> set[str]:
    """Return the exact browser origins allowed to call the deployed API."""

    configured = os.environ.get("ISHAARA_ALLOWED_ORIGINS", "")
    if configured.strip():
        return {origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()}
    return {"http://127.0.0.1:4173", "http://localhost:4173"}


ALLOWED_ORIGINS = configured_origins()


def origin_is_allowed(origin: str) -> bool:
    """Accept configured origins plus HTTPS Vercel production/preview domains."""

    normalized = origin.rstrip("/")
    if normalized in ALLOWED_ORIGINS:
        return True
    parsed = urlparse(normalized)
    return parsed.scheme == "https" and bool(parsed.hostname) and parsed.hostname.endswith(".vercel.app")


@app.after_request
def disable_development_cache(response):
    """Keep app metadata fresh and authorize the configured frontend origin."""

    response.headers["Cache-Control"] = "no-store"
    origin = request.headers.get("Origin", "").rstrip("/")
    if origin_is_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def save_temporary_clip(clip) -> Path:
    """Validate and save an uploaded clip until the current request completes."""

    suffix = Path(clip.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Use an MP4, MOV, WebM, M4V, or AVI video.")
    with tempfile.NamedTemporaryFile(prefix="ishaara-upload-", suffix=suffix, delete=False) as file:
        temporary_path = Path(file.name)
        clip.save(file)
    return temporary_path


def classify_clip(clip, recognizer) -> tuple[list[tuple[str, float]], Path | None]:
    """Run a temporary mobile or uploaded video through the native video path."""

    temporary_path = save_temporary_clip(clip)
    try:
        with landmark_extraction_lock:
            return recognizer(temporary_path, top_k=3), temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def personal_templates_from_request() -> dict[str, list[list[float]]]:
    """Validate browser-owned personal templates without retaining them."""

    raw_templates = request.form.get("personal_signs", "")
    if not raw_templates:
        return {}
    if len(raw_templates) > 2_000_000:
        raise ValueError("Too many personal gesture templates were sent.")
    try:
        templates = json.loads(raw_templates)
    except json.JSONDecodeError as error:
        raise ValueError("Personal gesture templates are invalid.") from error
    if not isinstance(templates, dict) or len(templates) > MAX_PERSONAL_SIGNS:
        raise ValueError(f"Store no more than {MAX_PERSONAL_SIGNS} personal gestures on one device.")
    validated = {}
    for label, examples in templates.items():
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '-]{0,38}", label):
            continue
        if isinstance(examples, list):
            validated[label] = examples[:EXAMPLES_PER_SIGN]
    return validated


def recognition_response(candidates: list[tuple[str, float]], minimum_confidence: float = MINIMUM_CONFIDENCE):
    """Avoid presenting an arbitrary low-probability class as a translation."""

    top_label, confidence = candidates[0]
    if confidence < minimum_confidence:
        return jsonify(
            status="no_confident_match",
            confidence=confidence,
            closest_label=top_label,
            error="No confident match in the current model vocabulary.",
        )
    return jsonify(
        status="recognized",
        candidates=[{"label": label, "confidence": confidence} for label, confidence in candidates],
    )


def discard_expired_sequences() -> None:
    expiry = monotonic() - SEQUENCE_IDLE_SECONDS
    for sequence_id, sequence in list(sequences.items()):
        if sequence.updated_at < expiry:
            sequence.close()
            del sequences[sequence_id]


@app.get("/")
def serve_index():
    return send_from_directory(DEMO_DIR, "index.html")


@app.get("/<path:asset_path>")
def serve_asset(asset_path: str):
    return send_from_directory(DEMO_DIR, asset_path)


@app.get("/api/vocabulary")
def prototype_vocabulary():
    """Expose the exact labels supported by the active mobile MVP model."""

    labels = sorted(refined_labels())
    return jsonify(label_count=len(labels), labels=labels)


@app.get("/api/model")
def active_model_status():
    return jsonify(
        model="Ishaara scoped ISL refinement",
        labels=refined_labels(),
        minimum_confidence=REFINED_MINIMUM_CONFIDENCE,
        metrics=refined_model_metrics(),
    )


@app.get("/api/health")
def health_check():
    """Small readiness response for the web app and review diagnostics."""

    return jsonify(
        status="ready",
        model_labels=len(refined_labels()),
        esp32_endpoint="/api/frames",
        esp32_raw_endpoint="/api/frames/raw",
    )


@app.post("/api/recognize")
def recognize_clip():
    clip = request.files.get("clip")
    if clip is None or not clip.filename:
        return jsonify(error="Choose a short signing video first."), 400

    temporary_path: Path | None = None
    try:
        candidates, temporary_path = classify_clip(clip, recognizer=recognize_refined)
        return recognition_response(candidates, minimum_confidence=REFINED_MINIMUM_CONFIDENCE)
    except ValueError as error:
        return jsonify(error=str(error)), 422
    except Exception:
        app.logger.exception("Recognition failed")
        return jsonify(error="The video could not be processed. Try a shorter, well-lit clip."), 500
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/api/mobile/recognize")
def recognize_mobile_clip():
    """Recognize trained vocabulary plus browser-owned personal gestures."""

    clip = request.files.get("clip")
    if clip is None or not clip.filename:
        return jsonify(error="Record a short sign with the mobile camera first."), 400

    temporary_path: Path | None = None
    try:
        temporary_path = save_temporary_clip(clip)
        with landmark_extraction_lock:
            feature_tensor = extract_feature_tensor(temporary_path)
        templates = personal_templates_from_request()
        if templates:
            personal_match = match_personal_sign(build_embedding(feature_tensor), templates)
            if personal_match is not None:
                label, confidence = personal_match
                return jsonify(
                    status="recognized",
                    source="personal",
                    candidates=[{"label": label, "confidence": confidence}],
                )
        candidates = recognize_feature_tensor(feature_tensor, top_k=3)
        return recognition_response(candidates, minimum_confidence=REFINED_MINIMUM_CONFIDENCE)
    except ValueError as error:
        return jsonify(error=str(error)), 422
    except Exception:
        app.logger.exception("Mobile recognition failed")
        return jsonify(error="The mobile recording could not be processed. Keep one signer, both hands, and upper body visible."), 500
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/api/personal-signs/embedding")
def create_personal_sign_embedding():
    """Convert one consented enrollment clip into a non-image landmark template."""

    clip = request.files.get("clip")
    if clip is None or not clip.filename:
        return jsonify(error="Record one example of the personal gesture first."), 400
    temporary_path: Path | None = None
    try:
        temporary_path = save_temporary_clip(clip)
        with landmark_extraction_lock:
            embedding = build_embedding(extract_feature_tensor(temporary_path))
        return jsonify(
            status="embedded",
            embedding=np.round(embedding, 4).tolist(),
            examples_required=EXAMPLES_PER_SIGN,
        )
    except ValueError as error:
        return jsonify(error=str(error)), 422
    except Exception:
        app.logger.exception("Personal sign enrollment failed")
        return jsonify(error="The gesture example could not be processed. Keep hands and upper body visible."), 500
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/api/training/captures")
def save_training_capture():
    """Store an explicitly labelled mobile recording for the ISL refinement dataset."""

    label = request.form.get("label", "").strip().lower()
    signer = request.form.get("signer", "").strip().lower()
    clip = request.files.get("clip")
    if not re.fullmatch(r"[a-z0-9][a-z0-9 -]{0,48}", label):
        return jsonify(error="Enter a short label using letters, numbers, spaces, or hyphens."), 400
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", signer):
        return jsonify(error="Enter a non-identifying signer code, such as signer-01."), 400
    if clip is None or not clip.filename:
        return jsonify(error="Record a labelled sign before saving it."), 400
    suffix = Path(clip.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        return jsonify(error="Save an MP4, MOV, WebM, M4V, or AVI recording."), 400

    FIELD_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    capture_id = uuid.uuid4().hex
    safe_label = re.sub(r"[^a-z0-9]+", "-", label).strip("-")
    filename = f"{safe_label}-{capture_id}{suffix}"
    target = FIELD_CAPTURE_DIR / filename
    clip.save(target)
    manifest_entry = {
        "id": capture_id,
        "label": label,
        "signer": signer,
        "file": filename,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": "mobile-camera",
    }
    with (FIELD_CAPTURE_DIR / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(manifest_entry) + "\n")
    return jsonify(status="saved", id=capture_id, label=label)


def store_device_result(device_id: str, payload: dict) -> dict:
    """Keep the newest finalized hardware result long enough for its phone to poll."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", device_id):
        return payload
    if device_id not in device_results and len(device_results) >= MAX_PAIRED_DEVICES:
        device_results.pop(next(iter(device_results)))
    result = {**payload, "event_id": uuid.uuid4().hex, "device_id": device_id}
    device_results[device_id] = result
    return result


def process_snapshot(sequence_id: str, image_bytes: bytes, mimetype: str, is_final: bool, device_id: str = ""):
    """Add one JPEG to an ESP32 sequence and optionally finalize inference."""
    discard_expired_sequences()
    if not sequence_id or len(sequence_id) > 64:
        return jsonify(error="Send a short sequence_id with every snapshot."), 400
    if not image_bytes:
        return jsonify(error="Send one JPEG snapshot in the frame field."), 400
    if mimetype not in {"image/jpeg", "image/jpg"}:
        return jsonify(error="ESP32 snapshots must use JPEG encoding."), 400

    sequence = sequences.get(sequence_id)
    if sequence is None:
        if len(sequences) >= MAX_ACTIVE_SEQUENCES:
            return jsonify(error="Too many active devices; retry shortly."), 429
        sequence = SnapshotSequence()
        sequences[sequence_id] = sequence

    try:
        with landmark_extraction_lock:
            detected = sequence.add_jpeg(image_bytes)
        if not is_final:
            return jsonify(
                status="collecting",
                frames_received=len(sequence.frames),
                detected=detected,
                pose_frames=sequence.pose_frames,
                signer_hand_frames=sequence.signer_hand_frames,
                minimum_frames=MINIMUM_FRAMES,
            ), 202

        candidates = sequence.recognize(top_k=3)
        if candidates[0][1] < MINIMUM_CONFIDENCE:
            payload = store_device_result(device_id, dict(
                status="no_confident_match",
                frames_received=len(sequence.frames),
                confidence=candidates[0][1],
                closest_label=candidates[0][0],
                pose_frames=sequence.pose_frames,
                signer_hand_frames=sequence.signer_hand_frames,
                error="No confident match in the current model vocabulary.",
            ))
            return jsonify(**payload)
        payload = store_device_result(device_id, dict(
            status="recognized",
            frames_received=len(sequence.frames),
            pose_frames=sequence.pose_frames,
            signer_hand_frames=sequence.signer_hand_frames,
            candidates=[{"label": label, "confidence": confidence} for label, confidence in candidates],
        ))
        return jsonify(**payload)
    except ValueError as error:
        return jsonify(error=str(error)), 422
    finally:
        if is_final:
            completed_sequence = sequences.pop(sequence_id, None)
            if completed_sequence is not None:
                completed_sequence.close()


@app.post("/api/frames")
def recognize_snapshot():
    """Receive one multipart ESP32 JPEG; final=true closes the sign sequence."""

    snapshot = request.files.get("frame")
    return process_snapshot(
        request.form.get("sequence_id", "").strip(),
        snapshot.read() if snapshot is not None else b"",
        snapshot.mimetype if snapshot is not None else "",
        request.form.get("final", "false").lower() == "true",
        request.form.get("device_id", "").strip(),
    )


@app.post("/api/frames/raw")
def recognize_raw_snapshot():
    """Receive an ESP32 frame buffer directly without multipart RAM overhead."""

    return process_snapshot(
        request.args.get("sequence_id", "").strip(),
        request.get_data(cache=False),
        request.mimetype,
        request.args.get("final", "false").lower() == "true",
        request.args.get("device_id", "").strip(),
    )


@app.get("/api/devices/<device_id>/latest")
def latest_device_result(device_id: str):
    """Return the newest finalized ESP32 result to the paired phone app."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", device_id):
        return jsonify(error="Use a device ID containing letters, numbers, underscores, or hyphens."), 400
    result = device_results.get(device_id)
    if result is None:
        return jsonify(status="waiting", device_id=device_id), 404
    return jsonify(**result)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Use a video smaller than 50 MB."), 413


if __name__ == "__main__":
    deployment_port = os.environ.get("PORT")
    host = os.environ.get("ISHAARA_HOST", "0.0.0.0" if deployment_port else "127.0.0.1")
    port = int(os.environ.get("ISHAARA_PORT", deployment_port or "4173"))
    app.run(host=host, port=port, debug=False)
