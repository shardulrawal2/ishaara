"""Local review server for Ishaara's real video-recognition pipeline.

The server is intentionally local-only. Uploaded clips are stored in a temporary
file just long enough to extract landmarks and run the ONNX model, then deleted.
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

from flask import Flask, jsonify, request, send_from_directory

from photo_stream_recognition import MINIMUM_CONFIDENCE, MINIMUM_FRAMES, SnapshotSequence
from recognize_refined import (
    MINIMUM_CONFIDENCE as REFINED_MINIMUM_CONFIDENCE,
    labels as refined_labels,
    model_metrics as refined_model_metrics,
    recognize as recognize_refined,
)


ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "review-demo"
FIELD_CAPTURE_DIR = ROOT / "data" / "isl-field-captures"
ALLOWED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".webm"}
MAX_ACTIVE_SEQUENCES = 8
SEQUENCE_IDLE_SECONDS = 120
sequences: dict[str, SnapshotSequence] = {}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.after_request
def disable_development_cache(response):
    """Keep rapidly-changing local app assets and model metadata fresh."""

    response.headers["Cache-Control"] = "no-store"
    return response


def classify_clip(clip, recognizer) -> tuple[list[tuple[str, float]], Path | None]:
    """Run a temporary mobile or uploaded video through the native video path."""

    suffix = Path(clip.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Use an MP4, MOV, WebM, M4V, or AVI video.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ishaara-upload-", suffix=suffix, delete=False) as file:
            temporary_path = Path(file.name)
            clip.save(file)
        return recognizer(temporary_path, top_k=3), temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


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

    return jsonify(status="ready", model_labels=len(refined_labels()), esp32_endpoint="/api/frames")


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
    """Mobile-camera backup: record locally, then use the model's video-native path."""

    clip = request.files.get("clip")
    if clip is None or not clip.filename:
        return jsonify(error="Record a short sign with the mobile camera first."), 400

    temporary_path: Path | None = None
    try:
        candidates, temporary_path = classify_clip(clip, recognizer=recognize_refined)
        return recognition_response(candidates, minimum_confidence=REFINED_MINIMUM_CONFIDENCE)
    except ValueError as error:
        return jsonify(error=str(error)), 422
    except Exception:
        app.logger.exception("Mobile recognition failed")
        return jsonify(error="The mobile recording could not be processed. Keep one signer, both hands, and upper body visible."), 500
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


@app.post("/api/frames")
def recognize_snapshot():
    """Receive one ESP32 JPEG; set final=true on the final image of one sign."""

    discard_expired_sequences()
    sequence_id = request.form.get("sequence_id", "").strip()
    snapshot = request.files.get("frame")
    if not sequence_id or len(sequence_id) > 64:
        return jsonify(error="Send a short sequence_id with every snapshot."), 400
    if snapshot is None or not snapshot.filename:
        return jsonify(error="Send one JPEG snapshot in the frame field."), 400
    if snapshot.mimetype not in {"image/jpeg", "image/jpg"}:
        return jsonify(error="ESP32 snapshots must use JPEG encoding."), 400

    sequence = sequences.get(sequence_id)
    if sequence is None:
        if len(sequences) >= MAX_ACTIVE_SEQUENCES:
            return jsonify(error="Too many active devices; retry shortly."), 429
        sequence = SnapshotSequence()
        sequences[sequence_id] = sequence

    try:
        detected = sequence.add_jpeg(snapshot.read())
        is_final = request.form.get("final", "false").lower() == "true"
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
            return jsonify(
                status="no_confident_match",
                frames_received=len(sequence.frames),
                confidence=candidates[0][1],
                closest_label=candidates[0][0],
                pose_frames=sequence.pose_frames,
                signer_hand_frames=sequence.signer_hand_frames,
                error="No confident match in the current model vocabulary.",
            )
        return jsonify(
            status="recognized",
            frames_received=len(sequence.frames),
            pose_frames=sequence.pose_frames,
            signer_hand_frames=sequence.signer_hand_frames,
            candidates=[{"label": label, "confidence": confidence} for label, confidence in candidates],
        )
    except ValueError as error:
        return jsonify(error=str(error)), 422
    finally:
        if request.form.get("final", "false").lower() == "true":
            completed_sequence = sequences.pop(sequence_id, None)
            if completed_sequence is not None:
                completed_sequence.close()


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Use a video smaller than 50 MB."), 413


if __name__ == "__main__":
    deployment_port = os.environ.get("PORT")
    host = os.environ.get("ISHAARA_HOST", "0.0.0.0" if deployment_port else "127.0.0.1")
    port = int(os.environ.get("ISHAARA_PORT", deployment_port or "4173"))
    app.run(host=host, port=port, debug=False)
