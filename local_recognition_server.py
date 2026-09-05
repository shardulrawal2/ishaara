"""Local review server for Ishaara's real video-recognition pipeline.

The server is intentionally local-only. Uploaded clips are stored in a temporary
file just long enough to extract landmarks and run the ONNX model, then deleted.
"""

from __future__ import annotations

import tempfile
from time import monotonic
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from photo_stream_recognition import MINIMUM_CONFIDENCE, MINIMUM_FRAMES, SnapshotSequence
from recognize_video import load_labels, recognize


ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "review-demo"
ALLOWED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".webm"}
MAX_ACTIVE_SEQUENCES = 8
SEQUENCE_IDLE_SECONDS = 120
sequences: dict[str, SnapshotSequence] = {}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


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
    """Expose the exact labels supported by the current checkpoint."""

    labels = sorted(load_labels().values())
    return jsonify(label_count=len(labels), labels=labels)


@app.post("/api/recognize")
def recognize_clip():
    clip = request.files.get("clip")
    if clip is None or not clip.filename:
        return jsonify(error="Choose a short signing video first."), 400

    suffix = Path(clip.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        return jsonify(error="Use an MP4, MOV, WebM, M4V, or AVI video."), 400

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ishaara-upload-", suffix=suffix, delete=False) as file:
            temporary_path = Path(file.name)
            clip.save(file)

        candidates = recognize(temporary_path, top_k=3)
        return jsonify(
            candidates=[{"label": label, "confidence": confidence} for label, confidence in candidates]
        )
    except ValueError as error:
        return jsonify(error=str(error)), 422
    except Exception:
        app.logger.exception("Recognition failed")
        return jsonify(error="The video could not be processed. Try a shorter, well-lit clip."), 500
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    app.run(host="127.0.0.1", port=4173, debug=False)
