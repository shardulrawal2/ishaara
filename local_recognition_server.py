"""Local review server for Ishaara's real video-recognition pipeline.

The server is intentionally local-only. Uploaded clips are stored in a temporary
file just long enough to extract landmarks and run the ONNX model, then deleted.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from recognize_video import recognize


ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "review-demo"
ALLOWED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.get("/")
def serve_index():
    return send_from_directory(DEMO_DIR, "index.html")


@app.get("/<path:asset_path>")
def serve_asset(asset_path: str):
    return send_from_directory(DEMO_DIR, asset_path)


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


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Use a video smaller than 50 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4173, debug=False)
