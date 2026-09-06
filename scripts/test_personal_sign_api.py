"""Smoke-test personal enrollment and personalized recognition over Flask."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_recognition_server import app  # noqa: E402


def main() -> None:
    captures = ROOT / "data" / "isl-field-captures"
    entries = [json.loads(line) for line in (captures / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    clip = next(captures / entry["file"] for entry in entries if (captures / entry["file"]).is_file())
    client = app.test_client()

    enrollment = client.post(
        "/api/personal-signs/embedding",
        data={"clip": (io.BytesIO(clip.read_bytes()), clip.name)},
        content_type="multipart/form-data",
    )
    assert enrollment.status_code == 200, enrollment.get_json()
    embedding = enrollment.get_json()["embedding"]
    templates = {"Demo Name": [embedding, embedding, embedding]}

    recognition = client.post(
        "/api/mobile/recognize",
        data={
            "clip": (io.BytesIO(clip.read_bytes()), clip.name),
            "personal_signs": json.dumps(templates),
        },
        content_type="multipart/form-data",
    )
    payload = recognition.get_json()
    assert recognition.status_code == 200, payload
    assert payload["status"] == "recognized", payload
    assert payload["source"] == "personal", payload
    assert payload["candidates"][0]["label"] == "Demo Name", payload
    print("Personal enrollment and recognition API smoke test passed.")


if __name__ == "__main__":
    main()
