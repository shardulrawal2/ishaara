"""Create clearly named local demo copies from the downloaded INCLUDE controls."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT / "data" / "include-control"
DEMO_ROOT = ROOT / "data" / "demo-clips"

DEMO_CLIPS = {
    "adult-1.mov": "People/80. Adult/MVI_4119.MOV",
    "hello-1.mov": "Greetings/48. Hello/MVI_9914.MOV",
    "hello-2.mov": "Greetings/48. Hello/MVI_0029.MOV",
    "skirt-1.mov": "Clothes/40. Skirt/MVI_3700.MOV",
    "skirt-2.mov": "Clothes/40. Skirt/MVI_3997.MOV",
    "book-1.mov": "Home/37. Book/MVI_4399.MOV",
    "teacher-1.mov": "Jobs/84. Teacher/MVI_5313.MOV",
    "teacher-2.mp4": "Jobs/84. Teacher/MVI_8866.MP4",
    "mother-1.mov": "People/60. Mother/MVI_3906.MOV",
    "mother-2.mov": "People/60. Mother/MVI_3907.MOV",
}


def main() -> None:
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    for demo_name, source_path in DEMO_CLIPS.items():
        source = CONTROL_ROOT / source_path
        if not source.is_file():
            raise FileNotFoundError(f"Missing control clip: {source}")
        destination = DEMO_ROOT / demo_name
        if not destination.is_file():
            shutil.copy2(source, destination)
    print(f"Prepared {len(DEMO_CLIPS)} clearly named clips in {DEMO_ROOT}")


if __name__ == "__main__":
    main()
