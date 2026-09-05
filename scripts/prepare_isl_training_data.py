"""Turn consented mobile recordings into a signer-separated ISL landmark dataset.

The recordings created by the review app live outside git in
``data/isl-field-captures``.  This script uses the same MediaPipe extraction
and 169 x 134 preprocessing schema as the deployed checkpoint, so the model is
trained on the exact kind of phone footage it will receive.

Example:
    INCLUDE\\.venv\\Scripts\\python.exe scripts\\prepare_isl_training_data.py

Use at least three different signer codes per label.  The split is by signer,
not by recording, preventing the misleading result where the same person's
hands appear in both training and evaluation data.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "INCLUDE"))


def normalise_label(label: str) -> str:
    value = "".join(character for character in label.lower() if character.isalpha())
    if not value:
        raise ValueError(f"Label {label!r} contains no letters after normalisation.")
    return value


def read_manifest(captures: Path) -> list[dict[str, str]]:
    manifest = captures / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"No capture manifest found at {manifest}. Save labelled mobile recordings first.")

    entries = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            entry["label"] = normalise_label(entry["label"])
            entry["signer"] = str(entry["signer"]).lower()
            filename = Path(entry["file"]).name
            if filename != entry["file"]:
                raise ValueError("capture file must be a plain filename")
            entry["path"] = str(captures / filename)
            if not Path(entry["path"]).is_file():
                raise FileNotFoundError(filename)
            entries.append(entry)
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid capture manifest entry on line {line_number}: {error}") from error
    if not entries:
        raise ValueError("The capture manifest has no usable recordings.")
    return entries


def split_by_signer(entries: list[dict[str, str]]) -> dict[str, str]:
    """Assign each signer to one split, consistently across every label."""

    signers = sorted({entry["signer"] for entry in entries})
    if len(signers) < 3:
        raise ValueError("Need recordings from at least three signer codes for an honest train/validation/test split.")
    assignments = {}
    for index, signer in enumerate(signers):
        if index % 5 == 0:
            assignments[signer] = "test"
        elif index % 5 == 1:
            assignments[signer] = "val"
        else:
            assignments[signer] = "train"
    # With exactly three people, ensure every split has one signer.
    assignments[signers[0]] = "train"
    assignments[signers[1]] = "val"
    assignments[signers[2]] = "test"
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a signer-separated ISL landmark dataset.")
    parser.add_argument("--captures", type=Path, default=ROOT / "data" / "isl-field-captures")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "isl-training-keypoints")
    args = parser.parse_args()

    captures = args.captures.resolve()
    output = args.output.resolve()
    entries = read_manifest(captures)
    assignments = split_by_signer(entries)
    labels = sorted({entry["label"] for entry in entries})
    labels_by_signer: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        labels_by_signer[entry["label"]].add(entry["signer"])
    undersampled = [label for label, signers in labels_by_signer.items() if len(signers) < 3]
    if undersampled:
        raise ValueError("Each label needs three or more distinct signers; currently insufficient: " + ", ".join(sorted(undersampled)))

    from generate_keypoints import process_video

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output directory to preserve prior runs.")
    for split in ("train", "val", "test"):
        (output / f"ishaara_{split}_keypoints").mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="ishaara-field-staging-") as temp_dir:
        staging = Path(temp_dir)
        for index, entry in enumerate(entries, start=1):
            label_dir = staging / entry["label"]
            label_dir.mkdir(exist_ok=True)
            staged_video = label_dir / Path(entry["path"]).name
            shutil.copy2(entry["path"], staged_video)
            extracted_dir = staging / "extracted"
            extracted_dir.mkdir(exist_ok=True)
            process_video(staged_video.as_posix(), str(extracted_dir))
            generated = next(extracted_dir.glob("*.json"), None)
            if generated is None:
                raise RuntimeError(f"No landmarks were written for {staged_video.name}.")
            split = assignments[entry["signer"]]
            target = output / f"ishaara_{split}_keypoints" / f"{entry['id']}.json"
            generated.replace(target)
            print(f"[{index}/{len(entries)}] {entry['label']} · {entry['signer']} -> {split}")

    (output / "label_map_ishaara.json").write_text(json.dumps({label: i for i, label in enumerate(labels)}, indent=2), encoding="utf-8")
    (output / "split_by_signer.json").write_text(json.dumps(assignments, indent=2), encoding="utf-8")
    print(f"Prepared {len(entries)} recordings across {len(labels)} labels in {output}.")


if __name__ == "__main__":
    main()
