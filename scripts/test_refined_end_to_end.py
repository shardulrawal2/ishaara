"""Run the scoped model against original videos from the held-out signer split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recognize_refined import MINIMUM_CONFIDENCE, recognize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test refined inference on held-out source videos.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "isl-training-keypoints-v2")
    parser.add_argument("--captures", type=Path, default=ROOT / "data" / "isl-field-captures")
    parser.add_argument("--samples-per-label", type=int, default=2)
    args = parser.parse_args()

    test_ids = {path.stem for path in (args.data_dir / "ishaara_test_keypoints").glob("*.json")}
    entries = [json.loads(line) for line in (args.captures / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: dict[str, list[dict]] = {}
    for entry in entries:
        if entry["id"] not in test_ids:
            continue
        label = "".join(character for character in entry["label"].lower() if character.isalpha())
        bucket = selected.setdefault(label, [])
        if len(bucket) < args.samples_per_label and (args.captures / entry["file"]).is_file():
            bucket.append(entry)

    total = 0
    correct = 0
    accepted = 0
    accepted_correct = 0
    for expected, samples in sorted(selected.items()):
        for entry in samples:
            candidates = recognize(args.captures / entry["file"], top_k=2)
            label, confidence = candidates[0]
            is_accepted = confidence >= MINIMUM_CONFIDENCE
            total += 1
            correct += int(label == expected)
            accepted += int(is_accepted)
            accepted_correct += int(is_accepted and label == expected)
            decision = label if is_accepted else "no_confident_match"
            print(f"expected={expected:8s} decision={decision:18s} top={label:8s} confidence={confidence:.1%}")

    print(f"tested={total} raw_accuracy={correct / total:.1%}")
    print(f"accepted={accepted} accepted_accuracy={accepted_correct / accepted:.1%}" if accepted else "accepted=0")


if __name__ == "__main__":
    main()
