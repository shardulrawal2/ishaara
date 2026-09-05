"""Fetch a tiny, labelled held-out control set from the official INCLUDE archive.

The public dataset is packaged as multi-gigabyte category ZIP files. This tool
uses HTTP byte ranges to download only the requested ZIP members, not the full
archives. The selected clips are entries from INCLUDE's supplied test split.

Usage:
    INCLUDE\\.venv\\Scripts\\python.exe scripts\\fetch_include_control_clips.py
"""

from __future__ import annotations

import json
import struct
import urllib.request
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "include-control"
RECORD_URL = "https://zenodo.org/api/records/4010759"

# Exact paths are listed in INCLUDE/train_test_paths/include_test.txt.
CONTROL_CLIPS = (
    "People/80. Adult/MVI_4119.MOV",
    "People/80. Adult/MVI_3823.MOV",
    "Greetings/48. Hello/MVI_0029.MOV",
    "Greetings/48. Hello/MVI_9914.MOV",
    "Clothes/40. Skirt/MVI_3700.MOV",
    "Clothes/40. Skirt/MVI_3997.MOV",
    "Electronics/54. Cell phone/MVI_4539.MOV",
    "Electronics/54. Cell phone/MVI_5391.MOV",
    "Home/37. Book/MVI_4399.MOV",
    "Home/37. Book/MVI_8805.MP4",
    "Jobs/84. Teacher/MVI_5313.MOV",
    "Jobs/84. Teacher/MVI_8866.MP4",
    "People/60. Mother/MVI_3906.MOV",
    "People/60. Mother/MVI_3907.MOV",
)


def get_bytes(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {} if start is None else {"Range": f"bytes={start}-{'' if end is None else end}"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        if start is not None and response.status != 206:
            raise RuntimeError(f"Server did not honour the byte-range request for {url}.")
        return data


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def central_directory(url: str) -> bytes:
    """Read only the normal ZIP central directory from a remote archive."""

    size = content_length(url)
    tail_start = max(0, size - 65_557)
    tail = get_bytes(url, tail_start, size - 1)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise RuntimeError("ZIP end-of-directory record was not found.")
    _, _, _, _, _, directory_size, directory_offset, _ = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
    return get_bytes(url, directory_offset, directory_offset + directory_size - 1)


def archive_entries(directory: bytes) -> dict[str, tuple[int, int, int]]:
    """Return name -> (compression method, compressed size, local header offset)."""

    entries: dict[str, tuple[int, int, int]] = {}
    offset = 0
    while offset < len(directory):
        if directory[offset : offset + 4] != b"PK\x01\x02":
            raise RuntimeError("Unexpected ZIP central-directory entry.")
        fields = struct.unpack_from("<4s6H3L5H2L", directory, offset)
        compression = fields[4]
        compressed_size = fields[8]
        name_length, extra_length, comment_length = fields[10:13]
        local_header_offset = fields[16]
        name_start = offset + 46
        name = directory[name_start : name_start + name_length].decode("utf-8")
        entries[name] = (compression, compressed_size, local_header_offset)
        offset = name_start + name_length + extra_length + comment_length
    return entries


def download_member(url: str, entry: tuple[int, int, int]) -> bytes:
    compression, compressed_size, local_offset = entry
    header = get_bytes(url, local_offset, local_offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError("Unexpected ZIP local-file header.")
    name_length, extra_length = fields[9:11]
    data_start = local_offset + 30 + name_length + extra_length
    compressed = get_bytes(url, data_start, data_start + compressed_size - 1)
    if compression == 0:
        return compressed
    if compression == 8:
        return zlib.decompress(compressed, -zlib.MAX_WBITS)
    raise RuntimeError(f"Unsupported ZIP compression method: {compression}")


def main() -> None:
    record = json.loads(get_bytes(RECORD_URL))
    archives = {file["key"]: file["links"]["self"] for file in record["files"] if file["key"].endswith(".zip")}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    remaining = {path for path in CONTROL_CLIPS if not (DESTINATION / path).is_file()}
    if not remaining:
        print(f"All {len(CONTROL_CLIPS)} labelled test clips are already present in {DESTINATION}")
        return

    for archive_name, url in archives.items():
        category = archive_name.split("_", 1)[0]
        relevant = [path for path in remaining if path.startswith(f"{category}/")]
        if not relevant:
            continue
        print(f"Inspecting {archive_name}…")
        entries = archive_entries(central_directory(url))
        for requested_path in relevant:
            matching = [name for name in entries if name.endswith(requested_path)]
            if not matching:
                continue
            target = DESTINATION / requested_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                print(f"Downloading {requested_path}…")
                target.write_bytes(download_member(url, entries[matching[0]]))
            remaining.remove(requested_path)

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise RuntimeError(f"Could not locate requested control clips: {missing}")
    print(f"Downloaded {len(CONTROL_CLIPS)} labelled test clips to {DESTINATION}")


if __name__ == "__main__":
    main()
