#!/usr/bin/env python3
"""Prepare DiTalker training metadata from EasyAnimate-style caption metadata.

The core mapping follows EasyAnimate `easyanimate/video_caption/filter_meta_train.py`:
`video_path -> file_path`, `caption -> text`, and `type = video`.
Optional DiTalker condition fields are preserved when already present in the input:
`audio_emb_path`, `phoneme_dir`, `3dmm_dir`, and `pose_video_path`.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


OPTIONAL_FIELDS = ("audio_emb_path", "phoneme_dir", "3dmm_dir", "pose_video_path")


def load_records(path: str) -> List[Dict[str, object]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON metadata must be a list of records.")
        return data
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError("Input metadata must be .json, .jsonl, or .csv")


def convert_records(
    records: Iterable[Dict[str, object]],
    video_folder: str,
    video_path_column: str,
    caption_column: str,
) -> List[Dict[str, object]]:
    converted = []
    for record in records:
        if video_path_column not in record:
            raise KeyError(f"Missing video path column: {video_path_column}")
        if caption_column not in record:
            raise KeyError(f"Missing caption column: {caption_column}")

        file_path = str(record[video_path_column])
        if video_folder:
            file_path = os.path.join(video_folder, file_path)

        item = {
            "file_path": file_path,
            "text": str(record[caption_column]),
            "type": "video",
        }
        for field in OPTIONAL_FIELDS:
            if field in record and record[field] not in (None, ""):
                item[field] = record[field]
        converted.append(item)
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DiTalker training metadata JSON.")
    parser.add_argument("--caption_metadata_path", required=True, help="Input EasyAnimate-style .json/.jsonl/.csv metadata.")
    parser.add_argument("--saved_path", required=True, help="Output DiTalker metadata JSON path.")
    parser.add_argument("--video_folder", default="", help="Optional prefix for relative video paths.")
    parser.add_argument("--video_path_column", default="video_path", help="Input column containing video paths.")
    parser.add_argument("--caption_column", default="caption", help="Input column containing captions/prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.caption_metadata_path)
    converted = convert_records(records, args.video_folder, args.video_path_column, args.caption_column)
    output_path = Path(args.saved_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved {len(converted)} records to {output_path}")


if __name__ == "__main__":
    main()
