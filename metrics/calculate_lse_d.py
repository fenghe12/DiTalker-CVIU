#!/usr/bin/env python3
"""Compute LSE-D from paired GT/generated landmark files."""

from __future__ import annotations

import argparse

from metrics.common import LANDMARK_EXTS, load_landmarks, paired_files, save_result
from metrics.landmark_distance import LIP_68, landmark_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LSE-D from paired landmarks.")
    parser.add_argument("--gt_dir", required=True, help="Directory of GT landmark .npy/.json files.")
    parser.add_argument("--generated_dir", required=True, help="Directory of generated landmark .npy/.json files.")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = []
    for gt_path, gen_path in paired_files(args.gt_dir, args.generated_dir, LANDMARK_EXTS):
        score = landmark_distance(load_landmarks(gt_path), load_landmarks(gen_path), indices=LIP_68)
        scores.append(score)
        print(f"{gt_path.name}: {score:.6f}")
    result = {"lse_d": float(sum(scores) / len(scores)), "num_pairs": len(scores)}
    print(f"LSE-D: {result['lse_d']:.6f}")
    save_result(args.output_json, result)


if __name__ == "__main__":
    main()
