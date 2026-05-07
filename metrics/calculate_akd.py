#!/usr/bin/env python3
"""Compute AKD from paired GT/generated keypoint files."""

from __future__ import annotations

import argparse

from metrics.common import LANDMARK_EXTS, load_landmarks, paired_files, save_result
from metrics.landmark_distance import landmark_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute AKD from paired keypoints.")
    parser.add_argument("--gt_dir", required=True, help="Directory of GT keypoint .npy/.json files.")
    parser.add_argument("--generated_dir", required=True, help="Directory of generated keypoint .npy/.json files.")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = []
    for gt_path, gen_path in paired_files(args.gt_dir, args.generated_dir, LANDMARK_EXTS):
        score = landmark_distance(load_landmarks(gt_path), load_landmarks(gen_path))
        scores.append(score)
        print(f"{gt_path.name}: {score:.6f}")
    result = {"akd": float(sum(scores) / len(scores)), "num_pairs": len(scores)}
    print(f"AKD: {result['akd']:.6f}")
    save_result(args.output_json, result)


if __name__ == "__main__":
    main()
