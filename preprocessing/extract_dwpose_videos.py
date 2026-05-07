#!/usr/bin/env python3
"""Run a local DWPose video extraction script for DiTalker preprocessing."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_DWPOSE_TOOL = os.environ.get(
    "DWPOSE_VIDEO_TOOL",
    "third_party/DWPose/tools/extract_dwpose_from_vid_tool.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DWPose videos for DiTalker.")
    parser.add_argument("--video_root", required=True, help="Root directory containing input .mp4 videos.")
    parser.add_argument("--save_dir", default=None, help="Output directory for DWPose videos.")
    parser.add_argument("-j", "--num_workers", type=int, default=1, help="Number of workers passed to the DWPose tool.")
    parser.add_argument(
        "--dwpose_tool",
        "--anigen_tool",
        dest="dwpose_tool",
        default=DEFAULT_DWPOSE_TOOL,
        help="Path to a local DWPose video extraction script. Can also be set with DWPOSE_VIDEO_TOOL.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print command without running it.")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = ["python", args.dwpose_tool, "--video_root", args.video_root, "-j", str(args.num_workers)]
    if args.save_dir:
        cmd.extend(["--save_dir", args.save_dir])
    return cmd


def main() -> None:
    args = parse_args()
    if not Path(args.dwpose_tool).exists():
        raise FileNotFoundError(
            f"DWPose extraction script not found: {args.dwpose_tool}. "
            "Pass --dwpose_tool or set DWPOSE_VIDEO_TOOL."
        )
    cmd = build_command(args)
    if args.dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
