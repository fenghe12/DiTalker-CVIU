#!/usr/bin/env python3
"""Sync-C evaluation note/wrapper.

DiTalker reports Sync-C using Bytedance LatentSync:
https://github.com/bytedance/LatentSync

Use LatentSync's `eval/eval_sync_conf.py` to compute SyncNet confidence. This
script only builds/runs that command; it does not reimplement or approximate
Sync-C locally.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from metrics.common import save_result


CONF_RE = re.compile(r"(?:SyncNet confidence:|average sync confidence is)\s*([-+]?\d*\.?\d+)", re.IGNORECASE)


def parse_sync_c(output: str) -> float:
    match = CONF_RE.search(output)
    if not match:
        raise RuntimeError("Could not parse Sync-C/SyncNet confidence from LatentSync output.")
    return float(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Sync-C using LatentSync eval_sync_conf.py.")
    parser.add_argument("--generated_video", default=None, help="Generated video path.")
    parser.add_argument("--generated_dir", default=None, help="Directory of generated videos.")
    parser.add_argument("--latentsync_root", required=True, help="Local checkout of https://github.com/bytedance/LatentSync")
    parser.add_argument("--initial_model", required=True, help="LatentSync SyncNet checkpoint path.")
    parser.add_argument("--temp_dir", default="temp")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.generated_video and not args.generated_dir:
        raise ValueError("Provide --generated_video or --generated_dir.")
    script = Path(args.latentsync_root) / "eval" / "eval_sync_conf.py"
    cmd = ["python", str(script), "--initial_model", args.initial_model, "--temp_dir", args.temp_dir]
    if args.generated_video:
        cmd.extend(["--video_path", args.generated_video])
    if args.generated_dir:
        cmd.extend(["--videos_dir", args.generated_dir])
    if args.dry_run:
        print(" ".join(cmd))
        return
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    value = parse_sync_c(proc.stdout + "\n" + proc.stderr)
    save_result(args.output_json, {"sync_c": value, "implementation": "https://github.com/bytedance/LatentSync"})


if __name__ == "__main__":
    main()
