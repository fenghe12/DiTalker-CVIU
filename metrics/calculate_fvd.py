#!/usr/bin/env python3
"""Compute FVD using songweige/content-debiased-fvd.

Reference implementation: https://github.com/songweige/content-debiased-fvd

This script is a thin wrapper because the paper numbers should come from that
implementation, not a reimplemented local approximation. Provide the path to the
content-debiased-fvd entry script in your local checkout.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from metrics.common import save_result


def parse_fvd_output(output: str) -> float:
    for raw in output.splitlines():
        line = raw.lower().replace("=", ":")
        if "fvd" in line and ":" in line:
            tail = line.rsplit(":", 1)[1].strip().split()[0]
            try:
                return float(tail)
            except ValueError:
                continue
    raise RuntimeError("Could not parse FVD from content-debiased-fvd output.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FVD with content-debiased-fvd.")
    parser.add_argument("--gt_dir", required=True, help="Directory of GT videos.")
    parser.add_argument("--generated_dir", required=True, help="Directory of generated videos.")
    parser.add_argument("--fvd_script", required=True, help="Path to the content-debiased-fvd evaluation script.")
    parser.add_argument("--output_json", default=None, help="Optional path to save {'fvd': value}.")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[], help="Arguments forwarded to the FVD script.")
    parser.add_argument("--dry_run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = ["python", args.fvd_script, "--gt_dir", args.gt_dir, "--generated_dir", args.generated_dir] + args.extra_args
    if args.dry_run:
        print(" ".join(cmd))
        return
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    value = parse_fvd_output(proc.stdout + "\n" + proc.stderr)
    save_result(args.output_json, {"fvd": value, "implementation": "https://github.com/songweige/content-debiased-fvd"})


if __name__ == "__main__":
    main()
