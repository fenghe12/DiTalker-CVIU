#!/usr/bin/env python3
"""Compute FID using mseitzer/pytorch-fid.

Reference implementation: https://github.com/mseitzer/pytorch-fid

Inputs are two directories containing GT and generated frames/images. If your
results are videos, first extract frames with the same sampling rule for both
GT and generated videos, then pass the two frame directories here.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from metrics.common import save_result


def build_command(gt_dir: str, generated_dir: str, batch_size: int, dims: int, device: str) -> list[str]:
    return [
        "python",
        "-m",
        "pytorch_fid",
        gt_dir,
        generated_dir,
        "--batch-size",
        str(batch_size),
        "--dims",
        str(dims),
        "--device",
        device,
    ]


def parse_fid_output(output: str) -> float:
    for line in output.splitlines():
        if "FID:" in line:
            return float(line.split("FID:", 1)[1].strip())
    raise RuntimeError("Could not parse FID from pytorch-fid output.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID with mseitzer/pytorch-fid.")
    parser.add_argument("--gt_dir", required=True, help="Directory of GT frames/images.")
    parser.add_argument("--generated_dir", required=True, help="Directory of generated frames/images.")
    parser.add_argument("--output_json", default=None, help="Optional path to save {'fid': value}.")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = build_command(args.gt_dir, args.generated_dir, args.batch_size, args.dims, args.device)
    if args.dry_run:
        print(" ".join(cmd))
        return
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    value = parse_fid_output(proc.stdout + "\n" + proc.stderr)
    save_result(args.output_json, {"fid": value, "implementation": "https://github.com/mseitzer/pytorch-fid"})


if __name__ == "__main__":
    main()
