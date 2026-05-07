#!/usr/bin/env python3
"""Convert lip/eye/face region masks to black-white PNGs with OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing.region_masks import convert_mask_tree, convert_region_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DiTalker region masks to binary PNGs.")
    parser.add_argument("--input_root", default=None, help="Root containing lip_mask/eye_masks/face_mask source folders.")
    parser.add_argument("--output_root", required=True, help="Root where binary lip_mask/eye_masks/face_mask PNGs are written.")
    parser.add_argument("--lip_mask_source", default=None, help="Single lip mask image/video source.")
    parser.add_argument("--eye_mask_source", default=None, help="Single eye mask image/video source.")
    parser.add_argument("--face_mask_source", default=None, help="Single face mask image/video source.")
    parser.add_argument("--stem", default=None, help="Output filename stem for single-file mode, e.g. demo_0001.")
    parser.add_argument("--threshold", type=int, default=1, help="Foreground threshold. Default treats non-black pixels as foreground.")
    parser.add_argument("--invert", action="store_true", help="Use when foreground is black and background is white.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Frame stride when a source is a mask video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_root:
        written = convert_mask_tree(
            args.input_root,
            args.output_root,
            threshold=args.threshold,
            invert=args.invert,
            frame_stride=args.frame_stride,
        )
    else:
        if not args.stem:
            raise ValueError("--stem is required in single-file mode.")
        sources = {
            "lip_mask": args.lip_mask_source,
            "eye_masks": args.eye_mask_source,
            "face_mask": args.face_mask_source,
        }
        written = convert_region_set(
            sources,
            args.output_root,
            stem=args.stem,
            threshold=args.threshold,
            invert=args.invert,
            frame_stride=args.frame_stride,
        )
    if not written:
        raise RuntimeError("No masks were converted. Check input paths.")
    print(f"Saved {len(written)} binary mask file(s) under {Path(args.output_root)}")


if __name__ == "__main__":
    main()
