"""Convert region mask images/videos to binary black-white PNG masks.

The EasyAnimate-derived training loader expects static PNG masks under
`lip_mask`, `eye_masks`, and `face_mask`. This module uses OpenCV to convert
existing mask images or mask videos into 0/255 binary masks. For videos, masks
are accumulated with a per-pixel union over sampled frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np


REGION_DIRS = ("lip_mask", "eye_masks", "face_mask")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def mask_to_binary(mask: np.ndarray, threshold: int = 1, invert: bool = False) -> np.ndarray:
    """Convert a grayscale/RGB/BGR mask array to a uint8 binary mask.

    Any pixel with grayscale value greater than `threshold` becomes 255, and
    the rest becomes 0. This matches the black-background / white-foreground
    masks expected by the training loader.
    """

    if mask is None:
        raise ValueError("Input mask is None.")
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif mask.ndim == 2:
        gray = mask
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, int(threshold), 255, flag)
    return binary.astype(np.uint8)


def read_mask_image(path: str | Path, threshold: int = 1, invert: bool = False) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read mask image: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = np.where(alpha[:, :, None] > 0, image[:, :, :3], 0).astype(np.uint8)
    return mask_to_binary(image, threshold=threshold, invert=invert)


def union_mask_video(
    path: str | Path,
    threshold: int = 1,
    invert: bool = False,
    frame_stride: int = 1,
) -> np.ndarray:
    """Read a mask video and return the union of its binary foreground masks."""

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open mask video: {path}")
    frame_stride = max(1, int(frame_stride))
    union = None
    frame_index = 0
    used = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                binary = mask_to_binary(frame, threshold=threshold, invert=invert)
                union = binary if union is None else cv2.bitwise_or(union, binary)
                used += 1
            frame_index += 1
    finally:
        cap.release()
    if union is None or used == 0:
        raise ValueError(f"No frames were read from mask video: {path}")
    return union


def convert_mask_file(
    source: str | Path,
    target: str | Path,
    threshold: int = 1,
    invert: bool = False,
    frame_stride: int = 1,
) -> Path:
    """Convert one mask image/video to a binary PNG."""

    source = Path(source)
    target = Path(target)
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTS:
        binary = read_mask_image(source, threshold=threshold, invert=invert)
    elif suffix in VIDEO_EXTS:
        binary = union_mask_video(source, threshold=threshold, invert=invert, frame_stride=frame_stride)
    else:
        raise ValueError(f"Unsupported mask file extension: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), binary):
        raise OSError(f"Failed to write binary mask: {target}")
    return target


def iter_mask_files(folder: str | Path) -> Iterable[Path]:
    folder = Path(folder)
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS):
            yield path


def convert_mask_tree(
    input_root: str | Path,
    output_root: str | Path,
    threshold: int = 1,
    invert: bool = False,
    frame_stride: int = 1,
) -> list[Path]:
    """Convert `lip_mask`, `eye_masks`, and `face_mask` folders to binary PNGs."""

    input_root = Path(input_root)
    output_root = Path(output_root)
    written: list[Path] = []
    for region in REGION_DIRS:
        region_dir = input_root / region
        if not region_dir.exists():
            continue
        for source in iter_mask_files(region_dir):
            target = output_root / region / f"{source.stem}.png"
            written.append(
                convert_mask_file(
                    source,
                    target,
                    threshold=threshold,
                    invert=invert,
                    frame_stride=frame_stride,
                )
            )
    return written


def convert_region_set(
    sources: Mapping[str, str | Path],
    output_root: str | Path,
    stem: str,
    threshold: int = 1,
    invert: bool = False,
    frame_stride: int = 1,
) -> list[Path]:
    """Convert one video's lip/eye/face source masks into loader-ready PNGs."""

    output_root = Path(output_root)
    written: list[Path] = []
    for region in REGION_DIRS:
        if region not in sources or not sources[region]:
            raise ValueError(f"Missing source for region: {region}")
        target = output_root / region / f"{stem}.png"
        written.append(
            convert_mask_file(
                sources[region],
                target,
                threshold=threshold,
                invert=invert,
                frame_stride=frame_stride,
            )
        )
    return written
