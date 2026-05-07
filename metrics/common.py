"""Common utilities for DiTalker metric scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LANDMARK_EXTS = {".npy", ".json"}


def list_files(root: str | Path, exts: Iterable[str]) -> list[Path]:
    root = Path(root)
    exts = {ext.lower() for ext in exts}
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No files with extensions {sorted(exts)} under {root}")
    return files


def paired_files(gt_dir: str | Path, generated_dir: str | Path, exts: Iterable[str]) -> list[tuple[Path, Path]]:
    gt_files = list_files(gt_dir, exts)
    generated_root = Path(generated_dir)
    pairs = []
    for gt_file in gt_files:
        rel = gt_file.relative_to(gt_dir) if Path(gt_dir).is_dir() else Path(gt_file.name)
        candidate = generated_root / rel
        if not candidate.exists():
            matches = list(generated_root.rglob(gt_file.name)) if generated_root.is_dir() else []
            if len(matches) == 1:
                candidate = matches[0]
            else:
                raise FileNotFoundError(f"Missing generated pair for {gt_file}: expected {candidate}")
        pairs.append((gt_file, candidate))
    return pairs


def read_video_rgb(path: str | Path, max_frames: int | None = None, size: tuple[int, int] | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if size is not None:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames read from video: {path}")
    return np.stack(frames, axis=0)


def load_landmarks(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        data = np.load(path)
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = np.asarray(json.load(f), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported landmark file: {path}")
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[-1] != 2:
        raise ValueError(f"Expected landmarks [T,K,2], got {data.shape} from {path}")
    return data


def save_result(path: str | Path | None, result: dict) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
