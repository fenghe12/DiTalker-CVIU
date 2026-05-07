#!/usr/bin/env python3
"""Extract Whisper audio embeddings and save DiTalker `.pt` feature files.

The output tensor is shaped `[num_video_frames, 50, 384]`, matching the
`AudioProjection` input used by DiTalker. The slicing logic follows the
MuseTalk-style Whisper feature processor: Whisper features are treated as 50 FPS
features and each 25 FPS video frame receives a 5-frame temporal audio window
(2 frames before, current frame, and 2 frames after; each step contributes two
Whisper feature rows, giving 50 x 384 after concatenation).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


class Audio2Feature:
    def __init__(self, whisper_model_type: str = "tiny", model_path: str | None = None, whisper_root: str | None = None):
        if whisper_root:
            sys.path.insert(0, str(Path(whisper_root).resolve()))
        whisper_module = importlib.import_module("whisper")
        load_model = getattr(whisper_module, "load_model")
        self.model = load_model(model_path or whisper_model_type)

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def audio2feat(self, audio_path: str | Path) -> np.ndarray:
        result = self.model.transcribe(str(audio_path))
        embed_list = []
        for segment in result.get("segments", []):
            if "encoder_embeddings" not in segment:
                raise KeyError(
                    "Whisper result does not contain `encoder_embeddings`. "
                    "Use a MuseTalk/ACTalker-style Whisper implementation that exposes encoder embeddings."
                )
            emb = self._to_numpy(segment["encoder_embeddings"])
            if emb.ndim == 4:
                emb = emb.transpose(0, 2, 1, 3).squeeze(0)
            if emb.ndim != 3 or emb.shape[-1] != 384:
                raise ValueError(f"Expected encoder embedding shape [T, *, 384], got {emb.shape}")
            start_idx = int(segment.get("start", 0))
            end_idx = int(segment.get("end", start_idx + emb.shape[0] * 2))
            emb_end_idx = max(1, int((end_idx - start_idx) / 2))
            embed_list.append(emb[:emb_end_idx])
        if not embed_list:
            raise ValueError(f"No encoder embeddings extracted from {audio_path}")
        return np.concatenate(embed_list, axis=0).astype(np.float32)

    @staticmethod
    def get_sliced_feature(feature_array: np.ndarray, vid_idx: int, audio_feat_length=(2, 2), fps: int = 25):
        length = len(feature_array)
        selected_feature = []
        selected_idx = []
        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2
        for idx in range(left_idx, right_idx):
            idx = max(0, min(length - 1, idx))
            selected_feature.append(feature_array[idx])
            selected_idx.append(idx)
        selected_feature = np.concatenate(selected_feature, axis=0).reshape(-1, 384)
        return selected_feature.astype(np.float32), selected_idx

    def feature2chunks(self, feature_array: np.ndarray, fps: int = 25, num_video_frames: int | None = None, audio_feat_length=(2, 2)) -> np.ndarray:
        chunks = []
        i = 0
        while True:
            start_idx = int(i * 50.0 / fps)
            if num_video_frames is not None and i >= num_video_frames:
                break
            if num_video_frames is None and start_idx > len(feature_array):
                break
            chunk, _ = self.get_sliced_feature(feature_array, i, audio_feat_length=audio_feat_length, fps=fps)
            chunks.append(chunk)
            i += 1
        if not chunks:
            raise ValueError("No audio chunks were generated.")
        return np.stack(chunks, axis=0).astype(np.float32)


def iter_audio_files(path: Path):
    if path.is_file():
        yield path
        return
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix.lower() in AUDIO_EXTS:
            yield file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DiTalker Whisper audio embedding .pt files.")
    parser.add_argument("--audio", required=True, help="Audio file or directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to save .pt files.")
    parser.add_argument("--whisper_root", default=None, help="Path containing a MuseTalk/ACTalker-style `whisper` package.")
    parser.add_argument("--model_path", default=None, help="Path to Whisper checkpoint, e.g. whisper_tiny.pt.")
    parser.add_argument("--whisper_model_type", default="tiny")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--num_video_frames", type=int, default=None, help="Optional fixed number of output video-frame chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = Audio2Feature(args.whisper_model_type, args.model_path, args.whisper_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for audio_path in iter_audio_files(Path(args.audio)):
        features = processor.audio2feat(audio_path)
        chunks = processor.feature2chunks(features, fps=args.fps, num_video_frames=args.num_video_frames)
        output_path = output_dir / f"{audio_path.stem}.pt"
        torch.save(torch.from_numpy(chunks), output_path)
        print(f"{audio_path} -> {output_path} {tuple(chunks.shape)}")
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No audio files found under {args.audio}")


if __name__ == "__main__":
    main()
