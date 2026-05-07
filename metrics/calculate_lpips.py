#!/usr/bin/env python3
"""Compute LPIPS between paired GT and generated videos."""

from __future__ import annotations

import argparse

import torch

from metrics.common import VIDEO_EXTS, paired_files, read_video_rgb, save_result


def video_to_tensor(video) -> torch.Tensor:
    # [T,H,W,C] uint8 -> [T,C,H,W] float in [-1, 1]
    tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return tensor


def compute_pair_lpips(gt_video, generated_video, model, device: str) -> float:
    frames = min(gt_video.shape[0], generated_video.shape[0])
    gt = video_to_tensor(gt_video[:frames]).to(device)
    gen = video_to_tensor(generated_video[:frames]).to(device)
    with torch.no_grad():
        score = model(gen, gt).mean()
    return float(score.detach().cpu())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LPIPS for paired videos.")
    parser.add_argument("--gt_dir", required=True, help="Directory of GT videos.")
    parser.add_argument("--generated_dir", required=True, help="Directory of generated videos with matching filenames.")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resize", type=int, default=None, help="Optional square resize before LPIPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import lpips  # type: ignore

    size = (args.resize, args.resize) if args.resize else None
    model = lpips.LPIPS(net=args.net).to(args.device).eval()
    pairs = paired_files(args.gt_dir, args.generated_dir, VIDEO_EXTS)
    scores = []
    for gt_path, gen_path in pairs:
        gt = read_video_rgb(gt_path, max_frames=args.max_frames, size=size)
        gen = read_video_rgb(gen_path, max_frames=args.max_frames, size=size)
        score = compute_pair_lpips(gt, gen, model, args.device)
        scores.append(score)
        print(f"{gt_path.name}: {score:.6f}")
    result = {"lpips": float(sum(scores) / len(scores)), "num_pairs": len(scores)}
    print(f"LPIPS: {result['lpips']:.6f}")
    save_result(args.output_json, result)


if __name__ == "__main__":
    main()
