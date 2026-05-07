#!/usr/bin/env python3
"""Predict DiTalker expression prompts from style videos with PLLaVA.

The paper uses an MLLM (PLLaVA) to predict an expression label from style frames
and then forms the prompt: "This person is [expression] and talks".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


EXPRESSIONS = ("happy", "sad", "angry", "disgusted", "surprised", "fearful", "neutral")
SYSTEM = """You are a powerful Video Magic ChatBot, a large vision-language assistant.
You are able to understand the video content that the user provides and assist the user in a video-language related task.
Follow the user's instruction and answer concisely.
"""
QUESTION = (
    "Please classify the person's facial expression in this video. "
    "Return exactly one label from: happy, sad, angry, disgusted, surprised, fearful, neutral."
)


def normalize_expression(answer: str) -> str:
    text = answer.lower()
    for label in EXPRESSIONS:
        if re.search(rf"\b{label}\b", text):
            return label
    return "neutral"


def load_chat(args: argparse.Namespace):
    sys.path.insert(0, str(Path(args.pllava_root).resolve()))
    from tasks.eval.eval_utils import ChatPllava, conv_plain_v1, conv_templates
    from tasks.eval.model_utils import load_pllava

    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        args.num_frames,
        use_lora=args.use_lora,
        weight_dir=args.weight_dir,
        lora_alpha=args.lora_alpha,
        use_multi_gpus=args.use_multi_gpus,
    )
    if not args.use_multi_gpus:
        model = model.to("cuda")
    conv_template = conv_templates.get(args.conv_mode, conv_plain_v1) if args.conv_mode else conv_plain_v1
    return ChatPllava(model, processor), conv_template


def predict_one(chat, conv_template, video_path: Path, num_segments: int | None, num_beams: int, temperature: float) -> dict:
    chat_state = conv_template.copy()
    img_list = []
    _, img_list, chat_state = chat.upload_video(str(video_path), chat_state, img_list, num_segments)
    chat_state = chat.ask(QUESTION, chat_state, SYSTEM)
    answer, _, chat_state = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=64,
        num_beams=num_beams,
        temperature=temperature,
    )
    expression = normalize_expression(answer.replace("<s>", ""))
    return {
        "video_path": str(video_path),
        "expression": expression,
        "prompt": f"This person is {expression} and talks",
        "raw_answer": answer,
    }


def iter_videos(path: Path):
    if path.is_file():
        yield path
        return
    for file in sorted(path.rglob("*.mp4")):
        yield file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict DiTalker expression prompts with PLLaVA.")
    parser.add_argument("--pllava_root", required=True, help="Local checkout of https://github.com/magic-research/PLLaVA")
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--style_video", required=True, help="Style video file or directory.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--num_segments", type=int, default=None)
    parser.add_argument("--conv_mode", default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_multi_gpus", action="store_true")
    parser.add_argument("--weight_dir", default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chat, conv_template = load_chat(args)
    results = [
        predict_one(chat, conv_template, video_path, args.num_segments, args.num_beams, args.temperature)
        for video_path in iter_videos(Path(args.style_video))
    ]
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved {len(results)} PLLaVA expression predictions to {output_path}")


if __name__ == "__main__":
    main()
