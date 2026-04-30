"""
DiTalker inference script.

This script keeps the current repository's actual inference path:
reference image + audio features + optional style / phoneme / pose conditions.
For the expression branch, the current codebase uses an explicit text prompt
at inference time rather than an MLLM predicted prompt from style frames.
"""
import argparse
import json
import os
import sys
from contextlib import nullcontext

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file
from scipy.io import loadmat
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5Tokenizer,
)

from ditalker.models.base.autoencoder_magvit import AutoencoderKLMagvit
from ditalker.models.base.transformer3d import Transformer3DModel
from ditalker.pipeline.pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline
from ditalker.utils.utils import get_image_to_video_latent, save_videos_grid


EXPRESSION_CHOICES = [
    "happy",
    "sad",
    "angry",
    "disgusted",
    "surprised",
    "fearful",
    "neutral",
    "surprise",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="DiTalker inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer.yaml",
        help="Path to inference config yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override transformer checkpoint path.",
    )
    parser.add_argument(
        "--reference_image",
        type=str,
        default=None,
        help="Path to the reference face image.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to whisper audio embedding (.pt/.npy), or a .wav path that can be resolved to an embedding file.",
    )
    parser.add_argument(
        "--phoneme_path",
        type=str,
        default=None,
        help="Path to phoneme labels (.json/.npy/.pt).",
    )
    parser.add_argument(
        "--style_clip_path",
        type=str,
        default=None,
        help="Path to 3DMM style coefficients (.mat/.txt/.npy).",
    )
    parser.add_argument(
        "--pose_path",
        type=str,
        default=None,
        help="Path to pose input (.npy/.pt or pose video).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output video path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Explicit expression prompt for the current inference implementation.",
    )
    parser.add_argument(
        "--expression",
        type=str,
        default=None,
        choices=EXPRESSION_CHOICES,
        help="Shortcut for building the prompt 'This person is [expression] and talks'.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt passed to the diffusion pipeline.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Number of output frames before VAE alignment.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Output resolution (square).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["bf16", "fp16", "fp32"],
        help="Inference dtype.",
    )
    return parser.parse_args()


def resolve_value(cli_value, config, key, default=None):
    if cli_value is not None:
        return cli_value
    value = config.get(key, default)
    return value


def normalize_expression(expression):
    if expression == "surprise":
        return "surprised"
    return expression


def build_prompt(prompt, expression):
    if prompt is not None:
        return prompt
    expression = normalize_expression(expression)
    if expression:
        return f"This person is {expression} and talks"
    return ""


def select_dtype(device, precision):
    if device != "cuda":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    return torch.bfloat16


def load_tensor_file(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".pt", ".pth", ".ckpt"}:
        tensor = torch.load(path, map_location="cpu")
    elif suffix == ".npy":
        tensor = torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported tensor file: {path}")

    if isinstance(tensor, dict):
        for key in ("audio_emb", "whisper_emb", "emb", "tensor"):
            if key in tensor and torch.is_tensor(tensor[key]):
                tensor = tensor[key]
                break
        else:
            raise ValueError(f"Unsupported tensor dict format: {path}")

    if not torch.is_tensor(tensor):
        raise ValueError(f"Failed to load tensor from: {path}")
    return tensor


def resolve_audio_feature_path(audio_path):
    if audio_path is None:
        raise ValueError("`audio_path` must be provided.")
    if os.path.isfile(audio_path) and os.path.splitext(audio_path)[1].lower() in {".pt", ".pth", ".ckpt", ".npy"}:
        return audio_path

    suffix = os.path.splitext(audio_path)[1].lower()
    if suffix != ".wav":
        raise FileNotFoundError(f"Audio feature file not found: {audio_path}")

    candidates = [
        audio_path.replace("audios", "whisper_audio_emb").replace(".wav", ".pt"),
        audio_path.replace("audios", "whisper_audio_emb").replace(".wav", ".npy"),
        audio_path.replace("audios", "whisper_embs").replace(".wav", ".pt"),
        audio_path.replace("audios", "whisper_embs").replace(".wav", ".npy"),
        os.path.splitext(audio_path)[0] + ".pt",
        os.path.splitext(audio_path)[0] + ".npy",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve a whisper embedding file from wav path: {audio_path}"
    )


def load_audio_features(audio_path, video_length, device, dtype):
    feature_path = resolve_audio_feature_path(audio_path)
    audio_features = load_tensor_file(feature_path).float()

    if audio_features.ndim == 2:
        audio_features = audio_features.unsqueeze(0)
    elif audio_features.ndim == 3:
        if audio_features.shape[0] != 1 and audio_features.shape[1] == 50:
            audio_features = audio_features.unsqueeze(0)
    elif audio_features.ndim != 4:
        raise ValueError(f"Unsupported audio feature shape: {tuple(audio_features.shape)}")

    if audio_features.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported for inference, got {tuple(audio_features.shape)}")
    if audio_features.shape[1] < video_length:
        raise ValueError(
            f"Audio feature length {audio_features.shape[1]} is shorter than requested video_length {video_length}."
        )

    audio_features = audio_features[:, :video_length].to(device=device, dtype=dtype)
    return audio_features, feature_path


def get_audio_window(audio, win_size=5):
    num_frames = len(audio)
    phoneme_frames = []
    for rid in range(num_frames):
        frame_window = []
        for idx in range(rid - win_size, rid + win_size + 1):
            if idx < 0 or idx >= num_frames:
                frame_window.append(74)
            else:
                frame_window.append(int(audio[idx]))
        phoneme_frames.append(frame_window)
    return np.array(phoneme_frames)


def load_phoneme_window(phoneme_path, video_length, device):
    if phoneme_path is None:
        return None

    suffix = os.path.splitext(phoneme_path)[1].lower()
    if suffix == ".json":
        with open(phoneme_path, "r") as f:
            phoneme = json.load(f)
        phoneme_win = torch.tensor(get_audio_window(phoneme, 5), dtype=torch.long)
    elif suffix in {".npy", ".pt", ".pth", ".ckpt"}:
        phoneme_win = load_tensor_file(phoneme_path).long()
        if phoneme_win.ndim == 1:
            phoneme_win = torch.tensor(get_audio_window(phoneme_win.tolist(), 5), dtype=torch.long)
    else:
        raise ValueError(f"Unsupported phoneme file: {phoneme_path}")

    if phoneme_win.ndim == 2:
        phoneme_win = phoneme_win.unsqueeze(0)
    if phoneme_win.ndim != 3:
        raise ValueError(f"Unsupported phoneme shape: {tuple(phoneme_win.shape)}")
    if phoneme_win.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported for phonemes, got {tuple(phoneme_win.shape)}")
    if phoneme_win.shape[1] < video_length:
        raise ValueError(
            f"Phoneme length {phoneme_win.shape[1]} is shorter than requested video_length {video_length}."
        )

    return phoneme_win[:, :video_length].to(device)


def load_style_clip(style_path, style_max_len=256):
    suffix = os.path.splitext(style_path)[1].lower()
    if suffix == ".mat":
        face3d_all = loadmat(style_path)["coeff"]
        face3d_exp = face3d_all[:, 80:144]
    elif suffix in {".txt", ".npy"}:
        face3d_exp = np.loadtxt(style_path) if suffix == ".txt" else np.load(style_path)
        if face3d_exp.ndim != 2:
            raise ValueError(f"Unsupported style clip shape: {face3d_exp.shape}")
        if face3d_exp.shape[1] >= 144:
            face3d_exp = face3d_exp[:, 80:144]
    else:
        raise ValueError(f"Unsupported style file: {style_path}")

    if face3d_exp.shape[1] != 64:
        raise ValueError(f"Style clip must have 64 expression coefficients, got {face3d_exp.shape}")

    style_clip = torch.tensor(face3d_exp, dtype=torch.float32)
    length = style_clip.shape[0]

    if length >= style_max_len:
        clip_start = (length - style_max_len) // 2
        style_clip = style_clip[clip_start : clip_start + style_max_len]
        pad_mask = torch.zeros(style_max_len, dtype=torch.bool)
    else:
        padding = torch.zeros(style_max_len - length, style_clip.shape[1], dtype=style_clip.dtype)
        style_clip = torch.cat((style_clip, padding), dim=0)
        pad_mask = torch.tensor([False] * length + [True] * (style_max_len - length), dtype=torch.bool)

    return style_clip.unsqueeze(0), pad_mask.unsqueeze(0)


def load_pose_input(pose_path, video_length, device, dtype):
    if pose_path is None:
        return None

    suffix = os.path.splitext(pose_path)[1].lower()
    if suffix in {".npy", ".pt", ".pth", ".ckpt"}:
        pose_tensor = load_tensor_file(pose_path).float()
    elif suffix in VIDEO_EXTENSIONS:
        from decord import VideoReader

        video_reader = VideoReader(pose_path)
        frame_count = min(len(video_reader), video_length)
        frames = video_reader.get_batch(list(range(frame_count))).asnumpy()
        pose_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    else:
        raise ValueError(f"Unsupported pose file: {pose_path}")

    if pose_tensor.ndim == 4:
        if pose_tensor.shape[0] == 3:
            pose_tensor = pose_tensor.unsqueeze(0)
        elif pose_tensor.shape[1] == 3:
            pose_tensor = pose_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported 4D pose shape: {tuple(pose_tensor.shape)}")
    elif pose_tensor.ndim == 5:
        if pose_tensor.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported for pose input, got {tuple(pose_tensor.shape)}")
        if pose_tensor.shape[1] == 3:
            pass
        elif pose_tensor.shape[2] == 3:
            pose_tensor = pose_tensor.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Unsupported 5D pose shape: {tuple(pose_tensor.shape)}")
    else:
        raise ValueError(f"Unsupported pose shape: {tuple(pose_tensor.shape)}")

    if pose_tensor.shape[2] < video_length:
        raise ValueError(
            f"Pose length {pose_tensor.shape[2]} is shorter than requested video_length {video_length}."
        )

    pose_tensor = pose_tensor[:, :, :video_length]
    if pose_tensor.shape[-2:] != (256, 256):
        pose_tensor = F.interpolate(
            pose_tensor,
            size=(pose_tensor.shape[2], 256, 256),
            mode="trilinear",
            align_corners=False,
        )
    return pose_tensor.to(device=device, dtype=dtype)


def validate_required_inputs(reference_image, audio_path, output_path):
    if not reference_image:
        raise ValueError("`reference_image` must be provided either in the config or on the command line.")
    if not os.path.isfile(reference_image):
        raise FileNotFoundError(f"Reference image not found: {reference_image}")
    if not audio_path:
        raise ValueError("`audio_path` must be provided either in the config or on the command line.")
    if not output_path:
        raise ValueError("`output_path` must be provided either in the config or on the command line.")


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = resolve_value(args.mixed_precision, config, "mixed_precision", "bf16")
    dtype = select_dtype(device, precision)

    pretrained_path = resolve_value(
        None, config, "pretrained_model_name_or_path", "models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512"
    )
    checkpoint_path = resolve_value(args.checkpoint, config, "transformer_checkpoint", None)
    transformer_kwargs = OmegaConf.to_container(
        config.get("transformer_additional_kwargs", {}), resolve=True
    )

    reference_image = resolve_value(args.reference_image, config, "reference_image", None)
    audio_path = resolve_value(args.audio_path, config, "audio_path", None)
    phoneme_path = resolve_value(args.phoneme_path, config, "phoneme_path", None)
    style_clip_path = resolve_value(args.style_clip_path, config, "style_clip_path", None)
    pose_path = resolve_value(args.pose_path, config, "pose_path", None)
    output_path = resolve_value(args.output_path, config, "output_path", None)
    prompt = build_prompt(
        resolve_value(args.prompt, config, "prompt", None),
        resolve_value(args.expression, config, "expression", None),
    )
    negative_prompt = resolve_value(args.negative_prompt, config, "negative_prompt", "") or ""
    num_inference_steps = int(resolve_value(args.num_inference_steps, config, "num_inference_steps", 25))
    guidance_scale = float(resolve_value(args.guidance_scale, config, "guidance_scale", 7.0))
    requested_video_length = int(resolve_value(args.video_length, config, "video_length", 16))
    resolution = int(resolve_value(args.resolution, config, "resolution", 512))
    seed = int(resolve_value(args.seed, config, "seed", 42))

    validate_required_inputs(reference_image, audio_path, output_path)

    print(f"Device: {device}, dtype: {dtype}")
    print("Loading VAE...")
    vae_kwargs = config.get("vae_kwargs", {"enable_magvit": True})
    vae_kwargs_dict = (
        vae_kwargs
        if isinstance(vae_kwargs, dict)
        else OmegaConf.to_container(vae_kwargs, resolve=True)
    )
    if vae_kwargs_dict.get("enable_magvit", True):
        vae = AutoencoderKLMagvit.from_pretrained(
            pretrained_path,
            subfolder="vae",
            vae_additional_kwargs=vae_kwargs_dict,
        ).to(device, dtype=dtype)
    else:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae").to(device, dtype=dtype)
    vae.eval()

    video_length = requested_video_length
    if video_length != 1 and hasattr(vae, "mini_batch_encoder"):
        aligned_video_length = int(video_length // vae.mini_batch_encoder * vae.mini_batch_encoder)
        if aligned_video_length == 0:
            raise ValueError(
                f"video_length {video_length} is smaller than VAE mini_batch_encoder {vae.mini_batch_encoder}."
            )
        if aligned_video_length != video_length:
            print(f"Aligning video_length from {video_length} to {aligned_video_length} for VAE temporal downsampling.")
            video_length = aligned_video_length

    print("Loading T5 text encoder...")
    tokenizer = T5Tokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder.eval()

    print("Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_path, subfolder="image_encoder"
    ).to(device, dtype=dtype)
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_path, subfolder="image_encoder")
    image_encoder.eval()

    print("Loading Transformer3DModel...")
    transformer = Transformer3DModel.from_pretrained_2d(
        pretrained_path,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_kwargs,
    ).to(dtype)

    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = (
            load_file(checkpoint_path)
            if checkpoint_path.endswith(".safetensors")
            else torch.load(checkpoint_path, map_location="cpu")
        )
        missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

    transformer = transformer.to(device)
    transformer.eval()

    print("Building inference pipeline...")
    pipeline = EasyAnimateInpaintPipeline.from_pretrained(
        pretrained_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        torch_dtype=dtype,
        clip_image_encoder=image_encoder,
        clip_image_processor=image_processor,
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    sample_size = [resolution, resolution]

    print(f"Preparing reference image: {reference_image}")
    input_video, input_video_mask, clip_image = get_image_to_video_latent(
        reference_image,
        None,
        video_length=video_length,
        sample_size=sample_size,
    )

    audio_features, audio_feature_path = load_audio_features(audio_path, video_length, device, dtype)
    print(f"Using audio features: {audio_feature_path}, shape={tuple(audio_features.shape)}")

    phoneme_win = load_phoneme_window(phoneme_path, video_length, device) if phoneme_path else None
    if phoneme_win is not None:
        print(f"Using phoneme window: {phoneme_path}, shape={tuple(phoneme_win.shape)}")

    style_clip = None
    pad_mask = None
    if style_clip_path:
        style_clip, pad_mask = load_style_clip(style_clip_path)
        style_clip = style_clip.to(device=device, dtype=dtype)
        pad_mask = pad_mask.to(device=device)
        print(f"Using style clip: {style_clip_path}, shape={tuple(style_clip.shape)}")

    pose_tensor = load_pose_input(pose_path, video_length, device, dtype) if pose_path else None
    if pose_tensor is not None:
        print(f"Using pose input: {pose_path}, shape={tuple(pose_tensor.shape)}")

    print(
        f'Running inference: prompt="{prompt}", video_length={video_length}, '
        f"guidance_scale={guidance_scale}, steps={num_inference_steps}"
    )
    autocast_context = (
        torch.autocast("cuda", dtype=dtype)
        if device == "cuda" and dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    )

    with torch.no_grad(), autocast_context:
        sample = pipeline(
            prompt=prompt,
            video_length=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            audio_encoder_hidden_states=audio_features,
            phoneme_win=phoneme_win,
            style_clip=style_clip,
            pad_mask=pad_mask,
            pixel_values_pose=pose_tensor,
        ).videos

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_videos_grid(sample, output_path)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
