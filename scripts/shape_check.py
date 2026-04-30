"""
DiTalker Shape Verification Script
Instantiates Transformer3DModel with all modules enabled and runs dummy forward pass.
Verifies: PoseAdapter, AudioProjection, StyleEncoder, ScaleAdapter, ASFM, ECA
"""
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import torch

from ditalker.models.base.transformer3d import Transformer3DModel


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Transformer config matching EasyAnimateV3-XL
    transformer_additional_kwargs = {
        'enable_audio_attention': True,
        'enable_style_attn': True,
        'enable_scale_fusion': True,
        'enable_audio_scale_fusion': True,
        'enable_eca': True,
        'enable_uvit': True,
        'basic_block_type': 'global_motionmodule',
        'enable_pose_adapter': True,
    }

    print('Loading Transformer3DModel from pretrained...')
    model = Transformer3DModel.from_pretrained_2d(
        'models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512',
        subfolder='transformer',
        transformer_additional_kwargs=transformer_additional_kwargs,
    ).to(device, dtype=dtype)
    model.eval()
    print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    # Dummy inputs
    B, C, F, H, W = 1, 4, 16, 64, 64  # 4 channels: noisy latent (inpaint_latents concatenated separately inside model)
    hidden_states = torch.randn(B, C, F, H, W, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=dtype)

    # T5 text encoder output: (B, seq_len=120, hidden_dim=1152)
    encoder_hidden_states = torch.randn(B, 120, 4096, device=device, dtype=dtype)  # T5 output dim is 4096, projected to 1152 inside model
    encoder_attention_mask = torch.ones(B, 120, device=device, dtype=dtype)

    # Audio whisper features: (B, F, 50, 384) where 50 is temporal window, 384 is whisper hidden dim
    audio_encoder_hidden_states = torch.randn(B, 64, 50, 384, device=device, dtype=dtype)  # (B, 64_video_frames, 50_temporal_window, 384_whisper_dim)

    # Style 3DMM clip: (B, max_len=256, 64) expression coefficients
    style_clip = torch.randn(B, 256, 64, device=device, dtype=dtype)
    # Phoneme window: (B, F, window=11) integer phoneme indices
    phoneme_win = torch.randint(0, 41, (B, 64, 11), device=device)  # (B, 64_video_frames, 11_window)
    # Padding mask for style clip: (B, 256) bool
    pad_mask = torch.zeros(B, 256, device=device, dtype=torch.bool)

    # DWPose keypoints at original resolution: (B, 3, F, 512, 512)
    pixel_values_pose = torch.randn(B, 3, F, 512, 512, device=device, dtype=dtype)

    # CLIP image embedding: (B, 768)
    clip_encoder_hidden_states = torch.randn(B, 768, device=device, dtype=dtype)
    clip_attention_mask = torch.ones(B, 8, device=device, dtype=dtype)

    # Inpaint latents: (B, 8, F, H, W) = concat(mask_4ch, masked_latent_4ch)
    inpaint_latents = torch.randn(B, 8, F, H, W, device=device, dtype=dtype)

    # Expression embedding c_emo: (B, seq_len, 1152)
    c_emo = torch.randn(B, 64, 1152, device=device, dtype=dtype)

    # Resolution / aspect ratio (not used for sample_size=64)
    added_cond_kwargs = {'resolution': None, 'aspect_ratio': None}

    print('Running forward pass with all modules enabled...')
    with torch.no_grad():
        output = model(
            hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            inpaint_latents=inpaint_latents,
            clip_encoder_hidden_states=clip_encoder_hidden_states,
            clip_attention_mask=clip_attention_mask,
            return_dict=False,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            phoneme_win=phoneme_win,
            style_clip=style_clip,
            pad_mask=pad_mask,
            pixel_values_pose=pixel_values_pose,
            c_emo=c_emo,
        )

    # Parse output
    if isinstance(output, tuple):
        model_output = output[0]
        repa_features = output[1] if len(output) > 1 else None
    else:
        model_output = output

    if isinstance(model_output, dict) and model_output.get('x', None) is not None:
        x = model_output['x']
    else:
        x = model_output

    print(f'Input shape:  (B={B}, C={C}, F={F}, H={H}, W={W})')
    print(f'Output shape: {tuple(x.shape)}')
    if repa_features is not None:
        print(f'REPA features: {len(repa_features)} layers')

    # Validate output dimensions
    assert x.shape[0] == B, f'Batch mismatch: {x.shape[0]} vs {B}'
    assert x.shape[2] == F, f'Frame mismatch: {x.shape[2]} vs {F}'
    assert x.shape[3] == H, f'Height mismatch: {x.shape[3]} vs {H}'
    assert x.shape[4] == W, f'Width mismatch: {x.shape[4]} vs {W}'
    # out_channels=8, with LEARNED_RANGE -> 16 channels
    print(f'Output channels: {x.shape[1]} (expected 16 for LEARNED_RANGE or 8)')

    print('='*50)
    print('Shape check PASSED!')
    print('='*50)


if __name__ == '__main__':
    main()
