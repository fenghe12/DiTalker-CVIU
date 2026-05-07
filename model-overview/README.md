# Model Overview

This page is a code map for the DiTalker implementation. It is meant for readers who want to check where each paper module is implemented, not just the high-level idea.

## End-To-End Flow

```text
reference image x
  -> CLIP image encoder in the EasyAnimate-style pipeline
  -> reference tokens c_ref

style frames V_S
  -> PLLaVA expression prediction
  -> prompt: "This person is [expression] and talks"
  -> T5 text encoder
  -> text tokens c_text

style frames V_S
  -> Deep3DFaceRecon 64D expression coefficients
  -> StyleEncoder + ContentEncoder + Decoder + StyleProjection
  -> style tokens c_s

driving audio a
  -> Whisper encoder features
  -> AudioProjection
  -> audio tokens c_a

driving audio a
  -> WhisperX phoneme ids
  -> ContentEncoder

style frames V_S
  -> DWPose facial keypoint video
  -> PoseGuider
  -> latent pose guidance

z_t + conditions
  -> Transformer3DModel
  -> TemporalTransformerBlock with ASFM and ECA
  -> denoised latent
  -> VAE decode
```

## Paper Module To Code

| Paper part | Code location | Exact implementation point |
| --- | --- | --- |
| DiT video backbone | `ditalker/models/base/transformer3d.py` | `Transformer3DModel` builds the EasyAnimate-style 3D transformer and owns the condition branches. The main forward path starts in `Transformer3DModel.forward()`. |
| DiT block | `ditalker/models/base/attention.py` | `TemporalTransformerBlock` contains self-attention, text/reference cross-attention, ASFM, ECA, and FFN in one block. |
| Audio projection | `ditalker/models/base/transformer3d.py` | `AudioProjection` projects Whisper features before they are passed to the audio attention branch. It is created when `enable_audio_attention` is set. |
| ACA in ASFM | `ditalker/models/base/attention.py` | In `TemporalTransformerBlock.forward()`, the `audio_attention` branch attends to `audio_encoder_hidden_states` and is scaled by `audio_scale_scalar`. |
| Style branch in SEEM | `ditalker/models/base/transformer3d.py` | `StyleEncoder` encodes 64D 3DMM coefficients, `ContentEncoder` embeds phoneme windows, `Decoder` combines content and style, and `StyleProjection` maps the generated expression stack to DiT token dimension. |
| SCA in ASFM | `ditalker/models/base/attention.py` | In `TemporalTransformerBlock.forward()`, the `style_attention` branch attends to `style_encoder_hidden_states` and is scaled by `scale_scalar`. |
| Scale Adapter | `ditalker/models/scale_adapter.py` | `ScaleAdapter.forward(c_s, c_a)` predicts the layer-wise style scale `s_phi` and audio scale `s_alpha`. `Transformer3DModel.forward()` calls it before looping over DiT blocks. |
| Compact ASFM reference | `ditalker/models/asfm.py` | `AudioStyleFusionModule` keeps a small standalone version of ACA + SCA + scale fusion. The training/inference path uses the integrated version in `attention.py`. |
| Emotion branch / ECA | `ditalker/models/base/transformer3d.py`, `ditalker/models/base/attention.py`, `ditalker/models/eca.py` | `Transformer3DModel.forward()` builds `c_emo` by concatenating projected T5 text tokens and CLIP reference tokens when ECA is enabled. `TemporalTransformerBlock.forward()` applies `emotion_cross_attention` after ASFM. `EmotionCrossAttention` is a compact standalone reference. |
| Pose Adapter | `ditalker/models/base/transformer3d.py` | `PoseGuider` is the 3-layer 3D CNN pose adapter. In `Transformer3DModel.forward()`, pose features are multiplied by `0.5` and added to the latent before denoising. |
| Dataset and condition loading | `ditalker/data/dataset_image_video.py` | `ImageVideoDataset.get_batch()` loads `audio_emb_path`, `phoneme_dir`, `3dmm_dir`, DWPose videos, and region masks. `__getitem__()` returns the tensors used by training. |
| Training config loading | `scripts/train.py` | `apply_config_overrides()` reads YAML values into CLI args. `main()` creates `ImageVideoDataset` and passes all condition tensors into the diffusion loss. |
| Diffusion losses | `ditalker/utils/gaussian_diffusion.py` | `GaussianDiffusion.training_losses()` computes denoising loss and optional region-mask, REPA, and sync losses. |
| Inference | `scripts/infer.py` | Loads the reference image, audio features, phoneme window, 3DMM style clip, pose input, and expression prompt from CLI/config before calling the DiTalker pipeline. |

## Important Code Fragments

### Dataset Inputs

`ditalker/data/dataset_image_video.py` is where the training sample is assembled.

- `ImageVideoDataset.__init__()` receives `text_drop_ratio`, `enable_audio_emb`, `enable_style_attn`, `enable_pose_adapter`, `finetune_emo`, and `enable_sync_lip_loss`.
- If `text_drop_ratio < 0`, the dataset uses the paper-style defaults: `0.99` when style attention is enabled and `0.5` when emotion fine-tuning is enabled.
- `get_batch()` loads `audio_emb_path` into `audio_emb`.
- `get_batch()` loads region masks by replacing `videos_25` with `lip_mask`, `eye_masks`, and `face_mask`.
- `get_batch()` loads `phoneme_dir`, converts phoneme ids to a temporal window with `get_audio_window()`, and loads `3dmm_dir` with `get_video_style_clip()`.
- `get_batch()` loads the corresponding DWPose video from the `videos_25_dwpose` convention when `enable_pose_adapter` is enabled.
- `__getitem__()` returns `audio_encoder_hidden_states`, `phoneme_win`, `style_clip`, `pad_mask`, `pixel_values_pose`, and region masks to the dataloader.

### Training Path

`scripts/train.py` is the main training entry.

- `apply_config_overrides()` reads the YAML keys for `learning_rate`, `max_train_steps`, `text_drop_ratio`, region loss, REPA, sync loss, and module switches.
- `main()` constructs `ImageVideoDataset` with `enable_audio_emb`, `enable_style_attn`, `enable_pose_adapter`, `enable_sync_lip_loss`, and `finetune_emo`.
- The collate function stacks audio embeddings, phoneme windows, style clips, pose tensors, and region masks.
- The training loop builds `region_mask` from `lip_mask_values`, `eye_mask_values`, and `full_mask_values`.
- The diffusion call passes `audio_encoder_hidden_states`, `phoneme_win`, `style_clip`, `pad_mask`, `pixel_values_pose`, and CLIP reference tokens through `model_kwargs`.

### Transformer Condition Branches

`ditalker/models/base/transformer3d.py` owns the condition encoders.

- `Transformer3DModel.__init__()` creates `AudioProjection` when `enable_audio_attention=True`.
- It creates `StyleEncoder`, `ContentEncoder`, `Decoder`, and `StyleProjection` when `enable_style_attn=True`.
- It creates `PoseGuider` when `enable_pose_adapter=True`.
- In `Transformer3DModel.forward()`, `pixel_values_pose` is processed by `PoseGuider`, scaled by `0.5`, padded/cropped to latent size, and added to `hidden_states`.
- Whisper audio features are projected by `self.audio_projection(audio_encoder_hidden_states)`.
- 3DMM style clips and phoneme windows are converted into `style_encoder_hidden_states` through the style/content encoder and decoder stack.
- When CLIP reference tokens and T5 text tokens are available, `c_emo` is built by concatenating them before the transformer block loop.
- `ScaleAdapter` receives style and audio tokens and returns `scale_scalar` and `audio_scale_scalar` for block-wise fusion.

### ASFM And ECA Inside Each DiT Block

`ditalker/models/base/attention.py` contains the fused block logic.

- `TemporalTransformerBlock.__init__()` creates `audio_attention`, `style_attention`, and `emotion_cross_attention` according to the module flags.
- In `TemporalTransformerBlock.forward()`, normal text/reference cross-attention is applied first.
- ASFM then saves the same residual state as the query for both branches.
- The audio branch computes `audio_attn_output = audio_attention(query=z_i, key/value=c_a)` and multiplies it by `s_alpha` when audio scale fusion is enabled.
- The style branch computes `style_attn_output = style_attention(query=z_i, key/value=c_s)` and multiplies it by `s_phi` when style scale fusion is enabled.
- The two branches are fused in parallel as `z_{i+1} = z_i + s_alpha * ACA(z_i, c_a) + s_phi * SCA(z_i, c_s)`.
- ECA is applied after ASFM: the block attends to `c_emo` and adds the result before the FFN.

### Losses

`ditalker/utils/gaussian_diffusion.py` contains the training losses used by `scripts/train.py`.

- `GaussianDiffusion.training_losses()` computes the denoising objective.
- When `enable_region_mask_loss=True`, eye/face masks are resized to latent/video resolution and applied to the reconstruction loss.
- When `enable_repa=True`, the projected visual representation is aligned with DINOv2 features prepared in the training loop.
- When sync loss is enabled, the code crops the masked mouth/face region and passes it with audio features to the sync network.

## Configs

| Config | Main role |
| --- | --- |
| `configs/ditalker_phase1.yaml` | Phase 1 backbone adaptation. |
| `configs/ditalker_phase2.yaml` | Phase 2 audio/style/pose training with high text-drop ratio. |
| `configs/ditalker_phase3.yaml` | Phase 3 expression/emotion fine-tuning with lower text-drop ratio. |
| `configs/infer.yaml` | Inference input paths and generation settings. |

## Inference Inputs

`scripts/infer.py` accepts the same condition types used during training:

- `--reference_image`: source portrait image.
- `--audio_path`: Whisper audio embedding `.pt`/`.npy`, or an audio path resolvable to a precomputed embedding.
- `--phoneme_path`: phoneme ids in `.json`, `.npy`, or `.pt` format.
- `--style_clip_path`: 64D 3DMM expression coefficients.
- `--pose_path`: DWPose video or tensor.
- `--prompt` / `--expression`: explicit expression text. In the paper setting, this text is produced from style frames by PLLaVA.
- `--output_path`: generated video path.
