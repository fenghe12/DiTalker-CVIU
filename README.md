<div align="center">

<h1>DiTalker: A Unified DiT-based Framework for High-Quality and Speaking Styles Controllable Portrait Animation</h1>

#### <p align="center">He Feng, Yongjia Ma, Donglin Di, Lei Fan, Tonghua Su, Xiangqian Wu</p>

<a href='https://thenameishope.github.io/DiTalker/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2508.06511'><img src='https://img.shields.io/badge/arXiv-2508.06511-b31b1b'></a>

</div>

## Introduction

> Portrait animation aims to synthesize talking videos from a static reference face, conditioned on audio and style frame cues (e.g., emotion and head poses), while ensuring precise lip synchronization and faithful reproduction of speaking styles. Existing diffusion-based portrait animation methods primarily focus on lip synchronization or static emotion transformation, often overlooking dynamic styles such as head movements. Moreover, most of these methods rely on a dual U-Net architecture, which preserves identity consistency but incurs additional computational overhead. To this end, we propose DiTalker, a unified DiT-based framework for speaking style controllable portrait animation. We design a Style-Emotion Encoding Module that employs two separate branches: a style branch extracting identity-specific style information like head poses and movements, and an emotion branch extracting identity-agnostic emotion features. We further introduce an Audio-Style Fusion Module that decouples audio and speaking styles via two parallel cross-attention layers, using these features to guide the animation process. To enhance the quality of results, we adopt and modify two optimization constraints: one to improve lip synchronization and the other to preserve fine-grained identity and background details. Extensive experiments demonstrate the superiority of DiTalker in terms of lip synchronization and speaking style controllability.

## Overview

DiTalker is a DiT-based portrait animation framework built on the EasyAnimate image-to-video backbone. Given a reference portrait, driving audio, and style-frame cues, it generates talking-head videos with controllable speaking style, head motion, and expression.

The main components are:

- A single DiT video backbone for latent denoising.
- SEEM for style and expression encoding from phonemes, 3DMM coefficients, style frames, and expression prompts.
- ASFM for parallel audio and style cross-attention inside each DiT block.
- A DWPose-based pose adapter for head pose and movement guidance.
- Region, synchronization, and representation-alignment losses used during training.

For implementation details, see `model-overview/README.md`.

## News

- Code, training configs, preprocessing utilities, inference scripts, and metric scripts are provided in this repository.
- Baseline inference videos are available through the request form below.
- DiTalker checkpoints and ablation weights will be shared through the same form upon paper acceptance.

## Installation

System and package versions used in our experiments:

- OS: Debian GNU/Linux 11
- GPU: 8 x NVIDIA A800-SXM4-80GB
- NVIDIA driver: 535.54.03
- CUDA: 11.8
- Python: 3.10.14
- PyTorch: 2.2.1
- TorchVision: 0.17.1
- TorchAudio: 2.2.1
- xformers: 0.0.25
- diffusers: 0.30.1
- transformers: 4.46.2
- accelerate: 0.34.0

Create the conda environment and install the required packages:

```bash
conda create -n ditalker python=3.10 -y
conda activate ditalker
pip install -r requirements.txt
```

`ffmpeg` is recommended for video I/O and audio processing. For example:

```bash
apt-get install ffmpeg
```

## Download Base Weights

DiTalker is initialized from the official EasyAnimateV3 image-to-video model. Please download `EasyAnimateV3-XL-2-InP-512x512` and place it under `models/Diffusion_Transformer/`:

```bash
mkdir -p models/Diffusion_Transformer
git lfs install
git clone https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-512x512 \
  models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512
```

The same model is also available from ModelScope:

```text
https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-512x512
```

According to the EasyAnimate model zoo, this checkpoint is the official EasyAnimateV3 512x512 text/image-to-video model, uses about 18.2 GB of storage, and was trained with 144 frames at 24 FPS. The default DiTalker configs expect this directory name:

```text
models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512
```

## Repository Structure

```text
configs/                 Training and inference YAML files
scripts/                 Training, inference, shape-checking, and metadata preparation
ditalker/                Model, dataset, pipeline, and diffusion utilities
preprocessing/           Data preprocessing wrappers
metrics/                 Metric computation scripts
examples/                Example metadata format
model-overview/          Mapping from paper modules to code
```

## Data Preparation

The training dataset **DH-FaceVId-1K** is publicly available. Please visit [this form](https://docs.google.com/forms/d/e/1FAIpQLSd92kS6ZdAGLoN6DvYUVUDCo7R3Oe6GNVPjQn4sDBPJH7_2_A/viewform) to request access.

DiTalker follows the EasyAnimate metadata format and adds paths for audio, phoneme, 3DMM, pose, and mask conditions. A minimal training record is:

```json
[
  {
    "file_path": "datasets/demo/videos_25/demo_0001.mp4",
    "text": "This person is neutral and talks.",
    "type": "video",
    "audio_emb_path": "datasets/demo/whisper_audio_emb/demo_0001.pt",
    "phoneme_dir": "datasets/demo/phoneme/demo_0001.json",
    "3dmm_dir": "datasets/demo/3dmm/demo_0001.mat",
    "pose_video_path": "datasets/demo/videos_25_dwpose/demo_0001.mp4"
  }
]
```

The same example is provided in `examples/training_metadata_format.json`.

For each training video, prepare the following files:

- `videos_25/*.mp4`: 25 FPS training videos.
- `whisper_audio_emb/*.pt`: Whisper audio embeddings.
- `phoneme/*.json`: phoneme id sequences. Chinese phonemes are extracted with WhisperX and mapped using the supplementary phoneme table.
- `3dmm/*.mat` or `3dmm/*.txt`: 64D expression coefficients extracted with Deep3DFaceRecon.
- `videos_25_dwpose/*.mp4`: DWPose facial keypoint videos.
- `lip_mask/*.png`, `eye_masks/*.png`, `face_mask/*.png`: binary region masks used by region and lip-related losses.

Caption metadata can be converted with:

```bash
python scripts/prepare_training_metadata.py \
  --caption_metadata_path path/to/captions.jsonl \
  --video_folder path/to/videos_25 \
  --saved_path path/to/ditalker_train.json
```

The input caption file should contain `video_path` and `caption`. If `audio_emb_path`, `phoneme_dir`, `3dmm_dir`, or `pose_video_path` already exists, the script keeps these fields.

## Preprocessing

External preprocessing tools are kept outside this repository. Clone the required projects and pass their paths to the corresponding DiTalker wrappers:

```bash
mkdir -p third_party
git clone https://github.com/TMElyralab/MuseTalk.git third_party/MuseTalk
git clone https://github.com/m-bain/whisperx.git third_party/whisperx
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git third_party/Deep3DFaceRecon_pytorch
git clone https://github.com/IDEA-Research/DWPose.git third_party/DWPose
git clone https://github.com/magic-research/PLLaVA.git third_party/PLLaVA
```

Extract Whisper audio embeddings:

```bash
python preprocessing/extract_audio_embeddings.py \
  --audio path/to/audio_or_audio_dir \
  --output_dir datasets/demo/whisper_audio_emb \
  --whisper_root third_party/MuseTalk/musetalk/whisper \
  --model_path path/to/whisper_tiny.pt \
  --fps 25
```

`extract_audio_embeddings.py` dynamically imports `whisper`, so `--whisper_root` should point to the MuseTalk/ACTalker-style Whisper package whose `transcribe()` output contains `encoder_embeddings`.

Predict expression prompts with PLLaVA:

```bash
python preprocessing/predict_expression_pllava.py \
  --pllava_root third_party/PLLaVA \
  --pretrained_model_name_or_path path/to/pllava_checkpoint \
  --style_video path/to/style_video_or_dir \
  --output_json datasets/demo/pllava_expression.json
```

Extract DWPose videos:

```bash
python preprocessing/extract_dwpose_videos.py \
  --video_root datasets/demo/videos_25 \
  --save_dir datasets/demo/videos_25_dwpose \
  --dwpose_tool path/to/extract_dwpose_from_vid_tool.py \
  -j 4
```

Convert existing mask images or mask videos to binary PNG masks:

```bash
python preprocessing/extract_region_masks.py \
  --input_root path/to/raw_region_masks \
  --output_root datasets/demo
```

More details are provided in `preprocessing/README.md`.

## Training

The three training phases are configured in `configs/`:

```bash
python scripts/train.py --config_path configs/ditalker_phase1.yaml
python scripts/train.py --config_path configs/ditalker_phase2.yaml
python scripts/train.py --config_path configs/ditalker_phase3.yaml
```

Before training, update the dataset paths and checkpoint paths in the YAML files. `scripts/train.py` reads the YAML values through `apply_config_overrides()` and passes the configured audio, style, pose, expression, and mask conditions to the training loop.

## Inference

Run inference with:

```bash
python scripts/infer.py --config_path configs/infer.yaml
```

`configs/infer.yaml` specifies the reference image, audio embedding, phoneme file, 3DMM style clip, DWPose pose input, expression prompt, and output path. In the paper setting, the expression prompt is predicted from style frames by PLLaVA; the inference script also accepts an explicit prompt for reproducible testing.

## Evaluation

Metric scripts are placed under `metrics/`:

- `calculate_fid.py`: wraps `pytorch-fid` from https://github.com/mseitzer/pytorch-fid.
- `calculate_fvd.py`: calls the implementation from https://github.com/songweige/content-debiased-fvd.
- `calculate_sync_c.py`: uses Bytedance LatentSync, especially the SyncNet confidence from `eval/eval_sync_conf.py`: https://github.com/bytedance/LatentSync.
- `calculate_lpips.py`: computes paired-video LPIPS with the official `lpips` package.
- `calculate_lse_d.py`, `calculate_akd.py`, and `calculate_f_lmd.py`: compute landmark/keypoint-based distances from paired GT and generated files.

See `metrics/README.md` for command examples.

## Results And Checkpoints

Baseline inference videos can be requested through the form below. Upon paper acceptance, the same form will also be used to provide DiTalker checkpoints and ablation weights for academic use.

[Baseline results and checkpoint request form](https://docs.google.com/forms/d/e/1FAIpQLSfPuQpCQsY__MZaGIKMY245gJGuKyX2nWhea6WxJgu8P9EqUg/viewform?usp=dialog)

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{feng2025ditalker,
  title={DiTalker: A Unified DiT-based Framework for High-Quality and Speaking Styles Controllable Portrait Animation},
  author={Feng, He and Ma, Yongjia and Di, Donglin and Fan, Lei and Su, Tonghua and Wu, Xiangqian},
  journal={arXiv preprint arXiv:2508.06511},
  year={2025}
}
```

## Acknowledgement

This repository is built on top of EasyAnimate. We thank the EasyAnimate authors for their open-source code and model weights.
