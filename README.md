<div align="center">

<h1>DiTalker: A Unified DiT-based Framework for High-Quality and Speaking Styles Controllable Portrait Animation</h1>

#### <p align="center">He Feng, Yongjia Ma, Donglin Di, Lei Fan, Tonghua Su, Xiangqian Wu</p>

<a href='https://thenameishope.github.io/DiTalker/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2508.06511'><img src='https://img.shields.io/badge/arXiv-2508.06511-b31b1b'></a>

</div>

## Introduction

> Portrait animation aims to synthesize talking videos from a static reference face, conditioned on audio and style frame cues (e.g., emotion and head poses), while ensuring precise lip synchronization and faithful reproduction of speaking styles. Existing diffusion-based portrait animation methods primarily focus on lip synchronization or static emotion transformation, often overlooking dynamic styles such as head movements. Moreover, most of these methods rely on a dual U-Net architecture, which preserves identity consistency but incurs additional computational overhead. To this end, we propose DiTalker, a unified DiT-based framework for speaking style controllable portrait animation. We design a Style-Emotion Encoding Module that employs two separate branches: a style branch extracting identity-specific style information like head poses and movements, and an emotion branch extracting identity-agnostic emotion features. We further introduce an Audio-Style Fusion Module that decouples audio and speaking styles via two parallel cross-attention layers, using these features to guide the animation process. To enhance the quality of results, we adopt and modify two optimization constraints: one to improve lip synchronization and the other to preserve fine-grained identity and background details. Extensive experiments demonstrate the superiority of DiTalker in terms of lip synchronization and speaking style controllability.

## Model Overview
The core design of DiTalker follows the formulation in our methodology section:

> DiTalker consists of three main components: a DiT backbone, a Style-Expression Encoding Module (SEEM), and an Audio-Style Fusion Module (ASFM) integrated into each DiT block.

> The SEEM takes style frames and phonemes extracted from the driving audio as inputs, producing style embeddings and expression embeddings to guide the DiT backbone's denoising process. The ASFM is integrated into each layer of the DiT backbone and injects the conditional features via two parallel cross-attention layers.

> To guide the head pose and movements from the style frames, facial keypoints are extracted using DWPose, and a lightweight 3-layer 3D CNN Pose Adapter transforms them into pose features before the denoising process begins.

In short, DiTalker uses a single DiT backbone to jointly model lip synchronization, emotional expression, and speaking style, instead of relying on an additional reference U-Net.

## Requirements
DiTalker was developed and tested in the following internal conda environment:

`/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/fenghe/conda_envs/easyanimate`

The accompanying `requirements.txt` is a cleaned dependency list organized from that environment and keeps the packages needed for DiTalker training and inference.

Main tested versions:

- Python 3.10.14
- CUDA 11.8
- PyTorch 2.2.1
- TorchVision 0.17.1
- TorchAudio 2.2.1
- xformers 0.0.25
- diffusers 0.30.1
- transformers 4.46.2
- accelerate 0.34.0

A typical installation flow is:

```bash
conda create -n ditalker python=3.10 -y
conda activate ditalker
pip install -r requirements.txt
```

During our internal experiments, the `models` directory was linked to EasyAnimate model assets:

`EasyAnimate/models`

## Tested Environment


- GPU: `8 x NVIDIA A800-SXM4-80GB`
- GPU memory: `80 GB` per GPU
- NVIDIA driver: `535.54.03`
- CPU: `2 x Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz`
- CPU cores: `128` logical CPUs
- Memory: `1.0 TiB`
- OS: `Debian GNU/Linux 11 (bullseye)`
- Kernel: `Linux 5.10.101-1.el8.ssai.x86_64`

Current shared conda environment used for this repository:

- Python `3.10.14`
- CUDA `11.8`
- PyTorch `2.2.1`
- TorchVision `0.17.1`
- TorchAudio `2.2.1`
- xformers `0.0.25`
- diffusers `0.30.1`
- transformers `4.46.2`
- accelerate `0.34.0`


For the current training and inference pipeline, several driving conditions are expected to be prepared externally before running the scripts:

- Whisper or WhisperX based audio or phoneme features
- DWPose keypoints
- 3DMM style clips

For data preprocessing, [StyleTalk](https://github.com/FuxiVirtualHuman/styletalk) can be referenced for extracting 3DMM coefficients and phoneme files, [WhisperX](https://github.com/m-bain/whisperx) can be referenced for Chinese phoneme extraction, the detailed phoneme mapping follows our supplementary material, and [DWPose](https://github.com/IDEA-Research/DWPose) can be referenced for extracting facial keypoint videos.




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
This repository is built on top of EasyAnimate. 
