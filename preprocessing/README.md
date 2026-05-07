# Preprocessing

This directory contains the preprocessing entry points used by the released DiTalker code. The descriptions below follow the method section of the paper: audio features come from Whisper, phonemes are obtained with WhisperX and mapped with the supplementary phoneme table, 3DMM expression coefficients are extracted with Deep3DFaceRecon, expression prompts are predicted with PLLaVA, and pose videos are extracted with DWPose.

## External Tool Setup
The full third-party preprocessing toolkits are not copied into this repository. Clone and configure them separately, then pass their local paths to the DiTalker wrappers:

```bash
mkdir -p third_party
git clone https://github.com/TMElyralab/MuseTalk.git third_party/MuseTalk
git clone https://github.com/m-bain/whisperx.git third_party/whisperx
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git third_party/Deep3DFaceRecon_pytorch
git clone https://github.com/IDEA-Research/DWPose.git third_party/DWPose
git clone https://github.com/magic-research/PLLaVA.git third_party/PLLaVA
```

Important path conventions:

- MuseTalk Whisper: `extract_audio_embeddings.py` uses `importlib.import_module("whisper")`, so pass `--whisper_root third_party/MuseTalk/musetalk/whisper` or otherwise add that package to `PYTHONPATH`; also provide `--model_path path/to/whisper_tiny.pt`.
- PLLaVA: install the PLLaVA environment following its repository, then pass `--pllava_root third_party/PLLaVA` and `--pretrained_model_name_or_path path/to/pllava_checkpoint`.
- DWPose: install DWPose following its repository; the provided wrapper can call a local DWPose extraction script when configured.
- WhisperX and Deep3DFaceRecon: run their original pipelines to generate phoneme JSON files and 64D 3DMM expression coefficients; DiTalker only requires the resulting files to be referenced by `phoneme_dir` and `3dmm_dir` in the training metadata.

## 1. Audio Embedding `.pt`

Paper correspondence: driving audio `a` is encoded by Whisper to obtain audio embeddings `c_a`, which are later projected before Audio Cross-Attention (ACA). The DiTalker loader expects `audio_emb_path` to point to a `.pt` tensor shaped `[num_video_frames, 50, 384]`.

Reference implementation: the slicing logic follows the MuseTalk-style Whisper feature processor, where Whisper features are aligned at 50 FPS and each 25 FPS video frame receives a 5-frame temporal audio window.

Original repository: https://github.com/TMElyralab/MuseTalk/tree/main/musetalk/whisper/whisper

```bash
python preprocessing/extract_audio_embeddings.py \
  --audio path/to/audio_or_audio_dir \
  --output_dir datasets/demo/whisper_audio_emb \
  --whisper_root path/to/MuseTalk/musetalk/whisper \
  --model_path path/to/whisper_tiny.pt \
  --fps 25
```

The script requires a MuseTalk/ACTalker-style Whisper package whose `transcribe()` output contains `encoder_embeddings`. It saves one `.pt` file per audio file.

## 2. Phoneme JSON

Paper correspondence: phonemes are extracted from the driving audio using WhisperX ASR, then mapped to an index sequence by the predefined phoneme table in the supplementary material. The DiTalker loader expects `phoneme_dir` to point to a JSON list of integer phoneme ids.

Original repository: https://github.com/m-bain/whisperx

Expected format:

```json
[12, 4, 18, 18, 31, 7, 9]
```

## 3. 3DMM Coefficients

Paper correspondence: 3DMM parameters `delta_{1:M} in R^{M x 64}` are extracted from the style frames using Deep3DFaceRecon. These 64-dimensional expression coefficients are encoded by the style branch of SEEM. The DiTalker loader expects `3dmm_dir` to point to either a `.mat` file containing `coeff` or a `.txt` file containing the expression coefficients.

Original repository: https://github.com/sicxu/Deep3DFaceRecon_pytorch

Practical data-organization reference: https://github.com/FuxiVirtualHuman/styletalk

## 4. DWPose Pose Videos

Paper correspondence: DWPose extracts facial keypoints from the style frames. Interior facial keypoints are removed in the method to avoid interfering with facial dynamics, and the resulting pose video is transformed by the Pose Adapter.

Original repository: https://github.com/IDEA-Research/DWPose

Wrapper command:

```bash
python preprocessing/extract_dwpose_videos.py \
  --video_root datasets/demo/videos_25 \
  --save_dir datasets/demo/videos_25_dwpose \
  --dwpose_tool path/to/extract_dwpose_from_vid_tool.py \
  -j 4
```

## 5. PLLaVA Expression Prompt

Paper correspondence: the expression branch uses an MLLM, PLLaVA, to predict an expression prompt `emo` from style frames. The prompt is then encoded with T5. The paper template is:

```text
This person is [expression] and talks
```

The label set is:

```text
happy, sad, angry, disgusted, surprised, fearful, neutral
```

Original repository: https://github.com/magic-research/PLLaVA

```bash
python preprocessing/predict_expression_pllava.py \
  --pllava_root path/to/PLLaVA \
  --pretrained_model_name_or_path path/to/pllava_checkpoint \
  --style_video path/to/style_video_or_dir \
  --output_json datasets/demo/pllava_expression.json
```

The released inference script also allows an explicit prompt/expression for reproducibility before checkpoint release.

## 6. Region Masks

The EasyAnimate-derived loader expects black-white PNG masks under `lip_mask/`, `eye_masks/`, and `face_mask/`. The mask foreground is white (`255`) and background is black (`0`). Existing mask images or mask videos can be converted with OpenCV:

```bash
python preprocessing/extract_region_masks.py \
  --input_root path/to/raw_region_masks \
  --output_root datasets/demo
```

Expected input folders are `lip_mask/`, `eye_masks/`, and `face_mask/`. For mask videos, the script accumulates a foreground union over sampled frames and writes static PNG masks.
