# Metrics

Each evaluation metric has its own Python entry script. The scripts use explicit GT/generated inputs so the evaluation setup is clear and easy to reproduce.

## FID

FID follows the implementation from mseitzer/pytorch-fid:
https://github.com/mseitzer/pytorch-fid

```bash
python metrics/calculate_fid.py \
  --gt_dir path/to/gt_frames \
  --generated_dir path/to/generated_frames \
  --output_json results/fid.json
```

If the results are videos, extract frames from GT and generated videos using the same sampling rule before running FID.

## FVD

FVD follows the implementation from songweige/content-debiased-fvd:
https://github.com/songweige/content-debiased-fvd

```bash
python metrics/calculate_fvd.py \
  --gt_dir path/to/gt_videos \
  --generated_dir path/to/generated_videos \
  --fvd_script path/to/content-debiased-fvd/eval_script.py \
  --output_json results/fvd.json
```

The wrapper intentionally delegates FVD to the external implementation used for reporting.

## LPIPS

```bash
python metrics/calculate_lpips.py \
  --gt_dir path/to/gt_videos \
  --generated_dir path/to/generated_videos \
  --output_json results/lpips.json
```

The script pairs videos by filename and uses the official `lpips` package.

## Sync-C

Sync-C is computed with Bytedance LatentSync:
https://github.com/bytedance/LatentSync

Specifically, use LatentSync's `eval/eval_sync_conf.py` and report its SyncNet confidence. We do not reimplement or approximate Sync-C locally.

```bash
python metrics/calculate_sync_c.py \
  --generated_dir path/to/generated_videos \
  --latentsync_root path/to/LatentSync \
  --initial_model path/to/syncnet_v2.model \
  --output_json results/sync_c.json
```

## LSE-D

```bash
python metrics/calculate_lse_d.py \
  --gt_dir path/to/gt_landmarks \
  --generated_dir path/to/generated_landmarks \
  --output_json results/lse_d.json
```

Landmarks should be `.npy` or `.json` arrays with shape `[T, K, 2]`; for 68-point landmarks, mouth indices 48..67 are used.

## AKD

```bash
python metrics/calculate_akd.py \
  --gt_dir path/to/gt_keypoints \
  --generated_dir path/to/generated_keypoints \
  --output_json results/akd.json
```

## F-LMD

```bash
python metrics/calculate_f_lmd.py \
  --gt_dir path/to/gt_landmarks \
  --generated_dir path/to/generated_landmarks \
  --output_json results/f_lmd.json
```
