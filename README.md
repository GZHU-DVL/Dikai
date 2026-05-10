# IDA Runtime Release

This is the clean runtime package for training, local Track 1-style evaluation, threshold selection, and final submission export. Paper-only assets, plotting scripts, ablation runner scripts, LaTeX sources, review notes, caches, and local tool configs are intentionally excluded.

## Included Files

- `Train.py`: main training implementation and post-training local Track 1 evaluation flow
- `run_reference.py`: non-interactive entrypoint for the long-schedule reference recipe
- `ida_plugin_trainer.py`: Track1 proxy fitness, SWAD selection, and trainer patches
- `custom_modules/hsfpn_dcn.py`: IBN-a, MixRand/DSU, RandConv, SPD, SimAM, and related modules
- `custom_modules/ida_loss.py`: NWD, VFL, and IDA segmentation loss
- `caculate_metric.py`: local Track 1 metric implementation
- `phase1_track1_rect_export.py`: prediction export in Track 1 rectangle format
- `phase1_track1_finalize_submission.py`: final submission package exporter
- `phase1_track1_submit_json.py`: lightweight JSON export wrapper
- `phase1_track1_metric_wrapper.py`: standalone metric wrapper
- `phase2_hard_negative_mining.py`, `phase2_merge_hard_negatives.py`: kept because `Train.py` imports them and the hard-negative menu path uses them
- `weights/yolo11l-seg.pt`, `weights/yolo11n.pt`: local YOLOv11 pretrained weights
- `requirements.txt`: Python dependency list

## Setup

Use Python 3.10+ on a Linux GPU server. Install the PyTorch build matching your CUDA driver first, then install project dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Adjust cu121/cu124/cpu for your server.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Dataset

Prepare IDA Track 1 data in YOLO segmentation format:

```text
datasets/ida_track1/
  data.yaml
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
```

Set runtime paths before launching:

```bash
export IDA_DATA_YAML=/absolute/path/to/datasets/ida_track1/data.yaml
export IDA_RESULT_ROOT=/absolute/path/to/ida_runs
export IDA_DEVICE=0
export IDA_MODEL_FAMILY=11
export IDA_MODEL_SCALE=l
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

## Quick Smoke Test

Run a tiny 3-epoch job first to verify the environment, dataset, export path, and local metric path:

```bash
python run_reference.py \
  --data-yaml "$IDA_DATA_YAML" \
  --result-root "$IDA_RESULT_ROOT" \
  --scale l \
  --device "$IDA_DEVICE" \
  --batch 2 \
  --epochs 3 \
  --patience 3 \
  --close-mosaic 1 \
  --save-period 1 \
  --project ida_smoke
```

## Main Run

Run the long-schedule reference recipe:

```bash
python run_reference.py \
  --data-yaml "$IDA_DATA_YAML" \
  --result-root "$IDA_RESULT_ROOT" \
  --scale l \
  --device "$IDA_DEVICE" \
  --epochs 2000 \
  --patience 300 \
  --close-mosaic 120 \
  --save-period 20
```

Important outputs:

- `weights/swad_best.pt`
- `results.csv`
- `official_eval/official_track1_best.json`

## Final Submission Export

```bash
python phase1_track1_finalize_submission.py \
  --best_json /path/to/official_eval/official_track1_best.json \
  --source /absolute/path/to/datasets/ida_track1/test/images \
  --source_root /absolute/path/to/datasets/ida_track1/test/images \
  --out_dir submissions \
  --device "$IDA_DEVICE" \
  --json_keep_rel_path
```

The output package contains `submission.json`, a config snapshot, and format validation results.
