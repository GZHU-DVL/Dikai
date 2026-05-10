#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train.py - YOLO11/YOLO26 IDA Track1 稳定版（含自动扫描）

支持：
1) Baseline / EFE / MixStyle / RandConv / MixStyle+RandConv
2) 训练后自动运行官方评测 sweep
3) checkpoint averaging
4) Priority-1 自动串行扫描：
   - Stage1: MixStyle 参数网格
   - Stage2: copy_paste 轻量扫描
   - Stage3: IBN ratio 轻量扫描
5) 自动输出 `summary.csv` / `summary.json` / 控制台表格
"""

import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_project_in_pythonpath(project_root: Path):
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    old_pp = os.environ.get("PYTHONPATH", "")
    parts = [p for p in old_pp.split(os.pathsep) if p]
    if root not in parts:
        os.environ["PYTHONPATH"] = root if not old_pp else root + os.pathsep + old_pp


ensure_project_in_pythonpath(PROJECT_ROOT)

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "Ultralytics"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
os.environ.setdefault("ULTRA_DISABLE_FUSE", "1")

from custom_modules.hsfpn_dcn import register_plugins
from ida_plugin_trainer import PluginSegTrainer, PLUGIN_ENV_KEY, LOSS_ENV_KEY, USE_TRACK1_PROXY_ENV_KEY
from phase1_track1_rect_export import export_rect_predictions
from phase2_hard_negative_mining import mine_hard_negatives
from phase2_merge_hard_negatives import merge_hard_negatives
from ultralytics import YOLO

# ===================== 用户配置 =====================
DATA_PATH = (
    os.environ.get("IDA_DATA_YAML", str(PROJECT_ROOT / "datasets" / "ida_track1" / "data.yaml")).strip()
    or str(PROJECT_ROOT / "datasets" / "ida_track1" / "data.yaml")
)
RESULT_ROOT = Path(
    os.environ.get("IDA_RESULT_ROOT", str(PROJECT_ROOT / "runs" / "ida_track1")).strip()
    or str(PROJECT_ROOT / "runs" / "ida_track1")
).expanduser()
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

CUSTOM_CFG_DIR = PROJECT_ROOT / "tmp_configs"
CUSTOM_CFG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SCALE = os.environ.get("IDA_MODEL_SCALE", "l").strip() or "l"
DEFAULT_MODEL_FAMILY = os.environ.get("IDA_MODEL_FAMILY", "11").strip() or "11"
ALLOW_REMOTE_WEIGHTS = str(os.environ.get("IDA_ALLOW_REMOTE_WEIGHTS", "0")).strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_DEVICE = os.environ.get("IDA_DEVICE", "0").strip() or "0"

PRETRAINED_SEG_WEIGHTS_BY_FAMILY = {
    "11": {
        "n": "yolo11n-seg.pt",
        "s": "yolo11s-seg.pt",
        "m": "yolo11m-seg.pt",
        "l": "yolo11l-seg.pt",
        "x": "yolo11x-seg.pt",
    },
    "26": {
        "n": "yolo26n-seg.pt",
        "s": "yolo26s-seg.pt",
        "m": "yolo26m-seg.pt",
        "l": "yolo26l-seg.pt",
        "x": "yolo26x-seg.pt",
    },
}

# ===================== 官方评测配置 =====================
RUN_OFFICIAL_EVAL_AFTER_TRAIN = True
RUN_OFFICIAL_SWEEP_AFTER_TRAIN = True
OFFICIAL_METRIC_SCRIPT = PROJECT_ROOT / "caculate_metric.py"
OFFICIAL_EVAL_SPLIT = "val"
OFFICIAL_EVAL_TRACK = 1

OFFICIAL_PRED_SAVE_CONF = False
OFFICIAL_PRED_RETINA_MASKS = False
OFFICIAL_PRED_CONF = 0.20
OFFICIAL_PRED_IOU = 0.50
OFFICIAL_PRED_MAX_DET = 50
OFFICIAL_PRED_TTA = False
OFFICIAL_PRED_TTA_SCALES = [0.83, 1.00, 1.17]
OFFICIAL_PRED_TENT = False
OFFICIAL_PRED_TENT_CFG = {
    "enabled": False,
    "mode": "lite",
    "steps": 1,
    "lr": 1e-4,
    "topk_ratio": 0.10,
    "topk_min": 64,
    "reset_mode": "scenario",
    "reset_each_image": False,
    "scope": "all",
    "shallow_stages": 4,
    "max_bn_layers": 0,
}
OFFICIAL_USE_RECT_EXPORT = True
OFFICIAL_RECT_DECIMALS = 6
OFFICIAL_EVAL_NOTE = "Local val-only official eval; not equal to hidden-domain official test."

OFFICIAL_SWEEP_CONF_LIST = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
OFFICIAL_SWEEP_IOU_LIST = [0.25, 0.30, 0.35, 0.40, 0.45]
OFFICIAL_SWEEP_MAXDET_LIST = [50]
OFFICIAL_EVAL_INCLUDE_PERIODIC = True
OFFICIAL_EVAL_PERIODIC_START_RATIO = 0.60
OFFICIAL_EVAL_PERIODIC_STRIDE = 1
OFFICIAL_EVAL_RESUME = True
OFFICIAL_EVAL_CANDIDATE_TYPES = None
OFFICIAL_EVAL_MAX_CANDIDATES = None
OFFICIAL_POSTPROCESS_PRESETS = {
    "base": {},
    "topk_only": {
        "per_class_topk": {"all": 2},
    },
    "gate_light": {
        "image_gate": {
            "enabled": True,
            "min_top1_score": 0.18,
            "min_count_above": {"0.12": 1},
        },
    },
    "classaware_light": {
        "per_class_conf": {
            "all": 0.10,
            "Particle": 0.08,
            "Damage": 0.10,
            "Bubble": 0.12,
            "Dent": 0.12,
            "Chipping": 0.12,
        },
        "per_class_topk": {
            "all": 2,
            "Particle": 3,
            "Damage": 3,
        },
    },
    "screen_loose": {
        "per_class_topk": {"all": 3},
        "image_gate": {
            "enabled": True,
            "min_top1_score": 0.25,
            "min_sum_score": 0.35,
            "min_count_above": {"0.20": 1},
        },
    },
    "screen_strict": {
        "per_class_topk": {"all": 2},
        "image_gate": {
            "enabled": True,
            "min_top1_score": 0.40,
            "min_sum_score": 0.55,
            "min_count_above": {"0.25": 2},
        },
    },
    "screen_balance": {
        "per_class_conf": {
            "all": 0.10,
            "Particle": 0.08,
            "Damage": 0.08,
            "Bubble": 0.12,
            "Dent": 0.12,
            "Chipping": 0.12,
        },
        "per_class_topk": {
            "all": 2,
            "Particle": 2,
            "Damage": 2,
        },
        "image_gate": {
            "enabled": True,
            "min_top1_score": 0.18,
            "min_sum_score": 0.28,
            "min_count_above": {"0.10": 1},
        },
    },
    "screen_aggressive": {
        "per_class_conf": {
            "all": 0.12,
            "Particle": 0.10,
            "Damage": 0.10,
            "Bubble": 0.14,
            "Dent": 0.14,
            "Chipping": 0.14,
        },
        "per_class_topk": {
            "all": 1,
            "Particle": 2,
            "Damage": 2,
        },
        "image_gate": {
            "enabled": True,
            "min_top1_score": 0.24,
            "min_sum_score": 0.34,
            "min_count_above": {"0.14": 1},
        },
    },
    "metric_safe": {
        "per_class_conf": {
            "all": 0.08,
            "Particle": 0.06,
            "Damage": 0.06,
            "Bubble": 0.10,
            "Dent": 0.10,
            "Chipping": 0.10,
        },
        "per_class_topk": {
            "all": 2,
            "Particle": 3,
            "Damage": 3,
        },
        "image_gate": {
            "enabled": True,
            "min_top1_score": 0.16,
            "min_sum_score": 0.24,
            "min_count_above": {"0.08": 1},
        },
    },
    "dense_positive": {
        "per_class_conf": {
            "all": 0.04,
            "Particle": 0.02,
            "Damage": 0.02,
            "Bubble": 0.05,
            "Dent": 0.05,
            "Chipping": 0.05,
        },
        "per_class_topk": {
            "all": 4,
            "Particle": 5,
            "Damage": 5,
        },
        "image_gate": {
            "enabled": True,
            "min_top1_score": 0.10,
            "min_sum_score": 0.16,
            "min_count_above": {"0.04": 1},
        },
    },
    "dense_positive_relaxed": {
        "per_class_conf": {
            "all": 0.02,
            "Particle": 0.01,
            "Damage": 0.01,
            "Bubble": 0.03,
            "Dent": 0.03,
            "Chipping": 0.03,
        },
        "per_class_topk": {
            "all": 6,
            "Particle": 8,
            "Damage": 8,
        },
        "image_gate": {
            "enabled": True,
            "min_top1_score": 0.08,
            "min_sum_score": 0.12,
            "min_count_above": {"0.03": 1},
        },
    },
}
OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES = [
    "base",
    "topk_only",
    "gate_light",
    "classaware_light",
    "screen_balance",
    "screen_aggressive",
]
TEXTURE_PRESERVE_POSTPROCESS_PRESETS = [
    "base",
    "gate_light",
    "classaware_light",
    "screen_balance",
    "screen_aggressive",
]
METRIC_ALIGNED_POSTPROCESS_PRESETS = [
    "metric_safe",
    "screen_balance",
    "classaware_light",
    "dense_positive",
    "dense_positive_relaxed",
    "base",
]

# ===================== Priority-1 扫描预设 =====================
P1_SWEEP_PRESETS = {
    "compact": {
        "mixstyle_prob": [0.15, 0.20, 0.25],
        "mixstyle_alpha": [0.10, 0.15, 0.20],
        "mixstyle_layers": [1, 2],
        "copy_paste": [0.10, 0.15],
        "ibn_ratio": [0.30, 0.50, 0.70],
    },
    "extended": {
        "mixstyle_prob": [0.15, 0.20, 0.25, 0.30],
        "mixstyle_alpha": [0.10, 0.15, 0.20, 0.30],
        "mixstyle_layers": [1, 2],
        "copy_paste": [0.10, 0.15],
        "ibn_ratio": [0.30, 0.50, 0.70],
    },
}

P1_BUDGET_PRESETS = {
    "quick": {"epochs": 120, "patience": 30},
    "full": {},
}

TRAIN_AUG_PRESETS = {
    "default": {},
    "industrial_soft": {
        "hsv_h": 0.01,
        "hsv_s": 0.18,
        "hsv_v": 0.12,
        "degrees": 1.0,
        "translate": 0.03,
        "scale": 0.10,
        "mosaic": 0.15,
        "close_mosaic": 30,
    },
    "industrial_conservative": {
        "hsv_h": 0.00,
        "hsv_s": 0.10,
        "hsv_v": 0.08,
        "degrees": 0.5,
        "translate": 0.02,
        "scale": 0.08,
        "mosaic": 0.10,
        "close_mosaic": 40,
    },
    "industrial_geometry_low": {
        "hsv_h": 0.01,
        "hsv_s": 0.16,
        "hsv_v": 0.10,
        "degrees": 0.5,
        "translate": 0.02,
        "scale": 0.06,
        "mosaic": 0.12,
        "close_mosaic": 40,
    },
}

P2_SWEEP_PRESETS = {
    "compact": {
        "aug_presets": ["default", "industrial_soft", "industrial_conservative"],
        "imgsz_list": [512],
        "cls_gain_list": [2.0, 2.2, 2.6],
    },
    "extended": {
        "aug_presets": ["default", "industrial_soft", "industrial_conservative", "industrial_geometry_low"],
        "imgsz_list": [512, 640],
        "cls_gain_list": [1.8, 2.0, 2.2, 2.6, 3.0],
    },
}

AUTO_BATCH_BY_IMGSZ = {
    512: 24,
    640: 20,
    768: 16,
    896: 12,
}
MANUAL_BATCH_OVERRIDE = False

P3_HN_FINETUNE_PRESETS = {
    "quick": {
        "epochs": 60,
        "patience": 20,
        "lr0": 1.5e-4,
        "lrf": 0.05,
        "mosaic": 0.10,
        "close_mosaic": 15,
    },
    "full": {
        "epochs": 100,
        "patience": 30,
        "lr0": 1.0e-4,
        "lrf": 0.03,
        "mosaic": 0.10,
        "close_mosaic": 20,
    },
}

# ===================== 默认训练配置 =====================
BASE_CFG = dict(
    data=DATA_PATH,
    imgsz=512,
    batch=24,
    epochs=600,
    patience=120,
    optimizer="AdamW",
    lr0=3e-4,
    lrf=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    warmup_epochs=8,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=2.2,
    dfl=1.5,
    hsv_h=0.02,
    hsv_s=0.45,
    hsv_v=0.35,
    degrees=2.0,
    translate=0.05,
    scale=0.15,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.25,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=20,
    device=DEFAULT_DEVICE,
    workers=4,
    seed=43,
    deterministic=False,
    save=True,
    save_period=5,
    plots=True,
    val=True,
    amp=True,
    cache=False,
    cos_lr=True,
    exist_ok=False,
    freeze=0,
    nbs=64,
)

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


# ===================== 基础工具 =====================
def make_run_id(prefix="R"):
    return f"{prefix}{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def parse_device_list(dev) -> List[str]:
    if dev is None:
        return []
    if isinstance(dev, str):
        return [x.strip() for x in dev.split(",") if x.strip()]
    if isinstance(dev, (list, tuple)):
        return [str(x).strip() for x in dev if str(x).strip()]
    return []


def dataset_id(data_yaml: str) -> str:
    return Path(data_yaml).stem


def find_ultralytics_path():
    try:
        import ultralytics
        return Path(ultralytics.__file__).parent
    except Exception:
        return None


ULTRALYTICS_PATH = find_ultralytics_path()


def load_yaml_file(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash_json(obj: Any, n: int = 8) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[: max(1, int(n))]


def normalize_model_family(model_family: Optional[str]) -> str:
    fallback = os.environ.get("IDA_MODEL_FAMILY", DEFAULT_MODEL_FAMILY).strip() or DEFAULT_MODEL_FAMILY
    family = str(model_family or fallback).strip()
    return family if family in PRETRAINED_SEG_WEIGHTS_BY_FAMILY else "11"


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def get_scale_cfg_path(scale_key: str, model_family: Optional[str] = None):
    family = normalize_model_family(model_family)
    if ULTRALYTICS_PATH is None:
        return None
    candidates = [
        ULTRALYTICS_PATH / f"cfg/models/{family}/yolo{family}{scale_key}-seg.yaml",
        ULTRALYTICS_PATH / f"cfg/models/{family}/yolo{family}-seg.yaml",
        ULTRALYTICS_PATH / f"cfg/models/v{family}/yolo{family}{scale_key}-seg.yaml",
        ULTRALYTICS_PATH / f"cfg/models/v{family}/yolo{family}-seg.yaml",
        ULTRALYTICS_PATH / f"models/{family}/yolo{family}{scale_key}-seg.yaml",
        ULTRALYTICS_PATH / f"models/{family}/yolo{family}-seg.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def render_custom_yaml(scale_key: str, data_yaml: Optional[str] = None, model_family: Optional[str] = None):
    family = normalize_model_family(model_family)
    data_yaml = data_yaml or BASE_CFG["data"]
    base_yaml = get_scale_cfg_path(scale_key, model_family=family)
    if base_yaml is None:
        return f"yolo{family}{scale_key}-seg.yaml"

    with open(base_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nc = 80
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            dy = yaml.safe_load(f) or {}
        if "nc" in dy:
            nc = int(dy["nc"])
        elif isinstance(dy.get("names"), (list, tuple)):
            nc = len(dy["names"])
        elif isinstance(dy.get("names"), dict):
            nc = len(dy["names"])
    except Exception:
        pass

    cfg["nc"] = nc
    cfg["task"] = "segment"

    out_path = CUSTOM_CFG_DIR / f"yolo{family}{scale_key}-seg-{dataset_id(data_yaml)}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return str(out_path)


def candidate_local_weight_paths(ckpt_name: str) -> List[Path]:
    cands = []
    p = Path(ckpt_name)
    if p.exists():
        cands.append(p.resolve())
    cands.append((PROJECT_ROOT / ckpt_name).resolve())
    cands.append((PROJECT_ROOT / "weights" / ckpt_name).resolve())
    cands.append((RESULT_ROOT / "weights" / ckpt_name).resolve())
    cands.append((Path.home() / ".cache" / "ultralytics" / ckpt_name).resolve())
    cands.append((Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / ckpt_name).resolve())

    extra_roots_raw = os.environ.get("IDA_PRETRAINED_ROOTS", "").strip()
    if extra_roots_raw:
        for root_str in extra_roots_raw.split(os.pathsep):
            root_str = root_str.strip()
            if not root_str:
                continue
            root = Path(root_str).expanduser()
            cands.append((root / ckpt_name).resolve())

    uniq = []
    seen = set()
    for x in cands:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            uniq.append(x)
    return uniq


def find_local_weight(ckpt_name: str) -> Optional[Path]:
    for p in candidate_local_weight_paths(ckpt_name):
        if p.exists() and p.is_file():
            return p
    return None


def clear_model_checkpoint_refs(model):
    for attr in ("ckpt", "ckpt_path"):
        try:
            if hasattr(model, attr):
                setattr(model, attr, None)
        except Exception:
            pass

    try:
        if hasattr(model, "overrides") and isinstance(model.overrides, dict):
            model.overrides.pop("resume", None)
            model.overrides["pretrained"] = False
    except Exception:
        pass

    try:
        args = getattr(model, "args", None)
        if isinstance(args, dict):
            args["pretrained"] = False
        elif args is not None and hasattr(args, "pretrained"):
            setattr(args, "pretrained", False)
    except Exception:
        pass

    try:
        inner = getattr(model, "model", None)
        args = getattr(inner, "args", None)
        if isinstance(args, dict):
            args["pretrained"] = False
        elif args is not None and hasattr(args, "pretrained"):
            setattr(args, "pretrained", False)
    except Exception:
        pass


def load_model(model_source: str, scale_key: str, pretrained: bool = True, model_family: Optional[str] = None):
    family = normalize_model_family(model_family)
    print(f"[Init] 使用 YAML 构建模型: {model_source}")
    model = YOLO(model_source)
    model.overrides["amp"] = BASE_CFG["amp"]
    model.overrides["pretrained"] = False

    family_weights = PRETRAINED_SEG_WEIGHTS_BY_FAMILY.get(family, {})
    if pretrained and scale_key in family_weights:
        ckpt = family_weights[scale_key]
        local_ckpt = find_local_weight(ckpt)
        if local_ckpt is not None:
            try_source = str(local_ckpt.resolve())
            try:
                print(f"[Init] 加载本地预训练权重: {try_source}")
                model.load(try_source)
                model.overrides["pretrained"] = True
                clear_model_checkpoint_refs(model)
            except Exception as e:
                print(f"[Init][WARN] 本地预训练权重加载失败，继续使用纯 YAML 初始化: {e}")
        elif ALLOW_REMOTE_WEIGHTS:
            try_source = ckpt
            try:
                print(f"[Init] 加载远程预训练权重: {try_source}")
                model.load(try_source)
                model.overrides["pretrained"] = True
                clear_model_checkpoint_refs(model)
            except Exception as e:
                print(f"[Init][WARN] 远程预训练权重加载失败，继续使用纯 YAML 初始化: {e}")
        else:
            print(
                f"[Init][WARN] 未找到本地预训练权重 {ckpt}，且 IDA_ALLOW_REMOTE_WEIGHTS=0，"
                "将继续使用纯 YAML 初始化。"
            )

    clear_model_checkpoint_refs(model)
    return model


def load_model_from_checkpoint(weights_path: str):
    print(f"[Init] 从已有权重继续训练: {weights_path}")
    model = YOLO(str(weights_path))
    model.overrides["amp"] = BASE_CFG["amp"]
    model.overrides["pretrained"] = False
    clear_model_checkpoint_refs(model)
    return model


def sanitize_cfg(cfg: dict):
    cleaned = dict(cfg)
    for k in ("ema", "ema_decay", "accumulate", "fuse", "sync_bn"):
        cleaned.pop(k, None)
    return cleaned


def resolve_save_dir(cfg: dict, res) -> Path:
    if res is not None:
        sd = getattr(res, "save_dir", None)
        if sd:
            return Path(sd)
        if isinstance(res, dict) and res.get("save_dir"):
            return Path(res["save_dir"])

    project = Path(cfg.get("project", Path.cwd() / "runs" / "segment"))
    name = cfg.get("name", "train")
    save_dir = project / name
    if save_dir.exists():
        return save_dir

    candidates = [p for p in project.glob(f"{name}*") if p.is_dir()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return save_dir


def read_float(row: dict, keys, default=0.0):
    for k in keys:
        v = row.get(k, None)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
    return default


def read_int(row: dict, keys, default=-1):
    for k in keys:
        v = row.get(k, None)
        if v not in (None, ""):
            try:
                return int(float(v))
            except Exception:
                pass
    return default


def dedup_tuples(seq):
    out = []
    seen = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def pretty_num(v, nd=4):
    if v is None:
        return "None"
    try:
        fv = float(v)
        if math.isfinite(fv):
            return f"{fv:.{nd}f}"
    except Exception:
        pass
    return str(v)


def safe_float(v, default=-1e9):
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else default
    except Exception:
        return default


def normalize_official_postprocess_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = deepcopy(cfg) if isinstance(cfg, dict) else {}
    out = {
        "per_class_conf": cfg.get("per_class_conf", None),
        "per_class_min_area": cfg.get("per_class_min_area", None),
        "per_class_topk": cfg.get("per_class_topk", None),
        "image_gate": cfg.get("image_gate", None),
    }
    return out


def build_official_postprocess_items() -> List[Dict[str, Any]]:
    preset_names = OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES or ["base"]
    items: List[Dict[str, Any]] = []
    seen_names = set()

    for raw_name in preset_names:
        name = str(raw_name).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        cfg = normalize_official_postprocess_cfg(OFFICIAL_POSTPROCESS_PRESETS.get(name, None))
        cfg_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
        cfg_hash = hashlib.md5(cfg_json.encode("utf-8")).hexdigest()[:8]
        safe_name = re.sub(r"[^0-9a-zA-Z_-]+", "_", name).strip("_") or "preset"
        items.append(
            {
                "name": name,
                "cfg": cfg,
                "cfg_hash": cfg_hash,
                "tag": f"pp_{safe_name}_{cfg_hash}",
            }
        )

    if not items:
        cfg = normalize_official_postprocess_cfg({})
        cfg_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
        cfg_hash = hashlib.md5(cfg_json.encode("utf-8")).hexdigest()[:8]
        items.append({"name": "base", "cfg": cfg, "cfg_hash": cfg_hash, "tag": f"pp_base_{cfg_hash}"})

    return items


def build_postprocess_items_from_names(preset_names: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen_names = set()

    for raw_name in preset_names:
        name = str(raw_name).strip()
        if not name or name in seen_names or name not in OFFICIAL_POSTPROCESS_PRESETS:
            continue
        seen_names.add(name)
        cfg = normalize_official_postprocess_cfg(OFFICIAL_POSTPROCESS_PRESETS.get(name, None))
        cfg_hash = stable_hash_json(cfg, n=10)
        safe_name = re.sub(r"[^0-9a-zA-Z_-]+", "_", name).strip("_") or "preset"
        items.append(
            {
                "name": name,
                "cfg": cfg,
                "cfg_hash": cfg_hash,
                "tag": f"pp_{safe_name}_{cfg_hash}",
            }
        )

    if not items:
        cfg = normalize_official_postprocess_cfg({})
        cfg_hash = stable_hash_json(cfg, n=10)
        items.append({"name": "base", "cfg": cfg, "cfg_hash": cfg_hash, "tag": f"pp_base_{cfg_hash}"})
    return items


def get_train_aug_preset(name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {}
    return deepcopy(TRAIN_AUG_PRESETS.get(str(name), {}))


def apply_profile_train_overrides(cfg: Dict[str, Any], profile: dict) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    aug_preset_name = profile.get("aug_preset", None)
    aug_cfg = get_train_aug_preset(aug_preset_name)
    if aug_cfg:
        cfg.update(aug_cfg)

    direct_float_keys = [
        "copy_paste",
        "cls_gain",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "train_scale",
        "mosaic",
        "mixup",
        "weight_decay",
        "lr0",
        "lrf",
    ]
    cfg_key_map = {"cls_gain": "cls", "train_scale": "scale"}
    for key in direct_float_keys:
        val = profile.get(key, None)
        if val is None:
            continue
        cfg[cfg_key_map.get(key, key)] = float(val)

    direct_int_keys = ["imgsz_override", "batch_override", "epochs_override", "patience_override", "close_mosaic"]
    cfg_int_map = {
        "imgsz_override": "imgsz",
        "batch_override": "batch",
        "epochs_override": "epochs",
        "patience_override": "patience",
    }
    for key in direct_int_keys:
        val = profile.get(key, None)
        if val is None:
            continue
        cfg[cfg_int_map.get(key, key)] = int(val)

    if profile.get("auto_batch_from_imgsz", False) and (not MANUAL_BATCH_OVERRIDE):
        imgsz_now = int(cfg.get("imgsz", BASE_CFG["imgsz"]))
        auto_batch = AUTO_BATCH_BY_IMGSZ.get(imgsz_now, None)
        if auto_batch is not None and profile.get("batch_override", None) is None:
            cfg["batch"] = int(auto_batch)

    return cfg


# ===================== Track1 代理分数 =====================
def track1_proxy_score_from_row(row: dict) -> float:
    for k in ("metrics/Track1Proxy", "metrics/Track1", "metrics/STrack1", "track1", "Strack1", "STrack1"):
        v = row.get(k, None)
        if v not in (None, ""):
            try:
                score = float(v)
                if math.isfinite(score):
                    return score
            except Exception:
                pass

    box_p = read_float(row, ["metrics/precision(B)", "metrics/precision"])
    box_r = read_float(row, ["metrics/recall(B)", "metrics/recall"])
    box_m50 = read_float(row, ["metrics/mAP50(B)", "metrics/mAP50"])
    box_map = read_float(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])

    mask_p = read_float(row, ["metrics/precision(M)"], default=box_p)
    mask_r = read_float(row, ["metrics/recall(M)"], default=box_r)
    mask_m50 = read_float(row, ["metrics/mAP50(M)"], default=box_m50)
    mask_map = read_float(row, ["metrics/mAP50-95(M)"], default=box_map)

    prec = 0.5 * (box_p + mask_p)
    rec = 0.5 * (box_r + mask_r)

    # Approximate Track1's emphasis:
    # 0.30 * S_loc + 0.30 * S_cls + 0.40 * S_screen
    loc_proxy = 0.65 * mask_map + 0.35 * box_map
    cls_proxy = 0.50 * mask_m50 + 0.50 * box_m50
    # Specificity is not directly available from results.csv, so precision is
    # used as a conservative proxy for false-positive control.
    screen_proxy = 0.55 * rec + 0.45 * prec
    score = 0.30 * loc_proxy + 0.30 * cls_proxy + 0.40 * screen_proxy

    if not math.isfinite(score):
        return -1e9
    return float(score)


def score_row(row: dict) -> float:
    mode = os.environ.get("IDA_CHECKPOINT_SCORE_MODE", "track1").strip().lower()
    if mode in {"ultra", "default", "stock", "ultralytics"}:
        ultra_fit = read_float(row, ["metrics/UltraFitness", "fitness"], default=-1.0)
        if ultra_fit >= 0:
            return ultra_fit
        box_m50 = read_float(row, ["metrics/mAP50(B)", "metrics/mAP50"])
        box_map = read_float(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
        mask_m50 = read_float(row, ["metrics/mAP50(M)"], default=box_m50)
        mask_map = read_float(row, ["metrics/mAP50-95(M)"], default=box_map)
        score = 0.1 * mask_m50 + 0.9 * mask_map
        return float(score) if math.isfinite(score) else -1e9
    return track1_proxy_score_from_row(row)


# ===================== 插件配置同步（DDP 友好） =====================
def build_plugin_cfg(profile: dict):
    return dict(
        enable_enhance=bool(profile.get("enhance", False)),
        enhance_edge_ks=int(profile.get("edge_ks", 5)),
        enhance_rates=tuple(profile.get("dilated_rates", (1, 2, 3))),
        enable_mixstyle=bool(profile.get("mixstyle", False)),
        mixstyle_prob=float(profile.get("mixstyle_prob", 0.3)),
        mixstyle_alpha=float(profile.get("mixstyle_alpha", 0.3)),
        mixstyle_layers=int(profile.get("mixstyle_layers", 1)),
        mixstyle_mode=str(profile.get("mixstyle_mode", "mixstyle")),
        mixstyle_low_freq_ratio=float(profile.get("mixstyle_low_freq_ratio", 0.10)),
        enable_randconv=bool(profile.get("randconv", False)),
        randconv_prob=float(profile.get("randconv_prob", 0.20)),
        randconv_prob_end=float(profile.get("randconv_prob_end", 0.45)),
        randconv_sigma=float(profile.get("randconv_sigma", 0.0)),
        randconv_sigma_end=float(profile.get("randconv_sigma_end", 0.15)),
        randconv_layers=int(profile.get("randconv_layers", 1)),
        randconv_kernel_sizes=tuple(profile.get("randconv_kernel_sizes", (3, 5))),
        randconv_refresh_interval=int(profile.get("randconv_refresh_interval", 1)),
        protect_shallow_textures=bool(profile.get("protect_shallow_textures", False)),
        enable_ibn=bool(profile.get("ibn", True)),
        ibn_ratio=float(profile.get("ibn_ratio", 0.5)),
        ibn_layers=int(profile.get("ibn_layers", 1)),
        enable_spd=bool(profile.get("spd", False)),
        spd_layers=int(profile.get("spd_layers", 2)),
        spd_scale=int(profile.get("spd_scale", 2)),
        spd_alpha_init=float(profile.get("spd_alpha_init", 0.10)),
        enable_simam=bool(profile.get("simam", False)),
        simam_layers=int(profile.get("simam_layers", 2)),
        simam_e_lambda=float(profile.get("simam_e_lambda", 1e-4)),
        enable_safe_concat=bool(profile.get("safe_concat", False)),
    )


def build_loss_cfg(profile: dict):
    return dict(
        use_vfl=bool(profile.get("use_vfl", False)),
        vfl_alpha=float(profile.get("vfl_alpha", 0.75)),
        vfl_gamma=float(profile.get("vfl_gamma", 2.0)),
        nwd_weight=float(profile.get("nwd_weight", 0.0)),
        nwd_constant=float(profile.get("nwd_constant", 12.8)),
        screen_loss_weight=float(profile.get("screen_loss_weight", 0.0)),
        screen_pos_weight=float(profile.get("screen_pos_weight", 1.0)),
        screen_neg_weight=float(profile.get("screen_neg_weight", 1.0)),
    )


def apply_official_eval_overrides_from_profile(profile: dict):
    global RUN_OFFICIAL_EVAL_AFTER_TRAIN, RUN_OFFICIAL_SWEEP_AFTER_TRAIN
    global OFFICIAL_PRED_CONF, OFFICIAL_PRED_IOU, OFFICIAL_PRED_MAX_DET, OFFICIAL_PRED_TTA
    global OFFICIAL_SWEEP_CONF_LIST, OFFICIAL_SWEEP_IOU_LIST, OFFICIAL_SWEEP_MAXDET_LIST
    global OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES
    global OFFICIAL_EVAL_INCLUDE_PERIODIC, OFFICIAL_EVAL_CANDIDATE_TYPES, OFFICIAL_EVAL_MAX_CANDIDATES

    backup = dict(
        RUN_OFFICIAL_EVAL_AFTER_TRAIN=RUN_OFFICIAL_EVAL_AFTER_TRAIN,
        RUN_OFFICIAL_SWEEP_AFTER_TRAIN=RUN_OFFICIAL_SWEEP_AFTER_TRAIN,
        OFFICIAL_PRED_CONF=OFFICIAL_PRED_CONF,
        OFFICIAL_PRED_IOU=OFFICIAL_PRED_IOU,
        OFFICIAL_PRED_MAX_DET=OFFICIAL_PRED_MAX_DET,
        OFFICIAL_PRED_TTA=OFFICIAL_PRED_TTA,
        OFFICIAL_SWEEP_CONF_LIST=deepcopy(OFFICIAL_SWEEP_CONF_LIST),
        OFFICIAL_SWEEP_IOU_LIST=deepcopy(OFFICIAL_SWEEP_IOU_LIST),
        OFFICIAL_SWEEP_MAXDET_LIST=deepcopy(OFFICIAL_SWEEP_MAXDET_LIST),
        OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES=deepcopy(OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES),
        OFFICIAL_EVAL_INCLUDE_PERIODIC=OFFICIAL_EVAL_INCLUDE_PERIODIC,
        OFFICIAL_EVAL_CANDIDATE_TYPES=None if OFFICIAL_EVAL_CANDIDATE_TYPES is None else deepcopy(OFFICIAL_EVAL_CANDIDATE_TYPES),
        OFFICIAL_EVAL_MAX_CANDIDATES=OFFICIAL_EVAL_MAX_CANDIDATES,
    )

    if "official_eval_enabled" in profile:
        RUN_OFFICIAL_EVAL_AFTER_TRAIN = bool(profile.get("official_eval_enabled"))
    if "official_sweep_enabled" in profile:
        RUN_OFFICIAL_SWEEP_AFTER_TRAIN = bool(profile.get("official_sweep_enabled"))
    if profile.get("official_pred_conf", None) is not None:
        OFFICIAL_PRED_CONF = float(profile.get("official_pred_conf"))
    if profile.get("official_pred_iou", None) is not None:
        OFFICIAL_PRED_IOU = float(profile.get("official_pred_iou"))
    if profile.get("official_pred_max_det", None) is not None:
        OFFICIAL_PRED_MAX_DET = int(profile.get("official_pred_max_det"))
    if profile.get("official_pred_tta", None) is not None:
        OFFICIAL_PRED_TTA = bool(profile.get("official_pred_tta"))
    if profile.get("official_sweep_confs", None) is not None:
        OFFICIAL_SWEEP_CONF_LIST = [float(x) for x in profile.get("official_sweep_confs", [])]
    if profile.get("official_sweep_ious", None) is not None:
        OFFICIAL_SWEEP_IOU_LIST = [float(x) for x in profile.get("official_sweep_ious", [])]
    if profile.get("official_sweep_maxdets", None) is not None:
        OFFICIAL_SWEEP_MAXDET_LIST = [int(x) for x in profile.get("official_sweep_maxdets", [])]
    if profile.get("official_postprocess_preset_names", None) is not None:
        OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES = [str(x) for x in profile.get("official_postprocess_preset_names", [])]
    if profile.get("official_eval_include_periodic", None) is not None:
        OFFICIAL_EVAL_INCLUDE_PERIODIC = bool(profile.get("official_eval_include_periodic"))
    if profile.get("official_eval_candidate_types", None) is not None:
        OFFICIAL_EVAL_CANDIDATE_TYPES = [str(x) for x in profile.get("official_eval_candidate_types", [])]
    if profile.get("official_eval_max_candidates", None) is not None:
        OFFICIAL_EVAL_MAX_CANDIDATES = int(profile.get("official_eval_max_candidates"))

    return backup


def restore_official_eval_overrides(backup: dict):
    global RUN_OFFICIAL_EVAL_AFTER_TRAIN, RUN_OFFICIAL_SWEEP_AFTER_TRAIN
    global OFFICIAL_PRED_CONF, OFFICIAL_PRED_IOU, OFFICIAL_PRED_MAX_DET, OFFICIAL_PRED_TTA
    global OFFICIAL_SWEEP_CONF_LIST, OFFICIAL_SWEEP_IOU_LIST, OFFICIAL_SWEEP_MAXDET_LIST
    global OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES
    global OFFICIAL_EVAL_INCLUDE_PERIODIC, OFFICIAL_EVAL_CANDIDATE_TYPES, OFFICIAL_EVAL_MAX_CANDIDATES

    RUN_OFFICIAL_EVAL_AFTER_TRAIN = bool(backup["RUN_OFFICIAL_EVAL_AFTER_TRAIN"])
    RUN_OFFICIAL_SWEEP_AFTER_TRAIN = bool(backup["RUN_OFFICIAL_SWEEP_AFTER_TRAIN"])
    OFFICIAL_PRED_CONF = float(backup["OFFICIAL_PRED_CONF"])
    OFFICIAL_PRED_IOU = float(backup["OFFICIAL_PRED_IOU"])
    OFFICIAL_PRED_MAX_DET = int(backup["OFFICIAL_PRED_MAX_DET"])
    OFFICIAL_PRED_TTA = bool(backup["OFFICIAL_PRED_TTA"])
    OFFICIAL_SWEEP_CONF_LIST = deepcopy(backup["OFFICIAL_SWEEP_CONF_LIST"])
    OFFICIAL_SWEEP_IOU_LIST = deepcopy(backup["OFFICIAL_SWEEP_IOU_LIST"])
    OFFICIAL_SWEEP_MAXDET_LIST = deepcopy(backup["OFFICIAL_SWEEP_MAXDET_LIST"])
    OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES = deepcopy(backup["OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES"])
    OFFICIAL_EVAL_INCLUDE_PERIODIC = bool(backup["OFFICIAL_EVAL_INCLUDE_PERIODIC"])
    OFFICIAL_EVAL_CANDIDATE_TYPES = None if backup["OFFICIAL_EVAL_CANDIDATE_TYPES"] is None else deepcopy(backup["OFFICIAL_EVAL_CANDIDATE_TYPES"])
    OFFICIAL_EVAL_MAX_CANDIDATES = backup["OFFICIAL_EVAL_MAX_CANDIDATES"]


def make_light_official_eval_settings(
    conf: float = 0.08,
    iou: float = 0.30,
    max_det: int = 50,
    postprocess_presets: Optional[List[str]] = None,
    tta: bool = False,
):
    return dict(
        official_eval_enabled=True,
        official_sweep_enabled=False,
        official_pred_conf=float(conf),
        official_pred_iou=float(iou),
        official_pred_max_det=int(max_det),
        official_pred_tta=bool(tta),
        official_postprocess_preset_names=list(postprocess_presets or ["base"]),
        official_eval_include_periodic=False,
        official_eval_candidate_types=["swad_best", "swa_best", "best"],
        official_eval_max_candidates=1,
    )


def make_texture_preserve_official_eval_settings(
    conf: float = 0.08,
    iou: float = 0.30,
    max_det: int = 50,
    tta: bool = False,
):
    return make_light_official_eval_settings(
        conf=conf,
        iou=iou,
        max_det=max_det,
        postprocess_presets=TEXTURE_PRESERVE_POSTPROCESS_PRESETS,
        tta=tta,
    )


# ===================== results.csv 读取 =====================
def load_results_rows(results_csv: Path):
    if not results_csv.exists():
        return []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {(k.strip() if isinstance(k, str) else k): v for k, v in row.items()}
            for row in reader
        ]


def pick_best_last_rows(rows):
    if not rows:
        return None, None
    best = max(rows, key=score_row)
    last = rows[-1]
    return best, last


def extract_metrics_from_row(row):
    if not row:
        return {
            "epoch": -1,
            "score": 0.0,
            "ultra_fitness": -1.0,
            "box_p": 0.0,
            "box_r": 0.0,
            "box_m50": 0.0,
            "box_map": 0.0,
            "mask_p": 0.0,
            "mask_r": 0.0,
            "mask_m50": 0.0,
            "mask_map": 0.0,
        }

    box_p = read_float(row, ["metrics/precision(B)", "metrics/precision"])
    box_r = read_float(row, ["metrics/recall(B)", "metrics/recall"])
    box_m50 = read_float(row, ["metrics/mAP50(B)", "metrics/mAP50"])
    box_map = read_float(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])

    return {
        "epoch": read_int(row, ["epoch"], -1),
        "score": score_row(row),
        "ultra_fitness": read_float(row, ["metrics/UltraFitness", "fitness"], default=-1.0),
        "box_p": box_p,
        "box_r": box_r,
        "box_m50": box_m50,
        "box_map": box_map,
        "mask_p": read_float(row, ["metrics/precision(M)"], default=box_p),
        "mask_r": read_float(row, ["metrics/recall(M)"], default=box_r),
        "mask_m50": read_float(row, ["metrics/mAP50(M)"], default=box_m50),
        "mask_map": read_float(row, ["metrics/mAP50-95(M)"], default=box_map),
    }


def extract_best_last_metrics(save_dir: Path):
    rows = load_results_rows(save_dir / "results.csv")
    best_row, last_row = pick_best_last_rows(rows)
    return extract_metrics_from_row(best_row), extract_metrics_from_row(last_row)


def print_best_summary(save_dir: Path):
    best, last = extract_best_last_metrics(save_dir)
    if best["epoch"] < 0:
        print("[Summary] 无法解析 results.csv")
        return

    print(
        f"[BEST@{best['epoch']}] "
        f"Track1Proxy={best['score']:.4f} "
        f"UltraFitness={best['ultra_fitness']:.4f} "
        f"Box(P/R/mAP50/mAP)={best['box_p']:.4f}/{best['box_r']:.4f}/{best['box_m50']:.4f}/{best['box_map']:.4f} "
        f"Mask(P/R/mAP50/mAP)={best['mask_p']:.4f}/{best['mask_r']:.4f}/{best['mask_m50']:.4f}/{best['mask_map']:.4f}"
    )
    print(
        f"[LAST@{last['epoch']}] "
        f"Track1Proxy={last['score']:.4f} "
        f"UltraFitness={last['ultra_fitness']:.4f} "
        f"Box mAP50-95={last['box_map']:.4f} "
        f"Mask mAP50-95={last['mask_map']:.4f}"
    )


def safe_torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def parse_epoch_checkpoint(path: Path) -> Optional[int]:
    m = re.fullmatch(r"epoch(\d+)", path.stem.lower())
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def list_periodic_checkpoints(weights_dir: Path):
    out = {}
    if not weights_dir.exists():
        return out
    for p in weights_dir.glob("epoch*.pt"):
        ep = parse_epoch_checkpoint(p)
        if ep is not None:
            out[int(ep)] = p
    return out


def collect_eval_candidate_items(save_dir: Path) -> List[Dict[str, Any]]:
    weights_dir = save_dir / "weights"
    rows = load_results_rows(save_dir / "results.csv")
    best_row, last_row = pick_best_last_rows(rows)
    best_epoch = read_int(best_row, ["epoch"], -1) if best_row else -1
    last_epoch = read_int(last_row, ["epoch"], -1) if last_row else -1

    recorded_epochs = []
    for row in rows:
        ep = read_int(row, ["epoch"], -1)
        if ep >= 0:
            recorded_epochs.append(int(ep))
    max_recorded_epoch = max(recorded_epochs) if recorded_epochs else -1

    periodic = list_periodic_checkpoints(weights_dir)
    periodic_epochs = sorted(periodic)
    selected_periodic_epochs: List[int] = []
    if OFFICIAL_EVAL_INCLUDE_PERIODIC and periodic_epochs:
        start_epoch = 0
        if max_recorded_epoch >= 0:
            start_epoch = int(math.ceil(max_recorded_epoch * float(OFFICIAL_EVAL_PERIODIC_START_RATIO)))
        selected_periodic_epochs = [ep for ep in periodic_epochs if ep >= start_epoch]
        stride = max(1, int(OFFICIAL_EVAL_PERIODIC_STRIDE))
        if stride > 1:
            selected_periodic_epochs = selected_periodic_epochs[::stride]

    items: List[Dict[str, Any]] = []
    seen = set()

    def _add(path: Path, candidate_type: str, candidate_epoch: Optional[int]) -> None:
        if path is None or (not path.exists()):
            return
        rp = path.resolve()
        sp = str(rp)
        if sp in seen:
            return
        seen.add(sp)
        items.append(
            {
                "path": rp,
                "weights": str(rp),
                "weights_name": rp.name,
                "candidate_type": str(candidate_type),
                "candidate_epoch": None if candidate_epoch is None or int(candidate_epoch) < 0 else int(candidate_epoch),
            }
        )

    _add(weights_dir / "best.pt", "best", best_epoch if best_epoch >= 0 else None)
    _add(weights_dir / "last.pt", "last", last_epoch if last_epoch >= 0 else None)
    _add(weights_dir / "swa_best.pt", "swa_best", None)
    _add(weights_dir / "swad_best.pt", "swad_best", None)
    for ep in selected_periodic_epochs:
        _add(periodic[ep], "epoch", ep)

    if OFFICIAL_EVAL_CANDIDATE_TYPES:
        preferred = [str(x) for x in OFFICIAL_EVAL_CANDIDATE_TYPES]
        items_by_type = {}
        for item in items:
            items_by_type.setdefault(str(item.get("candidate_type")), []).append(item)
        filtered = []
        seen_path = set()
        for ctype in preferred:
            for item in items_by_type.get(ctype, []):
                sp = str(item["path"])
                if sp in seen_path:
                    continue
                seen_path.add(sp)
                filtered.append(item)
        if filtered:
            items = filtered

    if OFFICIAL_EVAL_MAX_CANDIDATES is not None:
        items = items[: max(0, int(OFFICIAL_EVAL_MAX_CANDIDATES))]

    return items


# ===================== Checkpoint 加载自检 =====================
def collect_eval_candidates(save_dir: Path):
    return [item["path"] for item in collect_eval_candidate_items(save_dir)]


def verify_checkpoints_loadable(save_dir: Path, strict: bool = True):
    candidates = collect_eval_candidates(save_dir)
    summary = {"ok": True, "items": []}
    errors = []

    for p in candidates:
        item = {"path": str(p), "torch_load": False, "yolo_load": False, "error": None}
        try:
            obj = safe_torch_load(p)
            item["torch_load"] = True
            del obj
        except Exception as e:
            item["error"] = f"torch.load failed: {e}"
            errors.append(item["error"])

        if item["torch_load"]:
            try:
                model = YOLO(str(p))
                item["yolo_load"] = True
                del model
            except Exception as e:
                item["error"] = f"YOLO load failed: {e}"
                errors.append(item["error"])

        summary["items"].append(item)

    summary["ok"] = len(errors) == 0
    out_json = save_dir / "weights" / "checkpoint_loadability.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for item in summary["items"]:
        print(
            f"[CKPT-VERIFY] {Path(item['path']).name} "
            f"torch_load={item['torch_load']} yolo_load={item['yolo_load']} "
            f"{'' if item['error'] is None else 'error=' + item['error']}"
        )

    if strict and (not summary["ok"]):
        raise RuntimeError(f"发现不可加载的 checkpoint，详见: {out_json}")
    return summary


# ===================== Checkpoint Averaging =====================
def choose_swa_candidates(
    save_dir: Path,
    window: int = 4,
    max_models: int = 6,
    min_models: int = 3,
    include_best: bool = True,
    include_last: bool = False,
):
    weights_dir = save_dir / "weights"
    rows = load_results_rows(save_dir / "results.csv")
    best_row, last_row = pick_best_last_rows(rows)

    if best_row is None:
        return dict(ok=False, reason="results.csv is empty", paths=[])

    best_epoch = read_int(best_row, ["epoch"], -1)
    score_by_epoch = {
        read_int(row, ["epoch"], -1): score_row(row)
        for row in rows
        if read_int(row, ["epoch"], -1) >= 0
    }
    periodic = list_periodic_checkpoints(weights_dir)
    sorted_epochs = sorted(periodic)

    candidate_paths = []

    if include_best and (weights_dir / "best.pt").exists():
        candidate_paths.append(weights_dir / "best.pt")

    if sorted_epochs:
        anchor_idx = min(range(len(sorted_epochs)), key=lambda i: abs(sorted_epochs[i] - best_epoch))
        lo = max(0, anchor_idx - max(0, int(window)))
        hi = min(len(sorted_epochs), anchor_idx + max(0, int(window)) + 1)
        selected_epochs = sorted_epochs[lo:hi]

        if include_best and best_epoch in selected_epochs:
            selected_epochs = [e for e in selected_epochs if e != best_epoch]

        if len(selected_epochs) > max_models:
            selected_epochs = sorted(
                selected_epochs,
                key=lambda e: (abs(e - best_epoch), -score_by_epoch.get(e, -1e9)),
            )[:max_models]
            selected_epochs = sorted(selected_epochs)

        candidate_paths.extend(periodic[e] for e in selected_epochs)

    if include_last and (weights_dir / "last.pt").exists():
        candidate_paths.append(weights_dir / "last.pt")
    elif len(candidate_paths) < min_models and (weights_dir / "last.pt").exists():
        candidate_paths.append(weights_dir / "last.pt")

    uniq_paths = []
    seen = set()
    for p in candidate_paths:
        sp = str(p.resolve())
        if p.exists() and sp not in seen:
            seen.add(sp)
            uniq_paths.append(p.resolve())

    enough = len(uniq_paths) >= max(2, int(min_models))
    return dict(
        ok=enough,
        reason="" if enough else "not enough checkpoints",
        strategy="window",
        paths=uniq_paths,
        best_epoch=int(best_epoch),
        best_score=float(score_row(best_row)),
        last_epoch=read_int(last_row, ["epoch"], -1) if last_row else -1,
    )


def build_epoch_checkpoint_score_table(save_dir: Path):
    weights_dir = save_dir / "weights"
    rows = load_results_rows(save_dir / "results.csv")
    periodic = list_periodic_checkpoints(weights_dir)

    row_by_epoch = {}
    for row in rows:
        ep = read_int(row, ["epoch"], -1)
        if ep >= 0:
            row_by_epoch[int(ep)] = row

    table = []
    for ep in sorted(periodic):
        row = row_by_epoch.get(int(ep))
        if row is None:
            continue
        score = score_row(row)
        if not math.isfinite(score):
            continue
        table.append(
            {
                "epoch": int(ep),
                "path": periodic[int(ep)].resolve(),
                "score": float(score),
            }
        )
    return table


def centered_moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, int(window))
    radius = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        seg = values[lo:hi]
        out.append(float(sum(seg) / max(1, len(seg))))
    return out


def pick_evenly_spaced_indices(indices: List[int], target: int, anchor: Optional[int] = None) -> List[int]:
    if len(indices) <= max(0, int(target)):
        return list(indices)
    target = max(1, int(target))
    picks = {indices[0], indices[-1]}
    if anchor is not None and anchor in indices:
        picks.add(int(anchor))
    if len(picks) >= target:
        ordered = [x for x in indices if x in picks]
        return ordered[:target]

    if target > 1:
        denom = max(1, target - 1)
        for j in range(target):
            pos = int(round((len(indices) - 1) * (j / float(denom))))
            picks.add(indices[pos])
            if len(picks) >= target:
                break
    else:
        picks.add(indices[0])
    ordered = [x for x in indices if x in picks]
    if len(ordered) < target:
        for x in indices:
            if x in picks:
                continue
            ordered.append(x)
            if len(ordered) >= target:
                break
    return ordered[:target]


def choose_swad_candidates(
    save_dir: Path,
    max_models: int = 8,
    min_models: int = 4,
    smooth_window: int = 5,
    tolerance_ratio: float = 0.25,
    n_converge: int = 2,
    include_best: bool = True,
    include_last: bool = False,
):
    table = build_epoch_checkpoint_score_table(save_dir)
    weights_dir = save_dir / "weights"
    rows = load_results_rows(save_dir / "results.csv")
    best_row, last_row = pick_best_last_rows(rows)

    if len(table) < 2:
        fallback = choose_swa_candidates(
            save_dir=save_dir,
            window=max(2, int(max_models)),
            max_models=max_models,
            min_models=min_models,
            include_best=include_best,
            include_last=include_last,
        )
        fallback["strategy"] = "swad_fallback_window"
        fallback["reason"] = fallback.get("reason") or "not enough periodic checkpoints for swad"
        return fallback

    epochs = [item["epoch"] for item in table]
    scores = [item["score"] for item in table]
    smooth_scores = centered_moving_average(scores, window=smooth_window)
    best_idx = max(range(len(table)), key=lambda i: (smooth_scores[i], scores[i], -abs(epochs[i])))
    best_smooth = float(smooth_scores[best_idx])
    min_smooth = float(min(smooth_scores))
    score_span = max(best_smooth - min_smooth, 1e-6)
    tolerance_abs = max(1e-4, float(tolerance_ratio) * score_span)
    floor_score = best_smooth - tolerance_abs

    left = best_idx
    misses = 0
    for i in range(best_idx - 1, -1, -1):
        if smooth_scores[i] >= floor_score:
            left = i
            misses = 0
        else:
            misses += 1
            if misses >= max(1, int(n_converge)):
                break

    right = best_idx
    misses = 0
    for i in range(best_idx + 1, len(table)):
        if smooth_scores[i] >= floor_score:
            right = i
            misses = 0
        else:
            misses += 1
            if misses >= max(1, int(n_converge)):
                break

    candidate_indices = [i for i in range(left, right + 1) if smooth_scores[i] >= floor_score]
    if best_idx not in candidate_indices:
        candidate_indices.append(best_idx)
        candidate_indices = sorted(set(candidate_indices))

    if len(candidate_indices) < max(2, int(min_models)):
        proximity = sorted(
            range(len(table)),
            key=lambda i: (abs(i - best_idx), -smooth_scores[i], -scores[i]),
        )
        for i in proximity:
            if i not in candidate_indices:
                candidate_indices.append(i)
            if len(candidate_indices) >= max(2, int(min_models)):
                break
        candidate_indices = sorted(set(candidate_indices))

    if len(candidate_indices) > max_models:
        ranked = sorted(
            candidate_indices,
            key=lambda i: (smooth_scores[i], scores[i], -abs(i - best_idx)),
            reverse=True,
        )
        keep = []
        if best_idx in candidate_indices:
            keep.append(best_idx)
        for i in ranked:
            if i in keep:
                continue
            keep.append(i)
            if len(keep) >= max_models:
                break
        candidate_indices = sorted(keep)

    candidate_paths = [table[i]["path"] for i in candidate_indices]
    if include_best and (weights_dir / "best.pt").exists():
        candidate_paths.insert(0, (weights_dir / "best.pt").resolve())
    if include_last and (weights_dir / "last.pt").exists():
        candidate_paths.append((weights_dir / "last.pt").resolve())

    uniq_paths = []
    seen = set()
    for p in candidate_paths:
        sp = str(Path(p).resolve())
        if sp in seen or (not Path(p).exists()):
            continue
        seen.add(sp)
        uniq_paths.append(Path(p).resolve())

    enough = len(uniq_paths) >= max(2, int(min_models))
    valley_epochs = [epochs[i] for i in candidate_indices]
    selected_scores = [scores[i] for i in candidate_indices]
    best_epoch = read_int(best_row, ["epoch"], -1) if best_row else -1
    return dict(
        ok=enough,
        reason="" if enough else "not enough checkpoints selected by swad",
        strategy="swad",
        paths=uniq_paths,
        best_epoch=int(best_epoch),
        best_score=float(score_row(best_row)) if best_row else -1.0,
        last_epoch=read_int(last_row, ["epoch"], -1) if last_row else -1,
        valley_start_epoch=int(min(valley_epochs)) if valley_epochs else -1,
        valley_end_epoch=int(max(valley_epochs)) if valley_epochs else -1,
        valley_anchor_epoch=int(epochs[best_idx]),
        smooth_window=int(smooth_window),
        tolerance_ratio=float(tolerance_ratio),
        tolerance_abs=float(tolerance_abs),
        n_converge=int(n_converge),
        selected_epochs=[int(x) for x in valley_epochs],
        selected_scores=[float(x) for x in selected_scores],
        smoothed_best=float(best_smooth),
    )


def extract_checkpoint_state_source(ckpt):
    if isinstance(ckpt, dict):
        for key in ("ema", "model", "state_dict"):
            obj = ckpt.get(key, None)
            if obj is not None:
                return key, obj
    if hasattr(ckpt, "state_dict"):
        return "model", ckpt
    return None, None


def to_state_dict(source):
    if isinstance(source, dict):
        return source
    if hasattr(source, "state_dict"):
        return source.state_dict()
    raise TypeError(f"Unsupported checkpoint source: {type(source)}")


def average_state_dicts(state_dicts: List[dict]):
    if not state_dicts:
        raise ValueError("state_dicts is empty")

    base = state_dicts[0]
    out = {}
    n = len(state_dicts)

    for key, base_val in base.items():
        if torch.is_tensor(base_val) and torch.is_floating_point(base_val):
            acc = None
            compatible = True
            for sd in state_dicts:
                cur = sd.get(key, None)
                if (not torch.is_tensor(cur)) or cur.shape != base_val.shape:
                    compatible = False
                    break
                cur_f32 = cur.detach().float()
                acc = cur_f32 if acc is None else acc + cur_f32
            out[key] = (acc / float(n)).to(dtype=base_val.dtype) if compatible else base_val.clone()
        elif torch.is_tensor(base_val):
            out[key] = base_val.clone()
        else:
            out[key] = deepcopy(base_val)

    return out


def build_swa_checkpoint(
    save_dir: Path,
    window: int = 4,
    max_models: int = 6,
    min_models: int = 3,
    include_best: bool = True,
    include_last: bool = False,
    out_name: str = "swa_best.pt",
    strategy: str = "window",
    swad_smooth_window: int = 5,
    swad_tolerance_ratio: float = 0.25,
    swad_n_converge: int = 2,
):
    strategy = str(strategy).strip().lower()
    if strategy == "swad":
        selection = choose_swad_candidates(
            save_dir=save_dir,
            max_models=max_models,
            min_models=min_models,
            smooth_window=swad_smooth_window,
            tolerance_ratio=swad_tolerance_ratio,
            n_converge=swad_n_converge,
            include_best=include_best,
            include_last=include_last,
        )
    else:
        selection = choose_swa_candidates(
            save_dir=save_dir,
            window=window,
            max_models=max_models,
            min_models=min_models,
            include_best=include_best,
            include_last=include_last,
        )

    if not selection.get("ok", False):
        print(f"[CKPT-AVG] 跳过 checkpoint averaging: {selection.get('reason', 'unknown reason')}")
        return dict(ok=False, reason=selection.get("reason", "unknown"), paths=selection.get("paths", []))

    loaded = []
    for ckpt_path in selection["paths"]:
        ckpt = safe_torch_load(ckpt_path)
        source_key, source_obj = extract_checkpoint_state_source(ckpt)
        if source_obj is None:
            raise RuntimeError(f"无法从 checkpoint 中提取权重: {ckpt_path}")
        loaded.append((ckpt_path, ckpt, source_key, source_obj, to_state_dict(source_obj)))

    base_path, base_ckpt, base_key, base_source, _ = loaded[0]
    avg_state = average_state_dicts([item[4] for item in loaded])

    avg_source = deepcopy(base_source)
    if hasattr(avg_source, "load_state_dict"):
        avg_source.load_state_dict(avg_state, strict=False)
    else:
        avg_source = avg_state

    out_ckpt = deepcopy(base_ckpt)
    if isinstance(out_ckpt, dict):
        if base_key in ("ema", "model"):
            try:
                out_ckpt["ema"] = deepcopy(avg_source)
            except Exception:
                out_ckpt["ema"] = avg_source
            try:
                out_ckpt["model"] = deepcopy(avg_source)
            except Exception:
                out_ckpt["model"] = avg_source
        else:
            out_ckpt[base_key] = avg_source

        out_ckpt["epoch"] = int(selection.get("best_epoch", -1))
        out_ckpt["best_fitness"] = float(selection.get("best_score", -1.0))
        out_ckpt["swa_info"] = dict(
            enabled=True,
            type=str(selection.get("strategy", strategy)),
            source_paths=[str(p) for p in selection["paths"]],
            anchor_checkpoint=str(base_path),
            best_epoch=int(selection.get("best_epoch", -1)),
        )
        for key in (
            "strategy",
            "valley_start_epoch",
            "valley_end_epoch",
            "valley_anchor_epoch",
            "smooth_window",
            "tolerance_ratio",
            "tolerance_abs",
            "n_converge",
            "selected_epochs",
            "selected_scores",
            "smoothed_best",
        ):
            if key in selection:
                out_ckpt["swa_info"][key] = deepcopy(selection[key])

    out_path = save_dir / "weights" / out_name
    torch.save(out_ckpt, str(out_path))
    alias_path = None
    if str(selection.get("strategy", strategy)).startswith("swad") and out_name != "swad_best.pt":
        alias_path = save_dir / "weights" / "swad_best.pt"
        torch.save(out_ckpt, str(alias_path))

    summary = dict(
        ok=True,
        path=str(out_path),
        num_models=len(selection["paths"]),
        paths=[str(p) for p in selection["paths"]],
        best_epoch=int(selection.get("best_epoch", -1)),
        best_score=float(selection.get("best_score", -1.0)),
        strategy=str(selection.get("strategy", strategy)),
    )
    for key in (
        "valley_start_epoch",
        "valley_end_epoch",
        "valley_anchor_epoch",
        "smooth_window",
        "tolerance_ratio",
        "tolerance_abs",
        "n_converge",
        "selected_epochs",
        "selected_scores",
        "smoothed_best",
    ):
        if key in selection:
            summary[key] = deepcopy(selection[key])
    if alias_path is not None:
        summary["alias_path"] = str(alias_path)

    with open(save_dir / "weights" / "swa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if alias_path is not None:
        with open(save_dir / "weights" / "swad_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[CKPT-AVG] 已保存: {out_path} "
        f"(strategy={summary.get('strategy')} num_models={summary['num_models']})"
    )
    return summary


# ===================== 插件相关 =====================
def apply_plugins(profile: dict):
    plugin_cfg = build_plugin_cfg(profile)
    loss_cfg = build_loss_cfg(profile)
    register_plugins(**plugin_cfg, verbose=True)
    os.environ[PLUGIN_ENV_KEY] = json.dumps(plugin_cfg, ensure_ascii=False)
    os.environ[LOSS_ENV_KEY] = json.dumps(loss_cfg, ensure_ascii=False)


# ===================== 官方评测辅助函数 =====================
def load_python_module_from_path(module_path: Path, module_name: str):
    if not module_path.exists():
        raise FileNotFoundError(f"模块不存在: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_data_entries(data_yaml: str, split: str = "val") -> List[Path]:
    dy = load_yaml_file(data_yaml)
    if split not in dy:
        raise KeyError(f"data.yaml 中不存在 split='{split}'")

    entries = dy[split]
    if not isinstance(entries, (list, tuple)):
        entries = [entries]

    yaml_dir = Path(data_yaml).resolve().parent
    root = dy.get("path", None)
    root_dir = None
    if root:
        root_dir = Path(root)
        if not root_dir.is_absolute():
            root_dir = (yaml_dir / root_dir).resolve()

    out = []
    for e in entries:
        p = Path(str(e))
        if not p.is_absolute():
            if root_dir is not None:
                p1 = (root_dir / p).resolve()
                if p1.exists():
                    p = p1
                else:
                    p = (yaml_dir / p).resolve()
            else:
                p = (yaml_dir / p).resolve()
        out.append(p)
    return out


def read_images_from_source_entry(entry: Path) -> List[Path]:
    images = []

    if entry.is_dir():
        images = [p for p in entry.rglob("*") if p.is_file() and is_image_file(p)]
        return sorted(images)

    if entry.is_file() and entry.suffix.lower() == ".txt":
        with open(entry, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]

        for line in lines:
            p = Path(line)
            if not p.is_absolute():
                p1 = (entry.parent / p).resolve()
                if p1.exists():
                    p = p1
                else:
                    p = Path(line).resolve()
            if p.exists() and p.is_file() and is_image_file(p):
                images.append(p)
        return sorted(images)

    if entry.is_file() and is_image_file(entry):
        return [entry.resolve()]

    raise ValueError(f"不支持的数据源条目: {entry}")


def infer_label_path_from_image(image_path: Path) -> Path:
    image_path = image_path.resolve()
    parts = list(image_path.parts)

    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    cands = [
        image_path.with_suffix(".txt"),
        image_path.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if image_path.parent.parent != image_path.parent
        else image_path.with_suffix(".txt"),
    ]

    for c in cands:
        if c.exists():
            return c
    return cands[0]


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def materialize_eval_dataset(data_yaml: str, split: str, out_root: Path):
    sources = resolve_data_entries(data_yaml, split=split)
    image_files = []
    for s in sources:
        image_files.extend(read_images_from_source_entry(s))

    uniq_files = []
    seen = set()
    for p in image_files:
        sp = str(p.resolve())
        if sp not in seen:
            seen.add(sp)
            uniq_files.append(p.resolve())

    if not uniq_files:
        raise RuntimeError(f"no images found for split='{split}'")

    img_out = out_root / "images"
    gt_label_out = out_root / "labels"

    if out_root.exists():
        shutil.rmtree(out_root)
    img_out.mkdir(parents=True, exist_ok=True)
    gt_label_out.mkdir(parents=True, exist_ok=True)

    seen_stems = set()
    for img in uniq_files:
        stem = img.stem
        if stem in seen_stems:
            raise RuntimeError(
                f"duplicate image stem detected: '{stem}'. "
                f"Official eval matches predictions by stem, so stems must be unique. Example file: {img}"
            )
        seen_stems.add(stem)

        dst_img = img_out / img.name
        link_or_copy(img, dst_img)

        src_lab = infer_label_path_from_image(img)
        dst_lab = gt_label_out / f"{stem}.txt"
        if src_lab.exists() and src_lab.is_file():
            link_or_copy(src_lab, dst_lab)

    return img_out, gt_label_out, len(uniq_files)


def write_class_txt_from_data_yaml(data_yaml: str, out_txt: Path):
    dy = load_yaml_file(data_yaml)
    names = dy.get("names", None)

    if isinstance(names, dict):
        try:
            items = sorted(names.items(), key=lambda kv: int(kv[0]))
        except Exception:
            items = sorted(names.items(), key=lambda kv: str(kv[0]))
        class_names = [str(v) for _, v in items]
    elif isinstance(names, (list, tuple)):
        class_names = [str(x) for x in names]
    else:
        nc = int(dy.get("nc", 0))
        class_names = [str(i) for i in range(nc)]

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    return out_txt


def sanitize_score_dict(score_dict):
    out = {}
    if score_dict is None:
        return out
    for k, v in dict(score_dict).items():
        try:
            fv = float(v)
            out[str(k)] = fv if math.isfinite(fv) else None
        except Exception:
            out[str(k)] = None
    return out


def compute_track1_component_scores(metric_obj, fallback_s1: Optional[float] = None) -> Dict[str, Optional[float]]:
    try:
        screen = metric_obj.caculate_screen()
        fine = metric_obj.caculate_Sfine()
        loc = metric_obj.caculate_loc()
        s_loc = safe_float(loc.get("all"), default=None)
        s_screen = safe_float(screen.get("all"), default=None)
        s_fine = safe_float(fine.get("all"), default=None)
        if s_screen is None or s_fine is None:
            s_cls = None
        else:
            s_cls = 0.5 * s_fine + 0.5 * s_screen
        if s_loc is None or s_cls is None or s_screen is None:
            s1 = fallback_s1
        else:
            s1 = 0.3 * s_loc + 0.3 * s_cls + 0.4 * s_screen
        return {
            "S_loc": s_loc,
            "S_cls": s_cls,
            "S_screen": s_screen,
            "S_fine": s_fine,
            "S1": safe_float(s1, default=None),
        }
    except Exception as e:
        return {
            "S_loc": None,
            "S_cls": None,
            "S_screen": None,
            "S_fine": None,
            "S1": safe_float(fallback_s1, default=None),
            "component_error": str(e),
        }


def sort_official_eval_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(row: Dict[str, Any]):
        ok = 1 if row.get("ok") is True else 0
        score = safe_float(row.get("score_all"), default=-1e18)
        conf = safe_float(row.get("conf"), default=-1e18)
        iou = safe_float(row.get("iou"), default=-1e18)
        candidate_epoch = safe_float(row.get("candidate_epoch"), default=-1e18)
        gate_kept = -safe_float(row.get("num_images_suppressed_by_gate"), default=0.0)
        return (ok, score, conf, iou, gate_kept, candidate_epoch)

    return sorted(rows, key=_key, reverse=True)


def write_official_eval_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "ok",
        "cached",
        "score_all",
        "S_loc",
        "S_cls",
        "S_screen",
        "S1",
        "weights_name",
        "weights",
        "candidate_type",
        "candidate_epoch",
        "postprocess_preset",
        "postprocess_hash",
        "conf",
        "iou",
        "max_det",
        "tta",
        "tta_scales",
        "tent_cfg",
        "tag",
        "export_mode",
        "num_images",
        "raw_conf_used_for_model_predict",
        "image_gate_enabled",
        "num_images_suppressed_by_gate",
        "total_boxes_after_rule_filter",
        "total_boxes_after_filter",
        "pred_txt_dir",
        "summary_json",
        "official_best_pt",
        "error",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = [k for k in preferred if k in keys] + [k for k in sorted(keys) if k not in preferred]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            safe_row = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, Path):
                    value = str(value)
                elif value is None:
                    value = ""
                safe_row[key] = value
            writer.writerow(safe_row)


def export_predictions_for_official(
    weights_path: Path,
    source_img_dir: Path,
    out_dir: Path,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    postprocess_preset: str = "base",
    postprocess_cfg: Optional[Dict[str, Any]] = None,
    tta: bool = False,
    tta_scales: Optional[List[float]] = None,
    tent_cfg: Optional[Dict[str, Any]] = None,
):
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"[OfficialEval] 使用权重导出预测 txt: {weights_path}")
    print(f"[OfficialEval] source={source_img_dir}")
    print(f"[OfficialEval] conf={conf} iou={iou} max_det={max_det}")
    print(f"[OfficialEval] tta={bool(tta)}")
    if tta:
        print(f"[OfficialEval] tta_scales={tta_scales}")
    if tent_cfg and tent_cfg.get("enabled"):
        print(f"[OfficialEval] tent_cfg={tent_cfg}")
    export_mode = "rect_polygon" if OFFICIAL_USE_RECT_EXPORT else "ultralytics_save_txt"
    print(f"[OfficialEval] export_mode={export_mode}")
    print(f"[OfficialEval] postprocess_preset={postprocess_preset}")
    print(f"[OfficialEval][NOTE] {OFFICIAL_EVAL_NOTE}")

    pred_txt_dir = out_dir / "labels"
    pred_txt_dir.mkdir(parents=True, exist_ok=True)
    postprocess_cfg = normalize_official_postprocess_cfg(postprocess_cfg)
    export_summary = None

    if OFFICIAL_USE_RECT_EXPORT:
        export_summary = export_rect_predictions(
            model=weights_path,
            source=source_img_dir,
            save_dir=pred_txt_dir,
            source_root=source_img_dir,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            per_class_conf=postprocess_cfg.get("per_class_conf"),
            per_class_min_area=postprocess_cfg.get("per_class_min_area"),
            per_class_topk=postprocess_cfg.get("per_class_topk"),
            image_gate=postprocess_cfg.get("image_gate"),
            tta=tta,
            tta_scales=tta_scales,
            tent_cfg=tent_cfg,
            decimals=OFFICIAL_RECT_DECIMALS,
            clean_save_dir=True,
            touch_empty=True,
            verbose=False,
        )
    else:
        project = out_dir.parent
        name = out_dir.name
        project.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(weights_path))
        _ = model.predict(
            source=str(source_img_dir),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            save=False,
            augment=bool(tta),
            save_txt=True,
            save_conf=OFFICIAL_PRED_SAVE_CONF,
            retina_masks=OFFICIAL_PRED_RETINA_MASKS,
            project=str(project),
            name=name,
            exist_ok=True,
            verbose=False,
        )

    return out_dir, pred_txt_dir, export_summary


def prepare_official_eval_context(
    data_yaml: str,
    train_save_dir: Path,
    split: str,
):
    official_root = train_save_dir / "official_eval"
    official_root.mkdir(parents=True, exist_ok=True)

    gt_pack_dir = official_root / f"gt_{split}"
    gt_img_dir, gt_txt_dir, n_images = materialize_eval_dataset(
        data_yaml=data_yaml,
        split=split,
        out_root=gt_pack_dir,
    )
    class_txt_path = write_class_txt_from_data_yaml(
        data_yaml=data_yaml,
        out_txt=official_root / "classes_from_data.txt",
    )

    return dict(
        official_root=official_root,
        gt_img_dir=gt_img_dir,
        gt_txt_dir=gt_txt_dir,
        class_txt_path=class_txt_path,
        num_images=n_images,
    )


def run_official_track1_eval_once(
    eval_ctx: Dict[str, Any],
    weights_path: Path,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    tag: str,
    candidate_type: Optional[str] = None,
    candidate_epoch: Optional[int] = None,
    postprocess_preset: str = "base",
    postprocess_cfg: Optional[Dict[str, Any]] = None,
    postprocess_hash: Optional[str] = None,
    tta: bool = False,
    tta_scales: Optional[List[float]] = None,
    tent_cfg: Optional[Dict[str, Any]] = None,
):
    if not OFFICIAL_METRIC_SCRIPT.exists():
        raise FileNotFoundError(f"official metric script not found: {OFFICIAL_METRIC_SCRIPT}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    official_root = eval_ctx["official_root"]
    gt_img_dir = eval_ctx["gt_img_dir"]
    gt_txt_dir = eval_ctx["gt_txt_dir"]
    class_txt_path = eval_ctx["class_txt_path"]
    n_images = eval_ctx["num_images"]

    pred_dir = official_root / f"predict_{tag}"
    _, pred_txt_dir, export_summary = export_predictions_for_official(
        weights_path=weights_path,
        source_img_dir=gt_img_dir,
        out_dir=pred_dir,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        postprocess_preset=postprocess_preset,
        postprocess_cfg=postprocess_cfg,
        tta=tta,
        tta_scales=tta_scales,
        tent_cfg=tent_cfg,
    )

    metric_module = load_python_module_from_path(
        OFFICIAL_METRIC_SCRIPT,
        module_name=f"caculate_metric_{uuid.uuid4().hex[:8]}",
    )
    if not hasattr(metric_module, "CaculateMetric"):
        raise AttributeError("CaculateMetric not found in official metric script")

    cm = metric_module.CaculateMetric()
    score_dict = cm.process_data(
        gt_img_dir=str(gt_img_dir),
        gt_txt_dir=str(gt_txt_dir),
        pred_img_dir=str(gt_img_dir),
        pred_txt_dir=str(pred_txt_dir),
        class_txt_dir=str(class_txt_path),
        txt_shuffix=".txt",
        S=1,
    )

    official_scores = sanitize_score_dict(score_dict)
    official_all = official_scores.get("all", None)
    component_scores = compute_track1_component_scores(cm, fallback_s1=official_all)
    out_json = official_root / f"official_track1_{tag}.json"

    summary = {
        "ok": True,
        "tag": str(tag),
        "track": 1,
        "split": OFFICIAL_EVAL_SPLIT,
        "weights": str(weights_path),
        "weights_name": weights_path.name,
        "candidate_type": None if candidate_type is None else str(candidate_type),
        "candidate_epoch": None if candidate_epoch is None else int(candidate_epoch),
        "postprocess_preset": str(postprocess_preset),
        "postprocess_hash": None if postprocess_hash is None else str(postprocess_hash),
        "postprocess_cfg": normalize_official_postprocess_cfg(postprocess_cfg),
        "num_images": int(n_images),
        "gt_img_dir": str(gt_img_dir),
        "gt_txt_dir": str(gt_txt_dir),
        "pred_dir": str(pred_dir),
        "pred_txt_dir": str(pred_txt_dir),
        "class_txt_dir": str(class_txt_path),
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "tta": bool(tta),
        "tta_scales": [] if tta_scales is None else [float(x) for x in tta_scales],
        "tent_cfg": {} if not isinstance(tent_cfg, dict) else deepcopy(tent_cfg),
        "save_conf": bool(OFFICIAL_PRED_SAVE_CONF),
        "retina_masks": bool(OFFICIAL_PRED_RETINA_MASKS),
        "export_mode": "rect_polygon" if OFFICIAL_USE_RECT_EXPORT else "ultralytics_save_txt",
        "scores": official_scores,
        "score_all": official_all,
        "component_scores": component_scores,
        "S_loc": component_scores.get("S_loc"),
        "S_cls": component_scores.get("S_cls"),
        "S_screen": component_scores.get("S_screen"),
        "S1": component_scores.get("S1"),
        "summary_json": str(out_json),
        "note": OFFICIAL_EVAL_NOTE,
    }
    if isinstance(export_summary, dict):
        summary.update(
            {
                "raw_conf_used_for_model_predict": export_summary.get("raw_conf_used_for_model_predict"),
                "num_images_with_boxes_before_filter": export_summary.get("num_images_with_boxes_before_filter"),
                "num_images_with_boxes_after_filter": export_summary.get("num_images_with_boxes_after_filter"),
                "total_boxes_before_filter": export_summary.get("total_boxes_before_filter"),
                "total_boxes_after_rule_filter": export_summary.get("total_boxes_after_rule_filter"),
                "total_boxes_after_filter": export_summary.get("total_boxes_after_filter"),
                "image_gate_enabled": export_summary.get("image_gate_enabled"),
                "num_images_suppressed_by_gate": export_summary.get("num_images_suppressed_by_gate"),
            }
        )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[OfficialEval] Track1 done: weight={weights_path.name} "
        f"preset={postprocess_preset} conf={conf} iou={iou} max_det={max_det} score_all={official_all}"
    )
    return summary


def sweep_official_track1(
    train_save_dir: Path,
    data_yaml: str,
    imgsz: int,
    device: str,
):
    if not RUN_OFFICIAL_EVAL_AFTER_TRAIN:
        print("[OfficialEval] disabled")
        return None

    if OFFICIAL_EVAL_TRACK != 1:
        print(f"[OfficialEval][WARN] only Track1 is supported, got {OFFICIAL_EVAL_TRACK}")
        return None

    if not OFFICIAL_METRIC_SCRIPT.exists():
        print(f"[OfficialEval][WARN] official metric script not found: {OFFICIAL_METRIC_SCRIPT}")
        return None

    candidate_items = collect_eval_candidate_items(train_save_dir)
    if not candidate_items:
        print("[OfficialEval][WARN] no checkpoint candidates found")
        return None

    try:
        eval_ctx = prepare_official_eval_context(
            data_yaml=data_yaml,
            train_save_dir=train_save_dir,
            split=OFFICIAL_EVAL_SPLIT,
        )
    except Exception as e:
        print(f"[OfficialEval][ERROR] failed to prepare eval context: {e}")
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

    official_root = eval_ctx["official_root"]
    all_results = []
    best_result = None
    postprocess_items = build_official_postprocess_items()

    if RUN_OFFICIAL_SWEEP_AFTER_TRAIN:
        conf_list = OFFICIAL_SWEEP_CONF_LIST
        iou_list = OFFICIAL_SWEEP_IOU_LIST
        maxdet_list = OFFICIAL_SWEEP_MAXDET_LIST
    else:
        conf_list = [OFFICIAL_PRED_CONF]
        iou_list = [OFFICIAL_PRED_IOU]
        maxdet_list = [OFFICIAL_PRED_MAX_DET]

    tta_scales = [float(x) for x in OFFICIAL_PRED_TTA_SCALES] if OFFICIAL_PRED_TTA else None
    tent_cfg = deepcopy(OFFICIAL_PRED_TENT_CFG) if OFFICIAL_PRED_TENT else None
    if isinstance(tent_cfg, dict):
        tent_cfg["enabled"] = bool(tent_cfg.get("enabled", True))
    tta_tag = ""
    if OFFICIAL_PRED_TTA and tta_scales:
        tta_tag = "_tta_" + "_".join(str(x).replace(".", "p") for x in tta_scales)
    tent_tag = ""
    if isinstance(tent_cfg, dict) and tent_cfg.get("enabled"):
        tent_tag = f"_tent_{str(tent_cfg.get('mode', 'lite')).lower()}"

    planned_trials = len(candidate_items) * len(postprocess_items) * len(conf_list) * len(iou_list) * len(maxdet_list)
    print(
        f"[OfficialEval] candidate_checkpoints={len(candidate_items)} "
        f"postprocess_presets={len(postprocess_items)} planned_trials={planned_trials}"
    )

    for cand in candidate_items:
        wp = cand["path"]
        for post_item in postprocess_items:
            for conf in conf_list:
                for iou in iou_list:
                    for max_det in maxdet_list:
                        tag = (
                            f"{wp.stem}_"
                            f"{post_item['tag']}_"
                            f"c{str(conf).replace('.', 'p')}_"
                            f"i{str(iou).replace('.', 'p')}_"
                            f"m{max_det}"
                        )
                        if OFFICIAL_PRED_TTA:
                            tag += tta_tag or "_tta1"
                        if tent_tag:
                            tag += tent_tag
                        row_json = official_root / f"official_track1_{tag}.json"

                        if OFFICIAL_EVAL_RESUME and row_json.exists():
                            try:
                                with open(row_json, "r", encoding="utf-8") as f:
                                    cached = json.load(f)
                                if isinstance(cached, dict) and cached.get("ok") is True:
                                    cached.setdefault("tag", tag)
                                    cached.setdefault("weights", str(wp))
                                    cached.setdefault("weights_name", wp.name)
                                    cached.setdefault("candidate_type", cand.get("candidate_type"))
                                    cached.setdefault("candidate_epoch", cand.get("candidate_epoch"))
                                    cached.setdefault("postprocess_preset", post_item["name"])
                                    cached.setdefault("postprocess_hash", post_item["cfg_hash"])
                                    cached.setdefault("postprocess_cfg", post_item["cfg"])
                                    cached.setdefault("tta", bool(OFFICIAL_PRED_TTA))
                                    cached.setdefault("tta_scales", [] if tta_scales is None else [float(x) for x in tta_scales])
                                    cached.setdefault("tent_cfg", {} if tent_cfg is None else deepcopy(tent_cfg))
                                    cached.setdefault("summary_json", str(row_json))
                                    cached["cached"] = True
                                    cached["note"] = OFFICIAL_EVAL_NOTE
                                    all_results.append(cached)
                                    score = cached.get("score_all", None)
                                    if score is not None and (
                                        best_result is None or float(score) > float(best_result["score_all"])
                                    ):
                                        best_result = cached
                                    print(
                                        f"[OfficialEval][RESUME] weight={wp.name} "
                                        f"preset={post_item['name']} type={cand.get('candidate_type')} "
                                        f"epoch={cand.get('candidate_epoch')} "
                                        f"conf={conf} iou={iou} max_det={max_det} score_all={cached.get('score_all')}"
                                    )
                                    continue
                            except Exception:
                                pass

                        try:
                            summary = run_official_track1_eval_once(
                                eval_ctx=eval_ctx,
                                weights_path=wp,
                                imgsz=imgsz,
                                device=device,
                                conf=conf,
                                iou=iou,
                                max_det=max_det,
                                tag=tag,
                                candidate_type=cand.get("candidate_type"),
                                candidate_epoch=cand.get("candidate_epoch"),
                                postprocess_preset=post_item["name"],
                                postprocess_cfg=post_item["cfg"],
                                postprocess_hash=post_item["cfg_hash"],
                                tta=bool(OFFICIAL_PRED_TTA),
                                tta_scales=tta_scales,
                                tent_cfg=tent_cfg,
                            )
                            summary["cached"] = False
                            all_results.append(summary)
                            score = summary.get("score_all", None)
                            if score is not None and (
                                best_result is None or float(score) > float(best_result["score_all"])
                            ):
                                best_result = summary
                        except Exception as e:
                            err = {
                                "ok": False,
                                "tag": tag,
                                "weights": str(wp),
                                "weights_name": wp.name,
                                "candidate_type": cand.get("candidate_type"),
                                "candidate_epoch": cand.get("candidate_epoch"),
                                "postprocess_preset": post_item["name"],
                                "postprocess_hash": post_item["cfg_hash"],
                                "postprocess_cfg": post_item["cfg"],
                                "conf": conf,
                                "iou": iou,
                                "max_det": max_det,
                                "tta": bool(OFFICIAL_PRED_TTA),
                                "tta_scales": [] if tta_scales is None else [float(x) for x in tta_scales],
                                "tent_cfg": {} if tent_cfg is None else deepcopy(tent_cfg),
                                "summary_json": str(row_json),
                                "error": str(e),
                            }
                            all_results.append(err)
                            with open(row_json, "w", encoding="utf-8") as f:
                                json.dump(err, f, ensure_ascii=False, indent=2)
                            print(f"[OfficialEval][WARN] single eval failed: {err}")

    rows_sorted = sort_official_eval_rows(all_results)
    sweep_json = official_root / "official_track1_sweep_summary.json"
    sweep_csv = official_root / "official_track1_sweep_summary.csv"
    sweep_sorted_csv = official_root / "official_track1_sweep_summary_sorted.csv"
    write_official_eval_csv(all_results, sweep_csv)
    write_official_eval_csv(rows_sorted, sweep_sorted_csv)

    if best_result is not None:
        best_weights = Path(best_result["weights"])
        dst_best = train_save_dir / "weights" / "official_best.pt"
        try:
            shutil.copy2(best_weights, dst_best)
            best_result["official_best_pt"] = str(dst_best)
        except Exception as e:
            print(f"[OfficialEval][WARN] failed to copy official_best.pt: {e}")

        best_json = official_root / "official_track1_best.json"
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(best_result, f, ensure_ascii=False, indent=2)

    payload = {
        "ok": best_result is not None,
        "num_candidates": len(candidate_items),
        "num_trials_planned": int(planned_trials),
        "num_trials": len(all_results),
        "num_success": sum(1 for r in all_results if r.get("ok") is True),
        "num_failed": sum(1 for r in all_results if r.get("ok") is not True),
        "num_cached_success": sum(1 for r in all_results if r.get("cached") is True),
        "best": best_result,
        "candidates": [
            {
                "weights": item["weights"],
                "weights_name": item["weights_name"],
                "candidate_type": item["candidate_type"],
                "candidate_epoch": item["candidate_epoch"],
            }
            for item in candidate_items
        ],
        "postprocess_presets": [
            {
                "name": item["name"],
                "cfg_hash": item["cfg_hash"],
                "cfg": item["cfg"],
            }
            for item in postprocess_items
        ],
        "summary_csv": str(sweep_csv),
        "summary_sorted_csv": str(sweep_sorted_csv),
        "all_results": all_results,
        "note": OFFICIAL_EVAL_NOTE,
    }
    with open(sweep_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if best_result is None:
        print("[OfficialEval][WARN] sweep finished with no successful result")
        print(f"[OfficialEval] sweep csv = {sweep_csv}")
        print(f"[OfficialEval] sweep sorted csv = {sweep_sorted_csv}")
        return payload

    print(
        "[OfficialEval][BEST] "
        f"weights={Path(best_result['weights']).name} "
        f"preset={best_result.get('postprocess_preset')} "
        f"conf={best_result['conf']} iou={best_result['iou']} max_det={best_result['max_det']} "
        f"tta={best_result.get('tta')} "
        f"score_all={best_result['score_all']}"
    )
    print(f"[OfficialEval] sweep summary saved: {sweep_json}")
    print(f"[OfficialEval] sweep csv = {sweep_csv}")
    print(f"[OfficialEval] sweep sorted csv = {sweep_sorted_csv}")
    return payload


def train_once(model: YOLO, cfg: dict, profile: dict):
    clear_model_checkpoint_refs(model)

    print(
        f"[Train] family={profile.get('model_family', DEFAULT_MODEL_FAMILY)} "
        f"device={cfg.get('device')} batch={cfg.get('batch')} imgsz={cfg.get('imgsz')} "
        f"epochs={cfg.get('epochs')} patience={cfg.get('patience')} lr0={cfg.get('lr0')} "
        f"copy_paste={cfg.get('copy_paste', 0.0)} "
        f"fitness=Track1Proxy "
        f"enhance={profile.get('enhance', False)} "
        f"mixstyle={profile.get('mixstyle', False)} "
        f"(mode={profile.get('mixstyle_mode', 'mixstyle')}, p={profile.get('mixstyle_prob', None)}, a={profile.get('mixstyle_alpha', None)}, layers={profile.get('mixstyle_layers', None)}, lf={profile.get('mixstyle_low_freq_ratio', None)}) "
        f"randconv={profile.get('randconv', False)} "
        f"protect_shallow={profile.get('protect_shallow_textures', False)} "
        f"spd={profile.get('spd', False)}(layers={profile.get('spd_layers', 2)}, scale={profile.get('spd_scale', 2)}) "
        f"simam={profile.get('simam', False)}(layers={profile.get('simam_layers', 2)}) "
        f"vfl={profile.get('use_vfl', False)}(a={profile.get('vfl_alpha', 0.75)}, g={profile.get('vfl_gamma', 2.0)}) "
        f"nwd={profile.get('nwd_weight', 0.0)}(c={profile.get('nwd_constant', 12.8)}) "
        f"screen={profile.get('screen_loss_weight', 0.0)}(pos={profile.get('screen_pos_weight', 1.0)}, neg={profile.get('screen_neg_weight', 1.0)}) "
        f"ckpt_avg={profile.get('swa', True)}(strategy={profile.get('swa_strategy', 'window')}) "
        f"ibn={profile.get('ibn', True)}(ratio={profile.get('ibn_ratio', 0.5)}, layers={profile.get('ibn_layers', 1)}+stem) "
        f"safe_concat={profile.get('safe_concat', False)}"
    )

    res = model.train(trainer=PluginSegTrainer, **cfg)
    save_dir = resolve_save_dir(cfg, res)
    weights_dir = save_dir / "weights"

    return SimpleNamespace(
        save_dir=str(save_dir),
        best=str(weights_dir / "best.pt"),
        last=str(weights_dir / "last.pt"),
        result=res,
        official=None,
        official_score=None,
        official_conf=None,
        official_iou=None,
        official_max_det=None,
        official_tta=None,
        official_postprocess_preset=None,
        official_postprocess_hash=None,
        official_candidate_type=None,
        official_candidate_epoch=None,
        swa=None,
        eval_weights=None,
    )


def build_profile(
    label: str,
    enhance: bool,
    mixstyle: bool,
    randconv: bool = False,
    scale_key: str = DEFAULT_SCALE,
    model_family: Optional[str] = None,
    tag: str = "exp",
    pretrained: bool = True,
    mixstyle_prob: Optional[float] = None,
    mixstyle_alpha: Optional[float] = None,
    mixstyle_layers: int = 1,
    mixstyle_mode: str = "mixstyle",
    mixstyle_low_freq_ratio: float = 0.10,
    randconv_prob: Optional[float] = None,
    randconv_prob_end: Optional[float] = None,
    randconv_sigma: Optional[float] = None,
    randconv_sigma_end: Optional[float] = None,
    randconv_layers: int = 1,
    protect_shallow_textures: bool = False,
    ibn_ratio: float = 0.5,
    ibn_layers: int = 1,
    copy_paste: float = 0.0,
    aug_preset: str = "default",
    imgsz_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    cls_gain: Optional[float] = None,
    auto_batch_from_imgsz: bool = True,
    project: str = "ida_track1_official_sweep",
    swa: bool = True,
    spd: bool = False,
    simam: bool = False,
    spd_layers: int = 2,
    spd_scale: int = 2,
    spd_alpha_init: float = 0.10,
    simam_layers: int = 2,
    simam_e_lambda: float = 1e-4,
    swa_strategy: str = "window",
    swad_smooth_window: int = 5,
    swad_tolerance_ratio: float = 0.25,
    swad_n_converge: int = 2,
    use_vfl: bool = False,
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    nwd_weight: float = 0.0,
    nwd_constant: float = 12.8,
    screen_loss_weight: float = 0.0,
    screen_pos_weight: float = 1.0,
    screen_neg_weight: float = 1.0,
):
    run_id = make_run_id("R")
    model_family = normalize_model_family(model_family)
    mixstyle_mode = str(mixstyle_mode)
    tag_name = str(tag)
    if mixstyle and mixstyle_mode != "mixstyle":
        mode_token = re.sub(r"[^0-9a-zA-Z]+", "", mixstyle_mode.lower())
        if mode_token and mode_token not in tag_name.lower():
            tag_name = f"{tag_name}_{mode_token}"

    if mixstyle and randconv:
        _mixstyle_prob = 0.20 if mixstyle_prob is None else float(mixstyle_prob)
        _mixstyle_alpha = 0.20 if mixstyle_alpha is None else float(mixstyle_alpha)
        _randconv_prob = 0.12 if randconv_prob is None else float(randconv_prob)
        _randconv_prob_end = 0.30 if randconv_prob_end is None else float(randconv_prob_end)
        _randconv_sigma = 0.0 if randconv_sigma is None else float(randconv_sigma)
        _randconv_sigma_end = 0.10 if randconv_sigma_end is None else float(randconv_sigma_end)
    else:
        _mixstyle_prob = 0.30 if mixstyle_prob is None else float(mixstyle_prob)
        _mixstyle_alpha = 0.30 if mixstyle_alpha is None else float(mixstyle_alpha)
        _randconv_prob = 0.20 if randconv_prob is None else float(randconv_prob)
        _randconv_prob_end = 0.45 if randconv_prob_end is None else float(randconv_prob_end)
        _randconv_sigma = 0.0 if randconv_sigma is None else float(randconv_sigma)
        _randconv_sigma_end = 0.15 if randconv_sigma_end is None else float(randconv_sigma_end)

    return dict(
        label=label,
        model_family=model_family,
        enhance=enhance,
        mixstyle=mixstyle,
        randconv=randconv,
        spd=spd,
        simam=simam,
        edge_ks=5,
        dilated_rates=(1, 2, 3),
        mixstyle_prob=_mixstyle_prob,
        mixstyle_alpha=_mixstyle_alpha,
        mixstyle_layers=int(mixstyle_layers),
        mixstyle_mode=str(mixstyle_mode),
        mixstyle_low_freq_ratio=float(mixstyle_low_freq_ratio),
        randconv_prob=_randconv_prob,
        randconv_prob_end=_randconv_prob_end,
        randconv_sigma=_randconv_sigma,
        randconv_sigma_end=_randconv_sigma_end,
        randconv_layers=int(randconv_layers),
        randconv_kernel_sizes=(3, 5),
        randconv_refresh_interval=1,
        protect_shallow_textures=bool(protect_shallow_textures),
        spd_layers=int(spd_layers),
        spd_scale=int(spd_scale),
        spd_alpha_init=float(spd_alpha_init),
        simam_layers=int(simam_layers),
        simam_e_lambda=float(simam_e_lambda),
        ibn=True,
        ibn_ratio=float(ibn_ratio),
        ibn_layers=int(ibn_layers),
        safe_concat=False,
        copy_paste=float(copy_paste),
        aug_preset=str(aug_preset),
        imgsz_override=None if imgsz_override is None else int(imgsz_override),
        batch_override=None if batch_override is None else int(batch_override),
        cls_gain=None if cls_gain is None else float(cls_gain),
        auto_batch_from_imgsz=bool(auto_batch_from_imgsz),
        swa=bool(swa),
        swa_strategy=str(swa_strategy),
        swa_window=6,
        swa_max_models=8,
        swa_min_models=4,
        swa_include_last=True,
        swad_smooth_window=int(swad_smooth_window),
        swad_tolerance_ratio=float(swad_tolerance_ratio),
        swad_n_converge=int(swad_n_converge),
        use_vfl=bool(use_vfl),
        vfl_alpha=float(vfl_alpha),
        vfl_gamma=float(vfl_gamma),
        nwd_weight=float(nwd_weight),
        nwd_constant=float(nwd_constant),
        screen_loss_weight=float(screen_loss_weight),
        screen_pos_weight=float(screen_pos_weight),
        screen_neg_weight=float(screen_neg_weight),
        scale=scale_key,
        pretrained=pretrained,
        run_name=f"{tag_name}_{scale_key}_{run_id}",
        project=project,
    )


def infer_profile_kwargs_from_best_json(best_json_path: str, scale_key: str) -> Dict[str, Any]:
    p = str(best_json_path).lower()
    kwargs = dict(
        enhance=False,
        mixstyle=False,
        randconv=False,
        mixstyle_mode="mixstyle",
        scale_key=scale_key,
        pretrained=False,
    )
    if "fdsa" in p:
        kwargs["mixstyle"] = True
        kwargs["mixstyle_mode"] = "fdsa"
    elif "efdmix" in p:
        kwargs["mixstyle"] = True
        kwargs["mixstyle_mode"] = "efdmix"
    elif "dsu" in p:
        kwargs["mixstyle"] = True
        kwargs["mixstyle_mode"] = "dsu"
    if "mixrand" in p:
        kwargs["mixstyle"] = True
        kwargs["randconv"] = True
    elif "mixstyle" in p:
        kwargs["mixstyle"] = True
    elif "randconv" in p:
        kwargs["randconv"] = True
    elif "efe" in p:
        kwargs["enhance"] = True
    return kwargs


def compact_float_token(v: float) -> str:
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def run_efdmix_singlevar(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    cls_gain: float = 2.6,
    imgsz: int = 512,
    batch_override: Optional[int] = None,
    aug_preset: str = "default",
    copy_paste: float = 0.0,
    ibn_ratio: float = 0.5,
    randconv: bool = True,
    project: str = "ida_track1_efdmix_singlevar",
    official_tta: Optional[bool] = None,
    train_overrides: Optional[dict] = None,
):
    global OFFICIAL_PRED_TTA

    budget_key = budget_mode if budget_mode in P1_BUDGET_PRESETS else "quick"
    merged_overrides = deepcopy(P1_BUDGET_PRESETS[budget_key])
    if train_overrides:
        merged_overrides.update(train_overrides)

    mode_label = "MixRand+EFDMix" if randconv else "EFDMix"
    tag_prefix = "mixrand" if randconv else "mixstyle"
    profile = build_profile(
        label=f"[{scale_key.upper()}] {mode_label} cls={float(cls_gain):.2f} aug={aug_preset} imgsz={int(imgsz)}",
        enhance=False,
        mixstyle=True,
        randconv=bool(randconv),
        scale_key=scale_key,
        tag=f"{tag_prefix}_efdmix_cls{compact_float_token(cls_gain)}",
        pretrained=True,
        mixstyle_mode="efdmix",
        aug_preset=str(aug_preset),
        imgsz_override=int(imgsz),
        batch_override=None if batch_override is None else int(batch_override),
        auto_batch_from_imgsz=(batch_override is None),
        copy_paste=float(copy_paste),
        ibn_ratio=float(ibn_ratio),
        cls_gain=float(cls_gain),
        project=str(project),
    )

    old_official_tta = OFFICIAL_PRED_TTA
    if official_tta is not None:
        OFFICIAL_PRED_TTA = bool(official_tta)

    try:
        return run_experiment(
            profile,
            train_overrides=(merged_overrides if merged_overrides else None),
        )
    finally:
        OFFICIAL_PRED_TTA = old_official_tta


def run_fdsa_screen_singlevar(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    cls_gain: float = 2.6,
    imgsz: int = 512,
    batch_override: Optional[int] = None,
    aug_preset: str = "industrial_soft",
    copy_paste: float = 0.05,
    ibn_ratio: float = 0.60,
    randconv: bool = False,
    mixstyle_layers: int = 2,
    low_freq_ratio: float = 0.08,
    screen_loss_weight: float = 0.15,
    screen_pos_weight: float = 1.25,
    screen_neg_weight: float = 1.00,
    official_tta: Optional[bool] = None,
    project: str = "ida_track1_fdsa_screen",
    train_overrides: Optional[dict] = None,
):
    global OFFICIAL_PRED_TTA

    budget_key = budget_mode if budget_mode in P1_BUDGET_PRESETS else "quick"
    merged_overrides = deepcopy(P1_BUDGET_PRESETS[budget_key])
    if train_overrides:
        merged_overrides.update(train_overrides)

    mode_label = "MixRand+FDSA" if randconv else "FDSA"
    tag_prefix = "mixrand" if randconv else "mixstyle"
    profile = build_profile(
        label=(
            f"[{scale_key.upper()}] {mode_label}+Screen "
            f"cls={float(cls_gain):.2f} imgsz={int(imgsz)} "
            f"lf={float(low_freq_ratio):.2f} screen={float(screen_loss_weight):.2f}"
        ),
        enhance=False,
        mixstyle=True,
        randconv=bool(randconv),
        scale_key=scale_key,
        tag=f"{tag_prefix}_fdsa_screen_cls{compact_float_token(cls_gain)}",
        pretrained=True,
        mixstyle_layers=int(mixstyle_layers),
        mixstyle_mode="fdsa",
        mixstyle_low_freq_ratio=float(low_freq_ratio),
        aug_preset=str(aug_preset),
        imgsz_override=int(imgsz),
        batch_override=None if batch_override is None else int(batch_override),
        auto_batch_from_imgsz=(batch_override is None),
        copy_paste=float(copy_paste),
        ibn_ratio=float(ibn_ratio),
        ibn_layers=2,
        cls_gain=float(cls_gain),
        project=str(project),
        swa_strategy="swad",
        swad_smooth_window=5,
        swad_tolerance_ratio=0.20,
        swad_n_converge=2,
        screen_loss_weight=float(screen_loss_weight),
        screen_pos_weight=float(screen_pos_weight),
        screen_neg_weight=float(screen_neg_weight),
    )

    old_official_tta = OFFICIAL_PRED_TTA
    if official_tta is not None:
        OFFICIAL_PRED_TTA = bool(official_tta)

    try:
        return run_experiment(
            profile,
            train_overrides=(merged_overrides if merged_overrides else None),
        )
    finally:
        OFFICIAL_PRED_TTA = old_official_tta


def run_aggressive_upperbound(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    cls_gain: float = 2.6,
    imgsz: int = 512,
    batch_override: Optional[int] = None,
    aug_preset: str = "industrial_soft",
    copy_paste: float = 0.10,
    ibn_ratio: float = 0.60,
    mixstyle_mode: str = "efdmix",
    official_tta: Optional[bool] = True,
    project: str = "ida_track1_upperbound",
    train_overrides: Optional[dict] = None,
):
    global OFFICIAL_PRED_TTA

    budget_key = budget_mode if budget_mode in P1_BUDGET_PRESETS else "quick"
    merged_overrides = deepcopy(P1_BUDGET_PRESETS[budget_key])
    if train_overrides:
        merged_overrides.update(train_overrides)

    profile = build_profile(
        label=(
            f"[{scale_key.upper()}] UpperBound MixRand+{str(mixstyle_mode).upper()}+SPD+SimAM "
            f"cls={float(cls_gain):.2f} aug={aug_preset} imgsz={int(imgsz)}"
        ),
        enhance=False,
        mixstyle=True,
        randconv=True,
        scale_key=scale_key,
        tag=f"upperbound_mixrand_{str(mixstyle_mode).lower()}_spd_simam",
        pretrained=True,
        mixstyle_layers=2,
        mixstyle_mode=str(mixstyle_mode),
        randconv_layers=2,
        spd=True,
        spd_layers=3,
        spd_scale=2,
        spd_alpha_init=0.10,
        simam=True,
        simam_layers=2,
        simam_e_lambda=1e-4,
        aug_preset=str(aug_preset),
        imgsz_override=int(imgsz),
        batch_override=None if batch_override is None else int(batch_override),
        auto_batch_from_imgsz=(batch_override is None),
        copy_paste=float(copy_paste),
        ibn_ratio=float(ibn_ratio),
        ibn_layers=2,
        cls_gain=float(cls_gain),
        project=str(project),
        swa_strategy="swad",
        swad_smooth_window=5,
        swad_tolerance_ratio=0.20,
        swad_n_converge=2,
    )
    profile.update(make_light_official_eval_settings(conf=0.08, iou=0.30, postprocess_presets=["base"], tta=False))

    old_official_tta = OFFICIAL_PRED_TTA
    if official_tta is not None:
        OFFICIAL_PRED_TTA = bool(official_tta)

    try:
        return run_experiment(
            profile,
            train_overrides=(merged_overrides if merged_overrides else None),
        )
    finally:
        OFFICIAL_PRED_TTA = old_official_tta


def run_texture_preserve_upperbound(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    cls_gain: float = 2.6,
    imgsz: int = 512,
    batch_override: Optional[int] = None,
    aug_preset: str = "industrial_conservative",
    copy_paste: float = 0.10,
    ibn_ratio: float = 0.60,
    mixstyle_mode: str = "efdmix",
    official_tta: Optional[bool] = True,
    model_family: Optional[str] = None,
    project: str = "ida_track1_texture_preserve_upperbound",
    train_overrides: Optional[dict] = None,
):
    global OFFICIAL_PRED_TTA
    family = normalize_model_family(model_family)

    budget_key = budget_mode if budget_mode in P1_BUDGET_PRESETS else "quick"
    merged_overrides = deepcopy(P1_BUDGET_PRESETS[budget_key])
    if train_overrides:
        merged_overrides.update(train_overrides)

    profile = build_profile(
        label=(
            f"[{scale_key.upper()}] TexturePreserve MixRand+{str(mixstyle_mode).upper()}+SPD+SimAM "
            f"cls={float(cls_gain):.2f} aug={aug_preset} imgsz={int(imgsz)}"
        ),
        enhance=False,
        mixstyle=True,
        randconv=True,
        scale_key=scale_key,
        model_family=family,
        tag=f"texture_preserve_mixrand_{str(mixstyle_mode).lower()}_spd_simam",
        pretrained=True,
        mixstyle_layers=2,
        mixstyle_mode=str(mixstyle_mode),
        randconv_layers=2,
        protect_shallow_textures=True,
        spd=True,
        spd_layers=3,
        spd_scale=2,
        spd_alpha_init=0.10,
        simam=True,
        simam_layers=2,
        simam_e_lambda=1e-4,
        aug_preset=str(aug_preset),
        imgsz_override=int(imgsz),
        batch_override=None if batch_override is None else int(batch_override),
        auto_batch_from_imgsz=(batch_override is None),
        copy_paste=float(copy_paste),
        ibn_ratio=float(ibn_ratio),
        ibn_layers=2,
        cls_gain=float(cls_gain),
        project=str(project),
        swa_strategy="swad",
        swad_smooth_window=5,
        swad_tolerance_ratio=0.20,
        swad_n_converge=2,
    )
    profile.update(make_texture_preserve_official_eval_settings(conf=0.08, iou=0.30, tta=False))

    old_official_tta = OFFICIAL_PRED_TTA
    if official_tta is not None:
        OFFICIAL_PRED_TTA = bool(official_tta)

    try:
        return run_experiment(
            profile,
            train_overrides=(merged_overrides if merged_overrides else None),
        )
    finally:
        OFFICIAL_PRED_TTA = old_official_tta


def run_nwd_vfl_upperbound(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    cls_gain: float = 2.6,
    imgsz: int = 512,
    batch_override: Optional[int] = None,
    aug_preset: str = "industrial_soft",
    copy_paste: float = 0.10,
    ibn_ratio: float = 0.60,
    mixstyle_mode: str = "efdmix",
    nwd_weight: float = 0.35,
    nwd_constant: float = 12.8,
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    official_tta: Optional[bool] = True,
    project: str = "ida_track1_loss_upperbound",
    train_overrides: Optional[dict] = None,
):
    global OFFICIAL_PRED_TTA

    budget_key = budget_mode if budget_mode in P1_BUDGET_PRESETS else "quick"
    merged_overrides = deepcopy(P1_BUDGET_PRESETS[budget_key])
    if train_overrides:
        merged_overrides.update(train_overrides)

    profile = build_profile(
        label=(
            f"[{scale_key.upper()}] LossUpperBound MixRand+{str(mixstyle_mode).upper()}+"
            f"SPD+SimAM+NWD+VFL cls={float(cls_gain):.2f} imgsz={int(imgsz)}"
        ),
        enhance=False,
        mixstyle=True,
        randconv=True,
        scale_key=scale_key,
        tag=f"loss_upperbound_{str(mixstyle_mode).lower()}_nwd_vfl",
        pretrained=True,
        mixstyle_layers=2,
        mixstyle_mode=str(mixstyle_mode),
        randconv_layers=2,
        spd=True,
        spd_layers=3,
        spd_scale=2,
        spd_alpha_init=0.10,
        simam=True,
        simam_layers=2,
        simam_e_lambda=1e-4,
        aug_preset=str(aug_preset),
        imgsz_override=int(imgsz),
        batch_override=None if batch_override is None else int(batch_override),
        auto_batch_from_imgsz=(batch_override is None),
        copy_paste=float(copy_paste),
        ibn_ratio=float(ibn_ratio),
        ibn_layers=2,
        cls_gain=float(cls_gain),
        project=str(project),
        swa_strategy="swad",
        swad_smooth_window=5,
        swad_tolerance_ratio=0.20,
        swad_n_converge=2,
        use_vfl=True,
        vfl_alpha=float(vfl_alpha),
        vfl_gamma=float(vfl_gamma),
        nwd_weight=float(nwd_weight),
        nwd_constant=float(nwd_constant),
    )
    profile.update(make_light_official_eval_settings(conf=0.08, iou=0.30, postprocess_presets=["base"], tta=False))

    old_official_tta = OFFICIAL_PRED_TTA
    if official_tta is not None:
        OFFICIAL_PRED_TTA = bool(official_tta)

    try:
        return run_experiment(
            profile,
            train_overrides=(merged_overrides if merged_overrides else None),
        )
    finally:
        OFFICIAL_PRED_TTA = old_official_tta


def run_tent_official_resweep(
    best_json_path: str,
    tent_mode: str = "grad",
    tent_steps: int = 1,
    tent_lr: float = 1e-4,
    reset_mode: str = "scenario",
    search_mode: str = "aggressive",
):
    best = load_json_file(best_json_path)
    best_weights = best.get("official_best_pt") or best.get("weights") or best.get("eval_weights")
    if not best_weights or not Path(best_weights).exists():
        raise ValueError(f"weights not found in best_json: {best_json_path}")

    conf_center = float(best.get("conf", OFFICIAL_PRED_CONF))
    iou_center = float(best.get("iou", OFFICIAL_PRED_IOU))
    max_det = int(best.get("max_det", OFFICIAL_PRED_MAX_DET))
    imgsz = int(best.get("imgsz", BASE_CFG["imgsz"]))
    best_preset = str(best.get("postprocess_preset", "base"))
    tta = bool(best.get("tta", OFFICIAL_PRED_TTA))
    tta_scales = [float(x) for x in best.get("tta_scales", OFFICIAL_PRED_TTA_SCALES)] if tta else None
    best_tent_cfg = best.get("tent_cfg", {}) if isinstance(best.get("tent_cfg", {}), dict) else {}
    base_tent_scope = str(best_tent_cfg.get("scope", OFFICIAL_PRED_TENT_CFG.get("scope", "all"))).strip().lower()
    if base_tent_scope not in {"all", "backbone", "backbone_shallow"}:
        base_tent_scope = "all"
    base_tent_shallow_stages = max(1, int(best_tent_cfg.get("shallow_stages", OFFICIAL_PRED_TENT_CFG.get("shallow_stages", 4))))
    base_tent_max_bn_layers = max(0, int(best_tent_cfg.get("max_bn_layers", OFFICIAL_PRED_TENT_CFG.get("max_bn_layers", 0))))

    def _clip_values(center: float, deltas: List[float], lo: float, hi: float):
        vals = []
        for delta in deltas:
            v = max(lo, min(hi, center + delta))
            vals.append(round(float(v), 4))
        return sorted(set(vals))

    search_mode = str(search_mode).strip().lower()
    if search_mode not in {"compact", "aggressive"}:
        search_mode = "aggressive"

    if search_mode == "compact":
        conf_list = _clip_values(conf_center, [-0.03, 0.0, 0.03], 0.02, 0.30)
        iou_list = _clip_values(iou_center, [-0.05, 0.0, 0.05], 0.20, 0.50)
    else:
        conf_list = _clip_values(conf_center, [-0.05, -0.03, -0.01, 0.0, 0.02, 0.04], 0.02, 0.30)
        iou_list = _clip_values(iou_center, [-0.10, -0.05, 0.0, 0.05, 0.10], 0.20, 0.55)

    preset_names = []
    preset_candidates = [
        best_preset,
        "screen_balance",
        "screen_aggressive",
        "classaware_light",
        "gate_light",
        "topk_only",
        "base",
    ]
    for name in preset_candidates:
        if name not in preset_names and name in OFFICIAL_POSTPROCESS_PRESETS:
            preset_names.append(name)
    postprocess_items = [
        {
            "name": name,
            "cfg": normalize_official_postprocess_cfg(OFFICIAL_POSTPROCESS_PRESETS.get(name)),
            "cfg_hash": stable_hash_json(normalize_official_postprocess_cfg(OFFICIAL_POSTPROCESS_PRESETS.get(name)), n=10),
            "tag": re.sub(r"[^0-9a-zA-Z]+", "", name.lower()) or "base",
        }
        for name in preset_names
    ]

    base_tent_mode = str(tent_mode).strip().lower()
    base_reset_mode = str(reset_mode).strip().lower()
    if search_mode == "compact":
        tent_variants = [
            {
                "enabled": True,
                "mode": base_tent_mode,
                "steps": max(1, int(tent_steps)),
                "lr": float(tent_lr),
                "topk_ratio": 0.10,
                "topk_min": 64,
                "reset_mode": base_reset_mode,
                "reset_each_image": base_reset_mode == "image",
                "scope": base_tent_scope,
                "shallow_stages": base_tent_shallow_stages,
                "max_bn_layers": base_tent_max_bn_layers,
            }
        ]
        tta_variants = [
            {"enabled": tta, "scales": None if not tta else [float(x) for x in tta_scales]},
        ]
    else:
        if base_tent_mode == "lite":
            tent_variants = [
                {
                    "enabled": True,
                    "mode": "lite",
                    "steps": 1,
                    "lr": float(tent_lr),
                    "topk_ratio": 0.10,
                    "topk_min": 64,
                    "reset_mode": base_reset_mode,
                    "reset_each_image": base_reset_mode == "image",
                    "scope": base_tent_scope,
                    "shallow_stages": base_tent_shallow_stages,
                    "max_bn_layers": base_tent_max_bn_layers,
                },
                {
                    "enabled": True,
                    "mode": "lite",
                    "steps": 2,
                    "lr": float(tent_lr),
                    "topk_ratio": 0.15,
                    "topk_min": 96,
                    "reset_mode": base_reset_mode,
                    "reset_each_image": base_reset_mode == "image",
                    "scope": base_tent_scope,
                    "shallow_stages": base_tent_shallow_stages,
                    "max_bn_layers": base_tent_max_bn_layers,
                },
            ]
        else:
            tent_variants = [
                {
                    "enabled": True,
                    "mode": "grad",
                    "steps": max(1, int(tent_steps)),
                    "lr": float(tent_lr),
                    "topk_ratio": 0.10,
                    "topk_min": 64,
                    "reset_mode": base_reset_mode,
                    "reset_each_image": base_reset_mode == "image",
                    "scope": base_tent_scope,
                    "shallow_stages": base_tent_shallow_stages,
                    "max_bn_layers": base_tent_max_bn_layers,
                },
                {
                    "enabled": True,
                    "mode": "grad",
                    "steps": max(2, int(tent_steps) + 1),
                    "lr": max(5e-5, float(tent_lr) * 0.5),
                    "topk_ratio": 0.15,
                    "topk_min": 96,
                    "reset_mode": base_reset_mode,
                    "reset_each_image": base_reset_mode == "image",
                    "scope": base_tent_scope,
                    "shallow_stages": base_tent_shallow_stages,
                    "max_bn_layers": base_tent_max_bn_layers,
                },
                {
                    "enabled": True,
                    "mode": "grad",
                    "steps": max(1, int(tent_steps)),
                    "lr": min(3e-4, float(tent_lr) * 2.0),
                    "topk_ratio": 0.08,
                    "topk_min": 48,
                    "reset_mode": base_reset_mode,
                    "reset_each_image": base_reset_mode == "image",
                    "scope": base_tent_scope,
                    "shallow_stages": base_tent_shallow_stages,
                    "max_bn_layers": base_tent_max_bn_layers,
                },
            ]

        tta_variants = [
            {"enabled": False, "scales": None},
            {"enabled": True, "scales": [1.0, 0.83, 1.17]},
            {"enabled": True, "scales": [1.0, 0.90, 1.10]},
        ]
        if tta and tta_scales:
            tta_variants.insert(0, {"enabled": True, "scales": [float(x) for x in tta_scales]})

    uniq_tta = []
    seen_tta = set()
    for item in tta_variants:
        enabled = bool(item.get("enabled"))
        scales = None if not enabled else tuple(round(float(x), 4) for x in (item.get("scales") or [1.0]))
        key = (enabled, scales)
        if key in seen_tta:
            continue
        seen_tta.add(key)
        uniq_tta.append({"enabled": enabled, "scales": None if scales is None else [float(x) for x in scales]})
    tta_variants = uniq_tta

    suite_id = make_run_id("TENT")
    suite_root = RESULT_ROOT / "tent_resweeps" / suite_id
    suite_root.mkdir(parents=True, exist_ok=True)
    eval_ctx = prepare_official_eval_context(
        data_yaml=DATA_PATH,
        train_save_dir=suite_root,
        split=OFFICIAL_EVAL_SPLIT,
    )

    all_results = []
    best_result = None
    weights_path = Path(best_weights).resolve()
    planned_trials = len(postprocess_items) * len(conf_list) * len(iou_list) * len(tent_variants) * len(tta_variants)
    print(
        f"[TENT] search_mode={search_mode} presets={len(postprocess_items)} confs={len(conf_list)} "
        f"ious={len(iou_list)} tent_variants={len(tent_variants)} tta_variants={len(tta_variants)} "
        f"planned_trials={planned_trials}"
    )

    for post_item in postprocess_items:
        for conf in conf_list:
            for iou in iou_list:
                for tta_item in tta_variants:
                    for tent_cfg in tent_variants:
                        tag = (
                            f"{weights_path.stem}_{post_item['tag']}_"
                            f"c{str(conf).replace('.', 'p')}_i{str(iou).replace('.', 'p')}_"
                            f"m{max_det}_tent_{tent_cfg['mode']}_s{tent_cfg['steps']}_"
                            f"lr{str(tent_cfg['lr']).replace('.', 'p')}"
                        )
                        if tta_item["enabled"] and tta_item["scales"]:
                            tag += "_tta_" + "_".join(str(x).replace(".", "p") for x in tta_item["scales"])
                        try:
                            summary = run_official_track1_eval_once(
                                eval_ctx=eval_ctx,
                                weights_path=weights_path,
                                imgsz=imgsz,
                                device=str(BASE_CFG.get("device", DEFAULT_DEVICE)),
                                conf=conf,
                                iou=iou,
                                max_det=max_det,
                                tag=tag,
                                candidate_type="tent_resweep",
                                candidate_epoch=best.get("candidate_epoch"),
                                postprocess_preset=post_item["name"],
                                postprocess_cfg=post_item["cfg"],
                                postprocess_hash=post_item["cfg_hash"],
                                tta=bool(tta_item["enabled"]),
                                tta_scales=tta_item["scales"],
                                tent_cfg=tent_cfg,
                            )
                            summary["cached"] = False
                            all_results.append(summary)
                            score = summary.get("score_all", None)
                            if score is not None and (best_result is None or float(score) > float(best_result["score_all"])):
                                best_result = summary
                        except Exception as e:
                            err = {
                                "ok": False,
                                "tag": tag,
                                "weights": str(weights_path),
                                "weights_name": weights_path.name,
                                "candidate_type": "tent_resweep",
                                "candidate_epoch": best.get("candidate_epoch"),
                                "postprocess_preset": post_item["name"],
                                "postprocess_hash": post_item["cfg_hash"],
                                "postprocess_cfg": post_item["cfg"],
                                "conf": conf,
                                "iou": iou,
                                "max_det": max_det,
                                "tta": bool(tta_item["enabled"]),
                                "tta_scales": [] if tta_item["scales"] is None else [float(x) for x in tta_item["scales"]],
                                "tent_cfg": deepcopy(tent_cfg),
                                "error": str(e),
                            }
                            all_results.append(err)

    rows_sorted = sort_official_eval_rows(all_results)
    sweep_json = suite_root / "tent_official_sweep_summary.json"
    sweep_csv = suite_root / "tent_official_sweep_summary.csv"
    sweep_sorted_csv = suite_root / "tent_official_sweep_summary_sorted.csv"
    write_official_eval_csv(all_results, sweep_csv)
    write_official_eval_csv(rows_sorted, sweep_sorted_csv)

    payload = {
        "ok": best_result is not None,
        "best_json": str(best_json_path),
        "best_weights": str(weights_path),
        "search_mode": search_mode,
        "planned_trials": planned_trials,
        "tent_variants": tent_variants,
        "tta_variants": tta_variants,
        "conf_list": conf_list,
        "iou_list": iou_list,
        "postprocess_presets": preset_names,
        "best": best_result,
        "rows": all_results,
        "summary_csv": str(sweep_csv),
        "summary_sorted_csv": str(sweep_sorted_csv),
    }
    with open(sweep_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[TENT][OUT] summary.json = {sweep_json}")
    print(f"[TENT][OUT] summary.csv = {sweep_csv}")
    if best_result is not None:
        print(
            f"[TENT][BEST] score={best_result.get('score_all')} "
            f"conf={best_result.get('conf')} iou={best_result.get('iou')} "
            f"preset={best_result.get('postprocess_preset')} "
            f"tta={best_result.get('tta')} tent={best_result.get('tent_cfg')}"
        )
    return payload


def run_metric_aligned_official_resweep(
    best_json_path: str,
    search_mode: str = "aggressive",
):
    best = load_json_file(best_json_path)
    best_weights = best.get("official_best_pt") or best.get("weights") or best.get("eval_weights")
    if not best_weights or not Path(best_weights).exists():
        raise ValueError(f"weights not found in best_json: {best_json_path}")

    conf_center = float(best.get("conf", OFFICIAL_PRED_CONF))
    iou_center = float(best.get("iou", OFFICIAL_PRED_IOU))
    max_det = int(best.get("max_det", OFFICIAL_PRED_MAX_DET))
    imgsz = int(best.get("imgsz", BASE_CFG["imgsz"]))
    best_preset = str(best.get("postprocess_preset", "base"))
    base_tta = bool(best.get("tta", OFFICIAL_PRED_TTA))
    base_tta_scales = [float(x) for x in best.get("tta_scales", OFFICIAL_PRED_TTA_SCALES)] if base_tta else None

    search_mode = str(search_mode).strip().lower()
    if search_mode not in {"compact", "aggressive"}:
        search_mode = "aggressive"

    fixed_conf = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    fixed_iou = [0.20, 0.25, 0.30, 0.35]
    if search_mode == "compact":
        conf_list = sorted(set([round(conf_center, 4)] + [x for x in fixed_conf if 0.02 <= x <= 0.08]))
        iou_list = sorted(set([round(iou_center, 4)] + [0.25, 0.30, 0.35]))
        tta_variants = [
            {"enabled": base_tta, "scales": None if not base_tta else [float(x) for x in (base_tta_scales or [1.0])]},
            {"enabled": False, "scales": None},
        ]
    else:
        conf_list = sorted(set([round(conf_center, 4)] + fixed_conf))
        iou_list = sorted(set([round(iou_center, 4)] + fixed_iou))
        tta_variants = [
            {"enabled": False, "scales": None},
            {"enabled": True, "scales": [1.0, 0.83, 1.17]},
            {"enabled": True, "scales": [1.0, 0.90, 1.10]},
        ]
        if base_tta and base_tta_scales:
            tta_variants.insert(0, {"enabled": True, "scales": [float(x) for x in base_tta_scales]})

    uniq_tta = []
    seen_tta = set()
    for item in tta_variants:
        enabled = bool(item.get("enabled"))
        scales = None if not enabled else tuple(round(float(x), 4) for x in (item.get("scales") or [1.0]))
        key = (enabled, scales)
        if key in seen_tta:
            continue
        seen_tta.add(key)
        uniq_tta.append({"enabled": enabled, "scales": None if scales is None else [float(x) for x in scales]})
    tta_variants = uniq_tta

    preset_names = []
    preset_candidates = [best_preset] + METRIC_ALIGNED_POSTPROCESS_PRESETS
    for name in preset_candidates:
        if name not in preset_names and name in OFFICIAL_POSTPROCESS_PRESETS:
            preset_names.append(name)
    postprocess_items = build_postprocess_items_from_names(preset_names)

    suite_id = make_run_id("METRIC")
    suite_root = RESULT_ROOT / "metric_resweeps" / suite_id
    suite_root.mkdir(parents=True, exist_ok=True)
    eval_ctx = prepare_official_eval_context(
        data_yaml=DATA_PATH,
        train_save_dir=suite_root,
        split=OFFICIAL_EVAL_SPLIT,
    )

    all_results = []
    best_result = None
    weights_path = Path(best_weights).resolve()
    planned_trials = len(postprocess_items) * len(conf_list) * len(iou_list) * len(tta_variants)
    print(
        f"[MetricSweep] search_mode={search_mode} presets={len(postprocess_items)} "
        f"confs={len(conf_list)} ious={len(iou_list)} tta_variants={len(tta_variants)} "
        f"planned_trials={planned_trials}"
    )

    for post_item in postprocess_items:
        for conf in conf_list:
            for iou in iou_list:
                for tta_item in tta_variants:
                    tag = (
                        f"{weights_path.stem}_{post_item['tag']}_"
                        f"c{str(conf).replace('.', 'p')}_i{str(iou).replace('.', 'p')}_m{max_det}"
                    )
                    if tta_item["enabled"] and tta_item["scales"]:
                        tag += "_tta_" + "_".join(str(x).replace(".", "p") for x in tta_item["scales"])
                    try:
                        summary = run_official_track1_eval_once(
                            eval_ctx=eval_ctx,
                            weights_path=weights_path,
                            imgsz=imgsz,
                            device=str(BASE_CFG.get("device", DEFAULT_DEVICE)),
                            conf=conf,
                            iou=iou,
                            max_det=max_det,
                            tag=tag,
                            candidate_type="metric_resweep",
                            candidate_epoch=best.get("candidate_epoch"),
                            postprocess_preset=post_item["name"],
                            postprocess_cfg=post_item["cfg"],
                            postprocess_hash=post_item["cfg_hash"],
                            tta=bool(tta_item["enabled"]),
                            tta_scales=tta_item["scales"],
                            tent_cfg=None,
                        )
                        summary["cached"] = False
                        all_results.append(summary)
                        score = summary.get("score_all", None)
                        if score is not None and (best_result is None or float(score) > float(best_result["score_all"])):
                            best_result = summary
                    except Exception as e:
                        err = {
                            "ok": False,
                            "tag": tag,
                            "weights": str(weights_path),
                            "weights_name": weights_path.name,
                            "candidate_type": "metric_resweep",
                            "candidate_epoch": best.get("candidate_epoch"),
                            "postprocess_preset": post_item["name"],
                            "postprocess_hash": post_item["cfg_hash"],
                            "postprocess_cfg": post_item["cfg"],
                            "conf": conf,
                            "iou": iou,
                            "max_det": max_det,
                            "tta": bool(tta_item["enabled"]),
                            "tta_scales": [] if tta_item["scales"] is None else [float(x) for x in tta_item["scales"]],
                            "error": str(e),
                        }
                        all_results.append(err)

    rows_sorted = sort_official_eval_rows(all_results)
    sweep_json = suite_root / "metric_official_sweep_summary.json"
    sweep_csv = suite_root / "metric_official_sweep_summary.csv"
    sweep_sorted_csv = suite_root / "metric_official_sweep_summary_sorted.csv"
    write_official_eval_csv(all_results, sweep_csv)
    write_official_eval_csv(rows_sorted, sweep_sorted_csv)

    payload = {
        "ok": best_result is not None,
        "best_json": str(best_json_path),
        "best_weights": str(weights_path),
        "search_mode": search_mode,
        "planned_trials": planned_trials,
        "tta_variants": tta_variants,
        "conf_list": conf_list,
        "iou_list": iou_list,
        "postprocess_presets": preset_names,
        "best": best_result,
        "rows": all_results,
        "summary_csv": str(sweep_csv),
        "summary_sorted_csv": str(sweep_sorted_csv),
    }
    with open(sweep_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[MetricSweep][OUT] summary.json = {sweep_json}")
    print(f"[MetricSweep][OUT] summary.csv = {sweep_csv}")
    if best_result is not None:
        print(
            f"[MetricSweep][BEST] score={best_result.get('score_all')} "
            f"conf={best_result.get('conf')} iou={best_result.get('iou')} "
            f"preset={best_result.get('postprocess_preset')} "
            f"tta={best_result.get('tta')}"
        )
    return payload


def run_yolo26_texture_preserve_metric(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    search_mode: str = "aggressive",
    cls_gain: float = 2.6,
    imgsz: int = 512,
    batch_override: Optional[int] = None,
    aug_preset: str = "industrial_conservative",
    copy_paste: float = 0.10,
    ibn_ratio: float = 0.60,
    mixstyle_mode: str = "efdmix",
    official_tta: Optional[bool] = True,
    project: str = "ida_track1_yolo26_texture_metric",
    train_overrides: Optional[dict] = None,
):
    os.environ["IDA_MODEL_FAMILY"] = "26"
    train_res = run_texture_preserve_upperbound(
        scale_key=scale_key,
        budget_mode=budget_mode,
        cls_gain=cls_gain,
        imgsz=imgsz,
        batch_override=batch_override,
        aug_preset=aug_preset,
        copy_paste=copy_paste,
        ibn_ratio=ibn_ratio,
        mixstyle_mode=mixstyle_mode,
        official_tta=official_tta,
        model_family="26",
        project=project,
        train_overrides=train_overrides,
    )

    payload = {
        "ok": False,
        "model_family": "26",
        "train": None,
        "metric_resweep": None,
        "best_json": None,
        "final_best": None,
    }

    if train_res is None:
        return payload

    payload["train"] = {
        "save_dir": str(train_res.save_dir),
        "official_score": getattr(train_res, "official_score", None),
        "eval_weights": getattr(train_res, "eval_weights", None),
        "official_conf": getattr(train_res, "official_conf", None),
        "official_iou": getattr(train_res, "official_iou", None),
        "official_tta": getattr(train_res, "official_tta", None),
        "official_postprocess_preset": getattr(train_res, "official_postprocess_preset", None),
    }

    best_json_path = Path(str(train_res.save_dir)) / "official_eval" / "official_track1_best.json"
    if not best_json_path.exists():
        payload["error"] = f"best_json not found: {best_json_path}"
        summary_path = Path(str(train_res.save_dir)) / "yolo26_texture_metric_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[YOLO26][WARN] best_json missing: {best_json_path}")
        return payload

    payload["best_json"] = str(best_json_path)
    metric_payload = run_metric_aligned_official_resweep(
        best_json_path=str(best_json_path),
        search_mode=search_mode,
    )
    payload["metric_resweep"] = metric_payload
    best_metric = metric_payload.get("best", None) if isinstance(metric_payload, dict) else None
    if isinstance(best_metric, dict):
        payload["final_best"] = best_metric
        payload["ok"] = True

    summary_path = Path(str(train_res.save_dir)) / "yolo26_texture_metric_summary.json"
    payload["summary_json"] = str(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if isinstance(best_metric, dict):
        print(
            f"[YOLO26][BEST] score={best_metric.get('score_all')} "
            f"conf={best_metric.get('conf')} iou={best_metric.get('iou')} "
            f"preset={best_metric.get('postprocess_preset')} tta={best_metric.get('tta')}"
        )
    print(f"[YOLO26][OUT] summary.json = {summary_path}")
    return payload


def run_dsu_nwd_vfl_upperbound(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    cls_gain: float = 2.6,
    imgsz: int = 512,
    batch_override: Optional[int] = None,
    aug_preset: str = "industrial_soft",
    copy_paste: float = 0.10,
    ibn_ratio: float = 0.60,
    nwd_weight: float = 0.35,
    nwd_constant: float = 12.8,
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    official_tta: Optional[bool] = True,
    project: str = "ida_track1_dsu_loss_upperbound",
    train_overrides: Optional[dict] = None,
):
    return run_nwd_vfl_upperbound(
        scale_key=scale_key,
        budget_mode=budget_mode,
        cls_gain=cls_gain,
        imgsz=imgsz,
        batch_override=batch_override,
        aug_preset=aug_preset,
        copy_paste=copy_paste,
        ibn_ratio=ibn_ratio,
        mixstyle_mode="dsu",
        nwd_weight=nwd_weight,
        nwd_constant=nwd_constant,
        vfl_alpha=vfl_alpha,
        vfl_gamma=vfl_gamma,
        official_tta=official_tta,
        project=project,
        train_overrides=train_overrides,
    )


def run_balanced_screenloc_upperbound(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    cls_gain: float = 2.6,
    imgsz: int = 512,
    batch_override: Optional[int] = None,
    aug_preset: str = "industrial_soft",
    copy_paste: float = 0.05,
    ibn_ratio: float = 0.60,
    mixstyle_prob: float = 0.15,
    mixstyle_alpha: float = 0.12,
    vfl_alpha: float = 0.60,
    vfl_gamma: float = 1.5,
    nwd_weight: float = 0.0,
    nwd_constant: float = 12.8,
    official_tta: Optional[bool] = True,
    project: str = "ida_track1_balanced_upperbound",
    train_overrides: Optional[dict] = None,
):
    global OFFICIAL_PRED_TTA

    budget_key = budget_mode if budget_mode in P1_BUDGET_PRESETS else "quick"
    merged_overrides = deepcopy(P1_BUDGET_PRESETS[budget_key])
    if train_overrides:
        merged_overrides.update(train_overrides)

    profile = build_profile(
        label=(
            f"[{scale_key.upper()}] Balanced MixRand+DSU+lightVFL+SPD+SimAM "
            f"cls={float(cls_gain):.2f} imgsz={int(imgsz)}"
        ),
        enhance=False,
        mixstyle=True,
        randconv=True,
        scale_key=scale_key,
        tag="balanced_screenloc_dsu_vfl",
        pretrained=True,
        mixstyle_layers=2,
        mixstyle_mode="dsu",
        mixstyle_prob=float(mixstyle_prob),
        mixstyle_alpha=float(mixstyle_alpha),
        randconv_layers=2,
        spd=True,
        spd_layers=3,
        spd_scale=2,
        spd_alpha_init=0.10,
        simam=True,
        simam_layers=2,
        simam_e_lambda=1e-4,
        aug_preset=str(aug_preset),
        imgsz_override=int(imgsz),
        batch_override=None if batch_override is None else int(batch_override),
        auto_batch_from_imgsz=(batch_override is None),
        copy_paste=float(copy_paste),
        ibn_ratio=float(ibn_ratio),
        ibn_layers=2,
        cls_gain=float(cls_gain),
        project=str(project),
        swa_strategy="swad",
        swad_smooth_window=5,
        swad_tolerance_ratio=0.20,
        swad_n_converge=2,
        use_vfl=True,
        vfl_alpha=float(vfl_alpha),
        vfl_gamma=float(vfl_gamma),
        nwd_weight=float(nwd_weight),
        nwd_constant=float(nwd_constant),
    )
    profile.update(make_light_official_eval_settings(conf=0.08, iou=0.30, postprocess_presets=["base"], tta=False))

    old_official_tta = OFFICIAL_PRED_TTA
    if official_tta is not None:
        OFFICIAL_PRED_TTA = bool(official_tta)

    try:
        return run_experiment(
            profile,
            train_overrides=(merged_overrides if merged_overrides else None),
        )
    finally:
        OFFICIAL_PRED_TTA = old_official_tta


def run_balanced_screenloc_upperbound_conservative(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    official_tta: Optional[bool] = True,
    train_overrides: Optional[dict] = None,
):
    return run_balanced_screenloc_upperbound(
        scale_key=scale_key,
        budget_mode=budget_mode,
        copy_paste=0.00,
        ibn_ratio=0.55,
        mixstyle_prob=0.10,
        mixstyle_alpha=0.08,
        vfl_alpha=0.50,
        vfl_gamma=1.2,
        nwd_weight=0.0,
        official_tta=official_tta,
        project="ida_track1_balanced_upperbound_conservative",
        train_overrides=train_overrides,
    )


def run_balanced_screenloc_upperbound_aggressive(
    scale_key: str = DEFAULT_SCALE,
    budget_mode: str = "quick",
    official_tta: Optional[bool] = True,
    train_overrides: Optional[dict] = None,
):
    return run_balanced_screenloc_upperbound(
        scale_key=scale_key,
        budget_mode=budget_mode,
        copy_paste=0.10,
        ibn_ratio=0.65,
        mixstyle_prob=0.20,
        mixstyle_alpha=0.15,
        vfl_alpha=0.75,
        vfl_gamma=2.0,
        nwd_weight=0.12,
        official_tta=official_tta,
        project="ida_track1_balanced_upperbound_aggressive",
        train_overrides=train_overrides,
    )


ABLATION_ITEMS = [
    ("Baseline", dict(enhance=False, mixstyle=False, randconv=False)),
    ("EFE", dict(enhance=True, mixstyle=False, randconv=False)),
    ("RandConv", dict(enhance=False, mixstyle=False, randconv=True)),
    ("MixStyle", dict(enhance=False, mixstyle=True, randconv=False)),
    ("MixStyle+RandConv", dict(enhance=False, mixstyle=True, randconv=True)),
]


def run_experiment(
    profile: dict,
    data_override: Optional[str] = None,
    train_overrides: Optional[dict] = None,
    weights_override: Optional[str] = None,
):
    print("\n" + "#" * 80)
    print(f"# 开始实验: {profile['label']}")
    print("#" * 80)

    backup_official = apply_official_eval_overrides_from_profile(profile)
    print(
        f"[OfficialMode] eval={RUN_OFFICIAL_EVAL_AFTER_TRAIN} sweep={RUN_OFFICIAL_SWEEP_AFTER_TRAIN} "
        f"conf={OFFICIAL_PRED_CONF} iou={OFFICIAL_PRED_IOU} max_det={OFFICIAL_PRED_MAX_DET} "
        f"tta={OFFICIAL_PRED_TTA} presets={OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES} "
        f"periodic={OFFICIAL_EVAL_INCLUDE_PERIODIC} candidate_types={OFFICIAL_EVAL_CANDIDATE_TYPES} "
        f"max_candidates={OFFICIAL_EVAL_MAX_CANDIDATES}"
    )
    try:
        cfg = deepcopy(BASE_CFG)
        if data_override:
            cfg["data"] = data_override

        cfg["project"] = str(RESULT_ROOT / profile["project"])
        cfg["name"] = profile["run_name"]
        cfg = apply_profile_train_overrides(cfg, profile)

        if train_overrides:
            cfg.update(train_overrides)

        apply_plugins(profile)

        if weights_override:
            model = load_model_from_checkpoint(weights_override)
        else:
            model_source = render_custom_yaml(
                profile["scale"],
                data_yaml=cfg["data"],
                model_family=profile.get("model_family", DEFAULT_MODEL_FAMILY),
            )
            model = load_model(
                model_source,
                scale_key=profile["scale"],
                pretrained=profile.get("pretrained", True),
                model_family=profile.get("model_family", DEFAULT_MODEL_FAMILY),
            )

        start = time.time()
        res = train_once(model, sanitize_cfg(cfg), profile)
        minutes = (time.time() - start) / 60.0
    except Exception as e:
        print(f"[Error] 实验失败: {e}")
        traceback.print_exc()
        restore_official_eval_overrides(backup_official)
        return None

    try:
        print(f"[Done] {profile['label']}")
        print(f"  保存目录: {res.save_dir}")
        print(f"  耗时: {minutes:.1f} 分钟")
        print_best_summary(Path(res.save_dir))

        if profile.get("swa", True):
            try:
                res.swa = build_swa_checkpoint(
                    save_dir=Path(res.save_dir),
                    window=int(profile.get("swa_window", 4)),
                    max_models=int(profile.get("swa_max_models", 6)),
                    min_models=int(profile.get("swa_min_models", 3)),
                    include_best=True,
                    include_last=bool(profile.get("swa_include_last", False)),
                    out_name="swa_best.pt",
                    strategy=str(profile.get("swa_strategy", "window")),
                    swad_smooth_window=int(profile.get("swad_smooth_window", 5)),
                    swad_tolerance_ratio=float(profile.get("swad_tolerance_ratio", 0.25)),
                    swad_n_converge=int(profile.get("swad_n_converge", 2)),
                )
            except Exception as e:
                print(f"[CKPT-AVG][WARN] 失败: {e}")
                traceback.print_exc()
                res.swa = dict(ok=False, reason=str(e))

        try:
            verify_checkpoints_loadable(Path(res.save_dir), strict=True)
        except Exception as e:
            print(f"[CKPT-VERIFY][ERROR] {e}")
            traceback.print_exc()
            return res

        official_summary = sweep_official_track1(
            train_save_dir=Path(res.save_dir),
            data_yaml=cfg["data"],
            imgsz=int(cfg.get("imgsz", 512)),
            device=str(cfg.get("device", DEFAULT_DEVICE)),
        )
        res.official = official_summary

        try:
            best_official = official_summary.get("best", None) if isinstance(official_summary, dict) else None
            if best_official and best_official.get("score_all", None) is not None:
                res.official_score = float(best_official["score_all"])
                res.eval_weights = best_official.get("weights", None)
                res.official_conf = best_official.get("conf", None)
                res.official_iou = best_official.get("iou", None)
                res.official_max_det = best_official.get("max_det", None)
                res.official_tta = best_official.get("tta", None)
                res.official_postprocess_preset = best_official.get("postprocess_preset", None)
                res.official_postprocess_hash = best_official.get("postprocess_hash", None)
                res.official_candidate_type = best_official.get("candidate_type", None)
                res.official_candidate_epoch = best_official.get("candidate_epoch", None)
                print(f"[OfficialEval][ALL] Track1={res.official_score}")
                print(f"[OfficialEval][Chosen] weights={res.eval_weights}")
                print(
                    f"[OfficialEval][ChosenParams] conf={res.official_conf} "
                    f"iou={res.official_iou} max_det={res.official_max_det} tta={res.official_tta} "
                    f"preset={res.official_postprocess_preset} "
                    f"type={res.official_candidate_type} epoch={res.official_candidate_epoch}"
                )
            else:
                res.official_score = None
                res.official_conf = None
                res.official_iou = None
                res.official_max_det = None
                res.official_tta = None
                res.official_postprocess_preset = None
                res.official_postprocess_hash = None
                res.official_candidate_type = None
                res.official_candidate_epoch = None
        except Exception:
            res.official_score = None
            res.official_conf = None
            res.official_iou = None
            res.official_max_det = None
            res.official_tta = None
            res.official_postprocess_preset = None
            res.official_postprocess_hash = None
            res.official_candidate_type = None
            res.official_candidate_epoch = None
        return res
    finally:
        restore_official_eval_overrides(backup_official)


def run_ablation_suite(scale_key: str):
    results = []
    suite_id = make_run_id("S")

    for name, cfg_item in ABLATION_ITEMS:
        profile = build_profile(
            label=f"[{scale_key.upper()}] {name}",
            enhance=cfg_item["enhance"],
            mixstyle=cfg_item["mixstyle"],
            randconv=cfg_item["randconv"],
            scale_key=scale_key,
            tag=f"ablation_{name.replace('+', '_')}_{suite_id}",
            pretrained=True,
        )
        res = run_experiment(profile)

        row = dict(
            label=profile["label"],
            score=0.0,
            box_map=0.0,
            mask_map=0.0,
            official_track1=None,
            save_dir="(failed)",
        )

        if res is not None:
            best, _ = extract_best_last_metrics(Path(res.save_dir))
            row.update(
                score=best["score"],
                box_map=best["box_map"],
                mask_map=best["mask_map"],
                official_track1=res.official_score,
                save_dir=res.save_dir,
            )
        results.append(row)

    print("\n" + "=" * 126)
    print("Ablation summary (training uses Track1Proxy; final selection uses official sweep)")
    print("=" * 126)
    print(f"{'配置':<32} {'Track1Proxy':>12} {'OfficialTrack1':>16} {'Box mAP50-95':>14} {'Mask mAP50-95':>15}")
    print("-" * 126)

    for x in results:
        official_str = "None" if x["official_track1"] is None else f"{x['official_track1']:.4f}"
        print(
            f"{x['label']:<32} "
            f"{x['score']:>12.4f} "
            f"{official_str:>16} "
            f"{x['box_map']:>14.4f} "
            f"{x['mask_map']:>15.4f}"
        )
        print(f"  -> {x['save_dir']}")

    print("=" * 126)


# ===================== Priority-1 自动扫描 =====================
def ranking_key(row: dict):
    official_raw = row.get("official_track1", None)
    proxy_raw = row.get("track1_proxy", None)

    official = safe_float(official_raw, default=-1e9)
    proxy = safe_float(proxy_raw, default=-1e9)

    has_official = official_raw not in (None, "")
    return (1 if has_official else 0, official, proxy)


def pick_best_sweep_row(rows: List[dict]) -> Optional[dict]:
    valid = [r for r in rows if r.get("status") == "ok"]
    if not valid:
        return None
    return max(valid, key=ranking_key)


def sweep_row_from_result(
    stage: str,
    stage_order: int,
    index_in_stage: int,
    profile: dict,
    res,
    train_overrides: dict,
):
    row = dict(
        stage=stage,
        stage_order=int(stage_order),
        index_in_stage=int(index_in_stage),
        label=profile.get("label"),
        model_family=profile.get("model_family", DEFAULT_MODEL_FAMILY),
        status="failed",
        official_track1=None,
        track1_proxy=None,
        box_map=None,
        mask_map=None,
        best_epoch=None,
        mixstyle_mode=profile.get("mixstyle_mode", "mixstyle"),
        mixstyle_prob=profile.get("mixstyle_prob"),
        mixstyle_alpha=profile.get("mixstyle_alpha"),
        mixstyle_layers=profile.get("mixstyle_layers"),
        copy_paste=profile.get("copy_paste"),
        ibn_ratio=profile.get("ibn_ratio"),
        ibn_layers=profile.get("ibn_layers"),
        aug_preset=profile.get("aug_preset"),
        train_imgsz=profile.get("imgsz_override") if profile.get("imgsz_override") is not None else BASE_CFG["imgsz"],
        train_batch=profile.get("batch_override", None),
        cls_gain=profile.get("cls_gain", BASE_CFG["cls"]),
        enhance=profile.get("enhance"),
        mixstyle=profile.get("mixstyle"),
        randconv=profile.get("randconv"),
        protect_shallow_textures=profile.get("protect_shallow_textures", False),
        spd=profile.get("spd"),
        spd_layers=profile.get("spd_layers"),
        simam=profile.get("simam"),
        simam_layers=profile.get("simam_layers"),
        swa_strategy=profile.get("swa_strategy"),
        use_vfl=profile.get("use_vfl"),
        vfl_alpha=profile.get("vfl_alpha"),
        vfl_gamma=profile.get("vfl_gamma"),
        nwd_weight=profile.get("nwd_weight"),
        nwd_constant=profile.get("nwd_constant"),
        scale=profile.get("scale"),
        epochs=train_overrides.get("epochs", BASE_CFG["epochs"]) if train_overrides else BASE_CFG["epochs"],
        patience=train_overrides.get("patience", BASE_CFG["patience"]) if train_overrides else BASE_CFG["patience"],
        eval_weights=None,
        official_conf=None,
        official_iou=None,
        official_max_det=None,
        official_tta=None,
        official_postprocess_preset=None,
        official_postprocess_hash=None,
        official_candidate_type=None,
        official_candidate_epoch=None,
        save_dir=None,
    )

    if res is None:
        return row

    if row["train_batch"] is None and profile.get("auto_batch_from_imgsz", False):
        train_imgsz = row.get("train_imgsz", BASE_CFG["imgsz"])
        if train_imgsz is None:
            train_imgsz = BASE_CFG["imgsz"]
        row["train_imgsz"] = int(train_imgsz)
        row["train_batch"] = AUTO_BATCH_BY_IMGSZ.get(int(row["train_imgsz"]), BASE_CFG["batch"])

    try:
        best, _ = extract_best_last_metrics(Path(res.save_dir))
        row["track1_proxy"] = float(best["score"])
        row["box_map"] = float(best["box_map"])
        row["mask_map"] = float(best["mask_map"])
        row["best_epoch"] = int(best["epoch"])
    except Exception:
        pass

    row["official_track1"] = None if res.official_score is None else float(res.official_score)
    row["eval_weights"] = res.eval_weights
    row["official_conf"] = res.official_conf
    row["official_iou"] = res.official_iou
    row["official_max_det"] = res.official_max_det
    row["official_tta"] = getattr(res, "official_tta", None)
    row["official_postprocess_preset"] = res.official_postprocess_preset
    row["official_postprocess_hash"] = res.official_postprocess_hash
    row["official_candidate_type"] = res.official_candidate_type
    row["official_candidate_epoch"] = res.official_candidate_epoch
    row["save_dir"] = res.save_dir
    row["status"] = "ok"
    return row


def sort_sweep_rows(rows: List[dict]) -> List[dict]:
    def _key(r):
        rk = ranking_key(r)
        return (
            int(r.get("stage_order", 999)),
            -rk[0],
            -rk[1],
            -rk[2],
            int(r.get("index_in_stage", 999999)),
        )

    return sorted(rows, key=_key)


def stage_sorted_rows(rows: List[dict]) -> List[dict]:
    return sorted(rows, key=ranking_key, reverse=True)


def print_stage_ranking(stage_name: str, rows: List[dict], topk: Optional[int] = None):
    sorted_rows = stage_sorted_rows(rows)
    if topk is not None:
        sorted_rows = sorted_rows[:max(1, int(topk))]

    print("\n" + "=" * 132)
    print(f"[{stage_name}] Ranking")
    print("=" * 132)
    print(
        f"{'Rank':<6} {'Official':>10} {'Proxy':>10} {'mProb':>8} {'mAlpha':>8} {'mLayer':>8} "
        f"{'cPaste':>8} {'ibn':>8} {'bestEp':>8} {'标签':<34}"
    )
    print("-" * 132)

    for i, r in enumerate(sorted_rows, start=1):
        print(
            f"{i:<6} "
            f"{pretty_num(r.get('official_track1'), 4):>10} "
            f"{pretty_num(r.get('track1_proxy'), 4):>10} "
            f"{pretty_num(r.get('mixstyle_prob'), 2):>8} "
            f"{pretty_num(r.get('mixstyle_alpha'), 2):>8} "
            f"{str(r.get('mixstyle_layers')):>8} "
            f"{pretty_num(r.get('copy_paste'), 2):>8} "
            f"{pretty_num(r.get('ibn_ratio'), 2):>8} "
            f"{str(r.get('best_epoch')):>8} "
            f"{str(r.get('label', ''))[:34]:<34}"
        )

    print("=" * 132)


def write_sweep_summary(rows: List[dict], sweep_root: Path, meta: dict):
    sweep_root.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_root / "summary.csv"
    json_path = sweep_root / "summary.json"

    rows_sorted = sort_sweep_rows(rows)
    fieldnames = [
        "stage", "stage_order", "index_in_stage", "label", "model_family", "status",
        "official_track1", "track1_proxy", "box_map", "mask_map", "best_epoch",
        "mixstyle_mode", "mixstyle_prob", "mixstyle_alpha", "mixstyle_layers",
        "copy_paste", "ibn_ratio", "ibn_layers",
        "aug_preset", "train_imgsz", "train_batch", "cls_gain",
        "enhance", "mixstyle", "randconv", "protect_shallow_textures", "spd", "spd_layers", "simam", "simam_layers", "use_vfl", "vfl_alpha", "vfl_gamma", "nwd_weight", "nwd_constant", "swa_strategy", "scale",
        "epochs", "patience", "eval_weights",
        "official_conf", "official_iou", "official_max_det", "official_tta",
        "official_postprocess_preset", "official_postprocess_hash",
        "official_candidate_type", "official_candidate_epoch",
        "save_dir",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r.get(k, None) for k in fieldnames})

    payload = {
        "meta": meta,
        "rows": rows_sorted,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def clone_profile_for_stage(
    scale_key: str,
    suite_project: str,
    stage_tag: str,
    label_suffix: str,
    mixstyle_prob: float,
    mixstyle_alpha: float,
    mixstyle_layers: int,
    copy_paste: float,
    ibn_ratio: float,
):
    return build_profile(
        label=f"[{scale_key.upper()}] {label_suffix}",
        enhance=False,
        mixstyle=True,
        randconv=False,
        scale_key=scale_key,
        tag=stage_tag,
        pretrained=True,
        mixstyle_prob=float(mixstyle_prob),
        mixstyle_alpha=float(mixstyle_alpha),
        mixstyle_layers=int(mixstyle_layers),
        ibn_ratio=float(ibn_ratio),
        ibn_layers=1,
        copy_paste=float(copy_paste),
        project=suite_project,
        swa=True,
    )


def run_priority1_autoscan(scale_key: str, budget_mode: str = "quick", sweep_mode: str = "compact"):
    budget_mode = budget_mode if budget_mode in P1_BUDGET_PRESETS else "quick"
    sweep_mode = sweep_mode if sweep_mode in P1_SWEEP_PRESETS else "compact"

    budget_overrides = deepcopy(P1_BUDGET_PRESETS[budget_mode])
    preset = deepcopy(P1_SWEEP_PRESETS[sweep_mode])

    suite_id = make_run_id("P1")
    suite_project = f"priority1_sweeps/{suite_id}"
    sweep_root = RESULT_ROOT / suite_project

    print("\n" + "#" * 90)
    print("# Priority-1 automatic sweep start")
    print("# Stages: MixStyle grid -> copy_paste -> IBN ratio")
    print(f"# scale={scale_key} budget={budget_mode} sweep={sweep_mode}")
    print(f"# output_dir={sweep_root}")
    print("#" * 90)

    all_rows = []
    stage_best = {}

    # ---------- Stage 1: MixStyle 网格 ----------
    stage1_grid = []
    anchor = (0.30, 0.30, 1)
    stage1_grid.append(anchor)

    for p, a, l in product(
        preset["mixstyle_prob"],
        preset["mixstyle_alpha"],
        preset["mixstyle_layers"],
    ):
        stage1_grid.append((float(p), float(a), int(l)))

    stage1_grid = dedup_tuples(stage1_grid)
    print(f"[P1][Stage1] 共 {len(stage1_grid)} 组 MixStyle 参数")

    stage1_rows = []
    for idx, (p, a, l) in enumerate(stage1_grid, start=1):
        label_suffix = f"Stage1 MixStyle p={p:.2f} a={a:.2f} l={l}"
        profile = clone_profile_for_stage(
            scale_key=scale_key,
            suite_project=suite_project,
            stage_tag=f"stage1_mixstyle_{idx:02d}",
            label_suffix=label_suffix,
            mixstyle_prob=p,
            mixstyle_alpha=a,
            mixstyle_layers=l,
            copy_paste=0.0,
            ibn_ratio=0.5,
        )
        res = run_experiment(profile, train_overrides=budget_overrides)
        row = sweep_row_from_result("stage1_mixstyle", 1, idx, profile, res, budget_overrides)
        stage1_rows.append(row)
        all_rows.append(row)

        write_sweep_summary(
            all_rows,
            sweep_root,
            meta=dict(
                suite_id=suite_id,
                scale=scale_key,
                budget_mode=budget_mode,
                sweep_mode=sweep_mode,
                updated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                stage_best=stage_best,
            ),
        )

    print_stage_ranking("Stage1 MixStyle", stage1_rows)
    best1 = pick_best_sweep_row(stage1_rows)
    if best1 is None:
        raise RuntimeError("Stage1 produced no successful result; sweep stopped.")

    stage_best["stage1_mixstyle"] = best1
    print(
        "[P1][Stage1][BEST] "
        f"Official={pretty_num(best1.get('official_track1'))} "
        f"Proxy={pretty_num(best1.get('track1_proxy'))} "
        f"p={best1.get('mixstyle_prob')} a={best1.get('mixstyle_alpha')} l={best1.get('mixstyle_layers')}"
    )

    # ---------- Stage 2: copy_paste ----------
    cp_list = dedup_tuples([(float(x),) for x in preset["copy_paste"]])
    print(f"[P1][Stage2] 共 {len(cp_list)} 组 copy_paste")

    stage2_rows = []
    for idx, (cp,) in enumerate(cp_list, start=1):
        label_suffix = (
            f"Stage2 CopyPaste cp={cp:.2f} "
            f"(p={float(best1['mixstyle_prob']):.2f},a={float(best1['mixstyle_alpha']):.2f},l={int(best1['mixstyle_layers'])})"
        )
        profile = clone_profile_for_stage(
            scale_key=scale_key,
            suite_project=suite_project,
            stage_tag=f"stage2_copypaste_{idx:02d}",
            label_suffix=label_suffix,
            mixstyle_prob=float(best1["mixstyle_prob"]),
            mixstyle_alpha=float(best1["mixstyle_alpha"]),
            mixstyle_layers=int(best1["mixstyle_layers"]),
            copy_paste=float(cp),
            ibn_ratio=0.5,
        )
        res = run_experiment(profile, train_overrides=budget_overrides)
        row = sweep_row_from_result("stage2_copypaste", 2, idx, profile, res, budget_overrides)
        stage2_rows.append(row)
        all_rows.append(row)

        write_sweep_summary(
            all_rows,
            sweep_root,
            meta=dict(
                suite_id=suite_id,
                scale=scale_key,
                budget_mode=budget_mode,
                sweep_mode=sweep_mode,
                updated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                stage_best=stage_best,
            ),
        )

    print_stage_ranking("Stage2 CopyPaste", stage2_rows)
    best2 = pick_best_sweep_row(stage2_rows)
    if best2 is None:
        best2 = deepcopy(best1)
        best2["stage"] = "stage2_copypaste(fallback_stage1)"

    stage_best["stage2_copypaste"] = best2
    print(
        "[P1][Stage2][BEST] "
        f"Official={pretty_num(best2.get('official_track1'))} "
        f"Proxy={pretty_num(best2.get('track1_proxy'))} "
        f"cp={best2.get('copy_paste')} "
        f"p={best2.get('mixstyle_prob')} a={best2.get('mixstyle_alpha')} l={best2.get('mixstyle_layers')}"
    )

    # ---------- Stage 3: IBN ratio ----------
    ibn_list = dedup_tuples([(float(x),) for x in preset["ibn_ratio"]])
    print(f"[P1][Stage3] 共 {len(ibn_list)} 组 IBN ratio")

    stage3_rows = []
    for idx, (ratio,) in enumerate(ibn_list, start=1):
        cp_val = float(best2.get("copy_paste") or 0.0)
        label_suffix = (
            f"Stage3 IBN ratio={ratio:.2f} "
            f"(p={float(best2['mixstyle_prob']):.2f},a={float(best2['mixstyle_alpha']):.2f},"
            f"l={int(best2['mixstyle_layers'])},cp={cp_val:.2f})"
        )
        profile = clone_profile_for_stage(
            scale_key=scale_key,
            suite_project=suite_project,
            stage_tag=f"stage3_ibn_{idx:02d}",
            label_suffix=label_suffix,
            mixstyle_prob=float(best2["mixstyle_prob"]),
            mixstyle_alpha=float(best2["mixstyle_alpha"]),
            mixstyle_layers=int(best2["mixstyle_layers"]),
            copy_paste=cp_val,
            ibn_ratio=float(ratio),
        )
        res = run_experiment(profile, train_overrides=budget_overrides)
        row = sweep_row_from_result("stage3_ibnratio", 3, idx, profile, res, budget_overrides)
        stage3_rows.append(row)
        all_rows.append(row)

        write_sweep_summary(
            all_rows,
            sweep_root,
            meta=dict(
                suite_id=suite_id,
                scale=scale_key,
                budget_mode=budget_mode,
                sweep_mode=sweep_mode,
                updated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                stage_best=stage_best,
            ),
        )

    print_stage_ranking("Stage3 IBN Ratio", stage3_rows)
    best3 = pick_best_sweep_row(stage3_rows)
    if best3 is None:
        best3 = deepcopy(best2)
        best3["stage"] = "stage3_ibnratio(fallback_stage2)"

    stage_best["stage3_ibnratio"] = best3

    # ---------- Final summary ----------
    final_best = pick_best_sweep_row(all_rows)
    csv_path, json_path = write_sweep_summary(
        all_rows,
        sweep_root,
        meta=dict(
            suite_id=suite_id,
            scale=scale_key,
            budget_mode=budget_mode,
            sweep_mode=sweep_mode,
            updated_at=datetime.datetime.now().isoformat(timespec="seconds"),
            stage_best=stage_best,
            final_best=final_best,
        ),
    )

    print_stage_ranking("全量总榜", all_rows, topk=min(20, len(all_rows)))

    print("\n" + "#" * 90)
    print("# Priority-1 automatic sweep finished")
    print("#" * 90)
    print(f"[P1][OUT] summary.csv  = {csv_path}")
    print(f"[P1][OUT] summary.json = {json_path}")

    for k in ("stage1_mixstyle", "stage2_copypaste", "stage3_ibnratio"):
        b = stage_best.get(k, None)
        if b is None:
            continue
        print(
            f"[P1][BEST][{k}] "
            f"Official={pretty_num(b.get('official_track1'))} "
            f"Proxy={pretty_num(b.get('track1_proxy'))} "
            f"p={pretty_num(b.get('mixstyle_prob'), 2)} "
            f"a={pretty_num(b.get('mixstyle_alpha'), 2)} "
            f"l={b.get('mixstyle_layers')} "
            f"cp={pretty_num(b.get('copy_paste'), 2)} "
            f"ibn={pretty_num(b.get('ibn_ratio'), 2)}"
        )

    if final_best is not None:
        print(
            f"[P1][FINAL_BEST] "
            f"Official={pretty_num(final_best.get('official_track1'))} "
            f"Proxy={pretty_num(final_best.get('track1_proxy'))} "
            f"stage={final_best.get('stage')} "
            f"p={pretty_num(final_best.get('mixstyle_prob'), 2)} "
            f"a={pretty_num(final_best.get('mixstyle_alpha'), 2)} "
            f"l={final_best.get('mixstyle_layers')} "
            f"cp={pretty_num(final_best.get('copy_paste'), 2)} "
            f"ibn={pretty_num(final_best.get('ibn_ratio'), 2)} "
            f"save_dir={final_best.get('save_dir')}"
        )

    return dict(
        suite_id=suite_id,
        sweep_root=str(sweep_root),
        summary_csv=str(csv_path),
        summary_json=str(json_path),
        stage_best=stage_best,
        final_best=final_best,
        rows=all_rows,
    )


def run_priority2_autoscan(scale_key: str, budget_mode: str = "quick", sweep_mode: str = "compact"):
    preset = P2_SWEEP_PRESETS[sweep_mode]
    budget_overrides = deepcopy(P1_BUDGET_PRESETS[budget_mode])

    suite_id = make_run_id("P2")
    suite_project = f"priority2_augscans/{suite_id}"
    sweep_root = RESULT_ROOT / suite_project

    print("\n" + "#" * 90)
    print("# Priority-2 automatic sweep start")
    print("# Stages: aug preset -> imgsz -> cls gain")
    print(f"# scale={scale_key} budget={budget_mode} sweep={sweep_mode}")
    print(f"# output_dir={sweep_root}")
    print("#" * 90)

    all_rows = []
    stage_best = {}

    # ---------- Stage 1: augmentation preset ----------
    aug_presets = [str(x) for x in preset["aug_presets"]]
    stage1_rows = []
    print(f"[P2][Stage1] total {len(aug_presets)} augmentation presets")

    for idx, aug_name in enumerate(aug_presets, start=1):
        profile = build_profile(
            label=f"[{scale_key.upper()}] Stage1 aug={aug_name}",
            enhance=False,
            mixstyle=True,
            randconv=True,
            scale_key=scale_key,
            tag=f"p2_stage1_aug_{idx:02d}",
            aug_preset=aug_name,
            copy_paste=0.0,
            ibn_ratio=0.5,
            cls_gain=BASE_CFG["cls"],
            project=suite_project,
        )
        res = run_experiment(profile, train_overrides=budget_overrides)
        row = sweep_row_from_result("stage1_augpreset", 1, idx, profile, res, budget_overrides)
        stage1_rows.append(row)
        all_rows.append(row)
        write_sweep_summary(
            all_rows,
            sweep_root,
            meta=dict(
                suite_id=suite_id,
                scale=scale_key,
                budget_mode=budget_mode,
                sweep_mode=sweep_mode,
                updated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                stage_best=stage_best,
            ),
        )

    print_stage_ranking("Stage1 Aug Preset", stage1_rows)
    best1 = pick_best_sweep_row(stage1_rows)
    if best1 is None:
        raise RuntimeError("Stage1 aug preset produced no successful result; sweep stopped.")
    stage_best["stage1_augpreset"] = best1

    # ---------- Stage 2: imgsz ----------
    imgsz_list = [int(x) for x in preset["imgsz_list"]]
    stage2_rows = []
    print(f"[P2][Stage2] total {len(imgsz_list)} imgsz settings")

    for idx, imgsz_val in enumerate(imgsz_list, start=1):
        profile = build_profile(
            label=f"[{scale_key.upper()}] Stage2 imgsz={imgsz_val} aug={best1.get('aug_preset')}",
            enhance=False,
            mixstyle=True,
            randconv=True,
            scale_key=scale_key,
            tag=f"p2_stage2_imgsz_{idx:02d}",
            aug_preset=str(best1.get("aug_preset") or "default"),
            imgsz_override=int(imgsz_val),
            copy_paste=0.0,
            ibn_ratio=0.5,
            cls_gain=BASE_CFG["cls"],
            project=suite_project,
        )
        res = run_experiment(profile, train_overrides=budget_overrides)
        row = sweep_row_from_result("stage2_imgsz", 2, idx, profile, res, budget_overrides)
        stage2_rows.append(row)
        all_rows.append(row)
        write_sweep_summary(
            all_rows,
            sweep_root,
            meta=dict(
                suite_id=suite_id,
                scale=scale_key,
                budget_mode=budget_mode,
                sweep_mode=sweep_mode,
                updated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                stage_best=stage_best,
            ),
        )

    print_stage_ranking("Stage2 ImgSz", stage2_rows)
    best2 = pick_best_sweep_row(stage2_rows)
    if best2 is None:
        best2 = deepcopy(best1)
        best2["stage"] = "stage2_imgsz(fallback_stage1)"
    stage_best["stage2_imgsz"] = best2

    # ---------- Stage 3: cls gain ----------
    cls_gain_list = [float(x) for x in preset["cls_gain_list"]]
    stage3_rows = []
    print(f"[P2][Stage3] total {len(cls_gain_list)} cls gains")

    for idx, cls_gain in enumerate(cls_gain_list, start=1):
        profile = build_profile(
            label=(
                f"[{scale_key.upper()}] Stage3 cls={cls_gain:.2f} "
                f"aug={best2.get('aug_preset')} imgsz={int(best2.get('train_imgsz') or BASE_CFG['imgsz'])}"
            ),
            enhance=False,
            mixstyle=True,
            randconv=True,
            scale_key=scale_key,
            tag=f"p2_stage3_cls_{idx:02d}",
            aug_preset=str(best2.get("aug_preset") or "default"),
            imgsz_override=int(best2.get("train_imgsz") or BASE_CFG["imgsz"]),
            copy_paste=0.0,
            ibn_ratio=0.5,
            cls_gain=float(cls_gain),
            project=suite_project,
        )
        res = run_experiment(profile, train_overrides=budget_overrides)
        row = sweep_row_from_result("stage3_clsgain", 3, idx, profile, res, budget_overrides)
        stage3_rows.append(row)
        all_rows.append(row)
        write_sweep_summary(
            all_rows,
            sweep_root,
            meta=dict(
                suite_id=suite_id,
                scale=scale_key,
                budget_mode=budget_mode,
                sweep_mode=sweep_mode,
                updated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                stage_best=stage_best,
            ),
        )

    print_stage_ranking("Stage3 Cls Gain", stage3_rows)
    best3 = pick_best_sweep_row(stage3_rows)
    if best3 is None:
        best3 = deepcopy(best2)
        best3["stage"] = "stage3_clsgain(fallback_stage2)"
    stage_best["stage3_clsgain"] = best3

    final_best = pick_best_sweep_row(all_rows)
    csv_path, json_path = write_sweep_summary(
        all_rows,
        sweep_root,
        meta=dict(
            suite_id=suite_id,
            scale=scale_key,
            budget_mode=budget_mode,
            sweep_mode=sweep_mode,
            updated_at=datetime.datetime.now().isoformat(timespec="seconds"),
            stage_best=stage_best,
            final_best=final_best,
        ),
    )

    print_stage_ranking("Overall", all_rows, topk=min(20, len(all_rows)))
    print("\n" + "#" * 90)
    print("# Priority-2 automatic sweep finished")
    print("#" * 90)
    print(f"[P2][OUT] summary.csv  = {csv_path}")
    print(f"[P2][OUT] summary.json = {json_path}")

    for k in ("stage1_augpreset", "stage2_imgsz", "stage3_clsgain"):
        b = stage_best.get(k, None)
        if b is None:
            continue
        print(
            f"[P2][BEST][{k}] "
            f"Official={pretty_num(b.get('official_track1'))} "
            f"Proxy={pretty_num(b.get('track1_proxy'))} "
            f"aug={b.get('aug_preset')} "
            f"imgsz={b.get('train_imgsz')} "
            f"cls={pretty_num(b.get('cls_gain'), 2)}"
        )

    if final_best is not None:
        print(
            f"[P2][FINAL_BEST] "
            f"Official={pretty_num(final_best.get('official_track1'))} "
            f"Proxy={pretty_num(final_best.get('track1_proxy'))} "
            f"stage={final_best.get('stage')} "
            f"aug={final_best.get('aug_preset')} "
            f"imgsz={final_best.get('train_imgsz')} "
            f"cls={pretty_num(final_best.get('cls_gain'), 2)} "
            f"save_dir={final_best.get('save_dir')}"
        )

    return dict(
        suite_id=suite_id,
        sweep_root=str(sweep_root),
        summary_csv=str(csv_path),
        summary_json=str(json_path),
        stage_best=stage_best,
        final_best=final_best,
        rows=all_rows,
    )


def run_priority3_hn_finetune(
    scale_key: str,
    budget_mode: str,
    best_json_path: str,
    clean_image_source: str,
):
    best = load_json_file(best_json_path)
    best_weights = str(best.get("weights", "")).strip()
    if not best_weights:
        raise ValueError(f"weights not found in best_json: {best_json_path}")

    suite_id = make_run_id("P3")
    suite_project = f"priority3_hn_finetune/{suite_id}"
    suite_root = RESULT_ROOT / suite_project
    suite_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 90)
    print("# Priority-3 hard-negative short fine-tune start")
    print(f"# scale={scale_key} budget={budget_mode}")
    print(f"# best_json={best_json_path}")
    print(f"# clean_image_source={clean_image_source}")
    print(f"# output_dir={suite_root}")
    print("#" * 90)

    postprocess_cfg = best.get("postprocess_cfg", None) if isinstance(best, dict) else None
    hn_mining_root = suite_root / "hard_negative_mining"
    hn_bundle_root = suite_root / "hard_negative_bundle"
    merged_data_yaml = suite_root / "data_with_hn.yaml"

    mining_summary = mine_hard_negatives(
        weights=best_weights,
        source=clean_image_source,
        out_dir=hn_mining_root,
        imgsz=int(best.get("imgsz", BASE_CFG["imgsz"])),
        batch=int(BASE_CFG["batch"]),
        device=str(DEFAULT_DEVICE),
        conf=float(best.get("conf", OFFICIAL_PRED_CONF)),
        iou=float(best.get("iou", OFFICIAL_PRED_IOU)),
        max_det=int(best.get("max_det", OFFICIAL_PRED_MAX_DET)),
        per_class_conf=None if not isinstance(postprocess_cfg, dict) else postprocess_cfg.get("per_class_conf"),
        per_class_min_area=None if not isinstance(postprocess_cfg, dict) else postprocess_cfg.get("per_class_min_area"),
        per_class_topk=None if not isinstance(postprocess_cfg, dict) else postprocess_cfg.get("per_class_topk"),
        image_gate=None if not isinstance(postprocess_cfg, dict) else postprocess_cfg.get("image_gate"),
        min_score_to_save=float(best.get("conf", OFFICIAL_PRED_CONF)),
        copy_whole_image=True,
        no_save_crops=False,
        verbose=False,
    )

    if int(mining_summary.get("num_items", 0)) <= 0:
        payload = {
            "ok": False,
            "reason": "no hard negatives mined",
            "best_json": str(best_json_path),
            "best_weights": best_weights,
            "clean_image_source": str(clean_image_source),
            "hard_negative_mining": mining_summary,
        }
        out_json = suite_root / "priority3_hn_finetune_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("[P3][WARN] No hard negatives were mined; skip fine-tune.")
        print(f"[P3][OUT] summary.json = {out_json}")
        return payload

    merge_summary = merge_hard_negatives(
        data_yaml=DATA_PATH,
        hn_image_dir=hn_mining_root / "images",
        output_data_yaml=merged_data_yaml,
        output_dataset_root=hn_bundle_root,
        mode="copy",
        clear_output_root=True,
    )

    hn_overrides = deepcopy(P3_HN_FINETUNE_PRESETS[budget_mode])
    inferred = infer_profile_kwargs_from_best_json(best_json_path=best_json_path, scale_key=scale_key)
    profile = build_profile(
        **inferred,
        label=f"[{scale_key.upper()}] HN FineTune from {Path(best_weights).stem}",
        tag="hn_finetune",
        project=suite_project,
        aug_preset="default",
        copy_paste=0.0,
    )

    res = run_experiment(
        profile,
        data_override=str(merged_data_yaml),
        train_overrides=hn_overrides,
        weights_override=best_weights,
    )

    payload = {
        "ok": res is not None,
        "best_json": str(best_json_path),
        "best_weights": best_weights,
        "clean_image_source": str(clean_image_source),
        "hard_negative_mining": mining_summary,
        "merge_summary": {k: v for k, v in merge_summary.items() if k != "items"},
        "merged_data_yaml": str(merged_data_yaml),
        "train_overrides": hn_overrides,
        "profile": profile,
        "result_save_dir": None if res is None else res.save_dir,
        "official_score": None if res is None else res.official_score,
        "eval_weights": None if res is None else res.eval_weights,
    }
    out_json = suite_root / "priority3_hn_finetune_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[P3][OUT] summary.json = {out_json}")
    if res is not None:
        print(
            f"[P3][FINAL] Official={pretty_num(res.official_score)} "
            f"weights={res.eval_weights} save_dir={res.save_dir}"
        )
    return payload


# ===================== 交互输入 =====================
def input_gpu_devices():
    print("\nTraining device mode:")
    print("  1. Single GPU")
    print("  2. Multi GPU (DDP)")
    mode = input("Select (1/2, default 1): ").strip() or "1"
    if mode == "2":
        return input("Enter GPU ids, e.g. 0,1,2: ").strip() or "0,1"
    return input("Enter single GPU id (default 0): ").strip() or "0"


def input_model_family(default_family=DEFAULT_MODEL_FAMILY):
    fam = (input(f"Model family 11/26 (default {default_family}): ").strip() or default_family).lower()
    return normalize_model_family(fam)


def input_scale_key(default_key="l"):
    sc = (input(f"Model scale n/s/m/l/x (default {default_key}): ").strip() or default_key).lower()
    return sc if sc in ("n", "s", "m", "l", "x") else default_key


def input_batch(default_batch):
    global MANUAL_BATCH_OVERRIDE
    s = input(f"Batch size (default {default_batch}): ").strip()
    if not s:
        MANUAL_BATCH_OVERRIDE = False
        return default_batch
    try:
        MANUAL_BATCH_OVERRIDE = True
        return int(s)
    except Exception:
        MANUAL_BATCH_OVERRIDE = False
        return default_batch


def input_budget_mode():
    print("\nSweep training budget:")
    print("  1. quick  (epochs=120, patience=30)")
    print("  2. full   (use BASE_CFG: epochs=600, patience=120)")
    ch = input("Select (1/2, default 1): ").strip() or "1"
    return "full" if ch == "2" else "quick"


def input_sweep_mode():
    print("\nSweep grid size:")
    print("  1. compact  (recommended first)")
    print("  2. extended (broader search)")
    ch = input("Select (1/2, default 1): ").strip() or "1"
    return "extended" if ch == "2" else "compact"


def input_tent_mode():
    print("\nTENT mode:")
    print("  1. lite  (BN stats only)")
    print("  2. grad  (BN affine + entropy minimization)")
    ch = input("Select (1/2, default 2): ").strip() or "2"
    return "lite" if ch == "1" else "grad"


def input_tent_reset_mode():
    print("\nTENT reset strategy:")
    print("  1. scenario  (recommended, accumulate within folder/domain)")
    print("  2. image     (reset every image, safer but weaker)")
    print("  3. never     (accumulate across all images, most aggressive)")
    ch = input("Select (1/2/3, default 1): ").strip() or "1"
    if ch == "2":
        return "image"
    if ch == "3":
        return "never"
    return "scenario"


def input_tent_search_mode():
    print("\nTENT sweep size:")
    print("  1. compact     (faster)")
    print("  2. aggressive  (recommended for final push)")
    ch = input("Select (1/2, default 2): ").strip() or "2"
    return "compact" if ch == "1" else "aggressive"


def input_official_search_mode():
    print("\nOfficial sweep size:")
    print("  1. compact     (faster)")
    print("  2. aggressive  (wider metric-aligned search)")
    ch = input("Select (1/2, default 2): ").strip() or "2"
    return "compact" if ch == "1" else "aggressive"


def input_float_value(prompt: str, default: float) -> float:
    s = input(f"{prompt} (default {default}): ").strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def input_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    s = input(f"{prompt} ({hint}): ").strip().lower()
    if not s:
        return bool(default)
    return s in {"y", "yes", "1", "true", "t"}


def input_existing_path(prompt: str, expect: str = "any") -> str:
    while True:
        s = input(prompt).strip()
        p = Path(s)
        if not s:
            print("Path cannot be empty.")
            continue
        if expect == "file" and (not p.is_file()):
            print(f"File not found: {p}")
            continue
        if expect == "dir" and (not p.is_dir()):
            print(f"Directory not found: {p}")
            continue
        if expect == "any" and (not p.exists()):
            print(f"Path not found: {p}")
            continue
        return str(p)


def main():
    global DEFAULT_DEVICE, BASE_CFG

    print("\n" + "=" * 80)
    print("YOLO11/YOLO26 IDA Track1 training + official-eval sweep")
    print("=" * 80)
    print(f"Data: {DATA_PATH}")
    print(f"Result root: {RESULT_ROOT}")
    print(f"Official metric script: {OFFICIAL_METRIC_SCRIPT}")
    print(f"Official eval enabled: {RUN_OFFICIAL_EVAL_AFTER_TRAIN}")
    print(f"Official sweep enabled: {RUN_OFFICIAL_SWEEP_AFTER_TRAIN}")
    print(f"Official eval split: {OFFICIAL_EVAL_SPLIT}")
    print(f"Official eval track: {OFFICIAL_EVAL_TRACK}")
    print(f"Official export mode: {'rect_polygon' if OFFICIAL_USE_RECT_EXPORT else 'ultralytics_save_txt'}")
    print(f"Official sweep confs: {OFFICIAL_SWEEP_CONF_LIST}")
    print(f"Official sweep ious: {OFFICIAL_SWEEP_IOU_LIST}")
    print(f"Official sweep max_det: {OFFICIAL_SWEEP_MAXDET_LIST}")
    print(f"Official postprocess presets: {OFFICIAL_SWEEP_POSTPROCESS_PRESET_NAMES}")
    print(f"Official TTA: {OFFICIAL_PRED_TTA} scales={OFFICIAL_PRED_TTA_SCALES}")
    print(f"Official TENT: {OFFICIAL_PRED_TENT} cfg={OFFICIAL_PRED_TENT_CFG}")
    print(f"Note: {OFFICIAL_EVAL_NOTE}")

    DEFAULT_DEVICE = input_gpu_devices()
    BASE_CFG["device"] = DEFAULT_DEVICE
    BASE_CFG["batch"] = input_batch(BASE_CFG["batch"])

    while True:
        print(
            "\nSelect an action:\n"
            "  1. Run Baseline (with IBN-lite)\n"
            "  2. Run EFE (with IBN-lite)\n"
            "  3. Run MixStyle (with IBN-lite)\n"
            "  4. Run RandConv (with IBN-lite + checkpoint averaging)\n"
            "  5. Run MixStyle+RandConv (with IBN-lite)\n"
            "  6. Run MixStyle+RandConv+EFDMix (single-variable, cls=2.6)\n"
            "  7. Run small ablation suite (Baseline / EFE / RandConv / MixStyle / MixStyle+RandConv)\n"
            "  8. Run priority-1 automatic sweep (MixStyle -> copy_paste -> IBN ratio)\n"
            "  9. Run priority-2 automatic sweep (aug preset -> imgsz -> cls gain)\n"
            " 10. Run priority-3 hard-negative short fine-tune\n"
            " 11. Run aggressive upper-bound recipe (MixRand+EFDMix+SPD+SimAM+SWAD+TTA)\n"
            " 12. Run aggressive loss recipe (NWD+VFL+MixRand+SPD+SimAM+SWAD+TTA)\n"
            " 13. Re-sweep an existing best with TENT inference\n"
            " 14. Run DSU+NWD+VFL upper-bound recipe\n"
            " 15. Run texture-preserve upper-bound recipe (deep MixRand+SPD+screen-aware eval)\n"
            " 16. Re-sweep an existing best with metric-aligned aggressive presets\n"
            " 17. Run dedicated YOLO26 texture-preserve + metric-aligned chain\n"
            " 18. Run balanced screen+loc single-model recipe (2000e)\n"
            " 19. Run balanced screen+loc conservative variant\n"
            " 20. Run balanced screen+loc aggressive variant\n"
            " 21. Run FDSA + screening-aware recipe\n"
            " 22. Exit\n"
        )
        choice = input("Enter choice (1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22): ").strip()

        if choice in ("22", ""):
            print("Exit.")
            return

        model_family = input_model_family(DEFAULT_MODEL_FAMILY)
        os.environ["IDA_MODEL_FAMILY"] = model_family
        scale_key = input_scale_key(DEFAULT_SCALE)

        if choice == "1":
            profile = build_profile(
                f"[{scale_key.upper()}] Baseline",
                False,
                False,
                False,
                scale_key,
                tag="baseline",
            )
            run_experiment(profile)

        elif choice == "2":
            profile = build_profile(
                f"[{scale_key.upper()}] EFE",
                True,
                False,
                False,
                scale_key,
                tag="efe",
            )
            run_experiment(profile)

        elif choice == "3":
            profile = build_profile(
                f"[{scale_key.upper()}] MixStyle",
                False,
                True,
                False,
                scale_key,
                tag="mixstyle",
            )
            run_experiment(profile)

        elif choice == "4":
            profile = build_profile(
                f"[{scale_key.upper()}] RandConv",
                False,
                False,
                True,
                scale_key,
                tag="randconv",
            )
            run_experiment(profile)

        elif choice == "5":
            profile = build_profile(
                f"[{scale_key.upper()}] MixStyle+RandConv",
                False,
                True,
                True,
                scale_key,
                tag="mixrand",
            )
            run_experiment(profile)

        elif choice == "6":
            profile = build_profile(
                f"[{scale_key.upper()}] MixStyle+RandConv+EFDMix cls=2.6",
                False,
                True,
                True,
                scale_key,
                tag="mixrand_efdmix_cls26",
                mixstyle_mode="efdmix",
                cls_gain=2.6,
                aug_preset="default",
            )
            run_experiment(profile)

        elif choice == "7":
            run_ablation_suite(scale_key)

        elif choice == "8":
            budget_mode = input_budget_mode()
            sweep_mode = input_sweep_mode()
            run_priority1_autoscan(scale_key=scale_key, budget_mode=budget_mode, sweep_mode=sweep_mode)

        elif choice == "9":
            budget_mode = input_budget_mode()
            sweep_mode = input_sweep_mode()
            run_priority2_autoscan(scale_key=scale_key, budget_mode=budget_mode, sweep_mode=sweep_mode)

        elif choice == "10":
            budget_mode = input_budget_mode()
            best_json_path = input_existing_path("Enter path to official_track1_best.json: ", expect="file")
            clean_image_source = input_existing_path("Enter directory of clean images for hard-negative mining: ", expect="dir")
            run_priority3_hn_finetune(
                scale_key=scale_key,
                budget_mode=budget_mode,
                best_json_path=best_json_path,
                clean_image_source=clean_image_source,
            )

        elif choice == "11":
            budget_mode = input_budget_mode()
            run_aggressive_upperbound(scale_key=scale_key, budget_mode=budget_mode)

        elif choice == "12":
            budget_mode = input_budget_mode()
            run_nwd_vfl_upperbound(scale_key=scale_key, budget_mode=budget_mode)

        elif choice == "13":
            best_json_path = input_existing_path("Enter path to official_track1_best.json: ", expect="file")
            tent_mode = input_tent_mode()
            tent_reset_mode = input_tent_reset_mode()
            tent_search_mode = input_tent_search_mode()
            run_tent_official_resweep(
                best_json_path=best_json_path,
                tent_mode=tent_mode,
                reset_mode=tent_reset_mode,
                search_mode=tent_search_mode,
            )

        elif choice == "14":
            budget_mode = input_budget_mode()
            run_dsu_nwd_vfl_upperbound(scale_key=scale_key, budget_mode=budget_mode)

        elif choice == "15":
            budget_mode = input_budget_mode()
            run_texture_preserve_upperbound(scale_key=scale_key, budget_mode=budget_mode)

        elif choice == "16":
            best_json_path = input_existing_path("Enter path to official_track1_best.json: ", expect="file")
            search_mode = input_official_search_mode()
            run_metric_aligned_official_resweep(best_json_path=best_json_path, search_mode=search_mode)

        elif choice == "17":
            budget_mode = input_budget_mode()
            search_mode = input_official_search_mode()
            run_yolo26_texture_preserve_metric(
                scale_key=scale_key,
                budget_mode=budget_mode,
                search_mode=search_mode,
            )

        elif choice == "18":
            budget_mode = input_budget_mode()
            run_balanced_screenloc_upperbound(
                scale_key=scale_key,
                budget_mode=budget_mode,
                train_overrides={
                    "epochs": 2000,
                    "patience": 300,
                    "lr0": 1.5e-4,
                    "lrf": 0.03,
                    "close_mosaic": 120,
                    "save_period": 20,
                },
            )

        elif choice == "19":
            budget_mode = input_budget_mode()
            run_balanced_screenloc_upperbound_conservative(scale_key=scale_key, budget_mode=budget_mode)

        elif choice == "20":
            budget_mode = input_budget_mode()
            run_balanced_screenloc_upperbound_aggressive(scale_key=scale_key, budget_mode=budget_mode)

        elif choice == "21":
            budget_mode = input_budget_mode()
            randconv = input_yes_no("Combine FDSA with RandConv", default=False)
            low_freq_ratio = input_float_value("FDSA low-frequency ratio", 0.08)
            screen_loss_weight = input_float_value("Screening loss weight", 0.15)
            screen_pos_weight = input_float_value("Screening positive-image weight", 1.25)
            screen_neg_weight = input_float_value("Screening negative-image weight", 1.00)
            cls_gain = input_float_value("Classification gain", 2.6)
            run_fdsa_screen_singlevar(
                scale_key=scale_key,
                budget_mode=budget_mode,
                cls_gain=cls_gain,
                low_freq_ratio=low_freq_ratio,
                screen_loss_weight=screen_loss_weight,
                screen_pos_weight=screen_pos_weight,
                screen_neg_weight=screen_neg_weight,
                randconv=randconv,
            )

        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()

