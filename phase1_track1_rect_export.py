# -*- coding: utf-8 -*-
"""
Phase 1:
将 Ultralytics/YOLO 的检测结果统一导出为“矩形 polygon txt”，
用于替代默认 save_txt=True 的分割 polygon 导出方式。

特点：
1. 同时兼容 det / seg 模型，只取 result.boxes
2. 输出 YOLO segmentation 风格的 polygon txt：
   cls x1 y1 x2 y1 x2 y2 x1 y2
3. 默认输出归一化坐标（推荐）
4. 支持：
   - 全局 conf
   - per-class conf
   - per-class min_area
   - per-class topk
5. 支持保留 source_root 相对目录，避免重名文件冲突

用法示例：
python phase1_track1_rect_export.py \
  --model runs/train/exp/weights/best.pt \
  --source /path/to/images/val \
  --source_root /path/to/images/val \
  --save_dir runs/phase1_pred/best_rect \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.50 \
  --device 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from ultralytics.data.augment import LetterBox
from ultralytics import YOLO


_IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def load_json_dict(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是 dict: {path}")
    return data


def dump_json(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, ensure_ascii=False, indent=2)


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def normalize_names(names: Any) -> Dict[int, str]:
    if names is None:
        return {}
    if isinstance(names, (list, tuple)):
        return {int(i): str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        out = {}
        for k, v in names.items():
            try:
                ik = int(k)
                out[ik] = str(v)
            except Exception:
                # 忽略非数字 key
                continue
        return out
    return {}


def cfg_lookup(
    cfg: Optional[Dict[str, Any]],
    cls_id: int,
    names: Dict[int, str],
    default: Any = None,
) -> Any:
    if not cfg:
        return default

    candidates = [
        cls_id,
        str(cls_id),
        names.get(cls_id),
        "all",
        "__all__",
        "default",
        "__default__",
    ]

    for key in candidates:
        if key is None:
            continue
        if key in cfg:
            return cfg[key]

    return default


def numeric_values_from_cfg(cfg: Optional[Dict[str, Any]]) -> list[float]:
    if not cfg:
        return []
    vals = []
    for v in cfg.values():
        try:
            vals.append(float(v))
        except Exception:
            pass
    return vals


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def resolve_class_id(spec: Any, names: Dict[int, str]) -> Optional[int]:
    try:
        return int(spec)
    except Exception:
        pass

    spec_str = str(spec).strip()
    if not spec_str:
        return None

    for cls_id, cls_name in names.items():
        if spec_str == str(cls_name) or spec_str.lower() == str(cls_name).lower():
            return int(cls_id)
    return None


def collect_image_gate_stats(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float(row.get("score", 0.0)) for row in rows]
    areas = []
    class_counts: Dict[int, int] = {}
    class_top1_score: Dict[int, float] = {}

    for row in rows:
        cls_id = int(row["cls_id"])
        box_n = row["box_n"]
        x1, y1, x2, y2 = [float(v) for v in box_n.tolist()]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        areas.append(area)
        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        score = float(row.get("score", 0.0))
        if score > class_top1_score.get(cls_id, 0.0):
            class_top1_score[cls_id] = score

    scores_sorted = sorted(scores, reverse=True)
    return {
        "kept_box_count": int(len(rows)),
        "top1_score": float(scores_sorted[0]) if len(scores_sorted) >= 1 else 0.0,
        "top2_score": float(scores_sorted[1]) if len(scores_sorted) >= 2 else 0.0,
        "sum_score": float(sum(scores)),
        "max_area": float(max(areas)) if areas else 0.0,
        "sum_area": float(sum(areas)),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_top1_score": {str(k): float(v) for k, v in class_top1_score.items()},
        "scores": [float(x) for x in scores_sorted],
    }


def collect_min_count_above_cfg(image_gate: Optional[Dict[str, Any]]) -> Dict[float, int]:
    out: Dict[float, int] = {}
    if not image_gate:
        return out

    nested = image_gate.get("min_count_above", None)
    if isinstance(nested, dict):
        for k, v in nested.items():
            try:
                out[float(k)] = int(v)
            except Exception:
                pass

    for k, v in image_gate.items():
        if not isinstance(k, str) or (not k.startswith("min_count_above_")):
            continue
        thr_str = k[len("min_count_above_"):].replace("p", ".")
        try:
            out[float(thr_str)] = int(v)
        except Exception:
            pass
    return out


def evaluate_image_gate(
    rows: list[Dict[str, Any]],
    image_gate: Optional[Dict[str, Any]],
    names: Dict[int, str],
) -> tuple[bool, Dict[str, Any]]:
    stats = collect_image_gate_stats(rows)
    enabled = bool(image_gate) and bool(image_gate.get("enabled", True))
    stats["gate_enabled"] = bool(enabled)

    if not enabled:
        stats["gate_passed"] = True
        stats["gate_reason"] = "disabled"
        return True, stats

    def fail(reason: str) -> tuple[bool, Dict[str, Any]]:
        stats["gate_passed"] = False
        stats["gate_reason"] = reason
        return False, stats

    if stats["kept_box_count"] < safe_int(image_gate.get("min_kept_boxes", 0), 0):
        return fail("min_kept_boxes")
    if stats["top1_score"] < safe_float(image_gate.get("min_top1_score", 0.0), 0.0):
        return fail("min_top1_score")
    if stats["top2_score"] < safe_float(image_gate.get("min_top2_score", 0.0), 0.0):
        return fail("min_top2_score")
    if stats["sum_score"] < safe_float(image_gate.get("min_sum_score", 0.0), 0.0):
        return fail("min_sum_score")
    if stats["max_area"] < safe_float(image_gate.get("min_max_area", 0.0), 0.0):
        return fail("min_max_area")
    if stats["sum_area"] < safe_float(image_gate.get("min_sum_area", 0.0), 0.0):
        return fail("min_sum_area")

    min_count_above = collect_min_count_above_cfg(image_gate)
    for thr, need in min_count_above.items():
        got = sum(1 for score in stats["scores"] if float(score) >= float(thr))
        if got < int(need):
            return fail(f"min_count_above_{thr:g}")

    class_counts = {int(k): int(v) for k, v in stats["class_counts"].items()}
    class_top1_score = {int(k): float(v) for k, v in stats["class_top1_score"].items()}

    require_any = image_gate.get("require_classes_any", None)
    if isinstance(require_any, (list, tuple)) and require_any:
        cls_ids = [resolve_class_id(x, names) for x in require_any]
        cls_ids = [x for x in cls_ids if x is not None]
        if cls_ids and not any(class_counts.get(int(cls_id), 0) > 0 for cls_id in cls_ids):
            return fail("require_classes_any")

    require_all = image_gate.get("require_classes_all", None)
    if isinstance(require_all, (list, tuple)) and require_all:
        cls_ids = [resolve_class_id(x, names) for x in require_all]
        cls_ids = [x for x in cls_ids if x is not None]
        if cls_ids and not all(class_counts.get(int(cls_id), 0) > 0 for cls_id in cls_ids):
            return fail("require_classes_all")

    min_class_count = image_gate.get("min_class_count", None)
    if isinstance(min_class_count, dict):
        for cls_spec, need_raw in min_class_count.items():
            cls_id = resolve_class_id(cls_spec, names)
            if cls_id is None:
                continue
            if class_counts.get(int(cls_id), 0) < safe_int(need_raw, 0):
                return fail(f"min_class_count_{cls_id}")

    min_class_top1_score = image_gate.get("min_class_top1_score", None)
    if isinstance(min_class_top1_score, dict):
        for cls_spec, thr_raw in min_class_top1_score.items():
            cls_id = resolve_class_id(cls_spec, names)
            if cls_id is None:
                continue
            if class_top1_score.get(int(cls_id), 0.0) < safe_float(thr_raw, 0.0):
                return fail(f"min_class_top1_score_{cls_id}")

    stats["gate_passed"] = True
    stats["gate_reason"] = "passed"
    return True, stats


def resolve_output_txt_path(
    image_path: Union[str, Path],
    save_dir: Union[str, Path],
    source_root: Optional[Union[str, Path]] = None,
) -> Path:
    image_path = Path(str(image_path))
    save_dir = Path(save_dir)

    if source_root is not None:
        source_root = Path(str(source_root))
        try:
            rel = image_path.resolve().relative_to(source_root.resolve())
            return (save_dir / rel).with_suffix(".txt")
        except Exception:
            pass

    return save_dir / f"{image_path.stem}.txt"


def resolve_json_file_name(
    image_path: Union[str, Path],
    source_root: Optional[Union[str, Path]] = None,
    keep_rel_path: bool = False,
) -> str:
    image_path = Path(str(image_path))
    if keep_rel_path and source_root is not None:
        source_root = Path(str(source_root))
        try:
            rel = image_path.resolve().relative_to(source_root.resolve())
            return rel.as_posix()
        except Exception:
            pass
    return image_path.name


def round_float_list(values: list[float], decimals: int = 6) -> list[float]:
    out = []
    for v in values:
        fv = float(v)
        if abs(fv) < 0.5 * (10 ** (-decimals)):
            fv = 0.0
        out.append(round(fv, decimals))
    return out


def normalize_tta_scales(tta_scales: Optional[list[float]]) -> list[float]:
    if not tta_scales:
        return [1.0]
    out = []
    seen = set()
    for v in tta_scales:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv <= 0:
            continue
        key = round(fv, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(key))
    return out or [1.0]


def is_image_file(path: Union[str, Path]) -> bool:
    p = Path(str(path))
    return p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS


def resolve_source_image_paths(source: Union[str, Path]) -> list[Path]:
    source_path = Path(str(source))

    if source_path.is_dir():
        items = [p.resolve() for p in sorted(source_path.rglob("*")) if is_image_file(p)]
    elif source_path.is_file():
        if source_path.suffix.lower() == ".txt":
            items = []
            with open(source_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    p = Path(s)
                    if not p.is_absolute():
                        p = (source_path.parent / p).resolve()
                    if not is_image_file(p):
                        raise FileNotFoundError(f"image path from list does not exist or is not an image: {p}")
                    items.append(p.resolve())
        elif is_image_file(source_path):
            items = [source_path.resolve()]
        else:
            raise ValueError(f"manual TTA only supports image file, image directory, or .txt list: {source_path}")
    else:
        raise FileNotFoundError(f"{source_path} does not exist")

    if not items:
        raise FileNotFoundError(f"no image files found under source: {source_path}")

    uniq = []
    seen = set()
    for p in items:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def box_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def classwise_nms_rows(rows: list[Dict[str, Any]], iou_thr: float) -> list[Dict[str, Any]]:
    if len(rows) <= 1:
        return list(rows)

    grouped: Dict[int, list[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["cls_id"]), []).append(row)

    kept: list[Dict[str, Any]] = []
    for cls_id, cls_rows in grouped.items():
        _ = cls_id
        pending = sorted(cls_rows, key=lambda x: float(x["score"]), reverse=True)
        cls_keep = []
        while pending:
            cur = pending.pop(0)
            cls_keep.append(cur)
            pending = [
                other
                for other in pending
                if box_iou_xyxy(cur["box_n"], other["box_n"]) <= float(iou_thr)
            ]
        kept.extend(cls_keep)

    kept.sort(key=lambda x: float(x["score"]), reverse=True)
    return kept


def build_rows_from_box_arrays(
    xyxyn: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
    conf: float,
    per_class_conf: Optional[Dict[str, Any]],
    per_class_min_area: Optional[Dict[str, Any]],
    names: Dict[int, str],
) -> list[Dict[str, Any]]:
    rows = []
    for box_n, score, cls_id in zip(xyxyn, confs, clss):
        if not np.isfinite(box_n).all():
            continue

        cls_id = int(cls_id)
        thr_raw = cfg_lookup(per_class_conf, cls_id, names, conf)
        thr = safe_float(thr_raw, conf)
        if float(score) < thr:
            continue

        x1, y1, x2, y2 = [float(v) for v in box_n.tolist()]
        x1, x2 = sorted([clip01(x1), clip01(x2)])
        y1, y2 = sorted([clip01(y1), clip01(y2)])

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h

        min_area_raw = cfg_lookup(per_class_min_area, cls_id, names, 0.0)
        min_area = safe_float(min_area_raw, 0.0)
        if area < min_area:
            continue

        rows.append(
            {
                "cls_id": cls_id,
                "score": float(score),
                "box_n": np.array([x1, y1, x2, y2], dtype=np.float32),
            }
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def format_rect_polygon_line_from_norm_xyxy(
    cls_id: int,
    box_xyxyn: np.ndarray,
    decimals: int = 6,
) -> Optional[str]:
    if box_xyxyn is None or len(box_xyxyn) != 4:
        return None
    if not np.isfinite(box_xyxyn).all():
        return None

    x1, y1, x2, y2 = [float(v) for v in box_xyxyn.tolist()]
    x1, x2 = sorted([clip01(x1), clip01(x2)])
    y1, y2 = sorted([clip01(y1), clip01(y2)])

    if x2 <= x1 or y2 <= y1:
        return None

    fmt = f"{{:.{decimals}f}}"
    vals = [
        fmt.format(x1), fmt.format(y1),
        fmt.format(x2), fmt.format(y1),
        fmt.format(x2), fmt.format(y2),
        fmt.format(x1), fmt.format(y2),
    ]
    return f"{int(cls_id)} " + " ".join(vals)


class Track1RectExporter:
    def __init__(
        self,
        model: Union[str, Path],
        device: str = "",
        half: bool = False,
    ) -> None:
        self.model_path = str(model)
        self.model = YOLO(self.model_path)
        self.device = device
        self.half = half
        move_device = self.device
        if isinstance(move_device, str) and move_device.isdigit():
            move_device = f"cuda:{move_device}"
        if self.device:
            try:
                self.model.to(move_device)
            except Exception:
                pass
        self.core_model = self.model.model if hasattr(self.model, "model") else None
        self.names = normalize_names(getattr(self.model, "names", None))

    def _predict_single(
        self,
        source: Any,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        agnostic_nms: bool,
        verbose: bool,
    ):
        results = self.model.predict(
            source=source,
            save=False,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=self.device,
            batch=1,
            max_det=int(max_det),
            agnostic_nms=bool(agnostic_nms),
            half=bool(self.half),
            verbose=bool(verbose),
        )
        if isinstance(results, (list, tuple)):
            return results[0] if results else None
        try:
            return results[0]
        except Exception:
            return results

    def _resolved_device(self) -> torch.device:
        if isinstance(self.core_model, nn.Module):
            p = next(self.core_model.parameters(), None)
            if p is not None:
                return p.device
        return torch.device("cpu")

    def _resolved_stride(self) -> int:
        if isinstance(self.core_model, nn.Module):
            stride = getattr(self.core_model, "stride", None)
            if torch.is_tensor(stride) and stride.numel() > 0:
                return max(1, int(stride.max().item()))
        return 32

    def _snapshot_bn_state(self):
        if not isinstance(self.core_model, nn.Module):
            return {}
        snapshot = {}
        for name, module in self.core_model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            item = {
                "training": bool(module.training),
                "track_running_stats": bool(module.track_running_stats),
            }
            for attr in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, attr, None)
                if value is not None:
                    item[attr] = value.detach().clone()
            snapshot[name] = item
        return snapshot

    def _restore_bn_state(self, snapshot: Dict[str, Any]) -> None:
        if not isinstance(self.core_model, nn.Module) or not snapshot:
            return
        modules = dict(self.core_model.named_modules())
        for name, item in snapshot.items():
            module = modules.get(name)
            if module is None or not isinstance(module, nn.BatchNorm2d):
                continue
            module.track_running_stats = bool(item.get("track_running_stats", True))
            module.train(bool(item.get("training", False)))
            for attr in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                src = item.get(attr, None)
                dst = getattr(module, attr, None)
                if src is None or dst is None:
                    continue
                dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))

    def _tent_preprocess_image(self, image: Image.Image, imgsz: int) -> torch.Tensor:
        np_img = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        letterbox = LetterBox(new_shape=(int(imgsz), int(imgsz)), auto=False, stride=self._resolved_stride())
        np_img = letterbox(image=np_img)
        tensor = torch.from_numpy(np.ascontiguousarray(np_img.transpose((2, 0, 1)))).unsqueeze(0)
        tensor = tensor.to(self._resolved_device())
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        return tensor

    def _tent_entropy_loss(self, outputs: Any, tent_cfg: Dict[str, Any]) -> torch.Tensor:
        pred = outputs
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if not torch.is_tensor(pred) or pred.dim() != 3:
            ref = next(self.core_model.parameters())
            return torch.zeros((), device=ref.device, dtype=ref.dtype)

        head = self.core_model.model[-1] if isinstance(self.core_model, nn.Module) else None
        nc = int(getattr(head, "nc", 0)) if head is not None else 0
        if nc <= 0 or pred.shape[1] < 4 + nc:
            ref = pred if torch.is_tensor(pred) else next(self.core_model.parameters())
            return torch.zeros((), device=ref.device, dtype=ref.dtype)

        cls_prob = pred[:, 4 : 4 + nc, :].sigmoid().clamp(1e-6, 1.0 - 1e-6)
        cls_dist = cls_prob / cls_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        anchor_conf = cls_prob.max(dim=1).values

        topk_ratio = max(0.0, float(tent_cfg.get("topk_ratio", 0.10)))
        topk_min = max(1, int(tent_cfg.get("topk_min", 64)))
        topk = min(anchor_conf.shape[1], max(topk_min, int(round(anchor_conf.shape[1] * topk_ratio))))
        if topk <= 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        topk_scores, topk_idx = anchor_conf.topk(topk, dim=1)
        entropy = -(cls_dist * cls_dist.log()).sum(dim=1)
        entropy = torch.gather(entropy, 1, topk_idx)
        return (entropy * topk_scores.detach()).mean()

    def _normalize_tent_reset_mode(self, tent_cfg: Optional[Dict[str, Any]]) -> str:
        if not isinstance(tent_cfg, dict):
            return "image"
        mode = str(tent_cfg.get("reset_mode", "")).strip().lower()
        if mode in {"image", "scenario", "never"}:
            return mode
        if bool(tent_cfg.get("reset_each_image", False)):
            return "image"
        return "never"

    def _tent_parse_top_module_idx(self, module_name: str) -> Optional[int]:
        parts = str(module_name).split(".")
        if len(parts) < 2 or parts[0] != "model":
            return None
        try:
            return int(parts[1])
        except Exception:
            return None

    def _tent_backbone_stop_idx(self) -> Optional[int]:
        if not isinstance(self.core_model, nn.Module):
            return None
        top_modules = getattr(self.core_model, "model", None)
        if not isinstance(top_modules, (list, tuple, nn.ModuleList, nn.Sequential)):
            return None
        top_modules = list(top_modules)
        stop_idx = len(top_modules)
        for idx, module in enumerate(top_modules):
            cls_name = module.__class__.__name__.lower()
            if isinstance(module, nn.Upsample) or cls_name == "concat":
                stop_idx = idx
                break
        return stop_idx

    def _normalize_tent_name_tokens(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            return []
        out = []
        for item in value:
            token = str(item).strip()
            if token:
                out.append(token)
        return out

    def _select_tent_bn_modules(self, tent_cfg: Optional[Dict[str, Any]]) -> list[tuple[str, nn.BatchNorm2d]]:
        if not isinstance(self.core_model, nn.Module):
            return []

        named_bn = [(name, module) for name, module in self.core_model.named_modules() if isinstance(module, nn.BatchNorm2d)]
        if not named_bn:
            return []

        cfg = tent_cfg if isinstance(tent_cfg, dict) else {}
        scope = str(cfg.get("scope", "all")).strip().lower()
        if scope not in {"all", "backbone", "backbone_shallow"}:
            scope = "all"

        include_tokens = self._normalize_tent_name_tokens(cfg.get("name_include"))
        exclude_tokens = self._normalize_tent_name_tokens(cfg.get("name_exclude"))
        backbone_stop_idx = self._tent_backbone_stop_idx()
        shallow_stages = max(1, int(cfg.get("shallow_stages", 4)))

        filtered = []
        for name, module in named_bn:
            top_idx = self._tent_parse_top_module_idx(name)
            keep = True
            if scope == "backbone":
                keep = top_idx is not None and (backbone_stop_idx is None or top_idx < backbone_stop_idx)
            elif scope == "backbone_shallow":
                keep = top_idx is not None and top_idx < shallow_stages and (backbone_stop_idx is None or top_idx < backbone_stop_idx)

            if keep and include_tokens:
                keep = any(token in name for token in include_tokens)
            if keep and exclude_tokens:
                keep = not any(token in name for token in exclude_tokens)
            if keep:
                filtered.append((name, module))

        max_bn_layers = int(cfg.get("max_bn_layers", 0))
        if max_bn_layers > 0:
            filtered = filtered[:max_bn_layers]
        return filtered

    def _tent_context_key(
        self,
        image_path: Path,
        source_root: Optional[Union[str, Path]],
        reset_mode: str,
    ) -> str:
        if reset_mode == "image":
            return str(image_path.resolve())
        if reset_mode == "never":
            return "__global__"

        image_path = Path(str(image_path))
        if source_root is not None:
            try:
                rel = image_path.resolve().relative_to(Path(str(source_root)).resolve())
                parent = rel.parent.as_posix().strip()
                return parent if parent else "__root__"
            except Exception:
                pass
        parent = image_path.parent.name.strip()
        return parent if parent else "__root__"

    def _apply_tent(self, image: Image.Image, imgsz: int, tent_cfg: Optional[Dict[str, Any]], verbose: bool = False):
        if not isinstance(tent_cfg, dict) or (not tent_cfg.get("enabled")) or (not isinstance(self.core_model, nn.Module)):
            return None

        named_bn_modules = self._select_tent_bn_modules(tent_cfg)
        if not named_bn_modules:
            return None

        snapshot = self._snapshot_bn_state() if bool(tent_cfg.get("reset_each_image", False)) else None
        steps = max(1, int(tent_cfg.get("steps", 1)))
        mode = str(tent_cfg.get("mode", "lite")).strip().lower()
        lr = float(tent_cfg.get("lr", 1e-4))
        tensor = self._tent_preprocess_image(image, imgsz)

        self.core_model.eval()
        params = []
        for p in self.core_model.parameters():
            p.requires_grad_(False)
        for _, bn in named_bn_modules:
            bn.train()
            if mode == "grad" and bn.affine:
                bn.weight.requires_grad_(True)
                bn.bias.requires_grad_(True)
                params.extend([bn.weight, bn.bias])

        optimizer = torch.optim.Adam(params, lr=lr) if params else None

        for _ in range(steps):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    outputs = self.core_model(tensor)
                    loss = self._tent_entropy_loss(outputs, tent_cfg)
                    if torch.isfinite(loss):
                        loss.backward()
                        optimizer.step()
            else:
                with torch.no_grad():
                    _ = self.core_model(tensor)

        self.core_model.eval()
        for p in self.core_model.parameters():
            p.requires_grad_(False)
        self.model.predictor = None
        if verbose:
            scope = str(tent_cfg.get("scope", "all")).strip().lower()
            print(f"[TENT] mode={mode} steps={steps} lr={lr} scope={scope} bn={len(named_bn_modules)}")
        return snapshot

    def _rows_from_result_boxes(
        self,
        result: Any,
        conf: float,
        per_class_conf: Optional[Dict[str, Any]],
        per_class_min_area: Optional[Dict[str, Any]],
        flip_lr: bool = False,
    ) -> tuple[list[Dict[str, Any]], int]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) <= 0:
            return [], 0

        xyxyn = boxes.xyxyn.detach().cpu().numpy()
        if flip_lr:
            xyxyn = xyxyn.copy()
            old_x1 = xyxyn[:, 0].copy()
            old_x2 = xyxyn[:, 2].copy()
            xyxyn[:, 0] = 1.0 - old_x2
            xyxyn[:, 2] = 1.0 - old_x1

        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)
        rows = build_rows_from_box_arrays(
            xyxyn=xyxyn,
            confs=confs,
            clss=clss,
            conf=conf,
            per_class_conf=per_class_conf,
            per_class_min_area=per_class_min_area,
            names=self.names,
        )
        return rows, int(len(confs))

    def export(
        self,
        source: Union[str, Path],
        save_dir: Union[str, Path],
        source_root: Optional[Union[str, Path]] = None,
        imgsz: int = 512,
        conf: float = 0.25,
        iou: float = 0.50,
        batch: int = 16,
        max_det: int = 300,
        agnostic_nms: bool = False,
        per_class_conf: Optional[Dict[str, Any]] = None,
        per_class_min_area: Optional[Dict[str, Any]] = None,
        per_class_topk: Optional[Dict[str, Any]] = None,
        image_gate: Optional[Dict[str, Any]] = None,
        save_json: Optional[Union[str, Path]] = None,
        json_keep_rel_path: bool = False,
        json_decimals: int = 6,
        decimals: int = 6,
        tta: bool = False,
        tta_scales: Optional[list[float]] = None,
        tent_cfg: Optional[Dict[str, Any]] = None,
        clean_save_dir: bool = True,
        touch_empty: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        save_dir = Path(save_dir)
        source = str(source)

        if source_root is None:
            p = Path(source)
            if p.exists() and p.is_dir():
                source_root = str(p)

        if clean_save_dir and save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        raw_conf = conf
        per_class_conf_vals = numeric_values_from_cfg(per_class_conf)
        if per_class_conf_vals:
            raw_conf = min([conf] + per_class_conf_vals)
        raw_conf = max(1e-4, float(raw_conf))

        start_t = time.time()
        tta_scales = normalize_tta_scales(tta_scales if tta else [1.0])
        tent_enabled = bool(isinstance(tent_cfg, dict) and tent_cfg.get("enabled"))
        tent_reset_mode = self._normalize_tent_reset_mode(tent_cfg)
        base_tent_snapshot = self._snapshot_bn_state() if tent_enabled else None
        current_tent_context = None

        seen_txt_targets: Dict[Path, str] = {}
        num_images = 0
        num_images_with_boxes_before_filter = 0
        num_images_with_boxes_after_filter = 0
        total_boxes_before_filter = 0
        total_boxes_after_rule_filter = 0
        total_boxes_after_filter = 0
        num_images_suppressed_by_gate = 0
        json_results = []

        if tta or tent_enabled:
            image_items = []
            per_view_max_det = max(int(max_det), int(max_det) * 2)
            for image_path in resolve_source_image_paths(source):
                with Image.open(image_path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    img_w, img_h = img.size
                    if tent_enabled and base_tent_snapshot is not None:
                        context_key = self._tent_context_key(image_path, source_root=source_root, reset_mode=tent_reset_mode)
                        if current_tent_context is None or context_key != current_tent_context:
                            self._restore_bn_state(base_tent_snapshot)
                            self.model.predictor = None
                            current_tent_context = context_key
                    tent_snapshot = self._apply_tent(img, imgsz=imgsz, tent_cfg=tent_cfg, verbose=verbose)
                    view_rows = []
                    total_view_boxes = 0
                    flip_img = ImageOps.mirror(img)
                    scales = tta_scales if tta else [1.0]
                    flip_flags = [False, True] if tta else [False]
                    for scale in scales:
                        scaled_imgsz = max(32, int(round(float(imgsz) * float(scale) / 32.0)) * 32)
                        for flip_flag in flip_flags:
                            src = flip_img if flip_flag else img
                            result = self._predict_single(
                                source=src,
                                imgsz=scaled_imgsz,
                                conf=raw_conf,
                                iou=iou,
                                max_det=per_view_max_det,
                                agnostic_nms=agnostic_nms,
                                verbose=verbose,
                            )
                            rows_part, count_part = self._rows_from_result_boxes(
                                result=result,
                                conf=conf,
                                per_class_conf=per_class_conf,
                                per_class_min_area=per_class_min_area,
                                flip_lr=bool(flip_flag),
                            )
                            view_rows.extend(rows_part)
                            total_view_boxes += int(count_part)

                    if tent_snapshot is not None and tent_reset_mode == "image":
                        self._restore_bn_state(tent_snapshot)
                        self.model.predictor = None

                merged_rows = classwise_nms_rows(view_rows, iou_thr=float(iou))
                image_items.append(
                    {
                        "image_path": image_path,
                        "img_h": int(img_h),
                        "img_w": int(img_w),
                        "boxes_before_filter": int(total_view_boxes),
                        "rows_raw": merged_rows,
                    }
                )
        else:
            predict_kwargs = dict(
                source=source,
                stream=True,
                save=False,
                conf=raw_conf,
                iou=float(iou),
                imgsz=int(imgsz),
                device=self.device,
                batch=int(batch),
                max_det=int(max_det),
                agnostic_nms=bool(agnostic_nms),
                half=bool(self.half),
                verbose=bool(verbose),
            )
            results = self.model.predict(**predict_kwargs)
            image_items = []
            for idx, r in enumerate(results):
                image_path = getattr(r, "path", None)
                if image_path is None:
                    image_path = f"image_{idx:08d}.jpg"
                image_path = Path(str(image_path))
                orig_shape = getattr(r, "orig_shape", None)
                img_h = int(orig_shape[0]) if orig_shape is not None and len(orig_shape) >= 1 else None
                img_w = int(orig_shape[1]) if orig_shape is not None and len(orig_shape) >= 2 else None
                rows_raw, box_count = self._rows_from_result_boxes(
                    result=r,
                    conf=conf,
                    per_class_conf=per_class_conf,
                    per_class_min_area=per_class_min_area,
                    flip_lr=False,
                )
                image_items.append(
                    {
                        "image_path": image_path,
                        "img_h": img_h,
                        "img_w": img_w,
                        "boxes_before_filter": int(box_count),
                        "rows_raw": rows_raw,
                    }
                )

        for item in image_items:
            num_images += 1
            image_path = Path(str(item["image_path"]))

            txt_path = resolve_output_txt_path(
                image_path=image_path,
                save_dir=save_dir,
                source_root=source_root,
            )

            prev = seen_txt_targets.get(txt_path)
            if prev is not None and prev != str(image_path):
                raise RuntimeError(
                    f"检测到输出 txt 路径冲突：\n"
                    f"  txt_path = {txt_path}\n"
                    f"  image A  = {prev}\n"
                    f"  image B  = {image_path}\n"
                    f"请传入 --source_root 保留相对目录，避免同名文件覆盖。"
                )
            seen_txt_targets[txt_path] = str(image_path)

            txt_path.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            final_rows = []
            img_h = item.get("img_h")
            img_w = item.get("img_w")
            rows = list(item.get("rows_raw", []))
            box_count_before_filter = int(item.get("boxes_before_filter", 0))

            if box_count_before_filter > 0:
                num_images_with_boxes_before_filter += 1
            total_boxes_before_filter += box_count_before_filter

            if rows:
                rows.sort(key=lambda x: x["score"], reverse=True)

                if per_class_topk:
                    keep_rows = []
                    class_counter: Dict[int, int] = {}
                    for row in rows:
                        cls_id = int(row["cls_id"])
                        topk_raw = cfg_lookup(per_class_topk, cls_id, self.names, None)
                        if topk_raw is None:
                            keep_rows.append(row)
                            class_counter[cls_id] = class_counter.get(cls_id, 0) + 1
                            continue

                        topk = safe_int(topk_raw, 999999)
                        if topk <= 0:
                            continue

                        cur = class_counter.get(cls_id, 0)
                        if cur >= topk:
                            continue

                        keep_rows.append(row)
                        class_counter[cls_id] = cur + 1

                    rows = keep_rows

                if len(rows) > max_det:
                    rows = rows[:max_det]

                total_boxes_after_rule_filter += len(rows)
                gate_ok, _gate_stats = evaluate_image_gate(rows, image_gate, self.names)
                if (not gate_ok) and len(rows) > 0:
                    num_images_suppressed_by_gate += 1
                    rows = []
                final_rows = rows

                for row in rows:
                    line = format_rect_polygon_line_from_norm_xyxy(
                        cls_id=row["cls_id"],
                        box_xyxyn=row["box_n"],
                        decimals=decimals,
                    )
                    if line is not None:
                        lines.append(line)

                total_boxes_after_filter += len(lines)

                if len(lines) > 0:
                    num_images_with_boxes_after_filter += 1

            with open(txt_path, "w", encoding="utf-8") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")
                elif touch_empty:
                    # 创建空文件，表示该图无预测
                    pass

            if save_json is not None:
                defect_info = []
                for row in final_rows:
                    x1, y1, x2, y2 = [float(v) for v in row["box_n"].tolist()]
                    defect_info.append(
                        {
                            "category_id": int(row["cls_id"]),
                            "bbox": round_float_list([x1, y1, x2, y2], decimals=json_decimals),
                        }
                    )

                json_results.append(
                    {
                        "file_name": resolve_json_file_name(
                            image_path=image_path,
                            source_root=source_root,
                            keep_rel_path=json_keep_rel_path,
                        ),
                        "width": None if img_w is None else int(img_w),
                        "height": None if img_h is None else int(img_h),
                        "defect_info": defect_info,
                    }
                )

        elapsed = time.time() - start_t

        if tent_enabled and base_tent_snapshot is not None:
            self._restore_bn_state(base_tent_snapshot)
            self.model.predictor = None

        if save_json is not None:
            dump_json({"results": json_results}, Path(save_json))

        summary = {
            "model_path": self.model_path,
            "source": source,
            "source_root": None if source_root is None else str(source_root),
            "save_dir": str(save_dir),
            "imgsz": int(imgsz),
            "conf": float(conf),
            "iou": float(iou),
            "raw_conf_used_for_model_predict": float(raw_conf),
            "batch": int(batch),
            "max_det": int(max_det),
            "agnostic_nms": bool(agnostic_nms),
            "tta": bool(tta),
            "tta_scales": [float(x) for x in tta_scales],
            "tent_cfg": {} if not isinstance(tent_cfg, dict) else make_json_safe(tent_cfg),
            "num_images": int(num_images),
            "num_images_with_boxes_before_filter": int(num_images_with_boxes_before_filter),
            "num_images_with_boxes_after_filter": int(num_images_with_boxes_after_filter),
            "total_boxes_before_filter": int(total_boxes_before_filter),
            "total_boxes_after_rule_filter": int(total_boxes_after_rule_filter),
            "total_boxes_after_filter": int(total_boxes_after_filter),
            "image_gate_enabled": bool(image_gate),
            "num_images_suppressed_by_gate": int(num_images_suppressed_by_gate),
            "save_json": "" if save_json is None else str(save_json),
            "json_keep_rel_path": bool(json_keep_rel_path),
            "elapsed_seconds": float(elapsed),
        }

        dump_json(summary, save_dir / "_export_summary.json")
        return summary


def export_rect_predictions(
    model: Union[str, Path],
    source: Union[str, Path],
    save_dir: Union[str, Path],
    source_root: Optional[Union[str, Path]] = None,
    imgsz: int = 512,
    conf: float = 0.25,
    iou: float = 0.50,
    device: str = "",
    batch: int = 16,
    max_det: int = 300,
    agnostic_nms: bool = False,
    half: bool = False,
    per_class_conf: Optional[Dict[str, Any]] = None,
    per_class_min_area: Optional[Dict[str, Any]] = None,
    per_class_topk: Optional[Dict[str, Any]] = None,
    image_gate: Optional[Dict[str, Any]] = None,
    save_json: Optional[Union[str, Path]] = None,
    json_keep_rel_path: bool = False,
    json_decimals: int = 6,
    decimals: int = 6,
    tta: bool = False,
    tta_scales: Optional[list[float]] = None,
    tent_cfg: Optional[Dict[str, Any]] = None,
    clean_save_dir: bool = True,
    touch_empty: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    exporter = Track1RectExporter(model=model, device=device, half=half)
    return exporter.export(
        source=source,
        save_dir=save_dir,
        source_root=source_root,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        batch=batch,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        per_class_conf=per_class_conf,
        per_class_min_area=per_class_min_area,
        per_class_topk=per_class_topk,
        image_gate=image_gate,
        save_json=save_json,
        json_keep_rel_path=json_keep_rel_path,
        json_decimals=json_decimals,
        decimals=decimals,
        tta=tta,
        tta_scales=tta_scales,
        tent_cfg=tent_cfg,
        clean_save_dir=clean_save_dir,
        touch_empty=touch_empty,
        verbose=verbose,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Phase1 Track1 Rect Export")
    parser.add_argument("--model", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--source", type=str, required=True, help="图片目录 / txt 列表 / 任何 ultralytics 支持的 source")
    parser.add_argument("--save_dir", type=str, required=True, help="输出预测 txt 目录")
    parser.add_argument("--source_root", type=str, default=None, help="用于保留相对目录结构，强烈推荐传入")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--agnostic_nms", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--decimals", type=int, default=6)

    parser.add_argument("--per_class_conf_json", type=str, default=None, help='如 {"0":0.25,"1":0.30}')
    parser.add_argument("--per_class_min_area_json", type=str, default=None, help='如 {"0":0.00005}')
    parser.add_argument("--per_class_topk_json", type=str, default=None, help='如 {"0":3,"1":5}')
    parser.add_argument("--image_gate_json", type=str, default=None, help='如 {"enabled":true,"min_top1_score":0.35}')

    parser.add_argument("--no_clean_save_dir", action="store_true", help="不清空已有 save_dir")
    parser.add_argument("--no_touch_empty", action="store_true", help="无预测时不创建空 txt")
    parser.add_argument("--save_json", type=str, default=None, help="可选：导出比赛提交 JSON")
    parser.add_argument("--json_keep_rel_path", action="store_true", help="JSON 中 file_name 保留 source_root 下相对路径")
    parser.add_argument("--json_decimals", type=int, default=6, help="JSON bbox 小数位")
    parser.add_argument("--tta", action="store_true", help="Use manual horizontal-flip TTA")
    parser.add_argument("--tta_scales", type=str, default="1.0,0.83,1.17", help="Comma-separated TTA scales")
    parser.add_argument("--tent_json", type=str, default=None, help="Optional TENT config JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    per_class_conf = load_json_dict(args.per_class_conf_json)
    per_class_min_area = load_json_dict(args.per_class_min_area_json)
    per_class_topk = load_json_dict(args.per_class_topk_json)
    image_gate = load_json_dict(args.image_gate_json)
    tent_cfg = load_json_dict(args.tent_json)
    tta_scales = [float(x.strip()) for x in str(args.tta_scales).split(",") if x.strip()]

    summary = export_rect_predictions(
        model=args.model,
        source=args.source,
        save_dir=args.save_dir,
        source_root=args.source_root,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        half=args.half,
        per_class_conf=per_class_conf,
        per_class_min_area=per_class_min_area,
        per_class_topk=per_class_topk,
        image_gate=image_gate,
        save_json=args.save_json,
        json_keep_rel_path=args.json_keep_rel_path,
        json_decimals=args.json_decimals,
        decimals=args.decimals,
        tta=args.tta,
        tta_scales=tta_scales,
        tent_cfg=tent_cfg,
        clean_save_dir=not args.no_clean_save_dir,
        touch_empty=not args.no_touch_empty,
        verbose=args.verbose,
    )

    print(json.dumps(make_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
