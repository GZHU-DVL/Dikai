#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from phase1_track1_rect_export import (
    clip01,
    cfg_lookup,
    evaluate_image_gate,
    load_json_dict,
    make_json_safe,
    normalize_names,
    numeric_values_from_cfg,
    safe_float,
    safe_int,
)

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from clean images using current detector predictions."
    )
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="Directory or source containing clean images only")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max_det", type=int, default=50)
    parser.add_argument("--per_class_conf_json", type=str, default=None)
    parser.add_argument("--per_class_min_area_json", type=str, default=None)
    parser.add_argument("--per_class_topk_json", type=str, default=None)
    parser.add_argument("--image_gate_json", type=str, default=None)
    parser.add_argument("--min_score_to_save", type=float, default=0.0)
    parser.add_argument("--max_crops_per_image", type=int, default=20)
    parser.add_argument("--crop_pad_ratio", type=float, default=0.05)
    parser.add_argument("--copy_whole_image", action="store_true")
    parser.add_argument("--no_save_crops", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def ensure_dir(path: Path, clean: bool = False) -> Path:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def save_crop(image_path: Path, crop_path: Path, box_n: np.ndarray, crop_pad_ratio: float) -> bool:
    if Image is None:
        return False
    try:
        with Image.open(image_path) as im:
            w, h = im.size
            x1, y1, x2, y2 = [float(v) for v in box_n.tolist()]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            pad_x = bw * float(crop_pad_ratio)
            pad_y = bh * float(crop_pad_ratio)

            left = int(round(clip01(x1 - pad_x) * w))
            top = int(round(clip01(y1 - pad_y) * h))
            right = int(round(clip01(x2 + pad_x) * w))
            bottom = int(round(clip01(y2 + pad_y) * h))
            right = max(right, left + 1)
            bottom = max(bottom, top + 1)

            crop = im.crop((left, top, right, bottom))
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
            return True
    except Exception:
        return False


def mine_hard_negatives(
    weights: str | Path,
    source: str | Path,
    out_dir: str | Path,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    conf: float = 0.10,
    iou: float = 0.45,
    max_det: int = 50,
    per_class_conf: Optional[Dict[str, Any]] = None,
    per_class_min_area: Optional[Dict[str, Any]] = None,
    per_class_topk: Optional[Dict[str, Any]] = None,
    image_gate: Optional[Dict[str, Any]] = None,
    min_score_to_save: float = 0.0,
    max_crops_per_image: int = 20,
    crop_pad_ratio: float = 0.05,
    copy_whole_image: bool = False,
    no_save_crops: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    out_dir = ensure_dir(Path(out_dir), clean=True)
    crop_dir = ensure_dir(out_dir / "crops")
    image_dir = ensure_dir(out_dir / "images")

    model = YOLO(str(weights))
    names = normalize_names(getattr(model, "names", None))

    raw_conf = conf
    per_class_conf_vals = numeric_values_from_cfg(per_class_conf)
    if per_class_conf_vals:
        raw_conf = min([conf] + per_class_conf_vals)
    raw_conf = max(1e-4, float(raw_conf))

    results = model.predict(
        source=str(source),
        stream=True,
        save=False,
        conf=raw_conf,
        iou=float(iou),
        imgsz=int(imgsz),
        device=device if device else None,
        batch=int(batch),
        max_det=int(max_det),
        verbose=bool(verbose),
    )

    manifest_rows: List[Dict[str, Any]] = []
    image_summary_rows: List[Dict[str, Any]] = []

    for idx, r in enumerate(results):
        image_path = Path(str(getattr(r, "path", f"image_{idx:08d}.jpg")))
        boxes = getattr(r, "boxes", None)
        orig_shape = getattr(r, "orig_shape", None)
        img_h = int(orig_shape[0]) if orig_shape is not None and len(orig_shape) >= 1 else None
        img_w = int(orig_shape[1]) if orig_shape is not None and len(orig_shape) >= 2 else None

        rows: List[Dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            xyxyn = to_numpy(getattr(boxes, "xyxyn", None))
            confs = to_numpy(getattr(boxes, "conf", None))
            clss = to_numpy(getattr(boxes, "cls", None))
            if xyxyn is None:
                xyxyn = np.zeros((0, 4), dtype=np.float32)
            if confs is None:
                confs = np.zeros((len(xyxyn),), dtype=np.float32)
            if clss is None:
                clss = np.zeros((len(xyxyn),), dtype=np.int32)

            for box_n, score, cls_id in zip(xyxyn, confs, clss.astype(int)):
                if not np.isfinite(box_n).all():
                    continue
                thr_raw = cfg_lookup(per_class_conf, int(cls_id), names, conf)
                thr = safe_float(thr_raw, conf)
                if float(score) < thr:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box_n.tolist()]
                x1, x2 = sorted([clip01(x1), clip01(x2)])
                y1, y2 = sorted([clip01(y1), clip01(y2)])
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

                min_area_raw = cfg_lookup(per_class_min_area, int(cls_id), names, 0.0)
                min_area = safe_float(min_area_raw, 0.0)
                if area < min_area:
                    continue

                rows.append(
                    {
                        "cls_id": int(cls_id),
                        "score": float(score),
                        "box_n": np.array([x1, y1, x2, y2], dtype=np.float32),
                        "area": float(area),
                    }
                )

            rows.sort(key=lambda x: x["score"], reverse=True)

            if per_class_topk:
                keep_rows = []
                class_counter: Dict[int, int] = {}
                for row in rows:
                    cls_id = int(row["cls_id"])
                    topk_raw = cfg_lookup(per_class_topk, cls_id, names, None)
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

            if len(rows) > int(max_det):
                rows = rows[: int(max_det)]

            gate_ok, gate_stats = evaluate_image_gate(rows, image_gate, names)
            if (not gate_ok) and len(rows) > 0:
                rows = []
        else:
            gate_stats = {"gate_enabled": bool(image_gate), "gate_passed": True, "gate_reason": "no_boxes"}

        if rows and copy_whole_image and image_path.exists():
            shutil.copy2(image_path, image_dir / image_path.name)

        saved_crops = 0
        for det_idx, row in enumerate(rows, start=1):
            if float(row["score"]) < float(min_score_to_save):
                continue

            x1, y1, x2, y2 = [float(v) for v in row["box_n"].tolist()]
            crop_path = crop_dir / image_path.stem / f"{det_idx:02d}_cls{row['cls_id']}_s{row['score']:.4f}.jpg"
            crop_saved = False
            if (not no_save_crops) and saved_crops < int(max_crops_per_image) and image_path.exists():
                crop_saved = save_crop(
                    image_path=image_path,
                    crop_path=crop_path,
                    box_n=row["box_n"],
                    crop_pad_ratio=float(crop_pad_ratio),
                )
                if crop_saved:
                    saved_crops += 1

            manifest_rows.append(
                {
                    "image_path": str(image_path),
                    "file_name": image_path.name,
                    "cls_id": int(row["cls_id"]),
                    "cls_name": names.get(int(row["cls_id"]), str(row["cls_id"])),
                    "score": float(row["score"]),
                    "area": float(row["area"]),
                    "bbox_norm_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "width": img_w,
                    "height": img_h,
                    "crop_path": "" if not crop_saved else str(crop_path),
                }
            )

        image_summary_rows.append(
            {
                "image_path": str(image_path),
                "file_name": image_path.name,
                "num_hard_negatives": int(len(rows)),
                "top1_score": 0.0 if not rows else float(rows[0]["score"]),
                "gate_enabled": bool(gate_stats.get("gate_enabled", False)),
                "gate_passed": bool(gate_stats.get("gate_passed", True)),
                "gate_reason": str(gate_stats.get("gate_reason", "")),
            }
        )

    manifest_json = out_dir / "hard_negative_manifest.json"
    image_summary_json = out_dir / "hard_negative_image_summary.json"
    manifest_csv = out_dir / "hard_negative_manifest.csv"
    image_summary_csv = out_dir / "hard_negative_image_summary.csv"

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump({"items": make_json_safe(manifest_rows)}, f, ensure_ascii=False, indent=2)
    with open(image_summary_json, "w", encoding="utf-8") as f:
        json.dump({"images": make_json_safe(image_summary_rows)}, f, ensure_ascii=False, indent=2)

    if manifest_rows:
        with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    if image_summary_rows:
        with open(image_summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(image_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_summary_rows)

    return {
        "ok": True,
        "weights": str(weights),
        "source": str(source),
        "out_dir": str(out_dir),
        "num_images": len(image_summary_rows),
        "num_items": len(manifest_rows),
        "manifest_json": str(manifest_json),
        "image_summary_json": str(image_summary_json),
        "manifest_csv": str(manifest_csv),
        "image_summary_csv": str(image_summary_csv),
        "crops_saved": 0 if no_save_crops else sum(1 for row in manifest_rows if row.get("crop_path")),
    }


def main() -> None:
    args = build_argparser().parse_args()
    per_class_conf = load_json_dict(args.per_class_conf_json)
    per_class_min_area = load_json_dict(args.per_class_min_area_json)
    per_class_topk = load_json_dict(args.per_class_topk_json)
    image_gate = load_json_dict(args.image_gate_json)
    output = mine_hard_negatives(
        weights=args.weights,
        source=args.source,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        per_class_conf=per_class_conf,
        per_class_min_area=per_class_min_area,
        per_class_topk=per_class_topk,
        image_gate=image_gate,
        min_score_to_save=args.min_score_to_save,
        max_crops_per_image=args.max_crops_per_image,
        crop_pad_ratio=args.crop_pad_ratio,
        copy_whole_image=args.copy_whole_image,
        no_save_crops=args.no_save_crops,
        verbose=args.verbose,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
