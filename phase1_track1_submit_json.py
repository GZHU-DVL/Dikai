#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from phase1_track1_rect_export import export_rect_predictions


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be dict: {path}")
    return data


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load official_track1_best.json and export competition submission JSON."
    )
    parser.add_argument("--best_json", type=str, required=True, help="Path to official_track1_best.json")
    parser.add_argument("--source", type=str, required=True, help="Test image directory or source")
    parser.add_argument("--save_json", type=str, required=True, help="Output submission JSON path")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional debug txt output directory")
    parser.add_argument("--source_root", type=str, default=None, help="Optional root for relative JSON paths")
    parser.add_argument("--json_keep_rel_path", action="store_true", help="Keep relative path in JSON file_name")
    parser.add_argument("--device", type=str, default="", help="Optional device override")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=None, help="Optional imgsz override")
    parser.add_argument("--weights", type=str, default=None, help="Optional weights override")
    parser.add_argument("--conf", type=float, default=None, help="Optional conf override")
    parser.add_argument("--iou", type=float, default=None, help="Optional iou override")
    parser.add_argument("--max_det", type=int, default=None, help="Optional max_det override")
    parser.add_argument("--json_decimals", type=int, default=6)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    best = load_json(args.best_json)

    weights = args.weights or best.get("weights", None)
    conf = float(args.conf if args.conf is not None else best.get("conf", 0.1))
    iou = float(args.iou if args.iou is not None else best.get("iou", 0.45))
    max_det = int(args.max_det if args.max_det is not None else best.get("max_det", 50))
    imgsz = int(args.imgsz if args.imgsz is not None else best.get("imgsz", 512))
    postprocess_cfg: Optional[Dict[str, Any]] = best.get("postprocess_cfg", None)

    if not weights:
        raise ValueError("weights not found in best_json, and --weights was not provided")

    save_dir = args.save_dir or str(Path(args.save_json).with_suffix(""))

    summary = export_rect_predictions(
        model=weights,
        source=args.source,
        save_dir=save_dir,
        source_root=args.source_root,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=args.device,
        batch=args.batch,
        max_det=max_det,
        per_class_conf=None if not isinstance(postprocess_cfg, dict) else postprocess_cfg.get("per_class_conf"),
        per_class_min_area=None if not isinstance(postprocess_cfg, dict) else postprocess_cfg.get("per_class_min_area"),
        per_class_topk=None if not isinstance(postprocess_cfg, dict) else postprocess_cfg.get("per_class_topk"),
        image_gate=None if not isinstance(postprocess_cfg, dict) else postprocess_cfg.get("image_gate"),
        save_json=args.save_json,
        json_keep_rel_path=args.json_keep_rel_path,
        json_decimals=args.json_decimals,
        clean_save_dir=True,
        touch_empty=True,
        verbose=args.verbose,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "best_json": str(Path(args.best_json)),
                "weights": str(weights),
                "conf": conf,
                "iou": iou,
                "max_det": max_det,
                "imgsz": imgsz,
                "save_json": str(Path(args.save_json)),
                "save_dir": str(Path(save_dir)),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
