#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
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


def build_default_name(best: Dict[str, Any]) -> str:
    weights = Path(str(best.get("weights", "weights.pt"))).stem
    preset = str(best.get("postprocess_preset", "base"))
    conf = str(best.get("conf", "na")).replace(".", "p")
    iou = str(best.get("iou", "na")).replace(".", "p")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"track1_submit_{weights}_{preset}_c{conf}_i{iou}_{ts}"


def validate_submission_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    results = payload.get("results", None)
    if not isinstance(results, list):
        raise ValueError("submission JSON must contain a list field named 'results'")

    num_images = 0
    num_boxes = 0
    bad_items = []

    for idx, item in enumerate(results):
        num_images += 1
        if not isinstance(item, dict):
            bad_items.append(f"results[{idx}] is not a dict")
            continue

        for key in ("file_name", "width", "height", "defect_info"):
            if key not in item:
                bad_items.append(f"results[{idx}] missing key: {key}")

        defect_info = item.get("defect_info", [])
        if not isinstance(defect_info, list):
            bad_items.append(f"results[{idx}].defect_info is not a list")
            continue

        for j, det in enumerate(defect_info):
            num_boxes += 1
            if not isinstance(det, dict):
                bad_items.append(f"results[{idx}].defect_info[{j}] is not a dict")
                continue
            if "category_id" not in det:
                bad_items.append(f"results[{idx}].defect_info[{j}] missing category_id")
            if "bbox" not in det:
                bad_items.append(f"results[{idx}].defect_info[{j}] missing bbox")
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                bad_items.append(f"results[{idx}].defect_info[{j}].bbox must be a list of 4 numbers")
                continue
            try:
                x1, y1, x2, y2 = [float(x) for x in bbox]
            except Exception:
                bad_items.append(f"results[{idx}].defect_info[{j}].bbox contains non-numeric values")
                continue
            if not (0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0):
                bad_items.append(
                    f"results[{idx}].defect_info[{j}].bbox is outside normalized xyxy range: {bbox}"
                )

    return {
        "ok": len(bad_items) == 0,
        "num_images": int(num_images),
        "num_boxes": int(num_boxes),
        "issues": bad_items,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a final Track1 submission package from official_track1_best.json."
    )
    parser.add_argument("--best_json", type=str, required=True, help="Path to official_track1_best.json")
    parser.add_argument("--source", type=str, required=True, help="Test image directory or source")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for the final package")
    parser.add_argument("--name", type=str, default=None, help="Optional package folder name")
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
    parser.add_argument("--no_debug_txt", action="store_true", help="Delete intermediate txt outputs after JSON is created")
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

    out_dir = Path(args.out_dir)
    package_name = args.name or build_default_name(best)
    package_dir = out_dir / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    submission_json = package_dir / "submission.json"
    debug_txt_dir = package_dir / "pred_txt"
    meta_json = package_dir / "submission_meta.json"
    best_snapshot_json = package_dir / "official_track1_best.snapshot.json"
    validation_json = package_dir / "submission_validation.json"
    command_txt = package_dir / "COMMAND.txt"

    summary = export_rect_predictions(
        model=weights,
        source=args.source,
        save_dir=debug_txt_dir,
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
        save_json=submission_json,
        json_keep_rel_path=args.json_keep_rel_path,
        json_decimals=args.json_decimals,
        clean_save_dir=True,
        touch_empty=True,
        verbose=args.verbose,
    )

    with open(submission_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    validation = validate_submission_payload(payload)

    with open(best_snapshot_json, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    with open(validation_json, "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)

    meta = {
        "ok": True,
        "best_json": str(Path(args.best_json)),
        "weights": str(weights),
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "imgsz": imgsz,
        "postprocess_preset": best.get("postprocess_preset", None),
        "postprocess_hash": best.get("postprocess_hash", None),
        "source": str(args.source),
        "source_root": None if args.source_root is None else str(args.source_root),
        "submission_json": str(submission_json),
        "debug_txt_dir": str(debug_txt_dir),
        "validation_json": str(validation_json),
        "summary": summary,
        "validation": validation,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    command_txt.write_text(
        "\n".join(
            [
                "Track1 final submission package",
                f"best_json={Path(args.best_json)}",
                f"weights={weights}",
                f"conf={conf}",
                f"iou={iou}",
                f"max_det={max_det}",
                f"imgsz={imgsz}",
                f"postprocess_preset={best.get('postprocess_preset', None)}",
                f"submission_json={submission_json}",
                f"debug_txt_dir={debug_txt_dir}",
            ]
        ),
        encoding="utf-8",
    )

    if args.no_debug_txt:
        for p in sorted(debug_txt_dir.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        if debug_txt_dir.exists():
            debug_txt_dir.rmdir()

    print(
        json.dumps(
            {
                "ok": True,
                "package_dir": str(package_dir),
                "submission_json": str(submission_json),
                "meta_json": str(meta_json),
                "validation_json": str(validation_json),
                "validation_ok": validation["ok"],
                "num_images": validation["num_images"],
                "num_boxes": validation["num_boxes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
