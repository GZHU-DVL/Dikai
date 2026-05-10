#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from caculate_metric import CaculateMetric


def write_names_txt_from_data_yaml(data_yaml: Path) -> Path:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names", None)
    if names is None:
        raise KeyError(f"'names' not found in data yaml: {data_yaml}")

    if isinstance(names, dict):
        ordered = [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, (list, tuple)):
        ordered = [str(x) for x in names]
    else:
        raise TypeError(f"Unsupported names format in data yaml: {type(names).__name__}")

    fd, tmp_path = tempfile.mkstemp(prefix="ida_classes_", suffix=".txt")
    path = Path(tmp_path)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(ordered) + "\n")
    return path


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Track1 metric wrapper")
    parser.add_argument("--data_yaml", type=str, required=True)
    parser.add_argument("--gt_img_dir", type=str, required=True)
    parser.add_argument("--gt_txt_dir", type=str, required=True)
    parser.add_argument("--pred_txt_dir", type=str, required=True)
    parser.add_argument("--score_json", type=str, default=None)
    parser.add_argument("--track", type=int, default=1, choices=[1, 2])
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    data_yaml = Path(args.data_yaml)
    gt_img_dir = Path(args.gt_img_dir)
    gt_txt_dir = Path(args.gt_txt_dir)
    pred_txt_dir = Path(args.pred_txt_dir)

    if not data_yaml.is_file():
        raise FileNotFoundError(f"data yaml not found: {data_yaml}")
    if not gt_img_dir.is_dir():
        raise NotADirectoryError(f"gt_img_dir not found: {gt_img_dir}")
    if not gt_txt_dir.is_dir():
        raise NotADirectoryError(f"gt_txt_dir not found: {gt_txt_dir}")
    if not pred_txt_dir.is_dir():
        raise NotADirectoryError(f"pred_txt_dir not found: {pred_txt_dir}")

    class_txt = write_names_txt_from_data_yaml(data_yaml)
    try:
        cm = CaculateMetric()
        score_dict: Dict[str, Any] = cm.process_data(
            gt_img_dir=str(gt_img_dir),
            gt_txt_dir=str(gt_txt_dir),
            pred_img_dir=str(gt_img_dir),
            pred_txt_dir=str(pred_txt_dir),
            class_txt_dir=str(class_txt),
            txt_shuffix=".txt",
            S=int(args.track),
        )
    finally:
        try:
            class_txt.unlink(missing_ok=True)
        except Exception:
            pass

    payload = make_json_safe(score_dict)
    if args.score_json:
        score_json = Path(args.score_json)
        score_json.parent.mkdir(parents=True, exist_ok=True)
        score_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
