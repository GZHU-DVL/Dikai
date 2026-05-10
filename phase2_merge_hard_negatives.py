#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def dump_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def normalize_train_entries(train_value) -> List[str]:
    if train_value is None:
        return []
    if isinstance(train_value, str):
        return [train_value]
    if isinstance(train_value, (list, tuple)):
        return [str(x) for x in train_value]
    raise ValueError("data.yaml train must be a string or list")


def iter_image_files(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "link":
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def merge_hard_negatives(
    data_yaml: str | Path,
    hn_image_dir: str | Path,
    output_data_yaml: str | Path,
    output_dataset_root: str | Path,
    mode: str = "copy",
    clear_output_root: bool = False,
) -> Dict[str, Any]:
    data_yaml_obj = load_yaml(data_yaml)
    hn_image_dir = Path(hn_image_dir)
    output_root = Path(output_dataset_root)
    if not hn_image_dir.exists():
        raise FileNotFoundError(f"hard-negative image dir not found: {hn_image_dir}")

    if clear_output_root and output_root.exists():
        shutil.rmtree(output_root)

    bundle_images = output_root / "images"
    bundle_labels = output_root / "labels"
    bundle_images.mkdir(parents=True, exist_ok=True)
    bundle_labels.mkdir(parents=True, exist_ok=True)

    copied = []
    for src_img in iter_image_files(hn_image_dir):
        dst_img = bundle_images / src_img.name
        dst_lbl = bundle_labels / f"{src_img.stem}.txt"
        copy_or_link(src_img, dst_img, mode=mode)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.touch(exist_ok=True)
        copied.append(
            {
                "src_image": str(src_img),
                "dst_image": str(dst_img),
                "dst_label": str(dst_lbl),
            }
        )

    train_entries = normalize_train_entries(data_yaml_obj.get("train"))
    bundle_train_entry = str(bundle_images)
    if bundle_train_entry not in train_entries:
        train_entries.append(bundle_train_entry)
    data_yaml_obj["train"] = train_entries

    dump_yaml(data_yaml_obj, output_data_yaml)

    manifest = {
        "ok": True,
        "num_images": len(copied),
        "hn_image_dir": str(hn_image_dir),
        "output_dataset_root": str(output_root),
        "bundle_images": str(bundle_images),
        "bundle_labels": str(bundle_labels),
        "output_data_yaml": str(Path(output_data_yaml)),
        "items": copied,
    }
    manifest_path = output_root / "merge_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge hard-negative images back into the training dataset with empty labels."
    )
    parser.add_argument("--data_yaml", type=str, required=True, help="Original data.yaml")
    parser.add_argument("--hn_image_dir", type=str, required=True, help="Directory of hard-negative images")
    parser.add_argument("--output_data_yaml", type=str, required=True, help="Output merged data.yaml")
    parser.add_argument("--output_dataset_root", type=str, required=True, help="Where to build the hard-negative bundle")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "link"], help="Copy or hard-link images")
    parser.add_argument("--clear_output_root", action="store_true", help="Remove existing output_dataset_root first")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest = merge_hard_negatives(
        data_yaml=args.data_yaml,
        hn_image_dir=args.hn_image_dir,
        output_data_yaml=args.output_data_yaml,
        output_dataset_root=args.output_dataset_root,
        mode=args.mode,
        clear_output_root=args.clear_output_root,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
