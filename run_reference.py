"""Run the long-schedule reference recipe reported in the paper.

This wraps ``Train.run_dsu_nwd_vfl_upperbound`` with explicit CLI arguments
so the paper reference can be reproduced without stepping through the
interactive menu in Train.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import Train as T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-yaml", default=None, help="dataset YAML path; equivalent to IDA_DATA_YAML")
    ap.add_argument("--result-root", default=None, help="where run folders are written")
    ap.add_argument("--scale", default=T.DEFAULT_SCALE, choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--device", default=T.DEFAULT_DEVICE, help="CUDA device index, e.g. 0 or 0,1")
    ap.add_argument("--batch", type=int, default=None, help="optional batch override")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=300)
    ap.add_argument("--close-mosaic", type=int, default=120)
    ap.add_argument("--save-period", type=int, default=20)
    ap.add_argument("--project", default="ida_track1_dsu_loss_upperbound")
    args = ap.parse_args()

    if args.data_yaml:
        T.DATA_PATH = str(Path(args.data_yaml).expanduser())
        T.BASE_CFG["data"] = T.DATA_PATH
    if args.result_root:
        T.RESULT_ROOT = Path(args.result_root).expanduser()
        T.RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    T.DEFAULT_DEVICE = str(args.device)
    T.BASE_CFG["device"] = str(args.device)

    res = T.run_dsu_nwd_vfl_upperbound(
        scale_key=str(args.scale),
        budget_mode="full",
        batch_override=args.batch,
        project=str(args.project),
        train_overrides={
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "close_mosaic": int(args.close_mosaic),
            "save_period": int(args.save_period),
            "device": str(args.device),
        },
    )
    if res is None:
        raise SystemExit(1)

    print("\n[reference] finished")
    print(f"save_dir={res.save_dir}")
    print(f"official_score={getattr(res, 'official_score', None)}")
    print(f"official_conf={getattr(res, 'official_conf', None)}")
    print(f"official_iou={getattr(res, 'official_iou', None)}")
    print(f"official_max_det={getattr(res, 'official_max_det', None)}")


if __name__ == "__main__":
    main()
