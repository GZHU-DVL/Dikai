#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "Ultralytics"))

from ultralytics.models.yolo.segment import SegmentationTrainer


def ensure_project_in_pythonpath(project_root: Path):
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    old_pp = os.environ.get("PYTHONPATH", "")
    parts = [p for p in old_pp.split(os.pathsep) if p]
    if root not in parts:
        os.environ["PYTHONPATH"] = root if not old_pp else root + os.pathsep + old_pp


ensure_project_in_pythonpath(PROJECT_ROOT)

from custom_modules.hsfpn_dcn import (
    EFE,
    MixStyle,
    IBNa,
    SPDAdapter,
    SimAM,
    assert_plugins_in_optimizer,
    get_plugin_callbacks,
    prepare_model_plugins_before_train,
    register_plugins,
    restore_forward_patches,
    strip_forward_patches,
)
from custom_modules.ida_loss import configure_model_loss

PLUGIN_ENV_KEY = "IDA_PLUGIN_CFG_JSON"
LOSS_ENV_KEY = "IDA_LOSS_CFG_JSON"
USE_TRACK1_PROXY_ENV_KEY = "IDA_USE_TRACK1_PROXY"


def patch_skip_ultralytics_final_eval():
    try:
        from ultralytics.engine.trainer import BaseTrainer

        if getattr(BaseTrainer, "_ida_skip_final_eval_patched", False):
            return

        def _skip_final_eval(self):
            print("[Patch] 跳过 Ultralytics 默认 final_eval(best.pt)，训练后将按官方脚本做候选筛选。")

        BaseTrainer.final_eval = _skip_final_eval
        BaseTrainer._ida_skip_final_eval_patched = True
        print("[Patch] BaseTrainer.final_eval 已禁用")
    except Exception as e:
        print(f"[Patch] 禁用 final_eval 失败: {e}")


def patch_force_fp32_val():
    raw = os.environ.get("IDA_FORCE_FP32_VAL", "0").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        print("[Patch] in-training validation keeps Ultralytics AMP mode (set IDA_FORCE_FP32_VAL=1 to force FP32)")
        return

    try:
        from ultralytics.engine.validator import BaseValidator

        if getattr(BaseValidator, "_ida_force_fp32_val_patched", False):
            return

        old_call = BaseValidator.__call__

        def _call(self, trainer=None, model=None):
            if trainer is not None:
                old_amp = getattr(trainer, "amp", False)
                trainer.amp = False
                try:
                    return old_call(self, trainer=trainer, model=model)
                finally:
                    trainer.amp = old_amp
            return old_call(self, trainer=trainer, model=model)

        BaseValidator.__call__ = _call
        BaseValidator._ida_force_fp32_val_patched = True
        print("[Patch] 训练内验证已强制使用 FP32")
    except Exception as e:
        print(f"[Patch] 强制 FP32 验证失败: {e}")


patch_skip_ultralytics_final_eval()
patch_force_fp32_val()


def read_float(row: dict, keys, default=0.0):
    for k in keys:
        v = row.get(k, None)
        if v not in (None, ""):
            try:
                fv = float(v)
                if math.isfinite(fv):
                    return fv
            except Exception:
                pass
    return default


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

    box_p = read_float(row, ["metrics/precision(B)", "metrics/precision"], default=0.0)
    box_r = read_float(row, ["metrics/recall(B)", "metrics/recall"], default=0.0)
    box_m50 = read_float(row, ["metrics/mAP50(B)", "metrics/mAP50"], default=0.0)
    box_map = read_float(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"], default=0.0)

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
    screen_proxy = 0.55 * rec + 0.45 * prec
    score = 0.30 * loc_proxy + 0.30 * cls_proxy + 0.40 * screen_proxy

    if not math.isfinite(score):
        return -1e9
    return float(score)


def stock_ultralytics_fitness_from_row(row: dict) -> float:
    box_m50 = read_float(row, ["metrics/mAP50(B)", "metrics/mAP50"], default=0.0)
    box_map = read_float(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"], default=0.0)
    mask_m50 = read_float(row, ["metrics/mAP50(M)"], default=box_m50)
    mask_map = read_float(row, ["metrics/mAP50-95(M)"], default=box_map)
    score = 0.1 * mask_m50 + 0.9 * mask_map
    if not math.isfinite(score):
        return -1e9
    return float(score)


def use_track1_proxy_fitness() -> bool:
    raw = os.environ.get(USE_TRACK1_PROXY_ENV_KEY, "1").strip().lower()
    return raw not in {"0", "false", "no", "off", "ultra", "default"}


def ensure_plugins_registered_from_env(verbose=True):
    s = os.environ.get(PLUGIN_ENV_KEY, "").strip()
    if not s:
        return
    try:
        cfg = json.loads(s)
        register_plugins(**cfg, verbose=verbose)
    except Exception as e:
        print(f"[Plugins][WARN] 从环境变量恢复插件配置失败: {e}")


def ensure_loss_registered_from_env(model=None, verbose=True):
    s = os.environ.get(LOSS_ENV_KEY, "").strip()
    if not s or model is None:
        return
    try:
        cfg = json.loads(s)
        allowed = {"use_vfl", "vfl_alpha", "vfl_gamma", "nwd_weight", "nwd_constant"}
        cfg = {k: v for k, v in cfg.items() if k in allowed}
        configure_model_loss(model, verbose=verbose, **cfg)
    except Exception as e:
        print(f"[Loss][WARN] failed to restore loss cfg from env: {e}")


class PluginSegTrainer(SegmentationTrainer):
    def __init__(self, *args, **kwargs):
        ensure_plugins_registered_from_env(verbose=True)
        super().__init__(*args, **kwargs)
        self._register_plugin_callbacks_to_trainer()

    def _register_plugin_callbacks_to_trainer(self):
        if not hasattr(self, "callbacks") or not isinstance(self.callbacks, dict):
            return
        for cb_name, cb_fn in get_plugin_callbacks().items():
            cur = self.callbacks.get(cb_name, [])
            cur = [x for x in cur if x is not cb_fn]
            cur.append(cb_fn)
            self.callbacks[cb_name] = cur

    def get_model(self, cfg=None, weights=None, verbose=True):
        ensure_plugins_registered_from_env(verbose=True)
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        if not getattr(model, "_plugins_prepared", False):
            prepare_model_plugins_before_train(model, verbose=True)
            model._plugins_prepared = True

        ensure_loss_registered_from_env(model, verbose=True)

        root = model.model if hasattr(model, "model") else model
        efe_count = sum(1 for m in root.modules() if isinstance(m, EFE))
        mixstyle_count = sum(1 for m in root.modules() if isinstance(m, MixStyle))
        ibn_count = sum(1 for m in root.modules() if isinstance(m, IBNa))
        spd_count = sum(1 for m in root.modules() if isinstance(m, SPDAdapter))
        simam_count = sum(1 for m in root.modules() if isinstance(m, SimAM))
        wrapped_heads = int(getattr(root, "_efe_heads_wrapped", 0))

        print(
            "[PluginTrainer.get_model] "
            f"IBNa={ibn_count} EFE={efe_count} MixStyle={mixstyle_count} "
            f"SPD={spd_count} SimAM={simam_count} "
            f"EFE_heads_wrapped={wrapped_heads}"
        )
        return model

    def build_optimizer(self, model, *args, **kwargs):
        opt = super().build_optimizer(model, *args, **kwargs)
        try:
            assert_plugins_in_optimizer(self.model, opt, strict=True)
        except Exception as e:
            print(f"[Plugins][ERROR] 插件参数未完整进入优化器: {e}")
            raise
        return opt

    def validate(self):
        metrics = self.validator(self)
        if metrics is None:
            metrics = {}
        elif not isinstance(metrics, dict):
            try:
                metrics = dict(metrics)
            except Exception:
                metrics = {}

        ultra_fit = read_float(metrics, ["metrics/UltraFitness", "fitness"], default=-1.0)
        if ultra_fit < 0:
            ultra_fit = stock_ultralytics_fitness_from_row(metrics)
        proxy_fit = track1_proxy_score_from_row(metrics)
        if not math.isfinite(proxy_fit):
            proxy_fit = -1e9

        metrics["metrics/UltraFitness"] = float(ultra_fit)
        metrics["metrics/Track1Proxy"] = float(proxy_fit)
        selected_fit = proxy_fit if use_track1_proxy_fitness() else ultra_fit
        metrics["fitness"] = float(selected_fit)

        if self.best_fitness is None or self.best_fitness < selected_fit:
            self.best_fitness = float(selected_fit)

        return metrics, float(selected_fit)

    def save_model(self):
        strip_forward_patches(self.model)

        ema_model = None
        if getattr(self, "ema", None) is not None:
            ema_model = getattr(self.ema, "ema", None)
            if ema_model is not None:
                strip_forward_patches(ema_model)

        try:
            super().save_model()
        finally:
            restore_forward_patches(self.model)
            if ema_model is not None:
                restore_forward_patches(ema_model)


__all__ = [
    "PLUGIN_ENV_KEY",
    "LOSS_ENV_KEY",
    "USE_TRACK1_PROXY_ENV_KEY",
    "ensure_plugins_registered_from_env",
    "ensure_loss_registered_from_env",
    "PluginSegTrainer",
]
