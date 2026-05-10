# -*- coding: utf-8 -*-
"""
custom_modules/hsfpn_dcn.py
最终稳定修复版（A类优化版）

保留并扩展四类组件：
1) IBN-a：推理期也生效（结构级改动），默认 stem + 很浅层 stage
2) EFE：训练期边缘/上下文正则，eval/infer 旁路
3) MixStyle：训练期统计量扰动，eval/infer 旁路
4) Progressive RandConv：训练期浅层随机卷积扰动，eval/infer 旁路

本轮 A 类改动：
- 允许 MixStyle 与 RandConv 同时开启
- 组合模式下若参数仍为默认值，则自动降低两者强度
- 调整注入顺序为：IBN -> RandConv -> MixStyle
- 调整 restore 顺序，确保组合模式下真实执行顺序正确
- 增加 DDP / 反序列化兼容处理
- 增加 DCN 旧接口兼容占位，避免外部残留 import 报错
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules.head import Detect as UltralyticsDetect
    from ultralytics.nn.modules.head import Segment as UltralyticsSegment
except Exception:
    UltralyticsDetect = nn.Module
    UltralyticsSegment = nn.Module

try:
    from ultralytics.nn.modules.block import C2f as UltralyticsC2f
except Exception:
    UltralyticsC2f = None

try:
    from ultralytics.nn.modules.block import C3k2 as UltralyticsC3k2
except Exception:
    UltralyticsC3k2 = None

try:
    from ultralytics.nn.modules.block import C2PSA as UltralyticsC2PSA
except Exception:
    UltralyticsC2PSA = None


_DEFAULT_MIXSTYLE_PROB = 0.30
_DEFAULT_MIXSTYLE_ALPHA = 0.30
_DEFAULT_MIXSTYLE_LAYERS = 1
_DEFAULT_MIXSTYLE_MODE = "mixstyle"

_DEFAULT_RANDCONV_PROB = 0.20
_DEFAULT_RANDCONV_PROB_END = 0.45
_DEFAULT_RANDCONV_SIGMA = 0.00
_DEFAULT_RANDCONV_SIGMA_END = 0.15
_DEFAULT_RANDCONV_LAYERS = 1

_COMBO_MIXSTYLE_PROB = 0.20
_COMBO_MIXSTYLE_ALPHA = 0.20
_COMBO_RANDCONV_PROB = 0.12
_COMBO_RANDCONV_PROB_END = 0.30
_COMBO_RANDCONV_SIGMA = 0.00
_COMBO_RANDCONV_SIGMA_END = 0.10

_ENHANCE_ENABLED = False
_ENHANCE_EDGE_KS = 5
_ENHANCE_RATES = (1, 2, 3)

_MIXSTYLE_ENABLED = False
_MIXSTYLE_PROB = _DEFAULT_MIXSTYLE_PROB
_MIXSTYLE_ALPHA = _DEFAULT_MIXSTYLE_ALPHA
_MIXSTYLE_LAYERS = _DEFAULT_MIXSTYLE_LAYERS
_MIXSTYLE_MODE = _DEFAULT_MIXSTYLE_MODE

_RANDCONV_ENABLED = False
_RANDCONV_PROB = _DEFAULT_RANDCONV_PROB
_RANDCONV_PROB_END = _DEFAULT_RANDCONV_PROB_END
_RANDCONV_SIGMA = _DEFAULT_RANDCONV_SIGMA
_RANDCONV_SIGMA_END = _DEFAULT_RANDCONV_SIGMA_END
_RANDCONV_LAYERS = _DEFAULT_RANDCONV_LAYERS
_RANDCONV_KERNEL_SIZES = (3, 5)
_RANDCONV_REFRESH_INTERVAL = 1
_PROTECT_SHALLOW_TEXTURES = False

_IBN_ENABLED = False
_IBN_RATIO = 0.5
_IBN_LAYERS = 1

_SPD_ENABLED = False
_SPD_LAYERS = 2
_SPD_SCALE = 2
_SPD_ALPHA_INIT = 0.10

_SIMAM_ENABLED = False
_SIMAM_LAYERS = 2
_SIMAM_E_LAMBDA = 1e-4

_ENABLE_SAFE_CONCAT = False
_CONCAT_PATCHED = False


def _resolve_patch_root(model_or_module):
    root = model_or_module
    if hasattr(root, "module") and isinstance(root.module, nn.Module):
        root = root.module
    if hasattr(root, "model") and isinstance(root.model, nn.Module):
        root = root.model
    return root if isinstance(root, nn.Module) else None


def _safe_autocast_off_ctx(x: torch.Tensor):
    if not x.is_cuda:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=False)
    except Exception:
        return torch.cuda.amp.autocast(enabled=False)


def _module_floating_device_dtype(module: nn.Module, fallback: torch.Tensor):
    for p in module.parameters(recurse=True):
        if p is not None and torch.is_floating_point(p):
            return p.device, p.dtype
    for b in module.buffers(recurse=True):
        if b is not None and torch.is_floating_point(b):
            return b.device, b.dtype
    return fallback.device, fallback.dtype


def _align_model_floating_tensors(module: nn.Module, verbose: bool = True):
    ref = next((p for p in module.parameters() if p is not None and torch.is_floating_point(p)), None)
    if ref is None:
        return

    target_device = ref.device
    target_dtype = ref.dtype
    fixed_params = 0
    fixed_buffers = 0

    for m in module.modules():
        for name, p in list(m._parameters.items()):
            if p is not None and torch.is_floating_point(p):
                if p.device != target_device or p.dtype != target_dtype:
                    m._parameters[name] = nn.Parameter(
                        p.detach().to(device=target_device, dtype=target_dtype),
                        requires_grad=p.requires_grad,
                    )
                    fixed_params += 1

        for name, b in list(m._buffers.items()):
            if b is not None and torch.is_floating_point(b):
                if b.device != target_device or b.dtype != target_dtype:
                    m._buffers[name] = b.to(device=target_device, dtype=target_dtype)
                    fixed_buffers += 1

    if verbose and (fixed_params > 0 or fixed_buffers > 0):
        print(
            f"[DTypeAlign] fixed_params={fixed_params} "
            f"fixed_buffers={fixed_buffers} dtype={target_dtype} device={target_device}"
        )


def _iter_top_modules(net: nn.Module) -> List[nn.Module]:
    if hasattr(net, "model") and isinstance(net.model, (list, nn.Sequential, nn.ModuleList)):
        try:
            return list(net.model)
        except Exception:
            pass
    if isinstance(net, (nn.Sequential, nn.ModuleList)):
        return list(net)
    return list(net.children())


def _is_bound_to_impl(maybe_method, func) -> bool:
    return hasattr(maybe_method, "__func__") and maybe_method.__func__ is func


def _apply_mod_to_output(y, mod: nn.Module):
    if torch.is_tensor(y):
        return mod(y)

    if isinstance(y, tuple):
        y_list = list(y)
        for i, v in enumerate(y_list):
            if torch.is_tensor(v):
                y_list[i] = mod(v)
                break
        return tuple(y_list)

    if isinstance(y, list):
        out = list(y)
        for i, v in enumerate(out):
            if torch.is_tensor(v):
                out[i] = mod(v)
                break
        return out

    return y


def _apply_io_mod_to_output(y, x, mod: nn.Module):
    if torch.is_tensor(y):
        return mod(x, y)

    if isinstance(y, tuple):
        y_list = list(y)
        for i, v in enumerate(y_list):
            if torch.is_tensor(v):
                y_list[i] = mod(x, v)
                break
        return tuple(y_list)

    if isinstance(y, list):
        out = list(y)
        for i, v in enumerate(out):
            if torch.is_tensor(v):
                out[i] = mod(x, v)
                break
        return out

    return y


def _guess_backbone_stop_idx(top: List[nn.Module]) -> int:
    for idx, m in enumerate(top):
        cls_name = m.__class__.__name__.lower()
        if isinstance(m, nn.Upsample) or cls_name == "concat":
            return idx
    return len(top)


def _iter_backbone_stage_candidates(top: List[nn.Module], target_types: List[type]):
    stop_idx = _guess_backbone_stop_idx(top)
    candidates = []
    for idx, m in enumerate(top[1:stop_idx], start=1):
        if any(isinstance(m, t) for t in target_types):
            candidates.append((idx, m))
    return candidates


def _iter_stride_backbone_candidates(top: List[nn.Module]):
    stop_idx = _guess_backbone_stop_idx(top)
    candidates = []
    for idx, m in enumerate(top[:stop_idx]):
        stride = getattr(getattr(m, "conv", None), "stride", None)
        if stride is None:
            continue
        stride_h = int(stride[0]) if isinstance(stride, (list, tuple)) else int(stride)
        stride_w = int(stride[1]) if isinstance(stride, (list, tuple)) and len(stride) > 1 else stride_h
        if stride_h < 2 or stride_w < 2:
            continue
        candidates.append((idx, m))
    return candidates


def _select_stage_candidates(candidates, layers: int, prefer_deep: bool = False):
    count = max(0, int(layers))
    if count <= 0:
        return []

    picked = list(candidates)
    return picked[-count:] if prefer_deep else picked[:count]


def _materialize_dynamic_modules(model_or_yolo, imgsz: int = 256, verbose: bool = True) -> bool:
    outer = model_or_yolo
    if (
        isinstance(outer, nn.Module)
        and hasattr(outer, "predictor")
        and hasattr(outer, "overrides")
        and hasattr(outer, "model")
        and isinstance(outer.model, nn.Module)
    ):
        outer = outer.model
    if hasattr(outer, "module") and isinstance(outer.module, nn.Module):
        outer = outer.module
    if not isinstance(outer, nn.Module) and hasattr(outer, "model") and isinstance(outer.model, nn.Module):
        outer = outer.model
    if not isinstance(outer, nn.Module):
        return False

    p = next((pp for pp in outer.parameters() if pp is not None), None)
    device = p.device if p is not None else torch.device("cpu")
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)

    was_training = outer.training
    try:
        outer.eval()
        with torch.no_grad():
            with _safe_autocast_off_ctx(dummy):
                _ = outer(dummy)
        if verbose:
            print(f"[Materialize] dummy forward 完成，imgsz={imgsz}")
        return True
    except Exception as e:
        if verbose:
            print(f"[Materialize] dummy forward 失败: {e}")
        return False
    finally:
        outer.train(was_training)


def _float_eq(a: float, b: float, eps: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= eps


def _normalize_mixstyle_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode in {"mixstyle", "efdmix", "dsu"}:
        return mode
    return _DEFAULT_MIXSTYLE_MODE


def _safe_concat_forward(self, x):
    try:
        return torch.cat(x, dim=self.d)
    except RuntimeError as e:
        if "Sizes of tensors must match" not in str(e):
            raise

        hs = [t.shape[-2] for t in x]
        ws = [t.shape[-1] for t in x]
        target_h, target_w = max(hs), max(ws)

        aligned = []
        for t in x:
            h, w = t.shape[-2:]
            if h < target_h or w < target_w:
                pt = (target_h - h) // 2
                pb = target_h - h - pt
                pl = (target_w - w) // 2
                pr = target_w - w - pl
                t = F.pad(t, (pl, pr, pt, pb))

            h2, w2 = t.shape[-2:]
            if h2 > target_h or w2 > target_w:
                sh = max((h2 - target_h) // 2, 0)
                sw = max((w2 - target_w) // 2, 0)
                t = t[..., sh:sh + target_h, sw:sw + target_w]

            aligned.append(t)

        return torch.cat(aligned, dim=self.d)


def _patch_ultralytics_concat(verbose=True):
    global _CONCAT_PATCHED
    if _CONCAT_PATCHED or (not _ENABLE_SAFE_CONCAT):
        return

    try:
        import ultralytics.nn.modules.conv as u_conv
        if hasattr(u_conv, "Concat"):
            u_conv.Concat.forward = _safe_concat_forward
            _CONCAT_PATCHED = True
            if verbose:
                print("[Plugins] SafeConcatPatch=ON（仅建议 debug 使用）")
    except Exception as e:
        if verbose:
            print(f"[Plugins] SafeConcatPatch 失败: {e}")


class IBNa(nn.Module):
    def __init__(
        self,
        c: int,
        ratio: float = 0.5,
        eps: float = 1e-5,
        affine_bn: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        ratio = float(ratio)
        c1 = int(round(c * ratio))
        c1 = max(1, min(c - 1, c1))
        c2 = c - c1

        self.c = int(c)
        self.c1 = int(c1)
        self.c2 = int(c2)

        self.inorm = nn.InstanceNorm2d(self.c1, affine=False, eps=1e-5)
        self.bnorm = nn.BatchNorm2d(
            self.c2,
            eps=eps,
            affine=affine_bn,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.c1, self.c2], dim=1)
        x1 = self.inorm(x1)
        x2 = self.bnorm(x2)
        return torch.cat([x1, x2], dim=1)


class IBNApplier(nn.Module):
    def __init__(self, ratio: float = 0.5, eps: float = 1e-5):
        super().__init__()
        self.ratio = float(ratio)
        self.eps = float(eps)
        self.inner: Optional[IBNa] = None

    def _build_if_needed(self, x: torch.Tensor):
        c = int(x.shape[1])
        if self.inner is None or getattr(self.inner, "c", -1) != c:
            self.inner = IBNa(c, ratio=self.ratio, eps=self.eps).to(device=x.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(x)) or x.dim() != 4:
            return x
        self._build_if_needed(x)
        device, dtype = _module_floating_device_dtype(self.inner, x)
        return self.inner(x.to(device=device, dtype=dtype)).to(device=x.device, dtype=x.dtype)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, bias=False, act=True):
        super().__init__()
        if p is None:
            p = ((k - 1) // 2) * d
        self.conv = nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _make_log_kernel(ks: int, sigma: float):
    r = ks // 2
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1, dtype=torch.float32),
        torch.arange(-r, r + 1, dtype=torch.float32),
        indexing="ij",
    )
    dist2 = xs ** 2 + ys ** 2
    sigma2 = sigma ** 2
    kernel = ((dist2 - 2.0 * sigma2) / (sigma2 ** 2 + 1e-8)) * torch.exp(-dist2 / (2.0 * sigma2 + 1e-8))
    kernel = kernel - kernel.mean()
    kernel = kernel / (kernel.abs().sum() + 1e-8)
    return kernel.view(1, 1, ks, ks)


class EdgeAwareDefectAttention(nn.Module):
    def __init__(self, c: int, k: int = 5, sigma: float = 1.0):
        super().__init__()
        assert k in (3, 5, 7)

        self.inorm = nn.InstanceNorm2d(c, affine=False, eps=1e-5)
        self.edge = nn.Conv2d(c, c, kernel_size=k, padding=k // 2, groups=c, bias=False)

        log_kernel = _make_log_kernel(k, sigma)
        with torch.no_grad():
            w = torch.zeros_like(self.edge.weight)
            w[:, :, :, :] = log_kernel
            self.edge.weight.copy_(w)
        self.edge.weight.requires_grad = True

        self.mix = nn.Sequential(
            ConvBNAct(c, c, k=1, act=True),
            nn.Conv2d(c, c, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_norm = self.inorm(x)
        edge = self.edge(x_norm)
        gate = self.sigmoid(self.mix(edge))
        return x * (1.0 + gate)


class HRSDP(nn.Module):
    def __init__(self, c: int, rates=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList([ConvBNAct(c, c, k=3, d=r, p=r) for r in rates])
        self.fuse = ConvBNAct(c * len(rates), c, k=1)

    def forward(self, x):
        ys = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(ys, dim=1))


class EFE(nn.Module):
    def __init__(
        self,
        c: int,
        rates=(1, 2, 3),
        edge_ks: int = 5,
        spatial_dropout_p: float = 0.10,
        alpha_init: float = 0.05,
    ):
        super().__init__()
        self.sdrop = nn.Dropout2d(p=spatial_dropout_p) if spatial_dropout_p > 0 else nn.Identity()
        self.eada = EdgeAwareDefectAttention(c, k=edge_ks)
        self.ctx = HRSDP(c, rates=rates)
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_device = x.device
        device, dtype = _module_floating_device_dtype(self, x)
        with _safe_autocast_off_ctx(x):
            x_work = x.to(device=device, dtype=dtype)
            feat = x_work if not self.training else self.sdrop(x_work)
            y = self.ctx(self.eada(feat))
            if not torch.isfinite(y).all():
                y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
                y = torch.clamp(y, -1e4, 1e4)
            alpha = self.alpha.to(device=device, dtype=dtype).clamp(0.0, 0.3)
            out = x_work + alpha * y

        return out.to(device=x_device, dtype=x_dtype)


class SimAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = float(e_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(x)) or x.dim() != 4:
            return x

        x_dtype = x.dtype
        with _safe_autocast_off_ctx(x):
            x_f32 = x.float()
            _, _, h, w = x_f32.shape
            n = max(h * w - 1, 1)
            x_minus_mu_square = (x_f32 - x_f32.mean(dim=(2, 3), keepdim=True)).pow(2)
            denom = 4.0 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / float(n) + self.e_lambda)
            y = x_minus_mu_square / denom + 0.5
            out = x_f32 * torch.sigmoid(y)
            if not torch.isfinite(out).all():
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = torch.clamp(out, -1e4, 1e4)
        return out.to(dtype=x_dtype)


class MixStyle(nn.Module):
    def __init__(self, p=0.3, alpha=0.3, eps=1e-6, mode="mixstyle"):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.mode = _normalize_mixstyle_mode(mode)

    def _efdmix(self, x: torch.Tensor, perm: torch.Tensor, lmda: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, -1)
        x2_flat = x_flat[perm]

        x_sorted, x_rank = torch.sort(x_flat, dim=2)
        x2_sorted, _ = torch.sort(x2_flat, dim=2)

        lmda = lmda.reshape(b, 1, 1)
        mixed_sorted = lmda * x_sorted + (1.0 - lmda) * x2_sorted

        out = torch.empty_like(x_flat)
        out.scatter_(2, x_rank, mixed_sorted)
        return out.reshape(b, c, h, w)

    def _dsu(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(2, 3), keepdim=True)
        sig = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        x_normed = (x - mu) / sig

        mu_std = mu.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)
        sig_std = sig.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)

        factor = max(float(self.alpha), 1e-4)
        mu_noise = torch.randn_like(mu) * mu_std * factor
        sig_noise = torch.randn_like(sig) * sig_std * factor
        sig_perturbed = (sig + sig_noise).abs().clamp_min(self.eps)
        mu_perturbed = mu + mu_noise
        return x_normed * sig_perturbed + mu_perturbed

    def forward(self, x):
        if (not self.training) or x.dim() != 4:
            return x

        b = x.size(0)
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        if self.mode == "dsu":
            x_dtype = x.dtype
            with _safe_autocast_off_ctx(x):
                out = self._dsu(x.float())
                if not torch.isfinite(out).all():
                    out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                    out = torch.clamp(out, -1e4, 1e4)
            return out.to(dtype=x_dtype)

        if b < 2:
            return x

        perm = torch.randperm(b, device=x.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha)

        if self.mode == "efdmix":
            x_dtype = x.dtype
            with _safe_autocast_off_ctx(x):
                out = self._efdmix(
                    x.float(),
                    perm=perm,
                    lmda=beta.sample((b,)).to(device=x.device, dtype=torch.float32),
                )
                if not torch.isfinite(out).all():
                    out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                    out = torch.clamp(out, -1e4, 1e4)
            return out.to(dtype=x_dtype)

        mu = x.mean(dim=(2, 3), keepdim=True)
        sig = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        x_normed = (x - mu) / sig

        mu2 = mu[perm]
        sig2 = sig[perm]

        lmda = beta.sample((b, 1, 1, 1)).to(device=x.device, dtype=x.dtype)

        mu_mix = lmda * mu + (1.0 - lmda) * mu2
        sig_mix = lmda * sig + (1.0 - lmda) * sig2
        return x_normed * sig_mix + mu_mix


class ProgressiveRandConv(nn.Module):
    def __init__(
        self,
        p: float = 0.2,
        p_end: float = 0.45,
        sigma: float = 0.0,
        sigma_end: float = 0.15,
        kernel_sizes=(3, 5),
        refresh_interval: int = 1,
    ):
        super().__init__()

        ks = []
        for k in kernel_sizes:
            try:
                kk = int(k)
            except Exception:
                continue
            if kk >= 1 and kk % 2 == 1:
                ks.append(kk)

        self.kernel_sizes = tuple(ks or [3, 5])
        self.base_prob = float(p)
        self.final_prob = float(p_end)
        self.base_sigma = float(sigma)
        self.final_sigma = float(sigma_end)
        self.refresh_interval = max(1, int(refresh_interval))

        self.current_prob = float(self.base_prob)
        self.current_sigma = float(self.base_sigma)
        self.current_kernel_size = int(self.kernel_sizes[0])

        self._cached_weight: Optional[torch.Tensor] = None
        self._cached_meta = None
        self._since_refresh = 0

    def set_schedule(self, progress: float, tail_decay: float = 1.0):
        progress = float(max(0.0, min(1.0, progress)))
        tail_decay = float(max(0.0, min(1.0, tail_decay)))

        base_prob = self.base_prob + (self.final_prob - self.base_prob) * progress
        base_sigma = self.base_sigma + (self.final_sigma - self.base_sigma) * progress

        self.current_prob = base_prob * tail_decay
        self.current_sigma = base_sigma * tail_decay

        idx = min(len(self.kernel_sizes) - 1, int(progress * len(self.kernel_sizes)))
        new_kernel = int(self.kernel_sizes[idx])
        if new_kernel != self.current_kernel_size:
            self.current_kernel_size = new_kernel
            self._cached_weight = None
            self._cached_meta = None
            self._since_refresh = 0

    def schedule_signature(self) -> Tuple[float, float, int]:
        return (
            round(float(self.current_prob), 4),
            round(float(self.current_sigma), 4),
            int(self.current_kernel_size),
        )

    def _need_refresh(self, x: torch.Tensor) -> bool:
        meta = (
            int(x.shape[1]),
            int(self.current_kernel_size),
            x.device.type,
            -1 if x.device.index is None else int(x.device.index),
        )
        if self._cached_weight is None or self._cached_meta != meta:
            return True
        return self._since_refresh >= self.refresh_interval

    def _resample(self, x: torch.Tensor):
        c = int(x.shape[1])
        k = int(self.current_kernel_size)

        weight = torch.randn(c, 1, k, k, device=x.device, dtype=torch.float32)
        weight = weight - weight.mean(dim=(2, 3), keepdim=True)
        weight = weight / weight.abs().sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        self._cached_weight = weight
        self._cached_meta = (
            c,
            k,
            x.device.type,
            -1 if x.device.index is None else int(x.device.index),
        )
        self._since_refresh = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (not torch.is_tensor(x)) or x.dim() != 4:
            return x

        prob = float(self.current_prob)
        sigma = float(self.current_sigma)

        if prob <= 0.0 or sigma <= 0.0:
            return x

        if torch.rand(1, device=x.device).item() > prob:
            return x

        if self._need_refresh(x):
            self._resample(x)
        self._since_refresh += 1

        k = int(self.current_kernel_size)
        with _safe_autocast_off_ctx(x):
            x_f32 = x.float()
            y = F.conv2d(
                x_f32,
                self._cached_weight,
                stride=1,
                padding=k // 2,
                groups=int(x.shape[1]),
            )
            out = x_f32 + sigma * (y - x_f32)
            if not torch.isfinite(out).all():
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = torch.clamp(out, -1e4, 1e4)

        return out.to(dtype=x.dtype)


class SPDAdapter(nn.Module):
    def __init__(self, c1: int, c2: int, scale: int = 2, alpha_init: float = 0.10):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.scale = max(2, int(scale))
        self.proj = ConvBNAct(self.c1 * (self.scale ** 2), self.c2, k=1, s=1, act=True)
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(x)) or x.dim() != 4:
            return x

        b, c, h, w = x.shape
        s = self.scale
        if c != self.c1 or h < s or w < s:
            return x

        h_trim = h - (h % s)
        w_trim = w - (w % s)
        if h_trim < s or w_trim < s:
            return x

        x_dtype = x.dtype
        x_device = x.device
        device, dtype = _module_floating_device_dtype(self.proj, x)
        with _safe_autocast_off_ctx(x):
            x_work = x.to(device=device, dtype=dtype)
            if h_trim != h or w_trim != w:
                x_work = x_work[:, :, :h_trim, :w_trim]
            x_spd = x_work.view(b, c, h_trim // s, s, w_trim // s, s)
            x_spd = x_spd.permute(0, 1, 3, 5, 2, 4).contiguous()
            x_spd = x_spd.view(b, c * (s ** 2), h_trim // s, w_trim // s)
            y = self.proj(x_spd)
            alpha = self.alpha.to(device=device, dtype=dtype).clamp(0.0, 0.5)
            out = alpha * y
            if not torch.isfinite(out).all():
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = torch.clamp(out, -1e4, 1e4)
        return out.to(device=x_device, dtype=x_dtype)


class SPDApplier(nn.Module):
    def __init__(self, scale: int = 2, alpha_init: float = 0.10):
        super().__init__()
        self.scale = max(2, int(scale))
        self.alpha_init = float(alpha_init)
        self.inner: Optional[SPDAdapter] = None

    def _build_if_needed(self, x: torch.Tensor, y: torch.Tensor):
        c1 = int(x.shape[1])
        c2 = int(y.shape[1])
        if (
            self.inner is None
            or getattr(self.inner, "c1", -1) != c1
            or getattr(self.inner, "c2", -1) != c2
            or getattr(self.inner, "scale", -1) != self.scale
        ):
            self.inner = SPDAdapter(c1, c2, scale=self.scale, alpha_init=self.alpha_init).to(
                device=x.device,
                dtype=torch.float32,
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(x)) or (not torch.is_tensor(y)) or x.dim() != 4 or y.dim() != 4:
            return y

        self._build_if_needed(x, y)
        if self.inner is None:
            return y

        add = self.inner(x)
        if (not torch.is_tensor(add)) or add.dim() != 4:
            return y

        if add.shape[-2:] != y.shape[-2:]:
            add = F.interpolate(add.float(), size=y.shape[-2:], mode="bilinear", align_corners=False).to(dtype=y.dtype)
        return y + add.to(dtype=y.dtype)


def _ibna_forward_impl(self, x, *args, **kwargs):
    y = self._ibna_orig_forward(x, *args, **kwargs)
    applier = getattr(self, "_ibna", None)
    if applier is None:
        return y
    return _apply_mod_to_output(y, applier)


def _efe_head_forward_impl(self, x, *args, **kwargs):
    enhancers = getattr(self, "enhancers", None)
    feats = list(x) if isinstance(x, (list, tuple)) else [x]

    if not isinstance(enhancers, nn.ModuleList) or len(enhancers) == 0:
        channel_list = []
        valid = True
        for f in feats:
            if not (torch.is_tensor(f) and f.dim() == 4):
                valid = False
                break
            channel_list.append(int(f.shape[1]))

        if valid and channel_list:
            new_enh = nn.ModuleList(
                [
                    EFE(
                        c,
                        rates=_ENHANCE_RATES,
                        edge_ks=_ENHANCE_EDGE_KS,
                        alpha_init=0.05,
                    )
                    for c in channel_list
                ]
            )
            device = feats[0].device
            self.enhancers = new_enh.to(device=device)
            enhancers = self.enhancers

    if isinstance(enhancers, nn.ModuleList) and len(enhancers) == len(feats):
        new_feats = []
        for i, f in enumerate(feats):
            if torch.is_tensor(f) and f.dim() == 4:
                new_feats.append(enhancers[i](f))
            else:
                new_feats.append(f)
        x = tuple(new_feats) if isinstance(x, tuple) else new_feats

    return self._efe_orig_forward(x, *args, **kwargs)


def _mixstyle_forward_impl(self, x, *args, **kwargs):
    y = self._mixstyle_orig_forward(x, *args, **kwargs)
    ms_mod = getattr(self, "_mixstyle", None)
    if ms_mod is None:
        return y
    return _apply_mod_to_output(y, ms_mod)


def _randconv_forward_impl(self, x, *args, **kwargs):
    y = self._randconv_orig_forward(x, *args, **kwargs)
    rc_mod = getattr(self, "_randconv", None)
    if rc_mod is None:
        return y
    return _apply_mod_to_output(y, rc_mod)


def _spd_forward_impl(self, x, *args, **kwargs):
    y = self._spd_orig_forward(x, *args, **kwargs)
    spd_mod = getattr(self, "_spd", None)
    if spd_mod is None:
        return y
    return _apply_io_mod_to_output(y, x, spd_mod)


def _simam_forward_impl(self, x, *args, **kwargs):
    y = self._simam_orig_forward(x, *args, **kwargs)
    simam_mod = getattr(self, "_simam", None)
    if simam_mod is None:
        return y
    return _apply_mod_to_output(y, simam_mod)


def _attach_ibna_inplace(module: nn.Module, ratio=0.5):
    if getattr(module, "_ibna_attached", False):
        if getattr(module, "_ibna", None) is None:
            module.add_module("_ibna", IBNApplier(ratio=ratio))
        if getattr(module, "_ibna_orig_forward", None) is None:
            current = getattr(module, "forward", None)
            if _is_bound_to_impl(current, _ibna_forward_impl):
                module._ibna_orig_forward = module.__class__.forward.__get__(module, module.__class__)
            else:
                module._ibna_orig_forward = current
        if not _is_bound_to_impl(getattr(module, "forward", None), _ibna_forward_impl):
            module.forward = _ibna_forward_impl.__get__(module, module.__class__)
        return True

    module.add_module("_ibna", IBNApplier(ratio=ratio))
    module._ibna_orig_forward = module.forward
    module.forward = _ibna_forward_impl.__get__(module, module.__class__)
    module._ibna_attached = True
    return True


def _inject_ibna_to_backbone(root: nn.Module, verbose=True) -> int:
    if not _IBN_ENABLED:
        return 0

    top = _iter_top_modules(root)
    if not top:
        if verbose:
            print("[IBN] 未找到顶层模块")
        return 0

    ready = 0
    try:
        stem = top[0]
        if _attach_ibna_inplace(stem, ratio=_IBN_RATIO):
            ready += 1
            if verbose:
                print("[IBN] stem 已注入 IBN-a")
    except Exception as e:
        if verbose:
            print(f"[IBN] stem 注入失败: {e}")

    target_types = []
    if UltralyticsC2f is not None:
        target_types.append(UltralyticsC2f)
    if UltralyticsC3k2 is not None:
        target_types.append(UltralyticsC3k2)

    stage_candidates = _iter_backbone_stage_candidates(top, target_types)
    for i, (idx, m) in enumerate(stage_candidates[:max(0, int(_IBN_LAYERS))]):
        if _attach_ibna_inplace(m, ratio=_IBN_RATIO):
            ready += 1
            if verbose:
                print(f"[IBN] backbone stage[{i}] (top_idx={idx}, type={m.__class__.__name__}) 已注入 IBN-a")

    if verbose:
        print(f"[IBN] 注入完成: {ready} 处（stem + 前 {min(_IBN_LAYERS, len(stage_candidates))} 个 backbone stage）")
    return ready


def _efe_wrap_head(head: nn.Module):
    if getattr(head, "efe_wrapped", False):
        if not isinstance(getattr(head, "enhancers", None), nn.ModuleList):
            head.enhancers = nn.ModuleList()
        if getattr(head, "_efe_orig_forward", None) is None:
            current = getattr(head, "forward", None)
            if _is_bound_to_impl(current, _efe_head_forward_impl):
                head._efe_orig_forward = head.__class__.forward.__get__(head, head.__class__)
            else:
                head._efe_orig_forward = current
        if not _is_bound_to_impl(getattr(head, "forward", None), _efe_head_forward_impl):
            head.forward = _efe_head_forward_impl.__get__(head, head.__class__)
        return True

    head._efe_orig_forward = head.forward
    head.enhancers = nn.ModuleList()
    head.forward = _efe_head_forward_impl.__get__(head, head.__class__)
    head.efe_wrapped = True
    return True


def force_inject_efe(model_or_yolo, verbose=True):
    if not _ENHANCE_ENABLED:
        if verbose:
            print("[EFE] Enhance 未开启，跳过")
        return 0

    m = model_or_yolo
    if hasattr(m, "module") and isinstance(m.module, nn.Module):
        m = m.module
    if hasattr(m, "model"):
        m = m.model
    if not isinstance(m, nn.Module):
        return 0

    count = 0
    for mod in m.modules():
        if isinstance(mod, (UltralyticsDetect, UltralyticsSegment)):
            if _efe_wrap_head(mod):
                count += 1

    if verbose:
        print(f"[EFE] head 就绪数量: {count}")
    return count


def _attach_mixstyle_inplace(module: nn.Module, prob=0.3, alpha=0.3, mode="mixstyle"):
    if getattr(module, "_mixstyle_attached", False):
        if getattr(module, "_mixstyle", None) is None:
            ms = MixStyle(p=prob, alpha=alpha, mode=mode)
            try:
                p = next(module.parameters())
                ms = ms.to(device=p.device)
            except StopIteration:
                pass
            module.add_module("_mixstyle", ms)
        else:
            module._mixstyle.p = float(prob)
            module._mixstyle.alpha = float(alpha)
            module._mixstyle.mode = _normalize_mixstyle_mode(mode)

        if getattr(module, "_mixstyle_orig_forward", None) is None:
            current = getattr(module, "forward", None)
            if _is_bound_to_impl(current, _mixstyle_forward_impl):
                module._mixstyle_orig_forward = module.__class__.forward.__get__(module, module.__class__)
            else:
                module._mixstyle_orig_forward = current

        if not _is_bound_to_impl(getattr(module, "forward", None), _mixstyle_forward_impl):
            module.forward = _mixstyle_forward_impl.__get__(module, module.__class__)
        return True

    ms = MixStyle(p=prob, alpha=alpha, mode=mode)
    try:
        p = next(module.parameters())
        ms = ms.to(device=p.device)
    except StopIteration:
        pass

    module.add_module("_mixstyle", ms)
    module._mixstyle_orig_forward = module.forward
    module.forward = _mixstyle_forward_impl.__get__(module, module.__class__)
    module._mixstyle_attached = True
    return True


def _inject_mixstyle_to_backbone(root: nn.Module, verbose=True):
    if not _MIXSTYLE_ENABLED:
        return 0

    target_types = []
    if UltralyticsC2f is not None:
        target_types.append(UltralyticsC2f)
    if UltralyticsC3k2 is not None:
        target_types.append(UltralyticsC3k2)

    if not target_types:
        if verbose:
            print("[MixStyle] 未找到可注入的目标模块类型")
        return 0

    candidates = _iter_backbone_stage_candidates(_iter_top_modules(root), target_types)
    selected = _select_stage_candidates(candidates, _MIXSTYLE_LAYERS, prefer_deep=_PROTECT_SHALLOW_TEXTURES)
    ready = 0
    for idx, m in selected:
        if _attach_mixstyle_inplace(m, prob=_MIXSTYLE_PROB, alpha=_MIXSTYLE_ALPHA, mode=_MIXSTYLE_MODE):
            ready += 1
            if verbose:
                print(f"[MixStyle] backbone top_idx={idx} 已就绪 ({m.__class__.__name__})")

    if verbose:
        strategy = "deep" if _PROTECT_SHALLOW_TEXTURES else "front"
        print(f"[MixStyle] 就绪数量: {ready}/{len(candidates)} strategy={strategy}")
    return ready


def _attach_spd_inplace(module: nn.Module, scale=2, alpha_init=0.10):
    if getattr(module, "_spd_attached", False):
        if getattr(module, "_spd", None) is None:
            module.add_module("_spd", SPDApplier(scale=scale, alpha_init=alpha_init))
        if getattr(module, "_spd_orig_forward", None) is None:
            current = getattr(module, "forward", None)
            if _is_bound_to_impl(current, _spd_forward_impl):
                module._spd_orig_forward = module.__class__.forward.__get__(module, module.__class__)
            else:
                module._spd_orig_forward = current
        if not _is_bound_to_impl(getattr(module, "forward", None), _spd_forward_impl):
            module.forward = _spd_forward_impl.__get__(module, module.__class__)
        return True

    module.add_module("_spd", SPDApplier(scale=scale, alpha_init=alpha_init))
    module._spd_orig_forward = module.forward
    module.forward = _spd_forward_impl.__get__(module, module.__class__)
    module._spd_attached = True
    return True


def _inject_spd_to_backbone(root: nn.Module, verbose=True):
    if not _SPD_ENABLED:
        return 0

    top = _iter_top_modules(root)
    if not top:
        if verbose:
            print("[SPD] 未找到顶层模块")
        return 0

    ready = 0
    stride_candidates = [
        (idx, m) for idx, m in _iter_stride_backbone_candidates(top) if not getattr(m, "_randconv_attached", False)
    ]
    selected = _select_stage_candidates(stride_candidates, _SPD_LAYERS, prefer_deep=_PROTECT_SHALLOW_TEXTURES)
    for idx, m in selected:
        if _attach_spd_inplace(m, scale=_SPD_SCALE, alpha_init=_SPD_ALPHA_INIT):
            ready += 1
            if verbose:
                print(f"[SPD] top_idx={idx} ({m.__class__.__name__}) 已注入 SPDAdapter")
    if verbose:
        strategy = "deep" if _PROTECT_SHALLOW_TEXTURES else "front"
        print(f"[SPD] 注入完成: {ready} 处 strategy={strategy}")
    return ready


def _attach_simam_inplace(module: nn.Module, e_lambda=1e-4):
    if getattr(module, "_simam_attached", False):
        if getattr(module, "_simam", None) is None:
            module.add_module("_simam", SimAM(e_lambda=e_lambda))
        if getattr(module, "_simam_orig_forward", None) is None:
            current = getattr(module, "forward", None)
            if _is_bound_to_impl(current, _simam_forward_impl):
                module._simam_orig_forward = module.__class__.forward.__get__(module, module.__class__)
            else:
                module._simam_orig_forward = current
        if not _is_bound_to_impl(getattr(module, "forward", None), _simam_forward_impl):
            module.forward = _simam_forward_impl.__get__(module, module.__class__)
        return True

    module.add_module("_simam", SimAM(e_lambda=e_lambda))
    module._simam_orig_forward = module.forward
    module.forward = _simam_forward_impl.__get__(module, module.__class__)
    module._simam_attached = True
    return True


def _inject_simam_to_backbone(root: nn.Module, verbose=True):
    if not _SIMAM_ENABLED:
        return 0

    target_types = []
    if UltralyticsC2f is not None:
        target_types.append(UltralyticsC2f)
    if UltralyticsC3k2 is not None:
        target_types.append(UltralyticsC3k2)
    if UltralyticsC2PSA is not None:
        target_types.append(UltralyticsC2PSA)

    if not target_types:
        if verbose:
            print("[SimAM] 未找到可注入的目标模块")
        return 0

    candidates = _iter_backbone_stage_candidates(_iter_top_modules(root), target_types)
    ready = 0
    for idx, m in list(candidates)[-max(1, int(_SIMAM_LAYERS)):]:
        if _attach_simam_inplace(m, e_lambda=_SIMAM_E_LAMBDA):
            ready += 1
            if verbose:
                print(f"[SimAM] backbone top_idx={idx} ({m.__class__.__name__}) 已注入")

    if verbose:
        print(f"[SimAM] 注入完成: {ready}/{len(candidates)}")
    return ready


def _attach_randconv_inplace(
    module: nn.Module,
    prob=0.2,
    prob_end=0.45,
    sigma=0.0,
    sigma_end=0.15,
    kernel_sizes=(3, 5),
    refresh_interval=1,
):
    if getattr(module, "_randconv_attached", False):
        if getattr(module, "_randconv", None) is None:
            rc = ProgressiveRandConv(
                p=prob,
                p_end=prob_end,
                sigma=sigma,
                sigma_end=sigma_end,
                kernel_sizes=kernel_sizes,
                refresh_interval=refresh_interval,
            )
            module.add_module("_randconv", rc)

        if getattr(module, "_randconv_orig_forward", None) is None:
            current = getattr(module, "forward", None)
            if _is_bound_to_impl(current, _randconv_forward_impl):
                module._randconv_orig_forward = module.__class__.forward.__get__(module, module.__class__)
            else:
                module._randconv_orig_forward = current

        if not _is_bound_to_impl(getattr(module, "forward", None), _randconv_forward_impl):
            module.forward = _randconv_forward_impl.__get__(module, module.__class__)
        return True

    rc = ProgressiveRandConv(
        p=prob,
        p_end=prob_end,
        sigma=sigma,
        sigma_end=sigma_end,
        kernel_sizes=kernel_sizes,
        refresh_interval=refresh_interval,
    )
    module.add_module("_randconv", rc)
    module._randconv_orig_forward = module.forward
    module.forward = _randconv_forward_impl.__get__(module, module.__class__)
    module._randconv_attached = True
    return True


def _inject_randconv_to_backbone(root: nn.Module, verbose=True):
    if not _RANDCONV_ENABLED:
        return 0

    top = _iter_top_modules(root)
    if not top:
        if verbose:
            print("[RandConv] 未找到顶层模块")
        return 0

    ready = 0
    remain = max(0, int(_RANDCONV_LAYERS))
    if remain <= 0:
        return 0

    if not _PROTECT_SHALLOW_TEXTURES:
        try:
            stem = top[0]
            if _attach_randconv_inplace(
                stem,
                prob=_RANDCONV_PROB,
                prob_end=_RANDCONV_PROB_END,
                sigma=_RANDCONV_SIGMA,
                sigma_end=_RANDCONV_SIGMA_END,
                kernel_sizes=_RANDCONV_KERNEL_SIZES,
                refresh_interval=_RANDCONV_REFRESH_INTERVAL,
            ):
                ready += 1
                remain -= 1
                if verbose:
                    print("[RandConv] stem 已注入 Progressive RandConv")
        except Exception as e:
            if verbose:
                print(f"[RandConv] stem 注入失败: {e}")

    if remain <= 0:
        return ready

    target_types = []
    if UltralyticsC2f is not None:
        target_types.append(UltralyticsC2f)
    if UltralyticsC3k2 is not None:
        target_types.append(UltralyticsC3k2)

    stage_candidates = _iter_backbone_stage_candidates(top, target_types)
    selected = _select_stage_candidates(stage_candidates, remain, prefer_deep=_PROTECT_SHALLOW_TEXTURES)
    for idx, m in selected:
        if _attach_randconv_inplace(
            m,
            prob=_RANDCONV_PROB,
            prob_end=_RANDCONV_PROB_END,
            sigma=_RANDCONV_SIGMA,
            sigma_end=_RANDCONV_SIGMA_END,
            kernel_sizes=_RANDCONV_KERNEL_SIZES,
            refresh_interval=_RANDCONV_REFRESH_INTERVAL,
        ):
            ready += 1
            if verbose:
                print(f"[RandConv] backbone top_idx={idx} 已注入 Progressive RandConv ({m.__class__.__name__})")

    if verbose:
        strategy = "deep" if _PROTECT_SHALLOW_TEXTURES else "stem+front"
        print(f"[RandConv] 注入完成: {ready} 处 strategy={strategy}")
    return ready


def _collect_randconv_modules(model_or_module) -> List[ProgressiveRandConv]:
    root = _resolve_patch_root(model_or_module)
    if root is None and isinstance(model_or_module, nn.Module):
        root = model_or_module
    if root is None:
        return []
    return [m for m in root.modules() if isinstance(m, ProgressiveRandConv)]


def _update_randconv_schedule(
    model_or_module,
    epoch: int,
    total_epochs: int,
    close_mosaic: int = 0,
    verbose: bool = False,
):
    modules = _collect_randconv_modules(model_or_module)
    if not modules:
        return None

    total_epochs = max(1, int(total_epochs))
    epoch = max(0, int(epoch))
    close_mosaic = max(0, int(close_mosaic))

    progress = 0.0 if total_epochs <= 1 else min(1.0, max(0.0, epoch / float(total_epochs - 1)))
    tail_decay = 1.0

    if close_mosaic > 0 and total_epochs > close_mosaic:
        tail_start = total_epochs - close_mosaic
        if epoch >= tail_start:
            denom = max(close_mosaic - 1, 1)
            tail_ratio = min(1.0, max(0.0, (epoch - tail_start) / float(denom)))
            tail_decay = 1.0 - tail_ratio

    signature = None
    for mod in modules:
        mod.set_schedule(progress, tail_decay=tail_decay)
        signature = mod.schedule_signature()

    if verbose and signature is not None:
        prob, sigma, kernel = signature
        print(
            f"[RandConv] epoch={epoch + 1}/{total_epochs} "
            f"prob={prob:.3f} sigma={sigma:.3f} kernel={kernel} tail_decay={tail_decay:.3f}"
        )
    return signature


def _randconv_on_train_epoch_start(trainer):
    total_epochs = getattr(getattr(trainer, "args", None), "epochs", None)
    if total_epochs is None:
        total_epochs = getattr(trainer, "epochs", 1)

    close_mosaic = getattr(getattr(trainer, "args", None), "close_mosaic", 0)
    epoch = getattr(trainer, "epoch", 0)

    signature = _update_randconv_schedule(
        trainer.model,
        epoch=epoch,
        total_epochs=total_epochs,
        close_mosaic=close_mosaic,
        verbose=False,
    )
    if signature is None:
        return

    old_signature = getattr(trainer, "_randconv_logged_signature", None)
    if old_signature != signature or int(epoch) in (0, max(0, int(total_epochs) - 1)):
        prob, sigma, kernel = signature
        print(
            f"[RandConv] schedule epoch={int(epoch) + 1}/{int(total_epochs)} "
            f"prob={prob:.3f} sigma={sigma:.3f} kernel={kernel}"
        )
        trainer._randconv_logged_signature = signature


def assert_dynamic_plugins_materialized(model_or_yolo):
    root = _resolve_patch_root(model_or_yolo)
    if root is None:
        raise RuntimeError("无法解析模型根节点")

    issues = []
    for m in root.modules():
        if getattr(m, "_ibna_attached", False):
            applier = getattr(m, "_ibna", None)
            if isinstance(applier, IBNApplier) and applier.inner is None:
                issues.append("IBN inner 未 materialize")

        if getattr(m, "_spd_attached", False):
            applier = getattr(m, "_spd", None)
            if isinstance(applier, SPDApplier) and applier.inner is None:
                issues.append("SPD inner not materialized")

        if getattr(m, "efe_wrapped", False):
            enh = getattr(m, "enhancers", None)
            if not isinstance(enh, nn.ModuleList) or len(enh) == 0:
                issues.append("EFE enhancers 未 materialize")

    if issues:
        uniq = sorted(set(issues))
        raise RuntimeError(" ; ".join(uniq))


def collect_plugin_param_ids(model_or_yolo):
    root = _resolve_patch_root(model_or_yolo)
    if root is None:
        return set(), {"EFE": 0, "IBNa": 0, "MixStyle": 0, "RandConv": 0, "SPD": 0, "SimAM": 0}

    plugin_classes = (EFE, IBNa, MixStyle, ProgressiveRandConv, SPDAdapter, SimAM)
    ids = set()
    detail = {"EFE": 0, "IBNa": 0, "MixStyle": 0, "RandConv": 0, "SPD": 0, "SimAM": 0}

    for m in root.modules():
        if isinstance(m, plugin_classes):
            for p in m.parameters(recurse=True):
                if p is not None and p.requires_grad:
                    ids.add(id(p))
                    if isinstance(m, EFE):
                        detail["EFE"] += p.numel()
                    elif isinstance(m, IBNa):
                        detail["IBNa"] += p.numel()
                    elif isinstance(m, MixStyle):
                        detail["MixStyle"] += p.numel()
                    elif isinstance(m, ProgressiveRandConv):
                        detail["RandConv"] += p.numel()
                    elif isinstance(m, SPDAdapter):
                        detail["SPD"] += p.numel()
                    elif isinstance(m, SimAM):
                        detail["SimAM"] += p.numel()

    return ids, detail


def assert_plugins_in_optimizer(model_or_yolo, optimizer, strict=True):
    plugin_ids, detail = collect_plugin_param_ids(model_or_yolo)

    opt_ids = set()
    for g in getattr(optimizer, "param_groups", []):
        for p in g.get("params", []):
            if p is not None:
                opt_ids.add(id(p))

    missing = plugin_ids - opt_ids
    print(
        f"[Plugins][Check] plugin_params_numel={sum(detail.values())} "
        f"detail={detail} in_opt={len(plugin_ids) - len(missing)}/{len(plugin_ids)}"
    )

    if missing:
        msg = f"{len(missing)} 个插件参数未进入优化器"
        if strict:
            raise RuntimeError(msg)
        print(f"[Plugins][WARN] {msg}")


def prepare_model_plugins_before_train(model_or_yolo, verbose: bool = True):
    model = model_or_yolo.model if hasattr(model_or_yolo, "model") else model_or_yolo
    if hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module

    if not isinstance(model, nn.Module):
        raise TypeError("model is not nn.Module")

    _install_pickle_compat_shims(verbose=False)
    _patch_ultralytics_concat(verbose=verbose)

    changed = 0

    if _IBN_ENABLED:
        changed += _inject_ibna_to_backbone(model, verbose=verbose)

    if _ENHANCE_ENABLED:
        heads_wrapped = force_inject_efe(model_or_yolo, verbose=verbose)
        setattr(model, "_efe_heads_wrapped", int(heads_wrapped))
        changed += heads_wrapped
    else:
        setattr(model, "_efe_heads_wrapped", 0)

    if _RANDCONV_ENABLED:
        changed += _inject_randconv_to_backbone(model, verbose=verbose)
        _update_randconv_schedule(model, epoch=0, total_epochs=max(2, 10), close_mosaic=0, verbose=verbose)

    if _MIXSTYLE_ENABLED:
        changed += _inject_mixstyle_to_backbone(model, verbose=verbose)

    if _SPD_ENABLED:
        changed += _inject_spd_to_backbone(model, verbose=verbose)

    if _SIMAM_ENABLED:
        changed += _inject_simam_to_backbone(model, verbose=verbose)

    need_plugins = any([_IBN_ENABLED, _ENHANCE_ENABLED, _MIXSTYLE_ENABLED, _RANDCONV_ENABLED, _SPD_ENABLED, _SIMAM_ENABLED])
    if need_plugins:
        ok = _materialize_dynamic_modules(model_or_yolo, imgsz=256, verbose=verbose)
        if not ok:
            raise RuntimeError("插件注入后 dummy forward 失败，请检查模块兼容性。")
        assert_dynamic_plugins_materialized(model_or_yolo)

    _align_model_floating_tensors(model, verbose=verbose)

    efe_count = sum(1 for m in model.modules() if isinstance(m, EFE))
    mixstyle_count = sum(1 for m in model.modules() if isinstance(m, MixStyle))
    randconv_count = sum(1 for m in model.modules() if isinstance(m, ProgressiveRandConv))
    ibn_count = sum(1 for m in model.modules() if isinstance(m, IBNa))
    spd_count = sum(1 for m in model.modules() if isinstance(m, SPDAdapter))
    simam_count = sum(1 for m in model.modules() if isinstance(m, SimAM))
    setattr(model, "_efe_materialized", int(efe_count))

    if _ENHANCE_ENABLED and getattr(model, "_efe_heads_wrapped", 0) > 0 and efe_count <= 0:
        raise RuntimeError("EFE head 已注入，但未 materialize 成功。")

    if verbose:
        print(
            f"[Prepare] changed={changed} "
            f"EFE={efe_count} MixStyle={mixstyle_count} RandConv={randconv_count} "
            f"IBNa={ibn_count} SPD={spd_count} SimAM={simam_count}"
        )

    return changed


def _auto_tune_combo_defaults(
    enable_mixstyle: bool,
    mixstyle_prob: float,
    mixstyle_alpha: float,
    mixstyle_layers: int,
    enable_randconv: bool,
    randconv_prob: float,
    randconv_prob_end: float,
    randconv_sigma: float,
    randconv_sigma_end: float,
    randconv_layers: int,
):
    notes = []
    if not (enable_mixstyle and enable_randconv):
        return (
            mixstyle_prob,
            mixstyle_alpha,
            mixstyle_layers,
            randconv_prob,
            randconv_prob_end,
            randconv_sigma,
            randconv_sigma_end,
            randconv_layers,
            notes,
        )

    if _float_eq(mixstyle_prob, _DEFAULT_MIXSTYLE_PROB):
        mixstyle_prob = _COMBO_MIXSTYLE_PROB
        notes.append(f"mixstyle_prob->{mixstyle_prob}")

    if _float_eq(mixstyle_alpha, _DEFAULT_MIXSTYLE_ALPHA):
        mixstyle_alpha = _COMBO_MIXSTYLE_ALPHA
        notes.append(f"mixstyle_alpha->{mixstyle_alpha}")

    if _float_eq(randconv_prob, _DEFAULT_RANDCONV_PROB):
        randconv_prob = _COMBO_RANDCONV_PROB
        notes.append(f"randconv_prob->{randconv_prob}")

    if _float_eq(randconv_prob_end, _DEFAULT_RANDCONV_PROB_END):
        randconv_prob_end = _COMBO_RANDCONV_PROB_END
        notes.append(f"randconv_prob_end->{randconv_prob_end}")

    if _float_eq(randconv_sigma, _DEFAULT_RANDCONV_SIGMA):
        randconv_sigma = _COMBO_RANDCONV_SIGMA

    if _float_eq(randconv_sigma_end, _DEFAULT_RANDCONV_SIGMA_END):
        randconv_sigma_end = _COMBO_RANDCONV_SIGMA_END
        notes.append(f"randconv_sigma_end->{randconv_sigma_end}")

    return (
        mixstyle_prob,
        mixstyle_alpha,
        mixstyle_layers,
        randconv_prob,
        randconv_prob_end,
        randconv_sigma,
        randconv_sigma_end,
        randconv_layers,
        notes,
    )


def register_plugins(
    enable_enhance=False,
    enhance_edge_ks=5,
    enhance_rates=(1, 2, 3),
    enable_mixstyle=False,
    mixstyle_prob=_DEFAULT_MIXSTYLE_PROB,
    mixstyle_alpha=_DEFAULT_MIXSTYLE_ALPHA,
    mixstyle_layers=_DEFAULT_MIXSTYLE_LAYERS,
    mixstyle_mode=_DEFAULT_MIXSTYLE_MODE,
    enable_randconv=False,
    randconv_prob=_DEFAULT_RANDCONV_PROB,
    randconv_prob_end=_DEFAULT_RANDCONV_PROB_END,
    randconv_sigma=_DEFAULT_RANDCONV_SIGMA,
    randconv_sigma_end=_DEFAULT_RANDCONV_SIGMA_END,
    randconv_layers=_DEFAULT_RANDCONV_LAYERS,
    randconv_kernel_sizes=(3, 5),
    randconv_refresh_interval=1,
    protect_shallow_textures=False,
    enable_ibn=True,
    ibn_ratio=0.5,
    ibn_layers=1,
    enable_spd=False,
    spd_layers=2,
    spd_scale=2,
    spd_alpha_init=0.10,
    enable_simam=False,
    simam_layers=2,
    simam_e_lambda=1e-4,
    enable_safe_concat=False,
    enhance_use_hfp=None,
    verbose=True,
    **legacy_kwargs,
):
    global _ENHANCE_ENABLED, _ENHANCE_EDGE_KS, _ENHANCE_RATES
    global _MIXSTYLE_ENABLED, _MIXSTYLE_PROB, _MIXSTYLE_ALPHA, _MIXSTYLE_LAYERS, _MIXSTYLE_MODE
    global _RANDCONV_ENABLED, _RANDCONV_PROB, _RANDCONV_PROB_END
    global _RANDCONV_SIGMA, _RANDCONV_SIGMA_END, _RANDCONV_LAYERS
    global _RANDCONV_KERNEL_SIZES, _RANDCONV_REFRESH_INTERVAL
    global _PROTECT_SHALLOW_TEXTURES
    global _IBN_ENABLED, _IBN_RATIO, _IBN_LAYERS
    global _SPD_ENABLED, _SPD_LAYERS, _SPD_SCALE, _SPD_ALPHA_INIT
    global _SIMAM_ENABLED, _SIMAM_LAYERS, _SIMAM_E_LAMBDA
    global _ENABLE_SAFE_CONCAT

    (
        mixstyle_prob,
        mixstyle_alpha,
        mixstyle_layers,
        randconv_prob,
        randconv_prob_end,
        randconv_sigma,
        randconv_sigma_end,
        randconv_layers,
        combo_notes,
    ) = _auto_tune_combo_defaults(
        enable_mixstyle=bool(enable_mixstyle),
        mixstyle_prob=float(mixstyle_prob),
        mixstyle_alpha=float(mixstyle_alpha),
        mixstyle_layers=int(mixstyle_layers),
        enable_randconv=bool(enable_randconv),
        randconv_prob=float(randconv_prob),
        randconv_prob_end=float(randconv_prob_end),
        randconv_sigma=float(randconv_sigma),
        randconv_sigma_end=float(randconv_sigma_end),
        randconv_layers=int(randconv_layers),
    )

    _ENHANCE_ENABLED = bool(enable_enhance)
    _ENHANCE_EDGE_KS = int(enhance_edge_ks)
    _ENHANCE_RATES = tuple(enhance_rates)

    _MIXSTYLE_ENABLED = bool(enable_mixstyle)
    _MIXSTYLE_PROB = float(mixstyle_prob)
    _MIXSTYLE_ALPHA = float(mixstyle_alpha)
    _MIXSTYLE_LAYERS = int(mixstyle_layers)
    _MIXSTYLE_MODE = _normalize_mixstyle_mode(mixstyle_mode)

    _RANDCONV_ENABLED = bool(enable_randconv)
    _RANDCONV_PROB = float(randconv_prob)
    _RANDCONV_PROB_END = float(randconv_prob_end)
    _RANDCONV_SIGMA = float(randconv_sigma)
    _RANDCONV_SIGMA_END = float(randconv_sigma_end)
    _RANDCONV_LAYERS = int(randconv_layers)
    _RANDCONV_KERNEL_SIZES = tuple(randconv_kernel_sizes)
    _RANDCONV_REFRESH_INTERVAL = max(1, int(randconv_refresh_interval))
    _PROTECT_SHALLOW_TEXTURES = bool(protect_shallow_textures)

    _IBN_ENABLED = bool(enable_ibn)
    _IBN_RATIO = float(ibn_ratio)
    _IBN_LAYERS = int(ibn_layers)

    _SPD_ENABLED = bool(enable_spd)
    _SPD_LAYERS = max(0, int(spd_layers))
    _SPD_SCALE = max(2, int(spd_scale))
    _SPD_ALPHA_INIT = float(spd_alpha_init)

    _SIMAM_ENABLED = bool(enable_simam)
    _SIMAM_LAYERS = max(0, int(simam_layers))
    _SIMAM_E_LAMBDA = float(simam_e_lambda)

    _ENABLE_SAFE_CONCAT = bool(enable_safe_concat)

    _install_pickle_compat_shims(verbose=False)
    _patch_ultralytics_concat(verbose=verbose)

    ignored_legacy = {}
    if enhance_use_hfp is not None:
        ignored_legacy["enhance_use_hfp"] = enhance_use_hfp
    if legacy_kwargs:
        ignored_legacy.update(legacy_kwargs)

    if verbose and combo_notes:
        print(f"[Plugins] 检测到 MixStyle + RandConv 组合模式，已自动弱化默认强度: {', '.join(combo_notes)}")

    if verbose:
        print(
            f"[Plugins] Enhance={'ON' if _ENHANCE_ENABLED else 'OFF'} "
            f"MixStyle={'ON' if _MIXSTYLE_ENABLED else 'OFF'}(mode={_MIXSTYLE_MODE}, p={_MIXSTYLE_PROB}, a={_MIXSTYLE_ALPHA}, layers={_MIXSTYLE_LAYERS}) "
            f"RandConv={'ON' if _RANDCONV_ENABLED else 'OFF'}(p={_RANDCONV_PROB}, p_end={_RANDCONV_PROB_END}, "
            f"sigma={_RANDCONV_SIGMA}, sigma_end={_RANDCONV_SIGMA_END}, layers={_RANDCONV_LAYERS}) "
            f"IBN={'ON' if _IBN_ENABLED else 'OFF'} "
            f"SPD={'ON' if _SPD_ENABLED else 'OFF'}(layers={_SPD_LAYERS}, scale={_SPD_SCALE}, alpha={_SPD_ALPHA_INIT}) "
            f"SimAM={'ON' if _SIMAM_ENABLED else 'OFF'}(layers={_SIMAM_LAYERS}, lambda={_SIMAM_E_LAMBDA}) "
            f"ProtectShallowTextures={'ON' if _PROTECT_SHALLOW_TEXTURES else 'OFF'} "
            f"SafeConcat={'ON' if _ENABLE_SAFE_CONCAT else 'OFF'} "
            f"(ibn_ratio={_IBN_RATIO}, ibn_layers={_IBN_LAYERS}+stem)"
        )


def get_plugin_callbacks():
    cbs = {}
    if _RANDCONV_ENABLED:
        cbs["on_train_epoch_start"] = _randconv_on_train_epoch_start
    return cbs


def strip_forward_patches(model_or_module):
    root = _resolve_patch_root(model_or_module)
    if root is None:
        return

    for m in root.modules():
        orig = getattr(m, "_efe_orig_forward", None)
        if orig is not None and getattr(m, "efe_wrapped", False):
            m.forward = orig

        orig = getattr(m, "_mixstyle_orig_forward", None)
        if orig is not None and getattr(m, "_mixstyle_attached", False):
            m.forward = orig

        orig = getattr(m, "_randconv_orig_forward", None)
        if orig is not None and getattr(m, "_randconv_attached", False):
            m.forward = orig

        orig = getattr(m, "_spd_orig_forward", None)
        if orig is not None and getattr(m, "_spd_attached", False):
            m.forward = orig

        orig = getattr(m, "_simam_orig_forward", None)
        if orig is not None and getattr(m, "_simam_attached", False):
            m.forward = orig


def restore_forward_patches(model_or_module):
    root = _resolve_patch_root(model_or_module)
    if root is None:
        return

    for m in root.modules():
        if getattr(m, "efe_wrapped", False):
            if not isinstance(getattr(m, "enhancers", None), nn.ModuleList):
                m.enhancers = nn.ModuleList()

            orig = getattr(m, "_efe_orig_forward", None)
            if orig is None:
                current = getattr(m, "forward", None)
                if _is_bound_to_impl(current, _efe_head_forward_impl):
                    orig = m.__class__.forward.__get__(m, m.__class__)
                else:
                    orig = current
                m._efe_orig_forward = orig
            m.forward = _efe_head_forward_impl.__get__(m, m.__class__)

        if getattr(m, "_randconv_attached", False):
            orig = getattr(m, "_randconv_orig_forward", None)
            if orig is None:
                current = getattr(m, "forward", None)
                if _is_bound_to_impl(current, _randconv_forward_impl):
                    orig = m.__class__.forward.__get__(m, m.__class__)
                else:
                    orig = current
                m._randconv_orig_forward = orig
            m.forward = _randconv_forward_impl.__get__(m, m.__class__)

        if getattr(m, "_mixstyle_attached", False):
            orig = getattr(m, "_mixstyle_orig_forward", None)
            if orig is None:
                current = getattr(m, "forward", None)
                if _is_bound_to_impl(current, _mixstyle_forward_impl):
                    orig = m.__class__.forward.__get__(m, m.__class__)
                else:
                    orig = current
                m._mixstyle_orig_forward = orig
            m.forward = _mixstyle_forward_impl.__get__(m, m.__class__)

        if getattr(m, "_spd_attached", False):
            orig = getattr(m, "_spd_orig_forward", None)
            if orig is None:
                current = getattr(m, "forward", None)
                if _is_bound_to_impl(current, _spd_forward_impl):
                    orig = m.__class__.forward.__get__(m, m.__class__)
                else:
                    orig = current
                m._spd_orig_forward = orig
            m.forward = _spd_forward_impl.__get__(m, m.__class__)

        if getattr(m, "_simam_attached", False):
            orig = getattr(m, "_simam_orig_forward", None)
            if orig is None:
                current = getattr(m, "forward", None)
                if _is_bound_to_impl(current, _simam_forward_impl):
                    orig = m.__class__.forward.__get__(m, m.__class__)
                else:
                    orig = current
                m._simam_orig_forward = orig
            m.forward = _simam_forward_impl.__get__(m, m.__class__)


def _install_pickle_compat_shims(verbose: bool = True):
    shim_map = {
        "_ibna_forward_impl": _ibna_forward_impl,
        "_mixstyle_forward_impl": _mixstyle_forward_impl,
        "_randconv_forward_impl": _randconv_forward_impl,
        "_efe_head_forward_impl": _efe_head_forward_impl,
        "_spd_forward_impl": _spd_forward_impl,
        "_simam_forward_impl": _simam_forward_impl,
    }

    targets = [nn.Module]

    try:
        import ultralytics.nn.modules.conv as u_conv
        for cls_name in ("Conv", "Concat"):
            cls = getattr(u_conv, cls_name, None)
            if cls is not None:
                targets.append(cls)
    except Exception:
        pass

    try:
        import ultralytics.nn.modules.block as u_block
        for cls_name in ("C2f", "C3k2"):
            cls = getattr(u_block, cls_name, None)
            if cls is not None:
                targets.append(cls)
    except Exception:
        pass

    try:
        import ultralytics.nn.modules.head as u_head
        for cls_name in ("Detect", "Segment"):
            cls = getattr(u_head, cls_name, None)
            if cls is not None:
                targets.append(cls)
    except Exception:
        pass

    uniq_targets = []
    seen = set()
    for cls in targets:
        if cls is None:
            continue
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        uniq_targets.append(cls)

    patched = []
    for cls in uniq_targets:
        for name, func in shim_map.items():
            cur = getattr(cls, name, None)
            if cur is not func:
                setattr(cls, name, func)
                patched.append(f"{cls.__name__}.{name}")

    if verbose:
        if patched:
            preview = ", ".join(patched[:10])
            if len(patched) > 10:
                preview += " ..."
            print(f"[Plugins] 安装反序列化兼容实现: {preview}")
        else:
            print("[Plugins] 反序列化兼容实现已存在，跳过")


# 旧版 DCN 接口兼容占位
class DCNv2Pack(nn.Module):
    """仅用于兼容旧版外部 import；当前版本未启用 DCN。"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def setup_dcn_injection(*args, **kwargs):
    print("[DCN] setup_dcn_injection() 已被兼容占位，当前版本不执行任何操作。")
    return False


_install_pickle_compat_shims(verbose=True)

__all__ = [
    "register_plugins",
    "prepare_model_plugins_before_train",
    "assert_plugins_in_optimizer",
    "collect_plugin_param_ids",
    "assert_dynamic_plugins_materialized",
    "get_plugin_callbacks",
    "strip_forward_patches",
    "restore_forward_patches",
    "EFE",
    "MixStyle",
    "ProgressiveRandConv",
    "IBNa",
    "SPDAdapter",
    "SimAM",
    "DCNv2Pack",
    "setup_dcn_injection",
    "_ENHANCE_ENABLED",
    "_MIXSTYLE_ENABLED",
    "_RANDCONV_ENABLED",
    "_IBN_ENABLED",
    "_SPD_ENABLED",
    "_SIMAM_ENABLED",
]
