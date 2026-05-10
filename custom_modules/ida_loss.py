from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import DFLoss, VarifocalLoss, v8DetectionLoss, v8SegmentationLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import bbox2dist, make_anchors


_LOSS_HOOK_INSTALLED = False
_ORIG_SEG_INIT_CRITERION = None


def _resolve_root(model_or_yolo):
    root = model_or_yolo
    if hasattr(root, "module") and isinstance(root.module, nn.Module):
        root = root.module
    if not isinstance(root, nn.Module) and hasattr(root, "model") and isinstance(root.model, nn.Module):
        root = root.model
    return root if isinstance(root, nn.Module) else None


def _xyxy_to_cxcywh(boxes: torch.Tensor):
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return cx, cy, w, h


def normalized_wasserstein_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    constant: float = 12.8,
    eps: float = 1e-9,
) -> torch.Tensor:
    px, py, pw, ph = _xyxy_to_cxcywh(pred_boxes)
    gx, gy, gw, gh = _xyxy_to_cxcywh(target_boxes)
    center_dist = (px - gx).pow(2) + (py - gy).pow(2)
    shape_dist = 0.25 * ((pw - gw).pow(2) + (ph - gh).pow(2))
    wasserstein = torch.sqrt((center_dist + shape_dist).clamp_min(eps))
    similarity = torch.exp(-wasserstein / max(float(constant), eps))
    return 1.0 - similarity


def _is_feat_list(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(torch.is_tensor(v) and v.dim() == 4 for v in x)


def _build_feat_list_from_boxes_scores(boxes: Any, scores: Any):
    if isinstance(boxes, (list, tuple)) and isinstance(scores, (list, tuple)) and len(boxes) == len(scores) and len(boxes) > 0:
        feats = []
        for b, s in zip(boxes, scores):
            if not (torch.is_tensor(b) and torch.is_tensor(s)):
                return None
            if b.dim() != 4 or s.dim() != 4:
                return None
            if b.shape[0] != s.shape[0] or b.shape[2:] != s.shape[2:]:
                return None
            feats.append(torch.cat((b, s), dim=1))
        return feats

    if torch.is_tensor(boxes) and torch.is_tensor(scores):
        if boxes.dim() == 4 and scores.dim() == 4 and boxes.shape[0] == scores.shape[0] and boxes.shape[2:] == scores.shape[2:]:
            return [torch.cat((boxes, scores), dim=1)]

    return None


def _unpack_segmentation_preds(preds: Any):
    """Handle both older Ultralytics tuple outputs and newer dict/end2end-style outputs."""
    if isinstance(preds, dict):
        for key in ("one2many", "preds", "outputs", "main"):
            if key in preds:
                return _unpack_segmentation_preds(preds[key])

        boxes = preds.get("boxes")
        scores = preds.get("scores")
        pred_masks = preds.get(
            "pred_masks",
            preds.get(
                "mask_coeffs",
                preds.get(
                    "mask_coefficient",
                    preds.get("mask_coefficients"),
                ),
            ),
        )
        proto = preds.get("proto", preds.get("protos", preds.get("mask_proto")))
        feats = _build_feat_list_from_boxes_scores(boxes, scores)
        if feats is not None and torch.is_tensor(pred_masks) and torch.is_tensor(proto):
            return feats, pred_masks, proto

        feats = preds.get("feats", preds.get("features"))
        if _is_feat_list(feats) and torch.is_tensor(pred_masks) and torch.is_tensor(proto):
            return feats, pred_masks, proto

        for value in preds.values():
            try:
                return _unpack_segmentation_preds(value)
            except Exception:
                continue
        raise TypeError(f"Unsupported segmentation preds dict keys: {list(preds.keys())}")

    if isinstance(preds, (list, tuple)):
        if len(preds) == 3 and _is_feat_list(preds[0]) and torch.is_tensor(preds[1]) and torch.is_tensor(preds[2]):
            return preds[0], preds[1], preds[2]

        if len(preds) >= 2:
            inner = preds[1]
            if isinstance(inner, (list, tuple)):
                if len(inner) == 3 and _is_feat_list(inner[0]) and torch.is_tensor(inner[1]) and torch.is_tensor(inner[2]):
                    return inner[0], inner[1], inner[2]
                if len(inner) >= 3 and torch.is_tensor(inner[1]) and torch.is_tensor(inner[2]):
                    return inner[0], inner[1], inner[2]

            for item in preds:
                try:
                    return _unpack_segmentation_preds(item)
                except Exception:
                    continue

    raise TypeError(f"Unsupported segmentation preds type: {type(preds)}")


class IDABboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16, nwd_weight: float = 0.0, nwd_constant: float = 12.8):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.nwd_weight = max(0.0, min(1.0, float(nwd_weight)))
        self.nwd_constant = float(nwd_constant)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        base_loss = 1.0 - iou
        if self.nwd_weight > 0:
            nwd = normalized_wasserstein_loss(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                constant=self.nwd_constant,
            ).unsqueeze(-1)
            base_loss = (1.0 - self.nwd_weight) * base_loss + self.nwd_weight * nwd
        loss_iou = (base_loss * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        return loss_iou, loss_dfl


class IDASegmentationLoss(v8SegmentationLoss):
    def __init__(
        self,
        model,
        use_vfl: bool = False,
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        nwd_weight: float = 0.0,
        nwd_constant: float = 12.8,
    ):
        v8DetectionLoss.__init__(self, model)
        args = getattr(model, "args", None)
        self.overlap = bool(args.get("overlap_mask", True)) if isinstance(args, dict) else bool(getattr(args, "overlap_mask", True))
        self.use_vfl = bool(use_vfl)
        self.varifocal_loss = VarifocalLoss(gamma=float(vfl_gamma), alpha=float(vfl_alpha))
        self.bbox_loss = IDABboxLoss(model.model[-1].reg_max, nwd_weight=nwd_weight, nwd_constant=nwd_constant).to(
            self.device
        )

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """当前 Ultralytics(8.4.x) 的 Segment head 返回 dict: boxes/scores/feats/mask_coefficient/proto。

        原始 IDA loss 按旧版 tuple/list feat 输出解析，遇到新版 dict 会把 boxes+scores 拼回伪 feat，
        导致 channel 数与 spatial shape 不匹配。这里按新版 v8DetectionLoss 的解析方式计算目标分配，
        仅替换 cls(VFL) 与 bbox(NWD) 损失。
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        if self.use_vfl:
            target_labels = target_scores.gt(0).to(dtype)
            loss[1] = self.varifocal_loss(pred_scores, target_scores.to(dtype), target_labels) / target_scores_sum
        else:
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.parse_output(preds)
        if not isinstance(preds, dict):
            # 兼容旧版 Ultralytics 输出格式
            feats, pred_masks_raw, proto = _unpack_segmentation_preds(preds)
            pred_masks = pred_masks_raw.permute(0, 2, 1).contiguous()
            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1
            )
            preds = {
                "boxes": pred_distri,
                "scores": pred_scores,
                "feats": feats,
                "mask_coefficient": pred_masks_raw,
                "proto": proto,
            }
        else:
            pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
            proto = preds["proto"]

        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semseg
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None

        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape
        if fg_mask.sum():
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                # 与 Ultralytics 8.4.x 官方实现保持一致：优先上采样 proto，避免 mask 插值导致细节损失。
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()
                if self.overlap:
                    mask_zero = masks == 0
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0
                # bcedice_loss 只在部分 Ultralytics 版本存在，当前版本 __init__ 会创建。
                if hasattr(self, "bcedice_loss"):
                    loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                    loss[4] *= self.hyp.box
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box
        return loss * batch_size, loss.detach()


def _ida_init_segmentation_criterion(self):
    cfg = getattr(self, "_ida_loss_cfg", None)
    if isinstance(cfg, dict) and (cfg.get("use_vfl", False) or float(cfg.get("nwd_weight", 0.0)) > 0.0):
        return IDASegmentationLoss(self, **cfg)
    return _ORIG_SEG_INIT_CRITERION(self)


def install_loss_hook(verbose: bool = True):
    global _LOSS_HOOK_INSTALLED, _ORIG_SEG_INIT_CRITERION
    if _LOSS_HOOK_INSTALLED:
        return
    _ORIG_SEG_INIT_CRITERION = SegmentationModel.init_criterion
    SegmentationModel.init_criterion = _ida_init_segmentation_criterion
    _LOSS_HOOK_INSTALLED = True
    if verbose:
        print("[Loss] installed custom segmentation criterion hook")


def configure_model_loss(
    model_or_yolo,
    use_vfl: bool = False,
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    nwd_weight: float = 0.0,
    nwd_constant: float = 12.8,
    verbose: bool = True,
    **_ignored,
):
    root = _resolve_root(model_or_yolo)
    if root is None:
        raise TypeError("model is not nn.Module")

    install_loss_hook(verbose=False)
    cfg: Dict[str, Any] = {
        "use_vfl": bool(use_vfl),
        "vfl_alpha": float(vfl_alpha),
        "vfl_gamma": float(vfl_gamma),
        "nwd_weight": float(nwd_weight),
        "nwd_constant": float(nwd_constant),
    }
    root._ida_loss_cfg = cfg
    root.criterion = None
    if verbose:
        print(
            f"[Loss] VFL={'ON' if cfg['use_vfl'] else 'OFF'}"
            f"(alpha={cfg['vfl_alpha']}, gamma={cfg['vfl_gamma']}) "
            f"NWD={'ON' if cfg['nwd_weight'] > 0 else 'OFF'}"
            f"(weight={cfg['nwd_weight']}, constant={cfg['nwd_constant']})"
        )


__all__ = [
    "IDABboxLoss",
    "IDASegmentationLoss",
    "configure_model_loss",
    "install_loss_hook",
    "normalized_wasserstein_loss",
]
