#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

"""
Track 1:
    STrack1 = 0.3 * Sloc + 0.3 * Scls + 0.4 * Sscreen

Track 2:
    STrack2 = 0.2 * Sloc + 0.2 * Scls + 0.6 * Sgrade
"""


def convert_mask2bbox(points):
    if len(points) % 2 != 0:
        raise ValueError("points must contain an even number of values")

    npoints = np.zeros((2, int(len(points) / 2)), dtype=float)
    for i in range(0, int(len(points) / 2)):
        npoints[:, i] = np.array([points[2 * i], points[2 * i + 1]])

    xmin, xmax = float(min(npoints[0, :])), float(max(npoints[0, :]))
    ymin, ymax = float(min(npoints[1, :])), float(max(npoints[1, :]))
    w = xmax - xmin
    h = ymax - ymin
    return [xmin, ymin, w, h]


def xywh_to_xyxy(box):
    # convert_mask2bbox returns top-left xywh, not center xywh.
    x, y, w, h = box
    return [x, y, x + w, y + h]


def bbox_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
    area2 = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def safe_div(num, den, default=0.0):
    den = float(den)
    if den <= 0:
        return float(default)
    return float(num) / den


def greedy_match(gt_items, pred_items, iou_thresh=0.25):
    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_items):
        for pi, p in enumerate(pred_items):
            if int(g["cls"]) != int(p["cls"]):
                continue
            iou = bbox_iou(xywh_to_xyxy(g["bbox"]), xywh_to_xyxy(p["bbox"]))
            if iou >= float(iou_thresh):
                candidates.append((float(iou), gi, pi))

    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_gt = set()
    matched_pred = set()
    matches = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    return matches, matched_gt, matched_pred


class CaculateMetric:
    def __init__(self):
        self.gt_data = {}
        self.pred_data = {}
        self.grade_map = ["Acceptable", "Marginal NG", "NG", "Gross NG"]
        self.classes_index = []

    def read_data(self, img_dir, txt_dir, txt_shuffix=".json"):
        all_data = {}
        classes_index = []
        for img_name in os.listdir(img_dir):
            if img_name.split(".")[-1].lower() not in ["bmp", "jpg", "png", "jpeg"]:
                continue

            txt_name = img_name.split(".")[0] + txt_shuffix
            txt_path = os.path.join(txt_dir, txt_name)
            if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                all_data[img_name.split(".")[0]] = []
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                if txt_shuffix == ".txt":
                    lines = f.readlines()
                    data = []
                    for line in lines:
                        line = line.strip().split(" ")
                        bbox = convert_mask2bbox(line[1:])
                        data.append({"cls": int(line[0]), "bbox": bbox})
                        if int(line[0]) not in classes_index:
                            classes_index.append(int(line[0]))
                    all_data[img_name.split(".")[0]] = data
                elif txt_shuffix == ".json":
                    lines = json.load(f)
                    data = []
                    for line in lines:
                        points = [float(x.strip()) for x in line["points"].split(",")]
                        bbox = convert_mask2bbox(points)
                        grade = self.grade_map.index(line["severity"]) if "severity" in line else None
                        data.append({"cls": int(line["class"]), "bbox": bbox, "grade": grade})
                        if int(line["class"]) not in classes_index:
                            classes_index.append(int(line["class"]))
                    all_data[img_name.split(".")[0]] = data
        return all_data, classes_index

    def _image_class_view(self, items, cls_id):
        return [x for x in items if int(x["cls"]) == int(cls_id)]

    def caculate_screen(self, iou_thresh=0.25):
        tp_img = dict.fromkeys(self.classes_index, 0)
        fp_img = dict.fromkeys(self.classes_index, 0)
        fn_img = dict.fromkeys(self.classes_index, 0)
        tn_img = dict.fromkeys(self.classes_index, 0)

        tp_img_cnt = 0
        tn_img_cnt = 0
        fp_img_cnt = 0
        fn_img_cnt = 0

        all_names = sorted(set(self.gt_data.keys()) | set(self.pred_data.keys()))
        for img_name in all_names:
            gt_items = self.gt_data.get(img_name, [])
            pred_items = self.pred_data.get(img_name, [])

            matches_all, _, _ = greedy_match(gt_items, pred_items, iou_thresh=iou_thresh)
            gt_pos_all = len(gt_items) > 0
            pred_pos_all = len(pred_items) > 0
            matched_any = len(matches_all) > 0

            if gt_pos_all:
                if matched_any:
                    tp_img_cnt += 1
                else:
                    fn_img_cnt += 1
            else:
                if pred_pos_all:
                    fp_img_cnt += 1
                else:
                    tn_img_cnt += 1

            for cls in self.classes_index:
                gt_cls = self._image_class_view(gt_items, cls)
                pred_cls = self._image_class_view(pred_items, cls)
                matches_cls, _, _ = greedy_match(gt_cls, pred_cls, iou_thresh=iou_thresh)
                gt_pos = len(gt_cls) > 0
                pred_pos = len(pred_cls) > 0
                matched_pos = len(matches_cls) > 0

                if gt_pos:
                    if matched_pos:
                        tp_img[cls] += 1
                    else:
                        fn_img[cls] += 1
                else:
                    if pred_pos:
                        fp_img[cls] += 1
                    else:
                        tn_img[cls] += 1

        SscreenDict = defaultdict(float)
        for cls in self.classes_index:
            cls_recall_img = safe_div(tp_img[cls], tp_img[cls] + fn_img[cls])
            cls_specificity_img = safe_div(tn_img[cls], tn_img[cls] + fp_img[cls])
            SscreenDict[cls] = 0.5 * cls_recall_img + 0.5 * cls_specificity_img

        Recall_img = safe_div(tp_img_cnt, tp_img_cnt + fn_img_cnt)
        Specificity_img = safe_div(tn_img_cnt, tn_img_cnt + fp_img_cnt)
        SscreenDict["all"] = 0.5 * Recall_img + 0.5 * Specificity_img
        return SscreenDict

    def caculate_Sfine(self, iou_thresh=0.25):
        TP = dict.fromkeys(self.classes_index, 0)
        FP = dict.fromkeys(self.classes_index, 0)
        FN = dict.fromkeys(self.classes_index, 0)

        all_names = sorted(set(self.gt_data.keys()) | set(self.pred_data.keys()))
        for img_name in all_names:
            gt_items = self.gt_data.get(img_name, [])
            pred_items = self.pred_data.get(img_name, [])

            for cls in self.classes_index:
                gt_cls = self._image_class_view(gt_items, cls)
                pred_cls = self._image_class_view(pred_items, cls)
                matches, matched_gt, matched_pred = greedy_match(gt_cls, pred_cls, iou_thresh=iou_thresh)
                TP[cls] += len(matches)
                FN[cls] += max(0, len(gt_cls) - len(matched_gt))
                FP[cls] += max(0, len(pred_cls) - len(matched_pred))

        SfineDict = defaultdict(float)
        sfine_sum = 0.0
        for cls in self.classes_index:
            tp = TP[cls]
            fp = FP[cls]
            fn = FN[cls]
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall)
            SfineDict[cls] = f1
            sfine_sum += f1

        SfineDict["all"] = sfine_sum / max(1, len(self.classes_index))
        return SfineDict

    def caculate_cls(self):
        SfineDict = self.caculate_Sfine()
        SscreenDict = self.caculate_screen()
        SclsDict = defaultdict(float)
        for cls in self.classes_index:
            SclsDict[cls] = 0.5 * SfineDict[cls] + 0.5 * SscreenDict[cls]
        SclsDict["all"] = 0.5 * SfineDict["all"] + 0.5 * SscreenDict["all"]
        return SclsDict

    def caculate_loc(self, iou_thresh=0.25):
        ious_dict = defaultdict(list)
        all_ious = []

        all_names = sorted(set(self.gt_data.keys()) | set(self.pred_data.keys()))
        for img_name in all_names:
            gt_items = self.gt_data.get(img_name, [])
            pred_items = self.pred_data.get(img_name, [])
            if len(gt_items) == 0 or len(pred_items) == 0:
                continue
            matches, _, _ = greedy_match(gt_items, pred_items, iou_thresh=iou_thresh)
            for gi, _, iou in matches:
                cls = int(gt_items[gi]["cls"])
                ious_dict[cls].append(float(iou))
                all_ious.append(float(iou))

        SlocDict = defaultdict(float)
        for cls in self.classes_index:
            vals = ious_dict.get(cls, [])
            SlocDict[cls] = float(np.mean(vals)) if vals else 0.0
        SlocDict["all"] = float(np.mean(all_ious)) if all_ious else 0.0
        return SlocDict

    def severity_grading_from_confmat(self, conf_mat):
        conf_mat = np.asarray(conf_mat, dtype=float)
        K = conf_mat.shape[0]
        assert conf_mat.shape[0] == conf_mat.shape[1], "conf_mat must be KxK"

        N = conf_mat.sum()
        if N == 0:
            return np.nan

        idx = np.arange(K)
        I, J = np.meshgrid(idx, idx, indexing="ij")
        W = ((I - J) ** 2) / float((K - 1) ** 2)

        num = N * np.sum(W * conf_mat)
        n_i_dot = conf_mat.sum(axis=1)
        n_dot_j = conf_mat.sum(axis=0)
        expected = np.outer(n_i_dot, n_dot_j)
        denom = np.sum(W * expected)
        if denom == 0:
            return 0

        return 1.0 - num / denom

    def collect_triplets(self, gt_dict, pred_dict, iou_thres=0.25):
        triplets = []
        common_imgs = set(gt_dict.keys()) & set(pred_dict.keys())
        for img in common_imgs:
            gt_instances = gt_dict[img]
            pred_instances = pred_dict[img]
            matches, _, _ = greedy_match(gt_instances, pred_instances, iou_thresh=iou_thres)
            for gi, pi, _ in matches:
                gt_ins = gt_instances[gi]
                pred_ins = pred_instances[pi]
                if "grade" not in gt_ins or "grade" not in pred_ins:
                    continue
                if gt_ins.get("grade") is None or pred_ins.get("grade") is None:
                    continue
                triplets.append((int(gt_ins["cls"]), int(gt_ins["grade"]), int(pred_ins["grade"])))

        if not triplets:
            raise ValueError("no matched instances for severity grading")
        return triplets

    def caculate_grade(self, K=4):
        SgradeDict = defaultdict(float)
        triplets = self.collect_triplets(self.gt_data, self.pred_data)

        all_gt = np.array([g for _, g, _ in triplets], dtype=int)
        all_pred = np.array([p for _, _, p in triplets], dtype=int)

        if K is None:
            K = max(all_gt.max(), all_pred.max()) + 1
        if all_gt.min() == 1:
            all_gt -= 1
            all_pred -= 1
            triplets = [(c, g - 1, p - 1) for (c, g, p) in triplets]

        overall_conf = np.zeros((K, K), dtype=float)
        for g, p in zip(all_gt, all_pred):
            overall_conf[g, p] += 1
        SgradeDict["all"] = self.severity_grading_from_confmat(overall_conf)

        per_cls_pairs = defaultdict(list)
        for cls_id, gt_g, pred_g in triplets:
            per_cls_pairs[cls_id].append((gt_g, pred_g))

        for cls_id, pairs in per_cls_pairs.items():
            conf = np.zeros((K, K), dtype=float)
            for gt_g, pred_g in pairs:
                conf[gt_g, pred_g] += 1
            SgradeDict[cls_id] = self.severity_grading_from_confmat(conf)

        return SgradeDict

    def read_classes(self, class_txt_dir):
        with open(class_txt_dir, "r", encoding="utf-8") as f:
            classes_list = [i.strip() for i in f.readlines()]
        return classes_list

    def process_data(
        self,
        gt_img_dir,
        gt_txt_dir,
        pred_img_dir,
        pred_txt_dir,
        class_txt_dir,
        txt_shuffix,
        S=2,
    ):
        classes_list = self.read_classes(class_txt_dir)
        self.gt_data, gt_classes = self.read_data(gt_img_dir, gt_txt_dir, txt_shuffix=txt_shuffix)
        self.pred_data, pt_classes = self.read_data(pred_img_dir, pred_txt_dir, txt_shuffix=txt_shuffix)
        self.classes_index = sorted(list(set(gt_classes) | set(pt_classes)))

        SscreenDict = self.caculate_screen()
        SfineDict = self.caculate_Sfine()
        SclsDict = defaultdict(float)
        SclsDict["all"] = 0.5 * SfineDict["all"] + 0.5 * SscreenDict["all"]
        SlocDict = self.caculate_loc()

        if S == 1:
            S1Dict = defaultdict(float)
            S1 = 0.3 * SlocDict["all"] + 0.3 * SclsDict["all"] + 0.4 * SscreenDict["all"]
            S1Dict["all"] = S1
            print("class     Sloc     0.5*Sscreen     0.5*Sfine     Scls     Sscreen     Strack1")
            print(
                "all     "
                f"{SlocDict['all']:.3f}     {SscreenDict['all'] * 0.5:.3f}     {SfineDict['all'] * 0.5:.3f}     "
                f"{SclsDict['all']:.3f}     {SscreenDict['all']:.3f}     {S1:.3f}"
            )

            for cls_idx in self.classes_index:
                cls_sloc = SlocDict[cls_idx]
                cls_sfine = SfineDict[cls_idx]
                cls_screen = SscreenDict[cls_idx]
                cls_scls = 0.5 * cls_sfine + 0.5 * cls_screen
                SclsDict[cls_idx] = cls_scls
                cls_s1 = 0.3 * cls_sloc + 0.3 * cls_scls + 0.4 * cls_screen
                S1Dict[cls_idx] = cls_s1
                cls_name = classes_list[cls_idx] if cls_idx < len(classes_list) else str(cls_idx)
                print(
                    f"{cls_name}     {cls_sloc:.3f}     {cls_screen * 0.5:.3f}     {cls_sfine * 0.5:.3f}     "
                    f"{cls_scls:.3f}     {cls_screen:.3f}     {cls_s1:.3f}"
                )
            return S1Dict

        if S == 2:
            SgradeDict = self.caculate_grade(K=4)
            S2Dict = defaultdict(float)
            S2 = 0.2 * SlocDict["all"] + 0.2 * SclsDict["all"] + 0.6 * SgradeDict["all"]
            S2Dict["all"] = S2
            print("class     Sloc     0.5*Sscreen     0.5*Sfine     Scls     Sgrade     Strack2")
            print(
                "all     "
                f"{SlocDict['all']:.3f}     {SscreenDict['all'] * 0.5:.3f}     {SfineDict['all'] * 0.5:.3f}     "
                f"{SclsDict['all']:.3f}     {SgradeDict['all']:.3f}     {S2:.3f}"
            )
            for cls_idx in self.classes_index:
                cls_sloc = SlocDict[cls_idx]
                cls_sfine = SfineDict[cls_idx]
                cls_screen = SscreenDict[cls_idx]
                cls_scls = 0.5 * cls_sfine + 0.5 * cls_screen
                SclsDict[cls_idx] = cls_scls
                cls_grade = SgradeDict[cls_idx] if cls_idx in SgradeDict else 0.0
                cls_s2 = 0.2 * cls_sloc + 0.2 * cls_scls + 0.6 * cls_grade
                S2Dict[cls_idx] = cls_s2
                cls_name = classes_list[cls_idx] if cls_idx < len(classes_list) else str(cls_idx)
                print(
                    f"{cls_name}     {cls_sloc:.3f}     {cls_screen * 0.5:.3f}     {cls_sfine * 0.5:.3f}     "
                    f"{cls_scls:.3f}     {cls_grade:.3f}     {cls_s2:.3f}"
                )
            return S2Dict

        return None


if __name__ == "__main__":
    print("Use CaculateMetric from this module via phase1_track1_metric_wrapper.py.")
