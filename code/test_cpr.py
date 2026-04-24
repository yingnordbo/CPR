"""
Comprehensive CPR test/inference script with:
- mAP@50, mAR@100 at multiple confidence thresholds (bbox + segmentation)
- Per-class mAP@50
- Confusion matrix, per_image_analysis.csv
- BBox drawing on prediction defect areas
- FP16 inference
- Inference timing and GPU usage statistics
- Overkill/escape result saving
"""
# ── Bootstrap: locate CPR source and make it importable ───────────────────────
import os as _os, sys as _sys
_CPR_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'CPR')
if _CPR_DIR not in _sys.path:
    _sys.path.insert(0, _CPR_DIR)
_os.chdir(_CPR_DIR)  # all ./data/, log/, result/ paths resolve inside CPR/
# ─────────────────────────────────────────────────────────────────────────────
import contextlib
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from glob import glob
from itertools import chain
from pprint import pformat

import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from dataset import (
    DATASET_INFOS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    read_image,
    read_mask,
    register_custom_dataset,
    test_transform,
)
from metrics import (
    compute_ap_torch,
    compute_image_auc_torch,
    compute_pixel_auc_torch,
    compute_pro_torch,
)
from models import CPR, MODEL_INFOS, create_model
from utils import fix_seeds, plot_loss_curve

# ── Constants ─────────────────────────────────────────────────────────────────
CONF_THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.50]
_WARMUP_IMAGES = 10  # images to discard before timing / GPU stats


def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("-dn", "--dataset-name", type=str, default="mvtec",
                        choices=["mvtec", "mvtec_3d", "btad", "custom"], help="dataset name")
    parser.add_argument("-ss", "--scales", type=int, nargs="+", default=[4, 8])
    parser.add_argument("-kn", "--k-nearest", type=int, default=10)
    parser.add_argument("-r", "--resize", type=int, default=320)
    parser.add_argument("-fd", "--foreground-dir", type=str, default=None)
    parser.add_argument("-rd", "--retrieval-dir", type=str,
                        default='log/retrieval_mvtec_DenseNet_features.denseblock1_320')
    parser.add_argument("--sub-categories", type=str, nargs="+", default=None)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("-rs", "--region-sizes", type=int, nargs="+", default=[3, 1])
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet',
                        choices=list(MODEL_INFOS.keys()))
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None)
    # custom dataset
    parser.add_argument("--custom-data-dir", type=str, default=None)
    parser.add_argument("--object-categories", type=str, nargs="+", default=None)
    # output
    parser.add_argument("-lp", "--log-path", type=str, default=None)
    parser.add_argument("--save-root", type=str, default=None, help="save visualization root")
    # bbox
    parser.add_argument("--bbox-threshold", type=float, default=0.5, help="bbox threshold for drawing")
    # threshold mode
    parser.add_argument("--threshold-mode", type=str, default='auto',
                        choices=['auto', 'manual', 'bbox'], help="classification threshold mode")
    parser.add_argument("--manual-threshold", type=float, default=0.5)
    # inference
    parser.add_argument("--fp16", action="store_true", help="use FP16 inference")
    # saving options
    parser.add_argument("--save-correct", action="store_true", default=True,
                        help="save visualizations for correctly predicted images")
    parser.add_argument("--no-save-correct", dest="save_correct", action="store_false")
    parser.add_argument("--save-incorrect", action="store_true", default=True,
                        help="save overkill/escape images")
    parser.add_argument("--no-save-incorrect", dest="save_incorrect", action="store_false")
    return parser


# ── Utility functions ─────────────────────────────────────────────────────────

def denormalize(img_tensor):
    """Convert normalized tensor [C,H,W] to uint8 numpy [H,W,C] RGB."""
    std = np.array(IMAGENET_STD)
    mean = np.array(IMAGENET_MEAN)
    x = (((img_tensor.cpu().numpy().transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def _boxes_from_binary(binary_u8, max_dets=None):
    n, _, stats, _ = cv.connectedComponentsWithStats(binary_u8, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        boxes.append((x, y, x + w, y + h))
    if max_dets is not None:
        boxes = boxes[:max_dets]
    return boxes


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _draw_bboxes(overlay_rgb, ano_norm_resized, threshold, ano_score_map=None):
    binary = (ano_norm_resized >= threshold).astype(np.uint8) * 255
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    result = overlay_rgb.copy()
    h, w = result.shape[:2]
    line_thick = max(1, int(w / 300))
    font_scale = max(0.35, w / 900.0)
    font_thick = max(1, line_thick)
    for cnt in contours:
        x, y, bw, bh = cv.boundingRect(cnt)
        cv.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), line_thick)
        if ano_score_map is not None:
            region_score = float(ano_score_map[y:y + bh, x:x + bw].max())
            label = f'{region_score:.2f}'
            (tw, th_t), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            tx, ty = x, max(y - 4, th_t + 2)
            cv.rectangle(result, (tx, ty - th_t - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv.putText(result, label, (tx + 1, ty), cv.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 255, 0), font_thick, cv.LINE_AA)
    return result


# ── Detection metrics ─────────────────────────────────────────────────────────

def compute_detection_metrics(records, thresholds=None, iou_thr=0.5, max_dets=100):
    if thresholds is None:
        thresholds = CONF_THRESHOLDS
    classes = sorted({r['class_name'] for r in records})
    has_masks = any(r['gt_mask'].max() > 0 for r in records)

    out = {'has_gt_masks': has_masks, 'thresholds': thresholds,
           'mAP_bbox': {}, 'mAR_bbox': {}, 'mAP_seg': {}, 'mAR_seg': {},
           'per_class': {cls: {'mAP_bbox': {}, 'mAR_bbox': {},
                               'mAP_seg':  {}, 'mAR_seg':  {}} for cls in classes}}

    for thr in thresholds:
        bbox_g = {'tp': 0, 'fp': 0, 'fn': 0}
        seg_g  = {'tp': 0, 'fp': 0, 'fn': 0}
        bbox_c = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in classes}
        seg_c  = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in classes}

        for r in records:
            ano   = r['ano_map_norm']
            gt    = r['gt_mask'].astype(np.float32)
            if gt.shape != ano.shape:
                gt = cv.resize(gt, (ano.shape[1], ano.shape[0]), interpolation=cv.INTER_NEAREST)
            gt_bin   = (gt > 0.5).astype(np.uint8)
            pred_bin = (ano >= thr).astype(np.uint8)
            cls   = r['class_name']
            label = r['label']

            # bbox
            gt_boxes   = _boxes_from_binary(gt_bin)
            pred_boxes = _boxes_from_binary(pred_bin, max_dets=max_dets)
            if label == 0:
                tp_b, fp_b, fn_b = 0, len(pred_boxes), 0
            else:
                matched = set()
                tp_b = 0
                for pb in pred_boxes:
                    best_iou, best_j = 0.0, -1
                    for j, gb in enumerate(gt_boxes):
                        if j in matched:
                            continue
                        iou = _box_iou(pb, gb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou >= iou_thr and best_j >= 0:
                        tp_b += 1
                        matched.add(best_j)
                fp_b = len(pred_boxes) - tp_b
                fn_b = len(gt_boxes) - tp_b
            for d in (bbox_g, bbox_c[cls]):
                d['tp'] += tp_b; d['fp'] += fp_b; d['fn'] += fn_b

            # segmentation
            if has_masks:
                if label == 0:
                    tp_s, fp_s, fn_s = 0, int(pred_bin.sum()), 0
                else:
                    inter = int((pred_bin & gt_bin).sum())
                    tp_s  = inter
                    fp_s  = int(pred_bin.sum()) - inter
                    fn_s  = int(gt_bin.sum()) - inter
                for d in (seg_g, seg_c[cls]):
                    d['tp'] += tp_s; d['fp'] += fp_s; d['fn'] += fn_s

        def _pr(d):
            p = d['tp'] / (d['tp'] + d['fp'] + 1e-12)
            r = d['tp'] / (d['tp'] + d['fn'] + 1e-12)
            return float(p), float(r)

        gp_b, gr_b = _pr(bbox_g)
        out['mAP_bbox'][thr] = gp_b
        out['mAR_bbox'][thr] = gr_b
        if has_masks:
            gp_s, gr_s = _pr(seg_g)
        else:
            gp_s = gr_s = float('nan')
        out['mAP_seg'][thr]  = gp_s
        out['mAR_seg'][thr]  = gr_s

        for cls in classes:
            cp_b, cr_b = _pr(bbox_c[cls])
            out['per_class'][cls]['mAP_bbox'][thr] = cp_b
            out['per_class'][cls]['mAR_bbox'][thr] = cr_b
            if has_masks:
                cp_s, cr_s = _pr(seg_c[cls])
            else:
                cp_s = cr_s = float('nan')
            out['per_class'][cls]['mAP_seg'][thr]  = cp_s
            out['per_class'][cls]['mAR_seg'][thr]  = cr_s

    return out


def log_detection_metrics(det, print_fn):
    thrs = det['thresholds']
    has_seg = det.get('has_gt_masks', False)

    def _row(name, vals):
        cells = ['  N/A  ' if (v != v) else f'{v:.4f}' for v in vals]
        print_fn(f'    {name:<28}  ' + '  '.join(cells))

    print_fn('─' * 76)
    print_fn('Detection Metrics  (IoU thr=0.50,  confidence thresholds below)')
    print_fn('    {:28}  '.format('Metric') + '  '.join(f'@{t:.2f}' for t in thrs))
    _row('mAP@50  bbox (precision)',   [det['mAP_bbox'][t] for t in thrs])
    _row('mAR@100 bbox (recall)',      [det['mAR_bbox'][t] for t in thrs])
    if has_seg:
        _row('mAP@50  seg  (precision)', [det['mAP_seg'][t]  for t in thrs])
        _row('mAR@100 seg  (recall)',    [det['mAR_seg'][t]  for t in thrs])
    else:
        print_fn('    (segmentation metrics skipped — no GT pixel masks available)')
    print_fn('  Per-class breakdown:')
    for cls, cd in det.get('per_class', {}).items():
        print_fn(f'    [{cls}]')
        _row('  mAP@50  bbox',  [cd['mAP_bbox'][t] for t in thrs])
        _row('  mAR@100 bbox',  [cd['mAR_bbox'][t] for t in thrs])
        if has_seg:
            _row('  mAP@50  seg',   [cd['mAP_seg'][t]  for t in thrs])
            _row('  mAR@100 seg',   [cd['mAR_seg'][t]  for t in thrs])
    print_fn('─' * 76)


def save_metrics_csv(csv_path, det, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px,
                     timings_ms, gpu_mem_mb, gpu_util):
    thrs = det['thresholds']
    has_seg = det.get('has_gt_masks', False)

    def _s(lst):
        return (min(lst), sum(lst)/len(lst), max(lst)) if lst else (float('nan'),)*3

    t_min, t_mean, t_max = _s(timings_ms)
    m_min, m_mean, m_max = _s(gpu_mem_mb)
    u_min, u_mean, u_max = _s(gpu_util)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            base_cols = ['class', 'metric_type', 'metric_name']
            thr_cols  = [f'conf@{t:.2f}' for t in thrs]
            extra_cols = [
                'I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO',
                'infer_ms_min', 'infer_ms_mean', 'infer_ms_max',
                'gpu_mem_mb_min', 'gpu_mem_mb_mean', 'gpu_mem_mb_max',
                'gpu_util_pct_min', 'gpu_util_pct_mean', 'gpu_util_pct_max',
            ]
            writer.writerow(base_cols + thr_cols + extra_cols)

        def _nan(v):
            return '' if (isinstance(v, float) and math.isnan(v)) else v

        def _write(cls, mtype, mname, vals, extra=None):
            row = [cls, mtype, mname] + [_nan(v) for v in vals]
            if extra is None:
                extra = [''] * 16
            writer.writerow(row + extra)

        extra_first = [
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px,
            _nan(t_min), _nan(t_mean), _nan(t_max),
            _nan(m_min), _nan(m_mean), _nan(m_max),
            _nan(u_min), _nan(u_mean), _nan(u_max),
        ]

        _write('global', 'bbox', 'mAP@50',  [det['mAP_bbox'][t] for t in thrs], extra_first)
        _write('global', 'bbox', 'mAR@100', [det['mAR_bbox'][t] for t in thrs])
        if has_seg:
            _write('global', 'seg', 'mAP@50',  [det['mAP_seg'][t]  for t in thrs])
            _write('global', 'seg', 'mAR@100', [det['mAR_seg'][t]  for t in thrs])

        for cls, cd in det.get('per_class', {}).items():
            _write(cls, 'bbox', 'mAP@50',  [cd['mAP_bbox'][t] for t in thrs])
            _write(cls, 'bbox', 'mAR@100', [cd['mAR_bbox'][t] for t in thrs])
            if has_seg:
                _write(cls, 'seg', 'mAP@50',  [cd['mAP_seg'][t]  for t in thrs])
                _write(cls, 'seg', 'mAR@100', [cd['mAR_seg'][t]  for t in thrs])


def save_confusion_matrix(all_labels, all_preds_binary, save_dir, print_fn=None):
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(all_labels, all_preds_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[-1, -1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['OK', 'NG'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'confusion_matrix.png')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    total = len(all_labels)
    acc = (tn + tp) / max(total, 1)
    overkill_rate = fp / max(tn + fp, 1)
    escape_rate   = fn / max(tp + fn, 1)
    if print_fn:
        print_fn(f'Confusion Matrix: TP={tp} TN={tn} FP={fp} FN={fn} '
                 f'Acc={acc:.4f} Overkill={overkill_rate:.4f} Escape={escape_rate:.4f}')
        print_fn(f'Confusion matrix saved to: {path}')
    return path


def save_overkill_escape(per_image_records, save_root, print_fn=None):
    ok_dir  = os.path.join(save_root, 'overkill')
    esc_dir = os.path.join(save_root, 'escape')
    n_ok = n_esc = 0
    for r in per_image_records:
        gt, pred = r['gt_label'], r['pred']
        vis = r.get('vis_path')
        if vis is None or not os.path.exists(vis):
            continue
        if gt == 0 and pred == 1:  # overkill
            os.makedirs(ok_dir, exist_ok=True)
            dst = os.path.join(ok_dir, os.path.basename(vis))
            shutil.copy2(vis, dst)
            n_ok += 1
        elif gt == 1 and pred == 0:  # escape
            os.makedirs(esc_dir, exist_ok=True)
            dst = os.path.join(esc_dir, os.path.basename(vis))
            shutil.copy2(vis, dst)
            n_esc += 1
    if print_fn:
        print_fn(f'Overkill images saved ({n_ok}): {ok_dir}')
        print_fn(f'Escape   images saved ({n_esc}): {esc_dir}')


# ── Visualization ─────────────────────────────────────────────────────────────

def save_visualization(orig_rgb, ano_map, gt_mask, out_dir, idx_name,
                       bbox_threshold, pred_score, threshold, gt_label,
                       global_score_min, global_score_max):
    """Save visualization images: original, heatmap, GT, overlay, bbox+caption."""
    os.makedirs(out_dir, exist_ok=True)
    h, w = orig_rgb.shape[:2]

    # _0: original
    plt.imsave(os.path.join(out_dir, f'{idx_name}_0.png'), orig_rgb)

    # _1: per-image normalized jet heatmap
    ano_norm = (ano_map - ano_map.min()) / (ano_map.max() - ano_map.min() + 1e-8)
    plt.imsave(os.path.join(out_dir, f'{idx_name}_1.png'), ano_norm, cmap='jet')

    # _2: GT mask
    plt.imsave(os.path.join(out_dir, f'{idx_name}_2.png'), gt_mask, cmap='gray')

    # _3: heatmap overlay
    ano_resized = cv.resize(ano_norm, (w, h))
    heatmap_color = (plt.cm.jet(ano_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv.addWeighted(orig_rgb, 0.5, heatmap_color, 0.5, 0)
    plt.imsave(os.path.join(out_dir, f'{idx_name}_3.png'), overlay)

    # Globally-normalised anomaly map
    if global_score_min is not None and global_score_max is not None:
        _g_factor = (global_score_max - global_score_min) if global_score_max > global_score_min else 1.0
        ano_global = np.clip((cv.resize(ano_map, (w, h)) - global_score_min) / _g_factor, 0.0, 1.0)
    else:
        ano_global = ano_resized

    # _4: overlay + green bboxes + caption panel
    bbox_overlay = _draw_bboxes(overlay, ano_global, bbox_threshold, ano_score_map=ano_global)

    if pred_score is not None and threshold is not None:
        pred_lbl = 1 if pred_score >= threshold else 0
        gt_text   = 'GT: NG' if gt_label == 1 else ('GT: OK' if gt_label is not None else '')
        pred_text = 'Pred: NG' if pred_lbl == 1 else 'Pred: OK'
        pred_color = 'red' if (gt_label is not None and pred_lbl != gt_label) else 'green'

        dpi = 100
        scale = max(1.0, h / 256.0)
        text_px = int(max(40, 60 * scale))
        fig_w = (2 * w) / dpi
        fig_h = (h + text_px) / dpi
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = fig.add_gridspec(2, 2,
                              height_ratios=[text_px, h],
                              width_ratios=[1, 1],
                              hspace=0.0, wspace=0.02)

        ax_orig = fig.add_subplot(gs[1, 0])
        ax_overlay = fig.add_subplot(gs[1, 1])
        for ax in (ax_orig, ax_overlay):
            ax.set_axis_off()
        ax_orig.imshow(orig_rgb)
        ax_overlay.imshow(bbox_overlay)

        ax_text = fig.add_subplot(gs[0, 1])
        ax_text.set_axis_off()
        ax_text.set_facecolor('white')
        fontsize = max(5, 5 * scale)
        captions = [gt_text, pred_text, f'Score: {pred_score:.4f}', f'Thresh: {threshold:.4f}']
        colors   = ['black', pred_color, 'black', 'black']
        y_pos = 0.78
        for txt, col in zip(captions, colors):
            ax_text.text(0.02, y_pos, txt, fontsize=fontsize, fontweight='bold',
                         color=col, ha='left', va='center', transform=ax_text.transAxes)
            y_pos -= 0.20
        fig.add_subplot(gs[0, 0]).set_axis_off()

        fig.savefig(os.path.join(out_dir, f'{idx_name}_4.png'),
                    bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        plt.close(fig)
    else:
        plt.imsave(os.path.join(out_dir, f'{idx_name}_4.png'), bbox_overlay)
    plt.close('all')


# ── Core test function ────────────────────────────────────────────────────────

@torch.no_grad()
def test_comprehensive(model: CPR, train_fns, test_fns, retrieval_result,
                       foreground_result, resize, region_sizes, root_dir, knn, T,
                       sub_category='unknown',
                       fp16=False,
                       bbox_threshold=0.5,
                       threshold_mode='auto',
                       manual_threshold=0.5,
                       save_root=None,
                       save_correct=True,
                       save_incorrect=True,
                       print_fn=None,
                       csv_dir=None):
    """Extended test with comprehensive metrics, bbox, confusion matrix, CSV, FP16."""
    model.eval()
    
    _amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if fp16 else contextlib.nullcontext()
    if fp16 and print_fn:
        print_fn('[FP16] AMP float16 inference enabled')

    # Initialize pynvml for GPU utilization
    _pynvml_ok = False
    _nvml_h = None
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_h = pynvml.nvmlDeviceGetHandleByIndex(0)
        _pynvml_ok = True
    except Exception:
        pass

    with _amp_ctx:
        # Extract train features
        train_local_features = [
            torch.zeros((len(train_fns), out_channels, *shape[2:]), device='cuda')
            for shape, out_channels in zip(model.backbone.shapes, model.lrb.out_channels_list)
        ]
        train_foreground_weights = []
        k2id = {}
        for idx, image_fn in enumerate(tqdm(train_fns, desc='extract train local features', leave=False)):
            k = os.path.relpath(image_fn, root_dir)
            image = read_image(image_fn, (resize, resize))
            image_t = test_transform(image)
            features_list, ori_features_list = model(image_t[None].cuda())
            for i, features in enumerate(features_list):
                train_local_features[i][idx:idx+1] = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8)
            if k in foreground_result:
                train_foreground_weights.append(
                    torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda()
                )
            k2id[k] = idx
        if train_foreground_weights:
            train_foreground_weights = torch.stack(train_foreground_weights)

        # Predict test data
        gts = []
        i_gts = []
        preds_i = []
        preds_p = []
        image_paths = []
        anomaly_names = []
        _timings_ms = []
        _gpu_mem_mb = []
        _gpu_util = []
        _img_idx = 0

        for image_fn in tqdm(test_fns, desc='predict test data', leave=False):
            image = read_image(image_fn, (resize, resize))
            image_t = test_transform(image)
            k = os.path.relpath(image_fn, root_dir)
            image_name = os.path.basename(k)[:-4]
            anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
            mask_fn = os.path.join(root_dir, 'ground_truth', anomaly_name, image_name + '_mask.png')
            if os.path.exists(mask_fn):
                mask = read_mask(mask_fn, (resize, resize))
            else:
                mask = np.zeros((resize, resize))

            gts.append((mask > 127).astype(int))
            i_gts.append(int((mask > 127).sum() > 0))
            image_paths.append(image_fn)
            anomaly_names.append(anomaly_name)

            # Timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            features_list, ori_features_list = model(image_t[None].cuda())
            features_list = [features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8) for features in features_list]
            retrieval_idxs = [k2id[retrieval_k] for retrieval_k in retrieval_result[k][:knn]]
            retrieval_features_list = [train_local_features[i][retrieval_idxs] for i in range(len(features_list))]

            scores = []
            for features, retrieval_features, region_size in zip(features_list, retrieval_features_list, region_sizes):
                unfold = nn.Unfold(kernel_size=region_size, padding=region_size // 2)
                region_features = unfold(retrieval_features).reshape(
                    retrieval_features.shape[0], retrieval_features.shape[1], -1,
                    retrieval_features.shape[2], retrieval_features.shape[3]
                )
                dist = (1 - (features[:, :, None] * region_features).sum(1))
                dist = dist / (unfold(torch.ones(1, 1, retrieval_features.shape[2],
                                                 retrieval_features.shape[3],
                                                 device=retrieval_features.device)).reshape(
                    1, -1, retrieval_features.shape[2], retrieval_features.shape[3]) + 1e-8)
                score = dist.min(1)[0].min(0)[0]
                score = F.interpolate(
                    score[None, None],
                    size=(features_list[0].shape[2], features_list[0].shape[3]),
                    mode="bilinear", align_corners=False
                )[0, 0]
                scores.append(score)
            score = torch.stack(scores).sum(0)
            score = F.interpolate(
                score[None, None],
                size=(mask.shape[0], mask.shape[1]),
                mode="bilinear", align_corners=False
            )[0, 0]
            if k in foreground_result:
                foreground_weight = torch.from_numpy(
                    cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))
                ).cuda()
                foreground_weight = torch.cat([foreground_weight[None], train_foreground_weights[retrieval_idxs]]).max(0)[0]
                score = score * foreground_weight
            score_g = gaussian_blur(score[None], (33, 33), 4)[0]
            det_score = torch.topk(score_g.flatten(), k=T)[0].sum()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            preds_i.append(det_score)
            preds_p.append(score_g)

            # Collect timing/GPU stats after warmup
            if _img_idx >= _WARMUP_IMAGES:
                _timings_ms.append((t1 - t0) * 1000.0)
                if torch.cuda.is_available():
                    _gpu_mem_mb.append(torch.cuda.memory_allocated() / 1024 ** 2)
                if _pynvml_ok:
                    try:
                        _gpu_util.append(pynvml.nvmlDeviceGetUtilizationRates(_nvml_h).gpu)
                    except Exception:
                        pass
            _img_idx += 1

    # Compute original metrics
    gts_t = torch.from_numpy(np.stack(gts)).cuda()
    preds_p_t = torch.stack(preds_p)
    i_gts_t = torch.tensor(i_gts).long().cuda()
    preds_i_t = torch.stack(preds_i)

    pro = compute_pro_torch(gts_t, preds_p_t)
    ap = compute_ap_torch(gts_t, preds_p_t)
    pixel_auc = compute_pixel_auc_torch(gts_t, preds_p_t)
    image_auc = compute_image_auc_torch(i_gts_t, preds_i_t)

    # F1 scores
    try:
        precs, recs, thrs = precision_recall_curve(
            np.array(i_gts), preds_i_t.cpu().numpy()
        )
        f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-7)
        f1_sp = float(f1s.max()) if len(f1s) > 0 else 0.0
    except Exception:
        f1_sp = 0.0

    try:
        gts_flat = gts_t.cpu().numpy().ravel()
        preds_flat = preds_p_t.cpu().numpy().ravel()
        precs_px, recs_px, _ = precision_recall_curve(gts_flat, preds_flat)
        f1s_px = 2 * precs_px[:-1] * recs_px[:-1] / (precs_px[:-1] + recs_px[:-1] + 1e-7)
        f1_px = float(f1s_px.max()) if len(f1s_px) > 0 else 0.0
    except Exception:
        f1_px = 0.0

    ret = {
        'pro': pro, 'ap': ap, 'pixel-auc': pixel_auc, 'image-auc': image_auc,
    }

    if print_fn:
        print_fn(f'[{sub_category}] Core metrics: ' + ' '.join([f'{k}: {v:.4f}' for k, v in ret.items()]))
        print_fn(f'[{sub_category}] I-F1_max: {f1_sp:.4f}  P-F1_max: {f1_px:.4f}')

    # ── Pixel-level global bounds ─────────────────────────────────────────────
    preds_p_np = preds_p_t.cpu().numpy()
    preds_i_np = preds_i_t.cpu().numpy()
    _px_min = float(preds_p_np.min())
    _px_max = float(preds_p_np.max())
    _px_factor = (_px_max - _px_min) if _px_max > _px_min else 1.0

    # ── Classification threshold ──────────────────────────────────────────────
    if threshold_mode == 'manual':
        _best_thr_norm = float(manual_threshold)
    elif threshold_mode == 'bbox':
        _best_thr_norm = float(bbox_threshold)
    else:  # auto
        try:
            _prec, _rec, _thr_cands = precision_recall_curve(np.array(i_gts), preds_i_np)
            if len(_thr_cands) > 0:
                _f1s = 2 * _prec[:-1] * _rec[:-1] / (np.maximum(_prec[:-1] + _rec[:-1], 1e-8))
                _best_thr_raw = float(_thr_cands[np.argmax(_f1s)])
            else:
                _best_thr_raw = float(preds_i_np.min())
            _best_thr_norm = (_best_thr_raw - _px_min) / _px_factor
        except Exception:
            _best_thr_norm = 0.5

    if print_fn:
        print_fn(f'[threshold] mode={threshold_mode}  value={_best_thr_norm:.4f}')

    # ── Per-image analysis CSV ────────────────────────────────────────────────
    # Image score = highest pixel score, globally normalised
    img_scores_norm = np.array([(preds_p_np[i].max() - _px_min) / _px_factor for i in range(len(test_fns))])
    binary_preds = (img_scores_norm >= _best_thr_norm).astype(int)

    if csv_dir is not None:
        os.makedirs(csv_dir, exist_ok=True)
        try:
            pia_df = pd.DataFrame({
                'image_path': image_paths,
                'label': ['NG' if g == 1 else 'OK' for g in i_gts],
                'pred_score': img_scores_norm,
                'prediction': ['NG' if p == 1 else 'OK' for p in binary_preds],
                'anomaly_class': anomaly_names,
            })
            pia_df = pia_df.sort_values(by=['label', 'pred_score'], ascending=[True, False]).reset_index(drop=True)
            pia_csv = os.path.join(csv_dir, 'per_image_analysis.csv')
            pia_df.to_csv(pia_csv, index=False)
            if print_fn:
                print_fn(f'Per-image analysis saved to: {pia_csv}')
        except Exception as e:
            if print_fn:
                print_fn(f'WARNING: failed to write per_image_analysis.csv: {e}')

    # ── Detection records ─────────────────────────────────────────────────────
    _det_records = []
    for i in range(len(test_fns)):
        a = preds_p_np[i]
        a_norm = (a - a.min()) / (a.max() - a.min() + 1e-8)
        _det_records.append({
            'ano_map_norm': a_norm,
            'gt_mask': gts[i].astype(np.float32),
            'label': i_gts[i],
            'class_name': anomaly_names[i],
        })

    # ── Visualization ─────────────────────────────────────────────────────────
    _per_img_records = []
    if save_root is not None:
        cat_save_root = os.path.join(save_root, sub_category)
        for i in tqdm(range(len(test_fns)), desc='saving visualizations', leave=False):
            score_n = float(img_scores_norm[i])
            pred_lbl = int(binary_preds[i])
            gt_lbl = i_gts[i]

            # Skip correct predictions if save_correct is False
            if not save_correct and pred_lbl == gt_lbl:
                continue

            image = read_image(test_fns[i], (resize, resize))
            anomaly_name = anomaly_names[i]
            image_name = os.path.basename(test_fns[i])[:-4]
            out_dir = os.path.join(cat_save_root, anomaly_name)

            save_visualization(
                image, preds_p_np[i], gts[i].astype(np.float32),
                out_dir, image_name,
                bbox_threshold=_best_thr_norm,
                pred_score=score_n,
                threshold=_best_thr_norm,
                gt_label=gt_lbl,
                global_score_min=_px_min,
                global_score_max=_px_max,
            )

            vis_path = os.path.join(out_dir, f'{image_name}_4.png')
            _per_img_records.append({
                'img_path': test_fns[i],
                'gt_label': gt_lbl,
                'pred': pred_lbl,
                'vis_path': vis_path,
            })

        # Confusion matrix
        cm_dir = csv_dir if csv_dir is not None else cat_save_root
        save_confusion_matrix(i_gts, binary_preds.tolist(), cm_dir, print_fn)

        # Overkill/escape
        if save_incorrect and _per_img_records:
            save_overkill_escape(_per_img_records, cat_save_root, print_fn)

    # ── Detection metrics ─────────────────────────────────────────────────────
    if _det_records and print_fn:
        det = compute_detection_metrics(_det_records)
        log_detection_metrics(det, print_fn)

        if csv_dir is not None:
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, 'metrics.csv')
            save_metrics_csv(csv_path, det,
                             image_auc, ap, f1_sp, pixel_auc, ap, f1_px, pro,
                             _timings_ms, _gpu_mem_mb, _gpu_util)
            print_fn(f'Metrics CSV saved to: {csv_path}')

    # ── Timing & GPU stats ────────────────────────────────────────────────────
    if print_fn:
        def _stat(lst, unit=''):
            if not lst:
                return 'N/A (all warmup)'
            return f'min={min(lst):.1f}{unit}  mean={sum(lst)/len(lst):.1f}{unit}  max={max(lst):.1f}{unit}'

        warmup_note = f'(warmup={_WARMUP_IMAGES} images discarded)'
        print_fn(f'Inference time  {warmup_note}: {_stat(_timings_ms, " ms")}')
        if _gpu_mem_mb:
            print_fn(f'GPU memory      {warmup_note}: {_stat(_gpu_mem_mb, " MB")}')
        if _gpu_util:
            print_fn(f'GPU utilisation {warmup_note}: {_stat(_gpu_util, "%")}')

    return ret


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    if args.log_path:
        os.makedirs(args.log_path, exist_ok=True)
        logger.add(os.path.join(args.log_path, 'test_runtime.log'), mode='w')
    logger.info(f'run argv: {" ".join(sys.argv)}')
    logger.info('args: \n' + pformat(vars(args)))

    # Register custom dataset
    if args.dataset_name == 'custom':
        assert args.sub_categories is not None, "Must specify --sub-categories for custom dataset"
        register_custom_dataset(
            args.custom_data_dir or './data/custom',
            args.sub_categories,
            args.object_categories
        )

    all_categories, object_categories, texture_categories = DATASET_INFOS[args.dataset_name]
    sub_categories = all_categories if args.sub_categories is None else args.sub_categories
    assert all([sc in all_categories for sc in sub_categories])
    model_info = MODEL_INFOS[args.pretrained_model]
    layers = [model_info['layers'][model_info['scales'].index(scale)] for scale in args.scales]

    for sub_category_idx, sub_category in enumerate(sub_categories):
        fix_seeds(66)
        model = create_model(args.pretrained_model, layers).cuda()
        if args.checkpoints is not None:
            checkpoint_fn = args.checkpoints[0] if len(args.checkpoints) == 1 else args.checkpoints[sub_category_idx]
            if '{category}' in checkpoint_fn:
                checkpoint_fn = checkpoint_fn.format(category=sub_category)
            model.load_state_dict(torch.load(checkpoint_fn), strict=False)

        root_dir = os.path.join('./data', args.dataset_name, sub_category)
        train_fns = sorted(glob(os.path.join(root_dir, 'train/*/*')))
        test_fns = sorted(glob(os.path.join(root_dir, 'test/*/*')))
        with open(os.path.join(args.retrieval_dir, sub_category, 'r_result.json'), 'r') as f:
            retrieval_result = json.load(f)
        foreground_result = {}
        if args.foreground_dir is not None and sub_category in object_categories:
            for fn in train_fns + test_fns:
                k = os.path.relpath(fn, root_dir)
                foreground_result[k] = os.path.join(
                    args.foreground_dir, sub_category,
                    os.path.dirname(k), 'f_' + os.path.splitext(os.path.basename(k))[0] + '.npy'
                )

        csv_dir = os.path.join(args.log_path, sub_category) if args.log_path else None
        save_root = args.save_root

        logger.info(f'================={sub_category}=================')
        ret = test_comprehensive(
            model, train_fns, test_fns, retrieval_result, foreground_result,
            args.resize, args.region_sizes, root_dir, args.k_nearest, args.T,
            sub_category=sub_category,
            fp16=args.fp16,
            bbox_threshold=args.bbox_threshold,
            threshold_mode=args.threshold_mode,
            manual_threshold=args.manual_threshold,
            save_root=save_root,
            save_correct=args.save_correct,
            save_incorrect=args.save_incorrect,
            print_fn=lambda msg: logger.info(msg),
            csv_dir=csv_dir,
        )
        logger.info(f'{sub_category} result: {ret}')


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = f'log/test_{args.dataset_name}_{args.pretrained_model}'
    main(args)
