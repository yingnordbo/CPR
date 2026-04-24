"""
CPR inference script for running on unlabeled images (no GT required).
Produces anomaly heatmaps, bbox overlays, and per-image scores.
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
import sys
import time
from collections import defaultdict
from glob import glob
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
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from dataset import (
    DATASET_INFOS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    read_image,
    register_custom_dataset,
    test_transform,
)
from models import CPR, MODEL_INFOS, create_model
from utils import fix_seeds


def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("-dn", "--dataset-name", type=str, default="custom",
                        choices=["mvtec", "mvtec_3d", "btad", "custom"])
    parser.add_argument("-ss", "--scales", type=int, nargs="+", default=[4, 8])
    parser.add_argument("-kn", "--k-nearest", type=int, default=10)
    parser.add_argument("-r", "--resize", type=int, default=320)
    parser.add_argument("-fd", "--foreground-dir", type=str, default=None)
    parser.add_argument("-rd", "--retrieval-dir", type=str, default=None)
    parser.add_argument("--sub-categories", type=str, nargs="+", default=None)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("-rs", "--region-sizes", type=int, nargs="+", default=[3, 1])
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet',
                        choices=list(MODEL_INFOS.keys()))
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    # custom dataset
    parser.add_argument("--custom-data-dir", type=str, default=None)
    parser.add_argument("--object-categories", type=str, nargs="+", default=None)
    # inference input
    parser.add_argument("--infer-dir", type=str, required=True,
                        help="directory of images to run inference on")
    parser.add_argument("--train-dir", type=str, default=None,
                        help="directory of train images (for feature extraction). "
                             "If None, uses data/<dataset>/<category>/train/")
    # output
    parser.add_argument("-lp", "--log-path", type=str, default=None)
    parser.add_argument("--save-root", type=str, default='result/infer', help="save results root")
    # bbox
    parser.add_argument("--bbox-threshold", type=float, default=0.5)
    # inference
    parser.add_argument("--fp16", action="store_true", help="use FP16 inference")
    # classification threshold
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="image-level score threshold for OK/NG classification")
    return parser


def denormalize(img_tensor):
    std = np.array(IMAGENET_STD)
    mean = np.array(IMAGENET_MEAN)
    x = (((img_tensor.cpu().numpy().transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


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


@torch.no_grad()
def infer(model: CPR, train_fns, infer_fns, retrieval_result, foreground_result,
          resize, region_sizes, root_dir, knn, T,
          fp16=False, bbox_threshold=0.5, score_threshold=0.5, save_root=None):
    """Run inference on unlabeled images."""
    model.eval()
    _amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if fp16 else contextlib.nullcontext()

    with _amp_ctx:
        # Extract train features
        train_local_features = [
            torch.zeros((len(train_fns), out_channels, *shape[2:]), device='cuda')
            for shape, out_channels in zip(model.backbone.shapes, model.lrb.out_channels_list)
        ]
        train_foreground_weights = []
        k2id = {}
        for idx, image_fn in enumerate(tqdm(train_fns, desc='extract train features', leave=False)):
            k = os.path.relpath(image_fn, root_dir)
            image = read_image(image_fn, (resize, resize))
            image_t = test_transform(image)
            features_list, _ = model(image_t[None].cuda())
            for i, features in enumerate(features_list):
                train_local_features[i][idx:idx+1] = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8)
            if k in foreground_result:
                train_foreground_weights.append(
                    torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda()
                )
            k2id[k] = idx
        if train_foreground_weights:
            train_foreground_weights = torch.stack(train_foreground_weights)

        # For inference images, we need to find nearest train images for retrieval
        # We'll use all retrieval results if available, otherwise use first K train images
        results = []
        _timings_ms = []
        _WARMUP = 5

        for img_idx, image_fn in enumerate(tqdm(infer_fns, desc='inference', leave=False)):
            image = read_image(image_fn, (resize, resize))
            image_t = test_transform(image)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            features_list, _ = model(image_t[None].cuda())
            features_list = [features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8) for features in features_list]

            # Use first knn train images as retrieval (for inference without retrieval precomputed)
            retrieval_idxs = list(range(min(knn, len(train_fns))))

            scores = []
            for features, region_size in zip(features_list, region_sizes):
                retrieval_features = train_local_features[features_list.index(features)][retrieval_idxs]
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
                score[None, None], size=(resize, resize),
                mode="bilinear", align_corners=False
            )[0, 0]
            score_g = gaussian_blur(score[None], (33, 33), 4)[0]
            det_score = score_g.max().item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if img_idx >= _WARMUP:
                _timings_ms.append((t1 - t0) * 1000.0)

            score_np = score_g.cpu().numpy()
            results.append({
                'image_fn': image_fn,
                'score': det_score,
                'anomaly_map': score_np,
            })

    # Global normalization
    all_scores = np.array([r['score'] for r in results])
    _min = min(r['anomaly_map'].min() for r in results)
    _max = max(r['anomaly_map'].max() for r in results)
    _factor = (_max - _min) if _max > _min else 1.0

    # Save results
    if save_root is not None:
        os.makedirs(save_root, exist_ok=True)
        infer_records = []

        for r in tqdm(results, desc='saving results', leave=False):
            image_fn = r['image_fn']
            image = read_image(image_fn, (resize, resize))
            score_np = r['anomaly_map']
            score_norm = (r['score'] - _min) / _factor
            image_name = os.path.splitext(os.path.basename(image_fn))[0]
            pred_label = 'NG' if score_norm >= score_threshold else 'OK'

            out_dir = os.path.join(save_root, pred_label)
            os.makedirs(out_dir, exist_ok=True)

            h, w = image.shape[:2]

            # Original
            plt.imsave(os.path.join(out_dir, f'{image_name}_0.png'), image)

            # Heatmap
            ano_norm = (score_np - score_np.min()) / (score_np.max() - score_np.min() + 1e-8)
            plt.imsave(os.path.join(out_dir, f'{image_name}_1.png'), ano_norm, cmap='jet')

            # Overlay
            ano_resized = cv.resize(ano_norm, (w, h))
            heatmap_color = (plt.cm.jet(ano_resized)[:, :, :3] * 255).astype(np.uint8)
            overlay = cv.addWeighted(image, 0.5, heatmap_color, 0.5, 0)
            plt.imsave(os.path.join(out_dir, f'{image_name}_3.png'), overlay)

            # Global-norm map for bbox
            ano_global = np.clip((cv.resize(score_np, (w, h)) - _min) / _factor, 0.0, 1.0)

            # Bbox overlay
            bbox_overlay = _draw_bboxes(overlay, ano_global, bbox_threshold, ano_score_map=ano_global)

            # Caption panel
            dpi = 100
            scale = max(1.0, h / 256.0)
            text_px = int(max(40, 60 * scale))
            fig_w = (2 * w) / dpi
            fig_h = (h + text_px) / dpi
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            gs = fig.add_gridspec(2, 2, height_ratios=[text_px, h], width_ratios=[1, 1],
                                  hspace=0.0, wspace=0.02)
            ax_orig = fig.add_subplot(gs[1, 0])
            ax_overlay = fig.add_subplot(gs[1, 1])
            for ax in (ax_orig, ax_overlay):
                ax.set_axis_off()
            ax_orig.imshow(image)
            ax_overlay.imshow(bbox_overlay)

            ax_text = fig.add_subplot(gs[0, 1])
            ax_text.set_axis_off()
            fontsize = max(5, 5 * scale)
            pred_color = 'red' if pred_label == 'NG' else 'green'
            captions = [f'Pred: {pred_label}', f'Score: {score_norm:.4f}', f'Thresh: {score_threshold:.4f}']
            colors = [pred_color, 'black', 'black']
            y_pos = 0.78
            for txt, col in zip(captions, colors):
                ax_text.text(0.02, y_pos, txt, fontsize=fontsize, fontweight='bold',
                             color=col, ha='left', va='center', transform=ax_text.transAxes)
                y_pos -= 0.25
            fig.add_subplot(gs[0, 0]).set_axis_off()
            fig.savefig(os.path.join(out_dir, f'{image_name}_4.png'),
                        bbox_inches='tight', pad_inches=0.0, dpi=dpi)
            plt.close(fig)
            plt.close('all')

            infer_records.append({
                'image_path': image_fn,
                'pred_score': score_norm,
                'prediction': pred_label,
            })

        # Save CSV
        df = pd.DataFrame(infer_records)
        df = df.sort_values('pred_score', ascending=False).reset_index(drop=True)
        df.to_csv(os.path.join(save_root, 'per_image_analysis.csv'), index=False)

    # Print timing
    if _timings_ms:
        logger.info(f'Inference time (warmup={_WARMUP}): '
                     f'min={min(_timings_ms):.1f}ms  '
                     f'mean={sum(_timings_ms)/len(_timings_ms):.1f}ms  '
                     f'max={max(_timings_ms):.1f}ms')

    return results


def main(args):
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    if args.log_path:
        os.makedirs(args.log_path, exist_ok=True)
        logger.add(os.path.join(args.log_path, 'infer_runtime.log'), mode='w')
    logger.info(f'run argv: {" ".join(sys.argv)}')
    logger.info('args: \n' + pformat(vars(args)))

    if args.dataset_name == 'custom':
        assert args.sub_categories is not None
        register_custom_dataset(
            args.custom_data_dir or './data/custom',
            args.sub_categories,
            args.object_categories
        )

    all_categories, object_categories, texture_categories = DATASET_INFOS[args.dataset_name]
    sub_categories = all_categories if args.sub_categories is None else args.sub_categories
    model_info = MODEL_INFOS[args.pretrained_model]
    layers = [model_info['layers'][model_info['scales'].index(scale)] for scale in args.scales]

    for sub_category_idx, sub_category in enumerate(sub_categories):
        fix_seeds(66)
        model = create_model(args.pretrained_model, layers).cuda()
        checkpoint_fn = args.checkpoints[0] if len(args.checkpoints) == 1 else args.checkpoints[sub_category_idx]
        if '{category}' in checkpoint_fn:
            checkpoint_fn = checkpoint_fn.format(category=sub_category)
        model.load_state_dict(torch.load(checkpoint_fn), strict=False)

        root_dir = os.path.join('./data', args.dataset_name, sub_category)
        train_dir = args.train_dir or os.path.join(root_dir, 'train')
        train_fns = sorted(glob(os.path.join(train_dir, '*/*'))) or sorted(glob(os.path.join(train_dir, '*')))

        # Collect inference images
        infer_fns = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.JPG', '*.PNG']:
            infer_fns.extend(glob(os.path.join(args.infer_dir, '**', ext), recursive=True))
        infer_fns = sorted(set(infer_fns))
        logger.info(f'{sub_category}: {len(infer_fns)} inference images, {len(train_fns)} train images')

        # Load retrieval result if available
        retrieval_result = {}
        if args.retrieval_dir:
            r_path = os.path.join(args.retrieval_dir, sub_category, 'r_result.json')
            if os.path.exists(r_path):
                with open(r_path, 'r') as f:
                    retrieval_result = json.load(f)

        foreground_result = {}
        if args.foreground_dir and sub_category in object_categories:
            for fn in train_fns:
                k = os.path.relpath(fn, root_dir)
                foreground_result[k] = os.path.join(
                    args.foreground_dir, sub_category,
                    os.path.dirname(k), 'f_' + os.path.splitext(os.path.basename(k))[0] + '.npy'
                )

        save_root = os.path.join(args.save_root, sub_category) if args.save_root else None

        infer(model, train_fns, infer_fns, retrieval_result, foreground_result,
              args.resize, args.region_sizes, root_dir, args.k_nearest, args.T,
              fp16=args.fp16, bbox_threshold=args.bbox_threshold,
              score_threshold=args.score_threshold, save_root=save_root)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = f'log/infer_{args.dataset_name}_{args.pretrained_model}'
    main(args)
