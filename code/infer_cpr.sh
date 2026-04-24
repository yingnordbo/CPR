#!/bin/bash
# ============================================================================
# CPR Inference Pipeline
# ============================================================================
# Usage (from any directory):
#   bash /path/to/code/infer_cpr.sh
#
# This script runs CPR inference on unlabeled images.
# It requires a trained model checkpoint and the retrieval/foreground
# preprocessing results from run_cpr.sh.
# ============================================================================

set -e

# Resolve the CPR source directory relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPR_DIR="${SCRIPT_DIR}/CPR"

# Change to CPR directory so all relative paths (data/, log/, result/) resolve correctly
cd "${CPR_DIR}"

# ── Dataset Configuration ─────────────────────────────────────────────────────
DATASET_NAME="custom"

# Sub-categories
SUB_CATEGORIES=("my_product")

# Object categories
OBJECT_CATEGORIES=("my_product")

# Custom data directory (must contain train images for feature bank)
CUSTOM_DATA_DIR="./data/custom"

# ── Model Configuration ──────────────────────────────────────────────────────
PRETRAINED_MODEL="DenseNet"
SCALES="4 8"
REGION_SIZES="3 1"
K_NEAREST=10
T=512

# ── Image Configuration ──────────────────────────────────────────────────────
RESIZE=320

# ── Inference Configuration ──────────────────────────────────────────────────
# Directory of images to run inference on
INFER_DIR="./data/infer_images"

# Checkpoint path (use {category} placeholder for multi-category)
CHECKPOINT="log/${DATASET_NAME}_train/{category}/40000.pth"

# Score threshold for OK/NG classification
SCORE_THRESHOLD=0.5

# BBox threshold
BBOX_THRESHOLD=0.5

# FP16 inference
FP16="--fp16"                      # set to "" to disable

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_LAYER="features.denseblock1"
RETRIEVAL_DIR="log/retrieval_${DATASET_NAME}_${PRETRAINED_MODEL}_${FEATURE_LAYER}_${RESIZE}"
FOREGROUND_DIR="log/foreground_${DATASET_NAME}_${PRETRAINED_MODEL}_${FEATURE_LAYER}_${RESIZE}"
SAVE_ROOT="result/${DATASET_NAME}_infer"
LOG_PATH="log/infer_${DATASET_NAME}"

# ============================================================================
# Run Inference
# ============================================================================

echo "======================================================================"
echo "CPR Inference — ${DATASET_NAME} — ${SUB_CATEGORIES[*]}"
echo "  Input: ${INFER_DIR}"
echo "  Output: ${SAVE_ROOT}"
echo "======================================================================"

SUB_CAT_ARGS=""
for cat in "${SUB_CATEGORIES[@]}"; do
    SUB_CAT_ARGS="${SUB_CAT_ARGS} ${cat}"
done

OBJ_CAT_ARGS=""
for cat in "${OBJECT_CATEGORIES[@]}"; do
    OBJ_CAT_ARGS="${OBJ_CAT_ARGS} ${cat}"
done

CUSTOM_ARGS=""
if [ "${DATASET_NAME}" = "custom" ]; then
    CUSTOM_ARGS="--custom-data-dir ${CUSTOM_DATA_DIR} --object-categories ${OBJ_CAT_ARGS}"
fi

FG_ARG=""
if [ -d "${FOREGROUND_DIR}" ] && [ ${#OBJECT_CATEGORIES[@]} -gt 0 ]; then
    FG_ARG="-fd ${FOREGROUND_DIR}"
fi

python "${SCRIPT_DIR}/infer_cpr.py" \
    -dn "${DATASET_NAME}" \
    --sub-categories ${SUB_CAT_ARGS} \
    -pm "${PRETRAINED_MODEL}" \
    -ss ${SCALES} \
    -rs ${REGION_SIZES} \
    -kn "${K_NEAREST}" \
    -r "${RESIZE}" \
    --T "${T}" \
    -rd "${RETRIEVAL_DIR}" \
    ${FG_ARG} \
    --checkpoints "${CHECKPOINT}" \
    --infer-dir "${INFER_DIR}" \
    --bbox-threshold "${BBOX_THRESHOLD}" \
    --score-threshold "${SCORE_THRESHOLD}" \
    ${FP16} \
    ${CUSTOM_ARGS} \
    --save-root "${SAVE_ROOT}" \
    -lp "${LOG_PATH}"

echo ""
echo "======================================================================"
echo "Inference complete!"
echo "  Results: ${SAVE_ROOT}"
echo "  Logs:    ${LOG_PATH}"
echo "======================================================================"
