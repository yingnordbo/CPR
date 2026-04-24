#!/bin/bash
# ============================================================================
# CPR Training + Testing Pipeline
# ============================================================================
# Usage (from any directory):
#   bash /path/to/code/run_cpr.sh
#
# This script runs the full CPR pipeline:
#   1. Generate retrieval results (Global Retrieval Branch)
#   2. Generate foreground masks (Foreground Estimation Branch) [optional]
#   3. Generate synthetic anomaly data
#   4. Train CPR model
#   5. Test with comprehensive metrics
# ============================================================================

set -e

# Resolve the CPR source directory relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPR_DIR="${SCRIPT_DIR}/CPR"

# Change to CPR directory so all relative paths (data/, log/, result/) resolve correctly
cd "${CPR_DIR}"

# ── Dataset Configuration ─────────────────────────────────────────────────────
# For MVTec: "mvtec", for custom AOI: "custom"
DATASET_NAME="custom"

# Sub-categories to train/test (comma or space separated)
# For MVTec examples: "bottle cable capsule"
# For custom AOI: list your category names
SUB_CATEGORIES=("my_product")

# Object categories (for foreground estimation). Categories NOT listed here
# are treated as textures. Set empty array () if all are textures.
OBJECT_CATEGORIES=("my_product")

# Custom data directory (only used when DATASET_NAME=custom)
# Should contain: <category>/train/good/, <category>/test/good/, <category>/test/<defect_type>/
# and optionally: <category>/ground_truth/<defect_type>/
CUSTOM_DATA_DIR="./data/custom"

# ── Model Configuration ──────────────────────────────────────────────────────
PRETRAINED_MODEL="DenseNet"        # DenseNet | EfficientNet | ResNet
SCALES="4 8"                       # multiscale feature levels
REGION_SIZES="3 1"                 # local retrieval region sizes
OUTPUT_DIM=384                     # LRB output dimension
K_NEAREST=10                       # k-nearest for retrieval
T=512                              # top-k pixels for image-level score

# ── Image Configuration ──────────────────────────────────────────────────────
RESIZE=320                         # image resize for training/testing
SYNTHETIC_RESIZE=640               # image resize for synthetic generation

# ── Training Configuration ───────────────────────────────────────────────────
BATCH_SIZE=32
LEARNING_RATE=1e-3
STEPS=40000                        # total training steps (40000 for objects, 500 for textures)
TEST_PER_STEPS=2000                # test every N steps
NUM_WORKERS=8

# ── Testing Configuration ────────────────────────────────────────────────────
BBOX_THRESHOLD=0.5                 # threshold for drawing bounding boxes
THRESHOLD_MODE="auto"              # auto | manual | bbox
MANUAL_THRESHOLD=0.5               # used when THRESHOLD_MODE=manual
FP16="--fp16"                      # set to "" to disable FP16

# ── Saving Options ───────────────────────────────────────────────────────────
SAVE_CORRECT="--save-correct"      # save visualizations for correct predictions
SAVE_INCORRECT="--save-incorrect"  # save overkill/escape images
# Set to "--no-save-correct" or "--no-save-incorrect" to disable

# ── Synthetic Data Configuration ─────────────────────────────────────────────
SYNTHETIC_NUM=12000                # number of synthetic anomaly samples per category
SYNTHETIC_WORKERS=8

# ── Paths ─────────────────────────────────────────────────────────────────────
# Feature layer for retrieval/foreground (depends on PRETRAINED_MODEL)
# DenseNet: features.denseblock1, EfficientNet: features.2, ResNet: layer1
FEATURE_LAYER="features.denseblock1"

# Auto-generated paths (can be overridden)
RETRIEVAL_DIR="log/retrieval_${DATASET_NAME}_${PRETRAINED_MODEL}_${FEATURE_LAYER}_${RESIZE}"
FOREGROUND_DIR="log/foreground_${DATASET_NAME}_${PRETRAINED_MODEL}_${FEATURE_LAYER}_${RESIZE}"
SYNTHETIC_DIR="log/synthetic_${DATASET_NAME}_${SYNTHETIC_RESIZE}_${SYNTHETIC_NUM}_True_jpg"
LOG_PATH="log/${DATASET_NAME}_train"
TEST_LOG_PATH="log/${DATASET_NAME}_test"
SAVE_ROOT="result/${DATASET_NAME}_test"

# ── DTD texture dataset for synthetic anomaly generation ─────────────────────
# Download from: https://www.robots.ox.ac.uk/~vgg/data/dtd/
# Extract to: ./data/dtd/
DTD_PATH="./data/dtd"

# ============================================================================
# Pipeline Execution
# ============================================================================

echo "======================================================================"
echo "CPR Pipeline — ${DATASET_NAME} — ${SUB_CATEGORIES[*]}"
echo "======================================================================"

# Build sub-categories args
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

# ── Step 1: Generate Retrieval Results ────────────────────────────────────────
echo ""
echo "[Step 1/5] Generating retrieval results..."
python tools/generate_retrieval.py \
    --dataset-name "${DATASET_NAME}" \
    --resize "${RESIZE}" \
    -pm "${PRETRAINED_MODEL}" \
    --layer "${FEATURE_LAYER}" \
    -k "${K_NEAREST}" \
    --sub-categories ${SUB_CAT_ARGS} \
    ${CUSTOM_ARGS:+--object-categories ${OBJ_CAT_ARGS}} \
    -lp "${RETRIEVAL_DIR}"

# ── Step 2: Generate Foreground Masks (optional, for object categories) ──────
if [ ${#OBJECT_CATEGORIES[@]} -gt 0 ]; then
    echo ""
    echo "[Step 2/5] Generating foreground masks..."
    python tools/generate_foreground.py \
        --dataset-name "${DATASET_NAME}" \
        --resize "${RESIZE}" \
        -pm "${PRETRAINED_MODEL}" \
        --layer "${FEATURE_LAYER}" \
        --sub-categories ${SUB_CAT_ARGS} \
        ${CUSTOM_ARGS:+--object-categories ${OBJ_CAT_ARGS}} \
        -lp "${FOREGROUND_DIR}"
    FG_ARG="-fd ${FOREGROUND_DIR}"
else
    echo ""
    echo "[Step 2/5] Skipping foreground generation (no object categories)."
    FG_ARG=""
    FOREGROUND_DIR=""
fi

# ── Step 3: Generate Synthetic Anomaly Data ──────────────────────────────────
echo ""
echo "[Step 3/5] Generating synthetic anomaly data..."
python tools/generate_synthetic_data.py \
    --dataset-name "${DATASET_NAME}" \
    --resize "${SYNTHETIC_RESIZE}" \
    --num "${SYNTHETIC_NUM}" \
    --num-workers "${SYNTHETIC_WORKERS}" \
    --sub-categories ${SUB_CAT_ARGS} \
    ${CUSTOM_ARGS:+--object-categories ${OBJ_CAT_ARGS}} \
    ${FG_ARG} \
    -lp "${SYNTHETIC_DIR}"

# ── Step 4: Train ────────────────────────────────────────────────────────────
echo ""
echo "[Step 4/5] Training CPR model..."
python train.py \
    -dn "${DATASET_NAME}" \
    --sub-categories ${SUB_CAT_ARGS} \
    -pm "${PRETRAINED_MODEL}" \
    -ss ${SCALES} \
    -rs ${REGION_SIZES} \
    -kn "${K_NEAREST}" \
    -r "${RESIZE}" \
    --T "${T}" \
    -bs "${BATCH_SIZE}" \
    -lr "${LEARNING_RATE}" \
    --steps "${STEPS}" \
    -tps "${TEST_PER_STEPS}" \
    --num-workers "${NUM_WORKERS}" \
    -rd "${RETRIEVAL_DIR}" \
    ${FG_ARG} \
    --data-dir "${SYNTHETIC_DIR}" \
    ${CUSTOM_ARGS} \
    -lp "${LOG_PATH}"

# ── Step 5: Comprehensive Test ───────────────────────────────────────────────
echo ""
echo "[Step 5/5] Running comprehensive test..."
# Build checkpoint path (use last checkpoint)
CHECKPOINT_PATTERN="${LOG_PATH}/{category}/${STEPS:0:5}.pth"

python "${SCRIPT_DIR}/test_cpr.py" \
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
    --checkpoints "${LOG_PATH}/{category}/$(printf '%05d' ${STEPS}).pth" \
    --bbox-threshold "${BBOX_THRESHOLD}" \
    --threshold-mode "${THRESHOLD_MODE}" \
    --manual-threshold "${MANUAL_THRESHOLD}" \
    ${FP16} \
    ${SAVE_CORRECT} \
    ${SAVE_INCORRECT} \
    ${CUSTOM_ARGS} \
    --save-root "${SAVE_ROOT}" \
    -lp "${TEST_LOG_PATH}"

echo ""
echo "======================================================================"
echo "Pipeline complete!"
echo "  Training logs: ${LOG_PATH}"
echo "  Test results:  ${TEST_LOG_PATH}"
echo "  Visualizations: ${SAVE_ROOT}"
echo "======================================================================"
