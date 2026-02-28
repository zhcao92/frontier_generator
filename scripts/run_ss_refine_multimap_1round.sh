#!/bin/bash
# Multi-map self-supervised refinement: 1 round with full visualization
#
# This script performs the complete multi-map training pipeline:
# 1. Generates refined frontiers from multiple maps (set-cover selected)
# 2. Trains DETR on all selected frontiers from all maps
# 3. Tests the trained model on target map
# 4. Visualizes results
# 5. Re-runs refinement with trained model for evaluation
# 6. Generates coverage heatmap
#
# Usage:
#   bash scripts/run_ss_refine_multimap_1round.sh
#   FRONTIER_ID=8000 bash scripts/run_ss_refine_multimap_1round.sh  # Test on different map

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Source path config (edit config/defaults.yaml for your machine)
source "${PROJECT_ROOT}/config/env.sh"

# Configuration
TRAINING_MAPS=(7000 7018 7022 7044 7047)  # Maps to collect training data from
TARGET_MAP=${FRONTIER_ID}  # Map to test on (from env.sh or override)
EPOCHS=50
ROUNDS=1
DATA_DIR=results/multimap_data
TRAINING_DIR=results/multimap_training
OUT_DIR=results/${TARGET_MAP}_multimap

echo "============================================"
echo "  Multi-Map SS Refinement: 1 Round"
echo "============================================"
echo "Training maps:   ${TRAINING_MAPS[@]}"
echo "Target map:      ${TARGET_MAP}"
echo "Training epochs: ${EPOCHS}"
echo "Data dir:        ${DATA_DIR}"
echo "Model output:    ${TRAINING_DIR}"
echo "Test output:     ${OUT_DIR}"
echo ""

# ── Step 1: Generate refined frontiers from multiple maps ────────────────────
echo "[1/6] Generating high-quality training data from ${#TRAINING_MAPS[@]} maps ..."
echo ""

for fid in "${TRAINING_MAPS[@]}"; do
    echo ">>> Processing map $fid ..."

    MAP_OUT_DIR="${DATA_DIR}/${fid}"

    python3 self_training/self_supervised_refine.py \
        --frontier-id "$fid" \
        --detr-checkpoint "${DETR_CHECKPOINT}" \
        --gain-checkpoint "${GAIN_CHECKPOINT}" \
        --rounds 1 \
        --out-dir "${MAP_OUT_DIR}" \
        --no-finetune

    echo "    Generating visualization for map $fid ..."
    python3 visualization/plot_refinement.py \
        --frontier-id "$fid" \
        --json "${MAP_OUT_DIR}/refined_round1.json" \
        --out-dir "${MAP_OUT_DIR}"

    echo "    Saved to ${MAP_OUT_DIR}/refined_round1.json and .png"
    echo ""
done

echo "Data generation complete."
echo ""

# ── Step 2: Train DETR on multi-map data ──────────────────────────────────────
echo "[2/6] Training DETR on multi-map data (${EPOCHS} epochs) ..."
python3 self_training/multi_map_finetune.py \
    --frontier-ids "${TRAINING_MAPS[@]}" \
    --data-dir "${DATA_DIR}" \
    --detr-checkpoint "${DETR_CHECKPOINT}" \
    --gain-checkpoint "${GAIN_CHECKPOINT}" \
    --out-dir "${TRAINING_DIR}" \
    --epochs ${EPOCHS}

echo ""
echo "Training completed. Model saved to ${TRAINING_DIR}/detr_multi_map.pt"
echo ""

# ── Step 3: Test on target map (no finetune) ──────────────────────────────────
echo "[3/6] Testing trained model on map ${TARGET_MAP} ..."
python3 self_training/self_supervised_refine.py \
    --frontier-id "${TARGET_MAP}" \
    --detr-checkpoint "${TRAINING_DIR}/detr_multi_map.pt" \
    --gain-checkpoint "${GAIN_CHECKPOINT}" \
    --rounds 1 \
    --out-dir "${OUT_DIR}" \
    --no-finetune

echo ""
echo "Testing complete. Results saved to ${OUT_DIR}/"
echo ""

# ── Step 4: Batch plot with trained model ─────────────────────────────────────
echo "[4/6] Generating summary map with trained model ..."
python3 visualization/plot_predictions.py \
    --ids "${TARGET_MAP}" \
    --mode summary_map_gain_generation \
    --detr-checkpoint "${TRAINING_DIR}/detr_multi_map.pt" \
    --checkpoint "${GAIN_CHECKPOINT}" \
    --out-dir "${OUT_DIR}"

echo ""
echo "Summary map saved to ${OUT_DIR}/summary_map_gain_generation.png"
echo ""

# ── Step 5: Re-run refinement with trained DETR for evaluation ───────────────
FT_EVAL_DIR="${OUT_DIR}/finetuned_eval"
echo "[5/6] Re-running refinement with trained DETR (generates A0 frontiers) ..."
python3 self_training/self_supervised_refine.py \
    --frontier-id "${TARGET_MAP}" \
    --detr-checkpoint "${TRAINING_DIR}/detr_multi_map.pt" \
    --gain-checkpoint "${GAIN_CHECKPOINT}" \
    --rounds 1 \
    --out-dir "${FT_EVAL_DIR}" \
    --no-finetune

echo ""
echo "Finetuned evaluation saved to ${FT_EVAL_DIR}/"
echo ""

# ── Step 6: Coverage heatmap (trained DETR A0 predictions) ────────────────────
echo "[6/6] Generating coverage heatmap (trained DETR A0 frontiers) ..."
python3 visualization/plot_coverage_heatmap.py \
    --frontier-id "${TARGET_MAP}" \
    --a0-json "${FT_EVAL_DIR}/a0_frontiers_round1.json" \
    --evidence-pkl "${FT_EVAL_DIR}/a0_mean_evidence_round1.pkl" \
    --out-dir "${FT_EVAL_DIR}"

echo ""
echo "Coverage heatmap saved to ${FT_EVAL_DIR}/"
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "============================================"
echo "  Multi-Map Training Complete!"
echo "============================================"
echo "Training data (${#TRAINING_MAPS[@]} maps) with visualizations:"
for fid in "${TRAINING_MAPS[@]}"; do
    echo "  - Map ${fid}:"
    echo "      ${DATA_DIR}/${fid}/refined_round1.json"
    echo "      ${DATA_DIR}/${fid}/refined_round1.png"
done
echo ""
echo "Model outputs:"
echo "  - Trained DETR model:     ${TRAINING_DIR}/detr_multi_map.pt"
echo "  - Training metadata:      ${TRAINING_DIR}/training_meta.json"
echo ""
echo "Test results on map ${TARGET_MAP}:"
echo "  - Refined frontiers:      ${OUT_DIR}/refined_round1.json"
echo "  - Summary map:            ${OUT_DIR}/summary_map_gain_generation.png"
echo "  - Trained DETR A0 eval:   ${FT_EVAL_DIR}/"
echo "  - Coverage heatmap (A0):  ${FT_EVAL_DIR}/a0_frontiers_round1_coverage.png"
echo ""
echo "To test on a different map, run:"
echo "  FRONTIER_ID=8000 bash scripts/run_ss_refine_multimap_1round.sh"
echo ""
