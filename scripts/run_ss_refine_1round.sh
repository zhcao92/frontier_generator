#!/bin/bash
# Self-supervised refinement: 1 round with finetune + full visualization
#
# This script:
# 1. Runs 1-round SS training with DETR finetune
# 2. Visualizes pseudo-labels (vis_pseudo_labels.py)
# 3. Visualizes batch_plot with the fine-tuned model

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (one level up from scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root to ensure correct imports
cd "$PROJECT_ROOT"

# Add project root to PYTHONPATH so imports work correctly
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

export TIAMAT_DATA_DIR=/scratch/mcity_project_root/mcity_project/shared_data/DARPA_TIAMAT/frontier_generator/set2

FRONTIER_ID=7000
ROUNDS=1
OUT_DIR=results/${FRONTIER_ID}_1round

echo "============================================"
echo "  SS Refinement: 1 Round + Visualization"
echo "============================================"
echo "Frontier ID: ${FRONTIER_ID}"
echo "Output dir:  ${OUT_DIR}"
echo ""

# ── Step 1: Self-supervised training (1 round) ───────────────────────────────
echo "[1/3] Running self-supervised training (1 round with finetune) ..."
python3 self_training/self_supervised_refine.py \
    --frontier-id ${FRONTIER_ID} \
    --detr-checkpoint /scratch/mcity_project_root/mcity_project/zhcao/DARPA_TIAMAT/detr_cache/model_best.pt \
    --gain-checkpoint /home/zhcao/CodesLib/TIAMAT/frontier_predictor/gain_cache/model_final.pt \
    --rounds ${ROUNDS} \
    --out-dir ${OUT_DIR}

echo ""
echo "✓ Training completed. Model saved to ${OUT_DIR}/detr_ss_round1.pt"
echo ""

# ── Step 2: Visualize pseudo-labels ──────────────────────────────────────────
echo "[2/3] Visualizing pseudo-labels ..."
python3 visualization/plot_refinement.py \
    --frontier-id ${FRONTIER_ID} \
    --json ${OUT_DIR}/refined_round1.json \
    --out-dir ${OUT_DIR}

echo ""
echo "✓ Pseudo-label visualizations saved to ${OUT_DIR}/"
echo ""

# ── Step 3: Batch plot with fine-tuned DETR model ────────────────────────────
echo "[3/3] Running batch plot with fine-tuned model ..."
python3 visualization/plot_predictions.py \
    --ids ${FRONTIER_ID} \
    --mode summary_map_gain_generation \
    --detr-checkpoint ${OUT_DIR}/detr_ss_round1.pt \
    --checkpoint /home/zhcao/CodesLib/TIAMAT/frontier_predictor/gain_cache/model_final.pt \
    --out-dir ${OUT_DIR}

echo ""
echo "✓ Batch plot saved to ${OUT_DIR}/"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "============================================"
echo "  All tasks completed!"
echo "============================================"
echo "Outputs:"
echo "  - Fine-tuned DETR model:  ${OUT_DIR}/detr_ss_round1.pt"
echo "  - Pseudo-label JSON:      ${OUT_DIR}/refined_round1.json"
echo "  - All visualizations:     ${OUT_DIR}/"
echo ""
