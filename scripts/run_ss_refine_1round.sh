#!/bin/bash
# Self-supervised refinement: 1 round with finetune + full visualization
#
# This script:
# 1. Runs 1-round SS training with DETR finetune
# 2. Visualizes pseudo-labels
# 3. Batch plot with the fine-tuned model
# 4. Re-runs refinement with finetuned DETR to evaluate improvement
# 5. Visualizes coverage heatmap from finetuned evaluation

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Source path config (edit config/defaults.yaml for your machine)
source "${PROJECT_ROOT}/config/env.sh"

ROUNDS=1
OUT_DIR=results/${FRONTIER_ID}_1round

echo "============================================"
echo "  SS Refinement: 1 Round + Visualization"
echo "============================================"
echo "Frontier ID:     ${FRONTIER_ID}"
echo "Output dir:      ${OUT_DIR}"
echo ""

# ── Step 1: Self-supervised training (1 round) ───────────────────────────────
echo "[1/4] Running self-supervised training (1 round with finetune) ..."
python3 self_training/self_supervised_refine.py \
    --frontier-id "${FRONTIER_ID}" \
    --detr-checkpoint "${DETR_CHECKPOINT}" \
    --gain-checkpoint "${GAIN_CHECKPOINT}" \
    --rounds ${ROUNDS} \
    --out-dir "${OUT_DIR}"

echo ""
echo "Training completed. Model saved to ${OUT_DIR}/detr_ss_round1.pt"
echo ""

# ── Step 2: Visualize pseudo-labels ──────────────────────────────────────────
echo "[2/4] Visualizing pseudo-labels ..."
python3 visualization/plot_refinement.py \
    --frontier-id "${FRONTIER_ID}" \
    --json "${OUT_DIR}/refined_round1.json" \
    --out-dir "${OUT_DIR}"

echo ""
echo "Pseudo-label visualizations saved to ${OUT_DIR}/"
echo ""

# ── Step 3: Batch plot with fine-tuned DETR model ────────────────────────────
echo "[3/4] Running batch plot with fine-tuned model ..."
python3 visualization/plot_predictions.py \
    --ids "${FRONTIER_ID}" \
    --mode summary_map_gain_generation \
    --detr-checkpoint "${OUT_DIR}/detr_ss_round1.pt" \
    --checkpoint "${GAIN_CHECKPOINT}" \
    --out-dir "${OUT_DIR}"

echo ""
echo "Batch plot saved to ${OUT_DIR}/"
echo ""

# ── Step 4: Re-run refinement with finetuned DETR ────────────────────────────
FT_EVAL_DIR="${OUT_DIR}/finetuned_eval"
echo "[4/5] Re-running refinement with finetuned DETR (no-finetune) ..."
python3 self_training/self_supervised_refine.py \
    --frontier-id "${FRONTIER_ID}" \
    --detr-checkpoint "${OUT_DIR}/detr_ss_round1.pt" \
    --gain-checkpoint "${GAIN_CHECKPOINT}" \
    --rounds 1 \
    --out-dir "${FT_EVAL_DIR}" \
    --no-finetune
echo ""
echo "Finetuned evaluation saved to ${FT_EVAL_DIR}/"
echo ""

# ── Step 5: Coverage heatmap from finetuned eval ─────────────────────────────
echo "[5/5] Generating coverage heatmap (finetuned model) ..."
python3 visualization/plot_coverage_heatmap.py \
    --frontier-id "${FRONTIER_ID}" \
    --a0-json "${FT_EVAL_DIR}/a0_frontiers_round1.json" \
    --evidence-pkl "${FT_EVAL_DIR}/a0_mean_evidence_round1.pkl" \
    --out-dir "${FT_EVAL_DIR}"
echo ""
echo "Coverage heatmap saved to ${FT_EVAL_DIR}/"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "============================================"
echo "  All tasks completed!"
echo "============================================"
echo "Outputs:"
echo "  - Fine-tuned DETR model:  ${OUT_DIR}/detr_ss_round1.pt"
echo "  - Pseudo-label JSON:      ${OUT_DIR}/refined_round1.json"
echo "  - Pseudo-label vis:       ${OUT_DIR}/"
echo "  - Batch plot:             ${OUT_DIR}/"
echo "  - Finetuned eval:         ${FT_EVAL_DIR}/"
echo "  - Finetuned coverage:     ${FT_EVAL_DIR}/"
echo ""
