#!/bin/bash
# Self-supervised refinement: 1 round with finetune + full visualization
#
# This script:
# 1. Runs 1-round SS training with DETR finetune
# 2. Visualizes pseudo-labels
# 3. Batch plot with the fine-tuned model
# 4. (v3 only) Re-runs refinement with finetuned DETR to evaluate improvement
# 5. (v3 only) Visualizes coverage heatmap from finetuned evaluation

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Source path config (edit config/defaults.yaml for your machine)
source "${PROJECT_ROOT}/config/env.sh"

REFINE_METHOD="${REFINE_METHOD:-v2}"
ROUNDS=1
OUT_DIR=results/${FRONTIER_ID}_1round

echo "============================================"
echo "  SS Refinement: 1 Round + Visualization"
echo "============================================"
echo "Frontier ID:     ${FRONTIER_ID}"
echo "Refine method:   ${REFINE_METHOD}"
echo "Output dir:      ${OUT_DIR}"
echo ""

# ── Step 1: Self-supervised training (1 round) ───────────────────────────────
echo "[1/4] Running self-supervised training (1 round with finetune) ..."
python3 self_training/self_supervised_refine.py \
    --frontier-id "${FRONTIER_ID}" \
    --detr-checkpoint "${DETR_CHECKPOINT}" \
    --gain-checkpoint "${GAIN_CHECKPOINT}" \
    --rounds ${ROUNDS} \
    --out-dir "${OUT_DIR}" \
    --refine-method "${REFINE_METHOD}"

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

# ── Step 4: Re-run refinement with finetuned DETR (v3 only) ──────────────────
FT_EVAL_DIR="${OUT_DIR}/finetuned_eval"
if [ "${REFINE_METHOD}" = "v3" ]; then
    echo "[4/5] Re-running refinement with finetuned DETR (no-finetune) ..."
    python3 self_training/self_supervised_refine.py \
        --frontier-id "${FRONTIER_ID}" \
        --detr-checkpoint "${OUT_DIR}/detr_ss_round1.pt" \
        --gain-checkpoint "${GAIN_CHECKPOINT}" \
        --rounds 1 \
        --out-dir "${FT_EVAL_DIR}" \
        --no-finetune \
        --refine-method "${REFINE_METHOD}"
    echo ""
    echo "Finetuned evaluation saved to ${FT_EVAL_DIR}/"
    echo ""
else
    echo "[4/5] Skipping finetuned re-evaluation (v3 only)"
    echo ""
fi

# ── Step 5: Coverage heatmap from finetuned eval (v3 only) ───────────────────
if [ -f "${FT_EVAL_DIR}/a0_frontiers_round1.json" ] && \
   [ -f "${FT_EVAL_DIR}/a0_mean_evidence_round1.pkl" ]; then
    echo "[5/5] Generating coverage heatmap (finetuned model) ..."
    python3 visualization/plot_coverage_heatmap.py \
        --frontier-id "${FRONTIER_ID}" \
        --a0-json "${FT_EVAL_DIR}/a0_frontiers_round1.json" \
        --evidence-pkl "${FT_EVAL_DIR}/a0_mean_evidence_round1.pkl" \
        --out-dir "${FT_EVAL_DIR}"
    echo ""
    echo "Coverage heatmap saved to ${FT_EVAL_DIR}/"
else
    echo "[5/5] Skipping coverage heatmap (no A0 data — only generated with v3)"
fi
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
if [ "${REFINE_METHOD}" = "v3" ]; then
echo "  - Finetuned eval:         ${FT_EVAL_DIR}/"
echo "  - Finetuned coverage:     ${FT_EVAL_DIR}/"
fi
echo ""
