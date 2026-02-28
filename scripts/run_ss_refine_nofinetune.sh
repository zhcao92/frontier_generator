#!/bin/bash
# Inference-only run (no DETR fine-tuning) + visualization.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Source path config (edit config/env.sh for your machine)
source "${PROJECT_ROOT}/config/env.sh"

ROUNDS=1
OUT_DIR=results/${FRONTIER_ID}_nofinetune

python3 self_training/self_supervised_refine.py \
    --frontier-id "${FRONTIER_ID}" \
    --detr-checkpoint "${DETR_CHECKPOINT}" \
    --gain-checkpoint "${GAIN_CHECKPOINT}" \
    --rounds ${ROUNDS} \
    --out-dir "${OUT_DIR}" \
    --no-finetune

python3 visualization/plot_refinement.py \
    --frontier-id "${FRONTIER_ID}" \
    --json "${OUT_DIR}/refined_round1.json" \
    --out-dir "${OUT_DIR}"
