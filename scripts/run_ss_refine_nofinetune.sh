#!/bin/bash

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
OUT_DIR=results/${FRONTIER_ID}_nofinetune

python3 self_training/self_supervised_refine.py \
    --frontier-id ${FRONTIER_ID} \
    --detr-checkpoint /scratch/mcity_project_root/mcity_project/zhcao/DARPA_TIAMAT/detr_cache/model_best.pt \
    --gain-checkpoint /home/zhcao/CodesLib/TIAMAT/frontier_predictor/gain_cache/model_final.pt \
    --rounds ${ROUNDS} \
    --out-dir ${OUT_DIR} \
    --no-finetune

python3 visualization/plot_refinement.py \
    --frontier-id ${FRONTIER_ID} \
    --json ${OUT_DIR}/refined_round1.json \
    --out-dir ${OUT_DIR}
