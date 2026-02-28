# Frontier Generator

A self-supervised learning framework for robotic exploration frontier prediction using DETR and UNet models.

## Overview

This repository contains the self-supervised training pipeline for frontier exploration prediction. The system uses:

1. **DETR Model** - Predicts frontier positions as (x,y) coordinates using transformer architecture
2. **UNet Model** - Predicts exploration gain (m²) and per-cell occupancy (free/obstacle/unknown/unobserved) for each frontier
3. **Self-Supervised Refinement** - Iteratively improves predictions using pseudo-labels generated from model outputs

Two refinement algorithms are available:
- **v2 (DROP+MERGE+ADD)** — marginal-coverage filtering, greedy deduplication, boundary-based candidate addition
- **v3 (Set Cover)** — greedy submodular set cover on per-cell (cell, label) opinion pairs with shift-to-green positioning and proximity disqualification

## Project Structure

```
frontier_generator/
├── config/                           # Centralized configuration
│   ├── __init__.py                   # Python config loader (load_config)
│   ├── defaults.yaml                 # All paths, thresholds, and hyperparameters
│   └── env.sh                        # Shell-sourceable env vars (reads from defaults.yaml)
│
├── models/                           # ML model implementations
│   ├── frontier_detr_model.py        # DETR position prediction model
│   ├── frontier_gain_model.py        # UNet gain prediction model
│   └── map_utils.py                  # Data loading & coordinate utilities
│
├── self_training/                    # Self-supervised training pipeline
│   ├── self_supervised_refine.py     # Main training loop (v2/v3 dispatch)
│   └── frontier_refine.py            # Pseudo-label generation (v2 & v3)
│
├── visualization/                    # Visualization tools
│   ├── plot_refinement.py            # Visualize pseudo-labels & refinement process
│   ├── plot_predictions.py           # Visualize model predictions on map
│   └── plot_coverage_heatmap.py      # A0 mean-evidence coverage heatmap (v3)
│
├── scripts/                          # Quick-start scripts
│   ├── run_ss_refine_1round.sh       # 1-round training + visualization + finetuned eval
│   └── run_ss_refine_nofinetune.sh   # Inference only (no fine-tuning)
│
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, recommended for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

All paths, thresholds, and hyperparameters live in **`config/defaults.yaml`**. Edit this file for your machine:

```yaml
paths:
  tiamat_data_dir: /path/to/your/data
  detr_checkpoint: /path/to/detr_model_best.pt
  gain_checkpoint: /path/to/gain_model_final.pt
  frontier_id: 7000
```

The config system has three layers (later wins):
1. Hardcoded defaults in `config/__init__.py`
2. `config/defaults.yaml` overrides
3. Environment variables (`TIAMAT_DATA_DIR`, `DETR_CHECKPOINT`, `GAIN_CHECKPOINT`, `FRONTIER_ID`)

Shell scripts source `config/env.sh` automatically. Python code can use:

```python
from config import load_config
cfg = load_config()
```

### Key Configuration Sections

| Section | Examples |
|---------|----------|
| `paths` | `tiamat_data_dir`, `detr_checkpoint`, `gain_checkpoint`, `frontier_id` |
| `refinement` | `conf_thresh`, `kappa`, `q_min`, `boundary_radius` |
| `v2` | `delta_drop`, `delta_keep`, `delta_add`, `max_add_per_wp` |
| `v3` | `coverage_frac`, `shift_radius_cells`, `wall_margin_cells`, `proximity_disq_m` |
| `training` | `few_epochs`, `finetune_lr`, `n_rounds` |

## Data Format

```
data_directory/
├── atlas.pkl              # Global map: connectivity graph, waypoint positions, grid categories
├── frontiers/             # Frontier exploration states
│   └── {frontier_id}.json # {waypoint_ids, frontier_positions, frontier_observing_wps}
└── waypoints/             # Sensor data per waypoint
    └── {waypoint_id}/
        ├── meta.json      # Camera calibration & pose
        ├── *.png          # RGB images (5 cameras)
        └── *_depth.npy    # Depth maps
```

## Quick Start

### Option 1: 1-Round Training with Visualization

```bash
# v2 refinement (default)
bash scripts/run_ss_refine_1round.sh

# v3 set-cover refinement
REFINE_METHOD=v3 bash scripts/run_ss_refine_1round.sh
```

**Pipeline steps:**
1. Self-supervised DETR fine-tuning (1 round)
2. Pseudo-label visualization
3. Batch plot with fine-tuned model predictions
4. *(v3 only)* Re-run refinement with finetuned DETR to evaluate improvement
5. *(v3 only)* Coverage heatmap from finetuned evaluation

**Output structure:**
```
results/{frontier_id}_1round/
├── detr_ss_round1.pt                       # Fine-tuned DETR model
├── refined_round1.json                     # Pseudo-labels
├── refined_round1.png                      # Pseudo-label visualization
├── summary_map_gain_generation.png         # Predicted frontiers + gains on map
└── finetuned_eval/                         # (v3 only)
    ├── refined_round1.json                 # Re-evaluated with finetuned DETR
    ├── a0_frontiers_round1.json            # Pre-refinement A0 predictions
    ├── a0_mean_evidence_round1.pkl         # Per-cell mean evidence
    └── a0_frontiers_round1_coverage.png    # Coverage heatmap
```

### Option 2: Inference Only (No Fine-tuning)

```bash
# v3 refinement (default for nofinetune)
bash scripts/run_ss_refine_nofinetune.sh

# v2 refinement
REFINE_METHOD=v2 bash scripts/run_ss_refine_nofinetune.sh
```

**Outputs saved to:** `results/{frontier_id}_nofinetune/`

## Manual Usage

### Self-Supervised Training

```bash
python3 self_training/self_supervised_refine.py \
    --frontier-id 7000 \
    --detr-checkpoint /path/to/detr_model.pt \
    --gain-checkpoint /path/to/gain_model.pt \
    --rounds 3 \
    --refine-method v3 \
    --out-dir results/7000_3rounds
```

### Standalone Refinement (no training loop)

```bash
python3 self_training/frontier_refine.py \
    --frontier-id 7000 \
    --detr-checkpoint /path/to/detr_model.pt \
    --gain-checkpoint /path/to/gain_model.pt \
    --refine-method v3 \
    --coverage-frac 0.65 \
    --proximity-disq-m 2.0 \
    --out-dir results/7000_standalone
```

### Visualize Pseudo-Labels

```bash
python3 visualization/plot_refinement.py \
    --frontier-id 7000 \
    --json results/7000_3rounds/refined_round1.json \
    --out-dir results/7000_3rounds
```

### Coverage Heatmap (v3)

```bash
python3 visualization/plot_coverage_heatmap.py \
    --frontier-id 7000 \
    --a0-json results/finetuned_eval/a0_frontiers_round1.json \
    --evidence-pkl results/finetuned_eval/a0_mean_evidence_round1.pkl \
    --out-dir results/finetuned_eval
```

## Algorithm Details

### v2: DROP + MERGE + ADD

Each refinement round:
1. **DETR Inference** — generate raw frontier predictions per observing waypoint
2. **Score** — run UNet gain model; filter by `conf_thresh`
3. **DROP** — remove predictions with marginal gain < `delta_drop`
4. **MERGE** — greedy deduplication by cell overlap IoU; keep higher-gain frontier
5. **ADD** — scan free-unknown boundary regions for missed frontiers (gain > `delta_add`)
6. **Weights** — per-frontier training weight: `max(q_min, gain / (gain + kappa))`
7. **DETR Fine-tuning** — train on refined pseudo-labels (optional)

### v3: Set Cover

Each refinement round:
1. **Shift to Green** — move each frontier to the nearest safe-green region (eroded by `wall_margin_cells` to avoid visual wall overlap), via BFS anchor + local centroid
2. **Score** — run UNet gain model with extended output (per-cell evidence and predictions)
3. **Build Opinions** — each frontier produces hard opinions `{(cell, argmax_class)}` where class is free or obstacle
4. **Greedy Set Cover** — iteratively select frontier covering the most uncovered (cell, label) pairs; stop at `coverage_frac` (default 0.65); disqualify candidates within `proximity_disq_m` (default 2.0m) of any already-selected frontier
5. **Weights** — same formula as v2
6. **DETR Fine-tuning** — train on selected pseudo-labels (optional)

### Finetuned Evaluation (v3)

After DETR finetuning, the 1-round script re-runs the v3 pipeline with the finetuned model (no further finetuning). This produces a new set of A0 predictions — typically fewer and better-placed than the initial model's — along with a coverage heatmap showing prediction quality improvement.

### Key Concepts

- **Set-based prediction**: DETR predicts variable number of frontiers (max 10 per waypoint)
- **Gain-aware refinement**: UNet gain predictions filter and score frontiers
- **Nearest WP reassignment**: Each refined frontier is reassigned to its nearest observing waypoint for training
- **A0 data**: Pre-refinement scored DETR predictions; saved as diagnostic data for v3 coverage analysis

## Troubleshooting

### Missing Data Files

If you encounter `FileNotFoundError`:
- Verify `tiamat_data_dir` in `config/defaults.yaml` points to correct directory
- Check that `atlas.pkl`, `frontiers/`, and `waypoints/` exist

### CUDA Out of Memory

```bash
export CUDA_VISIBLE_DEVICES=""   # force CPU
```

### Import Errors

Ensure you're running from the repository root:
```bash
cd frontier_generator
python3 self_training/self_supervised_refine.py ...
```

The shell scripts automatically set `PYTHONPATH` correctly.

## Development

### Code Organization

- **config/**: Single source of truth for all paths and constants
- **models/**: Self-contained model definitions and data utilities
- **self_training/**: Training pipeline and pseudo-label generation logic
- **visualization/**: Standalone visualization scripts
- **scripts/**: High-level workflow automation

### Testing Your Changes

```bash
# Quick validation (~1 min)
bash scripts/run_ss_refine_nofinetune.sh

# Full pipeline (~5-10 min with GPU)
REFINE_METHOD=v3 bash scripts/run_ss_refine_1round.sh
```
