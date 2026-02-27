# Frontier Generator

A self-supervised learning framework for robotic exploration frontier prediction using DETR and UNet models.

## Overview

This repository contains the self-supervised training pipeline for frontier exploration prediction. The system uses:

1. **DETR Model** - Predicts frontier positions as (x,y) coordinates using transformer architecture
2. **UNet Model** - Predicts exploration gain (m²) for each frontier
3. **Self-Supervised Refinement** - Iteratively improves predictions using pseudo-labels generated from model outputs

The pipeline implements a three-stage refinement process (DROP → MERGE → ADD) to generate high-quality pseudo-labels for continuous model improvement.

## Project Structure

```
frontier_generator/
├── models/                        # ML model implementations
│   ├── frontier_detr_model.py     # DETR position prediction model
│   ├── frontier_gain_model.py     # UNet gain prediction model
│   └── map_utils.py               # Data loading & coordinate utilities
│
├── self_training/                 # Self-supervised training pipeline
│   ├── self_supervised_refine.py  # Main training loop
│   └── frontier_refine.py         # Pseudo-label generation (DROP/MERGE/ADD)
│
├── visualization/                 # Visualization tools
│   ├── plot_refinement.py         # Visualize pseudo-labels & refinement process
│   └── plot_predictions.py        # Visualize model predictions on map
│
├── scripts/                       # Quick-start scripts
│   ├── run_ss_refine_1round.sh    # 1-round training + full visualization
│   └── run_ss_refine_nofinetune.sh # Inference only (no fine-tuning)
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
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

For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/).

## Data Format

This codebase expects data in the following structure:

```
data_directory/
├── atlas.pkl              # Global map: connectivity graph, waypoint positions, grid categories
├── frontiers/             # Frontier exploration states
│   └── {frontier_id}.json # {waypoint_ids, frontier_positions, frontier_observing_wps}
└── waypoints/             # Sensor data per waypoint
    └── {waypoint_id}/
        ├── meta.json      # Camera calibration & pose
        ├── *.png          # RGB images (5 cameras: back, left, right, frontleft, frontright)
        └── *_depth.npy    # Depth maps (corresponding to RGB images)
```

### Environment Variable

Set your data directory via:

```bash
export TIAMAT_DATA_DIR=/path/to/your/data
```

## Quick Start

### Option 1: 1-Round Training with Visualization

Run self-supervised training for 1 round (10 epochs) with complete visualization:

```bash
bash scripts/run_ss_refine_1round.sh
```

**This script performs:**
1. Self-supervised DETR fine-tuning (10 epochs)
2. Pseudo-label visualization (color-coded by source: keep/add/drop/merge)
3. Batch plot generation with fine-tuned model predictions

**Outputs saved to:** `results/7000_1round/`
- `detr_ss_round1.pt` - Fine-tuned DETR model
- `refined_round1.json` - Pseudo-labels
- `refined_round1.png` - Pseudo-label visualization
- `summary_map_gain_generation.png` - Model predictions on map

### Option 2: Inference Only (No Fine-tuning)

Generate pseudo-labels without model fine-tuning:

```bash
bash scripts/run_ss_refine_nofinetune.sh
```

**This is useful for:**
- Testing the refinement pipeline
- Generating pseudo-labels with pre-trained models
- Quick validation runs

**Outputs saved to:** `results/7000_nofinetune/`

## Manual Usage

### Self-Supervised Training

```bash
python3 self_training/self_supervised_refine.py \
    --frontier-id 7000 \
    --detr-checkpoint /path/to/detr_model.pt \
    --gain-checkpoint /path/to/gain_model.pt \
    --rounds 3 \
    --out-dir results/7000_3rounds
```

### Visualize Pseudo-Labels

```bash
python3 visualization/plot_refinement.py \
    --frontier-id 7000 \
    --json results/7000_3rounds/refined_round1.json \
    --out-dir results/7000_3rounds
```

### Batch Visualization with Predictions

```bash
python3 visualization/plot_predictions.py \
    --ids 7000 \
    --mode summary_map_gain_generation \
    --detr-checkpoint results/7000_3rounds/detr_ss_round1.pt \
    --checkpoint /path/to/gain_model.pt \
    --out-dir results/7000_3rounds
```

## Key Parameters

### Training Parameters
- `--frontier-id`: Frontier ID to process (integer)
- `--rounds`: Number of self-supervised refinement rounds (default: 1)
- `--no-finetune`: Skip DETR fine-tuning (inference only)

### Model Checkpoints
- `--detr-checkpoint`: Pre-trained DETR model path (.pt file)
- `--gain-checkpoint`: Pre-trained UNet gain model path (.pt file)

### Output
- `--out-dir`: Output directory for results (models, JSON, visualizations)

### Visualization Modes
- `full`: Per-waypoint views + summary map (with RGB)
- `summary`: Summary map with RGB thumbnails
- `summary_map_only`: Center map only (fastest)
- `summary_map_gain_generation`: Map with frontier gain predictions

## Output Structure

After running training, outputs are organized as:

```
results/
└── {frontier_id}_{config}/
    ├── refined_round1.json                # Pseudo-labels (keep/add/drop/merge)
    ├── refined_round2.json                # Pseudo-labels round 2 (if multi-round)
    ├── detr_ss_round1.pt                  # Fine-tuned DETR model
    ├── detr_ss_round2.pt                  # Fine-tuned DETR model round 2
    ├── refined_round1.png                 # Pseudo-label visualization
    └── summary_map_gain_generation.png    # Predicted frontiers + gains on map
```

## Model Checkpoints

**Note**: Pre-trained model checkpoint files (`.pt`) are not included in this repository.

To use this codebase:
1. Train your own models using the training scripts in `models/`
2. Or obtain pre-trained checkpoints separately

## Configuration

### Hyperparameters

Training hyperparameters can be adjusted in `self_training/self_supervised_refine.py`:

```python
FEW_EPOCHS  = 10      # Epochs per refinement round
FINETUNE_LR = 1e-5    # Learning rate for DETR fine-tuning
N_ROUNDS    = 3       # Default number of refinement rounds
```

### Script Customization

Edit the shell scripts in `scripts/` to customize:
- Frontier IDs to process (`FRONTIER_ID=7000`)
- Number of refinement rounds (`ROUNDS=1`)
- Model checkpoint paths
- Output directory locations

### Environment Variables

- `TIAMAT_DATA_DIR`: Path to data directory (**required**)
- `DEBUG_WP`: Debug specific waypoint ID (optional, for development)

## Algorithm Details

### Self-Supervised Refinement Pipeline

Each refinement round consists of:

1. **DETR Inference**: Generate raw frontier position predictions
2. **Pseudo-Label Refinement**:
   - **DROP**: Remove low-gain predictions (gain < threshold)
   - **MERGE**: Eliminate overlapping predictions (keep higher gain)
   - **ADD**: Scan boundary regions for missed frontiers
3. **DETR Fine-tuning**: Train on refined pseudo-labels (optional)

The process repeats for multiple rounds, progressively improving prediction quality.

### Key Features

- **Set-based prediction**: DETR predicts variable number of frontiers (max 10 per waypoint)
- **Gain-aware refinement**: Uses UNet gain predictions to filter/score frontiers
- **Nearest WP assignment**: Each frontier associates with its nearest observing waypoint for training
- **Reachability checking**: Only counts gain reachable through free space

## Troubleshooting

### Missing Data Files

If you encounter `FileNotFoundError`:
- Verify `TIAMAT_DATA_DIR` points to correct directory
- Check that `atlas.pkl`, `frontiers/`, and `waypoints/` exist
- The code gracefully handles missing waypoint sensor data (assigns zero gain)

### CUDA Out of Memory

Reduce memory usage:
```bash
# Force CPU execution
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch size in training loop (edit self_supervised_refine.py)
```

### Import Errors

Ensure you're running from the repository root:
```bash
cd frontier_generator
python3 self_training/self_supervised_refine.py ...
```

The shell scripts automatically set `PYTHONPATH` correctly.

### Visualization Not Generated

Check that:
- matplotlib backend is available (`Agg` backend used for headless systems)
- Output directory has write permissions
- Frontier JSON file exists at specified path

## Development

### Code Organization

- **models/**: Self-contained model definitions and data utilities
- **self_training/**: Training pipeline and pseudo-label generation logic
- **visualization/**: Standalone visualization scripts (can run independently)
- **scripts/**: High-level workflow automation

### Testing Your Changes

Run the no-finetune script for quick validation:
```bash
bash scripts/run_ss_refine_nofinetune.sh  # ~1 minute
```

For full pipeline testing:
```bash
bash scripts/run_ss_refine_1round.sh      # ~5-10 minutes with GPU
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{frontier_generator,
  title = {Frontier Generator: Self-Supervised Refinement for Robotic Exploration},
  year = {2025},
  author = {[Your Name/Organization]},
  url = {https://github.com/[your-username]/frontier_generator}
}
```

## License

[Add your license here - e.g., MIT, Apache 2.0, etc.]

## Contact

For questions or issues, please open an issue on GitHub.
