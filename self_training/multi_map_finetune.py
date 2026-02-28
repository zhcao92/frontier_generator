#!/usr/bin/env python3
"""Multi-map DETR fine-tuning.

Collects refined frontiers from multiple maps and trains DETR on all of them.
This increases training data size while maintaining quality (only set-cover selected).

Usage:
    python3 multi_map_finetune.py \\
        --frontier-ids 7000 7018 7022 7044 7047 \\
        --data-dir results/multi_map_data \\
        --detr-checkpoint /path/to/detr.pt \\
        --gain-checkpoint /path/to/gain.pt \\
        --out-dir results/multi_map_training
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# Import from existing modules
from models.frontier_gain_model import (
    load_atlas, get_current_map_mask,
    build_atlas_grid, build_coverage_grid,
    build_sample,
    _build_model as _build_gain_model, _get_device,
    load_waypoint_meta, CAMERAS, WP_IMAGES_DIR,
    _project_depth_array,
)
from models.frontier_detr_model import (
    world_to_bev_normalized,
    _build_detr_model, MAX_QUERIES, LAMBDA_L1, LAMBDA_CONF,
)
from models.map_utils import load_frontier_data

from self_supervised_refine import (
    _hungarian_loss, _load_wp_data,
    FEW_EPOCHS, FINETUNE_LR,
)


def load_refined_frontiers(frontier_id, data_dir):
    """Load refined frontiers from a specific map.

    Returns
    -------
    frontiers : list[[x, y, conf, wp_id]] - only 'keep' frontiers
    wids      : list[int] - observing WP IDs for this map
    """
    json_path = os.path.join(data_dir, str(frontier_id), "refined_round1.json")

    with open(json_path) as f:
        data = json.load(f)

    # Only use 'keep' frontiers (set-cover selected)
    frontiers = []
    wids_set = set()
    for entry in data:
        if entry.get('source') == 'keep':
            frontiers.append([
                entry['x'], entry['y'],
                entry['conf'], entry['wp_id']
            ])
            wids_set.add(entry['wp_id'])

    return frontiers, sorted(wids_set)


def finetune_detr_multi_map(detr_model, device,
                            all_frontier_data,  # list of (frontier_id, frontiers, wids)
                            wp_positions,
                            covered_g2c_dict,   # {frontier_id: covered_g2c}
                            epochs=FEW_EPOCHS, lr=FINETUNE_LR):
    """Fine-tune DETR on frontiers from multiple maps.

    Parameters
    ----------
    all_frontier_data : list of (frontier_id, frontiers, wids)
        frontiers: [[x, y, conf, wp_id], ...]
        wids: [wp_id1, wp_id2, ...]
    covered_g2c_dict : {frontier_id: covered_g2c} dict
    """

    # Build training samples from all maps
    samples = []
    wp_data_cache = {}

    for frontier_id, frontiers, wids in all_frontier_data:
        print(f"  [map {frontier_id}] Loading {len(frontiers)} frontiers from {len(wids)} WPs")

        covered_g2c = covered_g2c_dict[frontier_id]
        atlas_cat, gx_min, gy_min = build_atlas_grid(covered_g2c)
        atlas_cov = (atlas_cat >= 0)
        atlas_gmin = (gx_min, gy_min)

        # Group frontiers by wp_id
        frontier_to_wp = {}
        all_fp_xy = []
        for i, entry in enumerate(frontiers):
            wp_id = int(entry[3])
            frontier_to_wp[i] = wp_id
            all_fp_xy.append([entry[0], entry[1]])

        all_fp_xy = np.array(all_fp_xy, dtype=np.float64) if all_fp_xy else None

        # Build samples for each WP
        for wp_id in wids:
            if wp_id not in wp_positions:
                continue

            wp_pos = np.asarray(wp_positions[wp_id], dtype=np.float64)
            center_xy = wp_pos[:2]
            theta = 0.0

            # Load WP data (cached across maps)
            if wp_id not in wp_data_cache:
                try:
                    wp_data_cache[wp_id] = _load_wp_data(wp_id)
                except Exception:
                    continue
            wp_data = wp_data_cache[wp_id]

            try:
                inp, _ = build_sample(center_xy, theta,
                                      atlas_cat, atlas_cov, atlas_gmin,
                                      wp_data=wp_data)
            except Exception:
                continue

            # Frontiers assigned to this WP
            assigned_indices = [i for i, assigned_wp in frontier_to_wp.items()
                                if assigned_wp == wp_id]
            if assigned_indices:
                assigned_fp_xy = all_fp_xy[assigned_indices]
                gt_norm = world_to_bev_normalized(assigned_fp_xy, center_xy, theta)
                if len(gt_norm) > MAX_QUERIES:
                    gt_norm = gt_norm[:MAX_QUERIES]
            else:
                gt_norm = np.zeros((0, 2), dtype=np.float32)

            samples.append((inp.astype(np.float32), gt_norm.astype(np.float32)))

    if not samples:
        print("    [finetune] No valid samples – skipping")
        return detr_model

    print(f"  [finetune] Total training samples: {len(samples)} (from {len(all_frontier_data)} maps)")

    # Training loop
    optimizer = optim.Adam(detr_model.parameters(), lr=lr)
    detr_model.train()
    for epoch in range(epochs):
        np.random.shuffle(samples)
        total_loss = 0.0
        for inp_np, gt_np in samples:
            inp_t = torch.tensor(inp_np[None], dtype=torch.float32, device=device)
            pred_xy_t, pred_conf_t = detr_model(inp_t)
            loss = _hungarian_loss(pred_xy_t[0], pred_conf_t[0],
                                   torch.tensor(gt_np, dtype=torch.float32, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"      epoch {epoch+1:02d}/{epochs}  loss={total_loss/len(samples):.4f}")

    detr_model.eval()
    return detr_model


def main():
    ap = argparse.ArgumentParser(
        description='Multi-map DETR fine-tuning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--frontier-ids', type=int, nargs='+', required=True,
                    help='List of frontier IDs to use for training')
    ap.add_argument('--data-dir', type=str, default='results/multi_map_data',
                    help='Directory containing refined_round1.json for each map')
    ap.add_argument('--detr-checkpoint', type=str, required=True)
    ap.add_argument('--gain-checkpoint', type=str, required=True)
    ap.add_argument('--out-dir', type=str, default='results/multi_map_training')
    ap.add_argument('--epochs', type=int, default=FEW_EPOCHS)
    ap.add_argument('--lr', type=float, default=FINETUNE_LR)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"[MultiMap] Device: {device}")
    print(f"[MultiMap] Training on {len(args.frontier_ids)} maps: {args.frontier_ids}")
    print("")

    # Load atlas and WP positions (shared across all maps)
    print("[MultiMap] Loading atlas ...")
    atlas_pts, wp_positions, g2c = load_atlas()

    # Load refined frontiers from each map
    print("[MultiMap] Loading refined frontiers from each map ...")
    all_frontier_data = []
    covered_g2c_dict = {}

    total_frontiers = 0
    for fid in args.frontier_ids:
        frontiers, wids = load_refined_frontiers(fid, args.data_dir)

        # Load frontier data to get covered_g2c
        frontier_data = load_frontier_data(fid)
        wids_full = frontier_data['waypoint_ids']
        cmask = get_current_map_mask(wids_full, wp_positions, atlas_pts)
        atlas_cat_f, gx_min, gy_min = build_atlas_grid(g2c)
        atlas_cov_grid = build_coverage_grid(
            atlas_pts, cmask, gx_min, gy_min, atlas_cat_f.shape)
        covered_g2c = {k: v for k, v in g2c.items()
                       if atlas_cov_grid[k[0] - gx_min, k[1] - gy_min]}

        covered_g2c_dict[fid] = covered_g2c
        all_frontier_data.append((fid, frontiers, wids))
        total_frontiers += len(frontiers)

        print(f"    Map {fid}: {len(frontiers)} selected frontiers")

    print(f"\n[MultiMap] Total: {total_frontiers} high-quality frontiers from {len(args.frontier_ids)} maps")
    print("")

    # Load DETR model
    print("[MultiMap] Loading DETR model ...")
    FrontierDETR = _build_detr_model()
    detr_model = FrontierDETR().to(device)
    detr_model.load_state_dict(
        torch.load(args.detr_checkpoint, map_location=device, weights_only=True))
    detr_model.eval()

    # Fine-tune on multi-map data
    print(f"[MultiMap] Fine-tuning DETR ({args.epochs} epochs, lr={args.lr}) ...")
    detr_model = finetune_detr_multi_map(
        detr_model, device, all_frontier_data,
        wp_positions, covered_g2c_dict,
        epochs=args.epochs, lr=args.lr)

    # Save fine-tuned model
    out_path = os.path.join(args.out_dir, "detr_multi_map.pt")
    torch.save(detr_model.state_dict(), out_path)
    print(f"\n[MultiMap] Saved fine-tuned model to {out_path}")

    # Save training metadata
    meta_path = os.path.join(args.out_dir, "training_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'frontier_ids': args.frontier_ids,
            'total_frontiers': total_frontiers,
            'epochs': args.epochs,
            'lr': args.lr,
        }, f, indent=2)
    print(f"[MultiMap] Saved metadata to {meta_path}")


if __name__ == '__main__':
    main()
