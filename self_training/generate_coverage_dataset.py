#!/usr/bin/env python3
"""generate_coverage_dataset.py
================================
Run the geometric pipeline exhaustively on every frontier step and save
selection order + marginal gains for coverage-aware DETR training.

Usage:
    python3 self_training/generate_coverage_dataset.py \
        --gain-checkpoint /path/to/model_final.pt \
        --out-dir results/coverage_dataset

Resume support: re-running skips already-processed step_{fid}.json files.
"""

import os, sys, json, time, argparse, glob
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from models.frontier_gain_model import (
    load_atlas, get_current_map_mask,
    build_atlas_grid, build_coverage_grid,
    RESOLUTION, CELL_AREA, FRONTIERS_DIR,
    _build_model as _build_gain_model, _get_device,
)
from models.map_utils import load_frontier_data
from self_training.geometric_pipeline import (
    find_global_boundary, assign_nearest_wp,
    shift_candidates_to_safe, score_candidates_batched,
    greedy_set_cover,
)


def discover_frontier_ids():
    """Return sorted list of frontier IDs from FRONTIERS_DIR/*.json."""
    paths = glob.glob(os.path.join(FRONTIERS_DIR, "*.json"))
    ids = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            ids.append(int(stem))
        except ValueError:
            continue
    return sorted(ids)


def process_step(fid, atlas_pts, wp_positions, g2c, gain_model, device,
                 wp_data_cache, proximity_disq_m):
    """Run full geometric pipeline on one frontier step. Return dict or None."""
    try:
        frontier_data = load_frontier_data(fid)
    except Exception as e:
        return None, f"load error: {e}"

    wids = frontier_data['waypoint_ids']
    if not wids:
        return None, "no waypoints"

    # Build covered_g2c
    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cat_f, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_cov_grid = build_coverage_grid(
        atlas_pts, cmask, gx_min, gy_min, atlas_cat_f.shape)
    covered_g2c = {k: v for k, v in g2c.items()
                   if atlas_cov_grid[k[0] - gx_min, k[1] - gy_min]}

    # Phase 1: boundary extraction
    boundary_cells = find_global_boundary(covered_g2c, include_absent=True)
    if not boundary_cells:
        return None, "no boundary cells"
    n_boundary = len(boundary_cells)

    # Phase 2: WP assignment
    candidates = assign_nearest_wp(boundary_cells, wids, wp_positions)
    if not candidates:
        return None, "no candidates assigned"

    # Phase 2b: shift to safe
    candidates, _ = shift_candidates_to_safe(candidates, covered_g2c)

    # Dedup: many boundary cells collapse to the same (x, y, wp_id) after shift
    seen = set()
    unique = []
    for c in candidates:
        key = (c[0], c[1], int(c[3]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique

    # Phase 3: batched UNet scoring
    (A0_frontiers, A0_gains, A0_connects, A0_masks,
     A0_pred_maps, n_zero) = score_candidates_batched(
        candidates, covered_g2c, wp_positions, gain_model, device,
        wp_data_cache)

    if not A0_frontiers:
        return None, "no candidates with gain > 0"

    # Phase 4: exhaustive greedy set-cover (coverage_frac=1.0)
    (sel_frontiers, sel_gains, sel_connects, sel_masks,
     sel_sources, dropped, sel_weights, cover_curve,
     all_selected_idx) = greedy_set_cover(
        A0_frontiers, A0_gains, A0_connects, A0_masks, A0_pred_maps,
        coverage_frac=1.0,
        proximity_disq_m=proximity_disq_m)

    # Build output — only exhaustively selected candidates
    selection_order = []
    candidates_out = []
    for pick_num, (pick_idx, mg_frac) in enumerate(
            zip(all_selected_idx, cover_curve), 1):
        _, marginal_gain, cum_frac = mg_frac
        f = A0_frontiers[pick_idx]
        entry = {
            "idx": pick_num - 1,
            "x": float(f[0]),
            "y": float(f[1]),
            "wp_id": int(f[3]),
            "gain_m2": float(A0_gains[pick_idx]),
            "connects": bool(A0_connects[pick_idx]),
        }
        candidates_out.append(entry)
        selection_order.append({
            "pick": pick_num,
            "candidate_idx": pick_num - 1,
            "x": entry["x"],
            "y": entry["y"],
            "wp_id": entry["wp_id"],
            "gain_m2": entry["gain_m2"],
            "connects": entry["connects"],
            "marginal_gain": int(marginal_gain),
            "cumulative_frac": float(cum_frac),
        })

    cover_curve_out = []
    for pick_num, mg, cf in cover_curve:
        cover_curve_out.append({
            "pick": pick_num,
            "marginal_gain": int(mg),
            "cumulative_frac": float(cf),
        })

    result = {
        "frontier_id": fid,
        "waypoint_ids": wids,
        "n_boundary_cells": n_boundary,
        "n_candidates_scored": len(A0_frontiers),
        "proximity_disq_m": proximity_disq_m,
        "candidates": candidates_out,
        "selection_order": selection_order,
        "cover_curve": cover_curve_out,
    }
    return result, None


def main():
    ap = argparse.ArgumentParser(
        description='Generate coverage dataset across all frontier steps')
    ap.add_argument('--gain-checkpoint', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--proximity-disq-m', type=float, default=2.0)
    ap.add_argument('--batch-size', type=int, default=64,
                    help='UNet inference batch size')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"[dataset] Device: {device}")

    # Load atlas once
    print("[dataset] Loading atlas ...")
    atlas_pts, wp_positions, g2c = load_atlas()

    # Load gain model once
    print("[dataset] Loading gain model ...")
    UNet = _build_gain_model()
    gain_model = UNet().to(device)
    gain_model.load_state_dict(
        torch.load(args.gain_checkpoint, map_location=device, weights_only=True))
    gain_model.eval()

    # Discover frontier IDs
    all_fids = discover_frontier_ids()
    print(f"[dataset] Found {len(all_fids)} frontier steps")

    # Resume support: skip already-processed
    existing = set()
    for p in out_dir.glob("step_*.json"):
        stem = p.stem  # step_7000
        try:
            existing.add(int(stem.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    todo_fids = [f for f in all_fids if f not in existing]
    if existing:
        print(f"[dataset] Skipping {len(existing)} already-processed steps")
    print(f"[dataset] Processing {len(todo_fids)} steps")

    # Shared cache
    wp_data_cache = {}

    processed_ids = sorted(existing)
    n_errors = 0
    t_start = time.time()

    for fid in tqdm(todo_fids, desc="frontier steps"):
        result, err = process_step(
            fid, atlas_pts, wp_positions, g2c, gain_model, device,
            wp_data_cache, args.proximity_disq_m)

        if result is None:
            n_errors += 1
            tqdm.write(f"  skip {fid}: {err}")
            continue

        out_path = out_dir / f"step_{fid}.json"
        with open(out_path, 'w') as f:
            json.dump(result, f)
        processed_ids.append(fid)

    processed_ids = sorted(processed_ids)
    elapsed = time.time() - t_start

    # Write manifest
    manifest = {
        "gain_checkpoint": os.path.abspath(args.gain_checkpoint),
        "proximity_disq_m": args.proximity_disq_m,
        "n_steps": len(processed_ids),
        "frontier_ids": processed_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[dataset] Done: {len(processed_ids)} steps in {elapsed:.1f}s "
          f"({n_errors} skipped)")
    print(f"[dataset] Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
