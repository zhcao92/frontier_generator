#!/usr/bin/env python3
"""Benchmark: Coverage DETR pipeline vs Geometric (UNet) pipeline.

Times each component of both methods on the same frontier step.
Both use batched model inference for fair comparison.

Usage:
    python3 scripts/benchmark_pipelines.py \
        --frontier-id 7000 \
        --detr-checkpoint cov_detr_checkpoints/model_best.pt \
        --gain-checkpoint /path/to/gain_model_final.pt \
        --coverage-rate 0.8
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.frontier_gain_model import (
    load_atlas, get_current_map_mask,
    build_atlas_grid, build_coverage_grid, build_sample,
    RESOLUTION, CELL_AREA, HALF_EXTENT, GRID_SIZE,
    _build_model as _build_gain_model, _get_device,
)
from models.map_utils import load_frontier_data
from models.frontier_detr_model import (
    world_to_bev_normalized, bev_normalized_to_world,
    BEV_EXTENT, MAX_QUERIES,
)
from models.coverage_detr_model import _build_coverage_detr

# geometric pipeline imports
from self_training.geometric_pipeline import (
    find_global_boundary, assign_nearest_wp,
    score_candidates_batched, shift_candidates_to_safe,
    greedy_set_cover,
)
from self_training.frontier_refine import (
    _load_wp_data, bev_upper_mask_to_world_cells,
    UPPER_HALF,
)


def benchmark_geometric(frontier_id, gain_checkpoint, wids, wp_positions,
                        g2c, atlas_pts, coverage_frac, device):
    """Run geometric pipeline with timing per phase."""
    timings = {}

    # Phase 1: Build covered_g2c
    t = time.time()
    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_cov_grid = build_coverage_grid(
        atlas_pts, cmask, gx_min, gy_min, atlas_cat.shape)
    covered_g2c = {k: v for k, v in g2c.items()
                   if atlas_cov_grid[k[0] - gx_min, k[1] - gy_min]}
    timings['[GEO] Coverage grid'] = time.time() - t

    # Phase 2: Boundary extraction
    t = time.time()
    boundary_cells = find_global_boundary(covered_g2c, include_absent=True)
    timings['[GEO] Boundary extraction'] = time.time() - t
    print(f"  Boundary cells: {len(boundary_cells):,}")

    # Phase 3: WP assignment
    t = time.time()
    candidates = assign_nearest_wp(boundary_cells, wids, wp_positions)
    timings['[GEO] WP assignment'] = time.time() - t
    print(f"  Candidates: {len(candidates):,}")

    # Phase 4: Shift to safe
    t = time.time()
    candidates, _ = shift_candidates_to_safe(candidates, covered_g2c)
    timings['[GEO] Shift to safe'] = time.time() - t
    print(f"  After shift+dedup: {len(candidates):,}")

    # Phase 5: UNet scoring (batched)
    t = time.time()
    UNet = _build_gain_model()
    gain_model = UNet().to(device)
    gain_model.load_state_dict(
        torch.load(gain_checkpoint, map_location=device, weights_only=True))
    gain_model.eval()
    t_model_load = time.time() - t

    t = time.time()
    wp_data_cache = {}
    (A0_frontiers, A0_gains, A0_connects, A0_masks,
     A0_pred_maps, n_zero) = score_candidates_batched(
        candidates, covered_g2c, wp_positions, gain_model, device,
        wp_data_cache, batch_size=64)
    timings['[GEO] UNet scoring (batched)'] = time.time() - t
    timings['[GEO] Model load'] = t_model_load
    print(f"  Scored: {len(A0_frontiers)} gain>0, {n_zero} zero")

    # Phase 6: Greedy set-cover
    t = time.time()
    (sel_frontiers, sel_gains, sel_connects, sel_masks,
     sel_sources, dropped, sel_weights, cover_curve, _) = greedy_set_cover(
        A0_frontiers, A0_gains, A0_connects, A0_masks, A0_pred_maps,
        coverage_frac=coverage_frac, proximity_disq_m=2.0)
    timings['[GEO] Greedy set-cover'] = time.time() - t
    print(f"  Selected: {len(sel_frontiers)}")

    return timings


def benchmark_detr(frontier_id, detr_checkpoint, wids, wp_positions,
                   g2c, atlas_pts, coverage_rate, device):
    """Run DETR pipeline with timing per phase."""
    timings = {}

    # Phase 1: Coverage grid
    t = time.time()
    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_cov = build_coverage_grid(
        atlas_pts, cmask, gx_min, gy_min, atlas_cat.shape)
    atlas_gmin = (gx_min, gy_min)
    timings['[DETR] Coverage grid'] = time.time() - t

    # Phase 2: Model load
    t = time.time()
    CoverageDETR = _build_coverage_detr()
    model = CoverageDETR().to(device)
    model.load_state_dict(
        torch.load(detr_checkpoint, map_location=device, weights_only=True))
    model.eval()
    timings['[DETR] Model load'] = time.time() - t

    # Phase 3: Build BEV grids (per WP)
    t = time.time()
    valid_wids = [w for w in wids if w in wp_positions]
    all_inps = []
    theta = 0.0
    for wp_id in valid_wids:
        wp_pos = wp_positions[wp_id]
        center_xy = np.array([wp_pos[0], wp_pos[1]])
        inp, _ = build_sample(center_xy, theta,
                              atlas_cat, atlas_cov, atlas_gmin,
                              obs_wp_id=wp_id)
        all_inps.append(inp)
    timings['[DETR] Build BEV grids'] = time.time() - t
    print(f"  BEV grids built: {len(all_inps)}")

    # Phase 4: Batched model forward
    t = time.time()
    import torch.nn.functional as F
    inp_batch = torch.from_numpy(np.stack(all_inps)).to(device)
    with torch.no_grad():
        pred_xy, pred_conf, pred_score = model(inp_batch)
    pred_xy_np = pred_xy.cpu().numpy()
    conf_np = torch.sigmoid(pred_conf).cpu().numpy()
    score_norm_np = F.softmax(pred_score, dim=1).cpu().numpy()
    timings['[DETR] Model forward (batched)'] = time.time() - t

    # Phase 5: Post-process + greedy selection
    t = time.time()
    conf_thresh = 0.3
    all_preds = []
    for wi, wp_id in enumerate(valid_wids):
        wp_pos = wp_positions[wp_id]
        center_xy = np.array([wp_pos[0], wp_pos[1]])
        inp = all_inps[wi]
        wp_weight = float((inp[3] > 0).sum())
        mask = conf_np[wi] > conf_thresh
        if not mask.any():
            continue
        preds_bev = pred_xy_np[wi][mask]
        confs = conf_np[wi][mask]
        scores = score_norm_np[wi][mask] * wp_weight
        world_xy = bev_normalized_to_world(preds_bev, center_xy, theta)
        for i in range(len(world_xy)):
            all_preds.append([world_xy[i, 0], world_xy[i, 1],
                              confs[i], scores[i]])

    n_total_preds = len(all_preds)

    # Greedy selection by normalised score
    if all_preds and coverage_rate is not None:
        pred_arr = np.array(all_preds)
        raw_scores = pred_arr[:, 3]
        score_sum = raw_scores.sum()
        norm_scores = raw_scores / score_sum if score_sum > 0 else raw_scores
        order = np.argsort(-norm_scores)
        cum = np.cumsum(norm_scores[order])
        n_select = min(int(np.searchsorted(cum, coverage_rate) + 1),
                       len(order))
    else:
        n_select = n_total_preds

    timings['[DETR] Post-process + selection'] = time.time() - t
    print(f"  Predictions: {n_total_preds}, selected: {n_select}")

    return timings


def main():
    ap = argparse.ArgumentParser(description='Benchmark DETR vs Geometric')
    ap.add_argument('--frontier-id', type=int, required=True)
    ap.add_argument('--detr-checkpoint', type=str, required=True)
    ap.add_argument('--gain-checkpoint', type=str, required=True)
    ap.add_argument('--coverage-rate', type=float, default=0.8)
    args = ap.parse_args()

    device = _get_device()
    print(f"Device: {device}")

    # Shared atlas load
    print("Loading atlas ...")
    t0 = time.time()
    atlas_pts, wp_positions, g2c = load_atlas()
    t_atlas = time.time() - t0
    print(f"  Atlas load: {t_atlas:.2f}s")

    fdata = load_frontier_data(args.frontier_id)
    wids = fdata['waypoint_ids']
    print(f"  Frontier {args.frontier_id}: {len(wids)} WPs\n")

    # ── Run Geometric pipeline ──
    print("=" * 60)
    print("GEOMETRIC (UNet) PIPELINE")
    print("=" * 60)
    geo_timings = benchmark_geometric(
        args.frontier_id, args.gain_checkpoint, wids, wp_positions,
        g2c, atlas_pts, args.coverage_rate, device)

    # ── Run DETR pipeline ──
    print()
    print("=" * 60)
    print("COVERAGE DETR PIPELINE")
    print("=" * 60)
    detr_timings = benchmark_detr(
        args.frontier_id, args.detr_checkpoint, wids, wp_positions,
        g2c, atlas_pts, args.coverage_rate, device)

    # ── Summary ──
    print()
    print("=" * 60)
    print("TIMING COMPARISON")
    print("=" * 60)
    print(f"{'Phase':<40s} {'Time (s)':>10s}")
    print("-" * 52)

    all_timings = {}
    all_timings.update(geo_timings)
    all_timings.update(detr_timings)

    geo_total = sum(v for k, v in all_timings.items() if k.startswith('[GEO]'))
    detr_total = sum(v for k, v in all_timings.items()
                     if k.startswith('[DETR]'))

    for phase, dt in all_timings.items():
        print(f"{phase:<40s} {dt:>10.3f}")
        if phase == list(geo_timings.keys())[-1]:
            print(f"{'[GEO] TOTAL':<40s} {geo_total:>10.3f}")
            print("-" * 52)

    print(f"{'[DETR] TOTAL':<40s} {detr_total:>10.3f}")
    print("-" * 52)
    print(f"{'Atlas load (shared)':<40s} {t_atlas:>10.3f}")

    if detr_total > 0:
        print(f"\nSpeedup: {geo_total / detr_total:.1f}x")


if __name__ == '__main__':
    main()
