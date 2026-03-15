#!/usr/bin/env python3
"""geometric_pipeline.py
========================
Replace DETR with geometric boundary extraction as the candidate source,
then score with UNet and select via greedy set-cover.

This is a timing / feasibility experiment.  No existing files are modified.

Usage:
    python3 self_training/geometric_pipeline.py \
        --frontier-id 7000 \
        --gain-checkpoint /path/to/model_final.pt \
        --out-dir results/7000_geometric
"""

import os, sys, json, math, argparse, time
from pathlib import Path

# Ensure project root is on sys.path (so `models.*` / `visualization.*` resolve)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── shared imports ──────────────────────────────────────────────────────────
from models.frontier_gain_model import (
    load_atlas, get_current_map_mask,
    build_atlas_grid, build_coverage_grid,
    RESOLUTION, CELL_AREA,
    _build_model as _build_gain_model, _get_device,
)
from models.map_utils import load_frontier_data
from self_training.frontier_refine import (
    predict_gain_with_mask, _load_wp_data, save_round_results,
    _eroded_green, _shift_to_green,
    KAPPA, Q_MIN, SHIFT_RADIUS_CELLS, WALL_MARGIN_CELLS,
)
from visualization import plot_predictions as _bp


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Geometric boundary extraction
# ═══════════════════════════════════════════════════════════════════════════

def find_global_boundary(covered_g2c, include_absent=True):
    """Return all boundary cells from the covered atlas grid.

    A cell qualifies if cat==1 (free) AND any 4-neighbor is cat==0 (unknown)
    or absent from the dict (when include_absent=True).
    """
    boundary = []
    for (gx, gy), cat in covered_g2c.items():
        if cat != 1:
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n_cat = covered_g2c.get((gx + dx, gy + dy))
            if n_cat == 0:
                boundary.append((gx, gy))
                break
            if include_absent and n_cat is None:
                boundary.append((gx, gy))
                break
    return boundary


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Assign each boundary cell to nearest observing WP
# ═══════════════════════════════════════════════════════════════════════════

def assign_nearest_wp(boundary_cells, wids, wp_positions):
    """For each boundary cell, find nearest observing WP.

    Returns list of [x, y, 1.0, wp_id] (conf=1.0 for geometric detections).
    """
    wp_xy = []
    wp_ids_valid = []
    for w in wids:
        if w in wp_positions:
            wp_xy.append([wp_positions[w][0], wp_positions[w][1]])
            wp_ids_valid.append(w)
    if not wp_xy:
        return []
    wp_xy = np.array(wp_xy)  # (W, 2)

    candidates = []
    for gx, gy in boundary_cells:
        x = gx * RESOLUTION
        y = gy * RESOLUTION
        d2 = (wp_xy[:, 0] - x) ** 2 + (wp_xy[:, 1] - y) ** 2
        nearest = int(np.argmin(d2))
        candidates.append([x, y, 1.0, wp_ids_valid[nearest]])
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# 3.  UNet scoring
# ═══════════════════════════════════════════════════════════════════════════

def score_candidates(candidates, covered_g2c, wp_positions, gain_model,
                     device, wp_data_cache):
    """Score each candidate with UNet gain model.

    Returns (A0_frontiers, A0_gains, A0_connects, A0_masks, A0_pred_maps)
    containing only candidates with gain > 0.
    """
    A0_frontiers = []
    A0_gains = []
    A0_connects = []
    A0_masks = []
    A0_pred_maps = []
    n_zero = 0

    for idx, cand in enumerate(candidates):
        if (idx + 1) % 100 == 0 or idx == len(candidates) - 1:
            print(f"    scoring {idx + 1}/{len(candidates)} "
                  f"({len(A0_frontiers)} gain>0, {n_zero} zero) ...")

        wp_id = int(cand[3])
        if wp_id not in wp_positions:
            n_zero += 1
            continue
        wp_pos = wp_positions[wp_id]

        gain_m2, connects, cells, evidence, preds = predict_gain_with_mask(
            cand[:2], wp_id, covered_g2c, wp_pos, gain_model, device,
            wp_data_cache=wp_data_cache, extended=True)

        if gain_m2 <= 0:
            n_zero += 1
            continue

        A0_frontiers.append(cand)
        A0_gains.append(gain_m2)
        A0_connects.append(connects)
        A0_masks.append(cells)
        A0_pred_maps.append(preds)

    return A0_frontiers, A0_gains, A0_connects, A0_masks, A0_pred_maps, n_zero


# ═══════════════════════════════════════════════════════════════════════════
# 2b. Shift candidates to safe green regions (before scoring)
# ═══════════════════════════════════════════════════════════════════════════

def shift_candidates_to_safe(candidates, covered_g2c,
                              shift_radius_cells=SHIFT_RADIUS_CELLS,
                              wall_margin_cells=WALL_MARGIN_CELLS):
    """Shift raw candidates to safe green regions (away from walls).

    Pure geometry — no UNet calls.  Returns (shifted_candidates, pre_positions)
    where pre_positions[i] = (ox, oy) original position for arrow plotting.
    """
    safe_green = _eroded_green(covered_g2c, margin=wall_margin_cells)
    print(f"    Eroded green: {len(safe_green)} safe cells "
          f"(margin={wall_margin_cells})")

    shifted = 0
    pre_positions = []  # (ox, oy) for each candidate, for plotting arrows
    for i, entry in enumerate(candidates):
        ox, oy = entry[0], entry[1]
        pre_positions.append((ox, oy))
        nx, ny = _shift_to_green(ox, oy, covered_g2c, safe_green,
                                 radius_cells=shift_radius_cells)
        if nx != ox or ny != oy:
            candidates[i] = [nx, ny, entry[2], entry[3]]
            shifted += 1

    print(f"    Shifted {shifted}/{len(candidates)} candidates")
    return candidates, pre_positions


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Greedy set-cover selection
# ═══════════════════════════════════════════════════════════════════════════

def greedy_set_cover(A0_frontiers, A0_gains, A0_connects, A0_masks,
                     A0_pred_maps, coverage_frac=0.65, proximity_disq_m=2.0):
    """Select frontiers via greedy set-cover over opinion sets.

    Returns (sel_frontiers, sel_gains, sel_connects, sel_masks, sel_sources,
             dropped, weights).
    """
    # Build opinion sets
    opinion_sets = []
    universe = set()
    for cp in A0_pred_maps:
        opinions = set()
        for cell, (cls, pf, po) in cp.items():
            if cls in (0, 1):
                opinions.add((cell, cls))
        opinion_sets.append(opinions)
        universe |= opinions
    n_universe = len(universe)
    print(f"    {n_universe} (cell, label) pairs in universe")

    if n_universe == 0:
        return [], [], [], [], [], [], []

    # Greedy selection
    disq_r2 = proximity_disq_m * proximity_disq_m
    covered = set()
    selected_idx = []
    candidates_set = set(range(len(A0_frontiers)))

    while candidates_set:
        best_idx = -1
        best_gain = 0
        for i in candidates_set:
            gain = len(opinion_sets[i] - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx < 0 or best_gain == 0:
            break

        selected_idx.append(best_idx)
        candidates_set.discard(best_idx)
        covered |= opinion_sets[best_idx]

        # Disqualify nearby candidates
        too_close = set()
        for i in candidates_set:
            cx, cy = A0_frontiers[i][0], A0_frontiers[i][1]
            for si in selected_idx:
                dx = cx - A0_frontiers[si][0]
                dy = cy - A0_frontiers[si][1]
                if dx * dx + dy * dy <= disq_r2:
                    too_close.add(i)
                    break
        candidates_set -= too_close

        frac = len(covered) / n_universe
        print(f"      pick #{len(selected_idx)}: A0[{best_idx}]  "
              f"+{best_gain} opinions  coverage={frac:.3f}  "
              f"disqualified={len(too_close)}")
        if frac >= coverage_frac:
            break

    print(f"    Selected {len(selected_idx)}/{len(A0_frontiers)} frontiers  "
          f"(coverage={len(covered)}/{n_universe} "
          f"= {len(covered)/n_universe:.3f})")

    # Build dropped list
    selected_set = set(selected_idx)
    dropped = []
    for i in range(len(A0_frontiers)):
        if i not in selected_set:
            f = A0_frontiers[i]
            dropped.append({
                'x': float(f[0]), 'y': float(f[1]),
                'conf': float(f[2]), 'wp_id': int(f[3]),
                'gain_m2': float(A0_gains[i]),
                'connects': bool(A0_connects[i]),
                'source': 'drop',
            })

    # Build output lists
    frontiers = [A0_frontiers[i] for i in selected_idx]
    gains     = [A0_gains[i]     for i in selected_idx]
    connects  = [A0_connects[i]  for i in selected_idx]
    masks     = [A0_masks[i]     for i in selected_idx]
    sources   = ['keep'] * len(frontiers)

    # Compute weights (unique-cell formula)
    final_cell_count = {}
    for m in masks:
        for cell in m:
            final_cell_count[cell] = final_cell_count.get(cell, 0) + 1
    weights = []
    for m in masks:
        delta_final = sum(
            1 for cell in m if final_cell_count.get(cell, 0) == 1
        ) * CELL_AREA
        q = delta_final / (delta_final + KAPPA)
        weights.append(float(max(Q_MIN, min(1.0, q))))

    return frontiers, gains, connects, masks, sources, dropped, weights


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Atlas-level visualization helpers
# ═══════════════════════════════════════════════════════════════════════════

def _draw_atlas_background(ax, waypoint_ids, wp_positions=None):
    """Draw atlas background (cells colored by category) and WP graph.

    Uses atlas globals from plot_predictions module.
    Returns the set of neighborhood WP IDs for later use.
    """
    WP_POS    = wp_positions or _bp._WP_POSITIONS
    ATLAS_PTS = _bp._ATLAS_POINTS_2D
    ATLAS_GI  = _bp._ATLAS_GRID_IDX
    G2C       = _bp._GRID_TO_CAT
    WP_GRAPH  = _bp._WP_GRAPH

    NEIGHBORHOOD_RADIUS = 5.0
    all_wp_xy = np.array([[WP_POS[w][0], WP_POS[w][1]]
                          for w in waypoint_ids if w in WP_POS])

    # Atlas scatter
    if len(all_wp_xy) > 0:
        R = NEIGHBORHOOD_RADIUS
        R2 = R * R
        bbox_min = all_wp_xy.min(axis=0) - R
        bbox_max = all_wp_xy.max(axis=0) + R
        box_mask = ((ATLAS_PTS[:, 0] >= bbox_min[0]) &
                    (ATLAS_PTS[:, 0] <= bbox_max[0]) &
                    (ATLAS_PTS[:, 1] >= bbox_min[1]) &
                    (ATLAS_PTS[:, 1] <= bbox_max[1]))
        box_idx = np.where(box_mask)[0]
        box_pts = ATLAS_PTS[box_idx]
        near_sub = np.zeros(len(box_pts), dtype=bool)
        for wxy in all_wp_xy:
            d2 = (box_pts[:, 0] - wxy[0]) ** 2 + (box_pts[:, 1] - wxy[1]) ** 2
            near_sub |= (d2 <= R2)
        near_mask = np.zeros(len(ATLAS_PTS), dtype=bool)
        near_mask[box_idx[near_sub]] = True
        local_pts = ATLAS_PTS[near_mask]
        local_gi  = ATLAS_GI[near_mask]
        cats = np.array([int(G2C.get((g[0], g[1]), 0)) for g in local_gi],
                        dtype=np.int8)
        cat_colors = {0: '#c8b8e8', 1: '#50c850', 2: '#222222', 3: '#222222'}
        cat_alpha  = {0: 0.45,      1: 0.55,      2: 0.8,       3: 0.8}
        for cat_val in [0, 3, 1, 2]:
            m = cats == cat_val
            if np.any(m):
                ax.scatter(local_pts[m, 0], local_pts[m, 1],
                           c=cat_colors[cat_val], s=64,
                           alpha=cat_alpha[cat_val],
                           zorder=1, marker='s', linewidths=0)

    # WP graph edges
    neighborhood_wps = set()
    observing_set = set(waypoint_ids)
    for ref_wp in waypoint_ids:
        if ref_wp not in WP_POS:
            continue
        ox, oy = WP_POS[ref_wp][0], WP_POS[ref_wp][1]
        for wp_id in observing_set:
            if wp_id not in WP_POS:
                continue
            pos = WP_POS[wp_id]
            dx, dy = pos[0] - ox, pos[1] - oy
            if dx * dx + dy * dy <= NEIGHBORHOOD_RADIUS ** 2:
                neighborhood_wps.add(wp_id)

    for u_node, v_node in WP_GRAPH.edges():
        if u_node in neighborhood_wps and v_node in neighborhood_wps:
            if u_node in WP_POS and v_node in WP_POS:
                ux, uy = WP_POS[u_node][0], WP_POS[u_node][1]
                vx, vy = WP_POS[v_node][0], WP_POS[v_node][1]
                ax.plot([ux, vx], [uy, vy],
                        color='#bbbbbb', lw=1.0, alpha=0.6, zorder=2)

    # WP dots with labels
    for wp_id in neighborhood_wps:
        if wp_id not in WP_POS:
            continue
        wx, wy = WP_POS[wp_id][0], WP_POS[wp_id][1]
        ax.scatter(wx, wy, c='#006600', s=60, marker='o', zorder=4,
                   edgecolors='darkgreen', linewidths=1.2)
        ax.annotate(str(wp_id), (wx, wy), fontsize=6, color='#222222',
                    fontweight='bold', xytext=(3, 3),
                    textcoords='offset points')

    return neighborhood_wps


def _finalize_plot(fig, ax, title, out_path):
    """Set common axis properties and save."""
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X (meters)', fontsize=13)
    ax.set_ylabel('Y (meters)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.04), loc='upper center',
              ncol=6, fontsize=9, markerscale=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_step1_boundary(frontier_id, wids, boundary_cells, out_dir):
    """Plot boundary extraction result on atlas map."""
    fig = plt.figure(figsize=(24, 20))
    ax  = fig.add_subplot(111)
    _draw_atlas_background(ax, wids)

    # Boundary cells as small blue dots
    if boundary_cells:
        bx = [gx * RESOLUTION for gx, gy in boundary_cells]
        by = [gy * RESOLUTION for gx, gy in boundary_cells]
        ax.scatter(bx, by, c='dodgerblue', s=8, alpha=0.6, zorder=5,
                   label=f'Boundary cells ({len(boundary_cells)})')

    # Legend placeholders
    ax.scatter([], [], c='#50c850', s=40, marker='s', label='Free')
    ax.scatter([], [], c='#222222', s=40, marker='s', label='Obstacle')
    ax.scatter([], [], c='#c8b8e8', s=40, marker='s', label='Unknown')
    ax.scatter([], [], c='#006600', s=40, marker='o', label='Observing WPs')

    _finalize_plot(fig, ax,
                   f'Geometric Boundary \u2014 Frontier {frontier_id} '
                   f'({len(boundary_cells)} boundary cells)',
                   os.path.join(out_dir, 'step1_boundary.png'))


def plot_step2_scored(frontier_id, wids, candidates, A0_frontiers, A0_gains,
                      n_zero, out_dir, filename='step2_scored.png'):
    """Plot scored candidates (gain>0 as cyan stars, zero as gray dots)."""
    fig = plt.figure(figsize=(24, 20))
    ax  = fig.add_subplot(111)
    _draw_atlas_background(ax, wids)

    # Zero-gain candidates as gray dots
    scored_set = set()
    for f in A0_frontiers:
        scored_set.add((f[0], f[1]))
    zero_x = [c[0] for c in candidates if (c[0], c[1]) not in scored_set]
    zero_y = [c[1] for c in candidates if (c[0], c[1]) not in scored_set]
    if zero_x:
        ax.scatter(zero_x, zero_y, c='#aaaaaa', s=6, alpha=0.4, zorder=4,
                   label=f'Zero gain ({len(zero_x)})')

    # Scored candidates sized by gain
    if A0_frontiers:
        sx = [f[0] for f in A0_frontiers]
        sy = [f[1] for f in A0_frontiers]
        sg = np.array(A0_gains)
        sizes = np.clip(30 + 60 * np.sqrt(sg), 30, 500)
        ax.scatter(sx, sy, c='cyan', s=sizes, marker='*', zorder=6,
                   edgecolors='darkcyan', linewidths=0.5, alpha=0.9,
                   label=f'Gain>0 ({len(A0_frontiers)})')

    ax.scatter([], [], c='#50c850', s=40, marker='s', label='Free')
    ax.scatter([], [], c='#222222', s=40, marker='s', label='Obstacle')
    ax.scatter([], [], c='#c8b8e8', s=40, marker='s', label='Unknown')

    n_total = len(candidates)
    n_scored = len(A0_frontiers)
    _finalize_plot(fig, ax,
                   f'{n_scored}/{n_total} candidates with gain>0',
                   os.path.join(out_dir, filename))


def plot_step2b_shifted(frontier_id, wids, pre_positions, candidates, out_dir):
    """Plot shift-to-safe result: arrows from pre to post positions.

    pre_positions and candidates have the same length (1:1 pairing).
    """
    fig = plt.figure(figsize=(24, 20))
    ax  = fig.add_subplot(111)
    _draw_atlas_background(ax, wids)

    n_shifted = 0
    for (ox, oy), cand in zip(pre_positions, candidates):
        nx, ny = cand[0], cand[1]
        moved = (abs(nx - ox) > 1e-6 or abs(ny - oy) > 1e-6)
        if moved:
            n_shifted += 1
            ax.annotate('', xy=(nx, ny), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle='->', color='#888888',
                                        lw=1.0, alpha=0.6),
                        zorder=5)
        ax.scatter(nx, ny, c='dodgerblue', s=8, alpha=0.6, zorder=6)

    ax.scatter([], [], c='dodgerblue', s=40,
               label=f'After shift ({len(candidates)}, {n_shifted} moved)')
    ax.scatter([], [], c='#50c850', s=40, marker='s', label='Free')
    ax.scatter([], [], c='#222222', s=40, marker='s', label='Obstacle')
    ax.scatter([], [], c='#c8b8e8', s=40, marker='s', label='Unknown')
    ax.scatter([], [], c='#006600', s=40, marker='o', label='Observing WPs')

    _finalize_plot(fig, ax,
                   f'Shift to Safe \u2014 {n_shifted}/{len(candidates)} shifted',
                   os.path.join(out_dir, 'step2b_shifted.png'))


def plot_step4_selected(frontier_id, wids, sel_frontiers, sel_gains,
                        dropped, coverage_frac, out_dir):
    """Plot selected vs unselected frontiers after set-cover."""
    fig = plt.figure(figsize=(24, 20))
    ax  = fig.add_subplot(111)
    _draw_atlas_background(ax, wids)

    # Unselected (dropped) as orange X
    if dropped:
        dx = [d['x'] for d in dropped]
        dy = [d['y'] for d in dropped]
        ax.scatter(dx, dy, c='orange', s=80, marker='X', zorder=5,
                   edgecolors='darkorange', linewidths=0.8, alpha=0.65,
                   label=f'Unselected ({len(dropped)})')

    # Selected as red stars with labels
    if sel_frontiers:
        for i, (f, g) in enumerate(zip(sel_frontiers, sel_gains)):
            sz = max(30, 30 + 60 * np.sqrt(g))
            ax.scatter(f[0], f[1], c='red', s=sz, marker='*', zorder=7,
                       edgecolors='darkred', linewidths=0.8)
            label_txt = f'#{i} {g:.0f}m\u00b2'
            ax.annotate(label_txt, (f[0], f[1]), xytext=(8, 8),
                        textcoords='offset points', fontsize=9,
                        color='darkred', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=3,
                                                    foreground='white')])
        # Legend entry for selected
        ax.scatter([], [], c='red', s=120, marker='*', edgecolors='darkred',
                   linewidths=0.5, label=f'Selected ({len(sel_frontiers)})')

    ax.scatter([], [], c='#50c850', s=40, marker='s', label='Free')
    ax.scatter([], [], c='#222222', s=40, marker='s', label='Obstacle')
    ax.scatter([], [], c='#c8b8e8', s=40, marker='s', label='Unknown')
    ax.scatter([], [], c='#006600', s=40, marker='o', label='Observing WPs')

    n_sel = len(sel_frontiers)
    _finalize_plot(fig, ax,
                   f'{n_sel} selected (coverage target={coverage_frac})',
                   os.path.join(out_dir, 'step4_selected.png'))


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Geometric boundary → UNet → set-cover pipeline')
    ap.add_argument('--frontier-id', type=int, required=True)
    ap.add_argument('--gain-checkpoint', type=str, required=True)
    ap.add_argument('--out-dir', type=str, default=None)
    ap.add_argument('--coverage-frac', type=float, default=0.40)
    ap.add_argument('--proximity-disq-m', type=float, default=2.0)
    ap.add_argument('--round', type=int, default=1)
    args = ap.parse_args()

    out_dir = args.out_dir or f"results/{args.frontier_id}_geometric"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"[geo] Device: {device}")

    timings = {}

    # ── Phase 0: Atlas + data load ────────────────────────────────────────
    t0 = time.time()
    print("[geo] Loading atlas ...")
    atlas_pts, wp_positions, g2c = load_atlas()

    print(f"[geo] Loading frontier {args.frontier_id} ...")
    frontier_data = load_frontier_data(args.frontier_id)
    wids = frontier_data['waypoint_ids']
    print(f"      obs_WPs={len(wids)}  total_WPs={len(wp_positions)}")

    # Build covered_g2c (same as frontier_refine.py main)
    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cat_f, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_cov_grid = build_coverage_grid(
        atlas_pts, cmask, gx_min, gy_min, atlas_cat_f.shape)
    covered_g2c = {k: v for k, v in g2c.items()
                   if atlas_cov_grid[k[0] - gx_min, k[1] - gy_min]}
    print(f"      covered atlas cells: {len(covered_g2c):,}")

    # Load gain model
    print("[geo] Loading gain model ...")
    UNet = _build_gain_model()
    gain_model = UNet().to(device)
    gain_model.load_state_dict(
        torch.load(args.gain_checkpoint, map_location=device, weights_only=True))
    gain_model.eval()
    timings['Atlas + data load'] = (time.time() - t0, '-')

    # ── Phase 1: Boundary extraction ──────────────────────────────────────
    t1 = time.time()
    print("[geo] Phase 1: Boundary extraction ...")
    boundary_cells = find_global_boundary(covered_g2c, include_absent=True)
    n_boundary = len(boundary_cells)
    timings['Boundary extraction'] = (time.time() - t1, f'{n_boundary:,}')
    print(f"      {n_boundary:,} boundary cells found")

    # ── Phase 2: WP assignment ────────────────────────────────────────────
    t2 = time.time()
    print("[geo] Phase 2: WP assignment ...")
    candidates = assign_nearest_wp(boundary_cells, wids, wp_positions)
    timings['WP assignment'] = (time.time() - t2, f'{len(candidates):,}')
    print(f"      {len(candidates):,} candidates assigned")

    # ── Step 1 plot ───────────────────────────────────────────────────────
    print("[geo] Plotting step 1 ...")
    plot_step1_boundary(args.frontier_id, wids, boundary_cells, out_dir)

    # ── Phase 2b: Shift to safe green ─────────────────────────────────────
    t2b = time.time()
    print("[geo] Phase 2b: Shift to safe green ...")
    candidates, pre_positions = shift_candidates_to_safe(
        candidates, covered_g2c)
    timings['Shift to safe'] = (time.time() - t2b, f'{len(candidates):,}')

    # ── Step 2b plot ──────────────────────────────────────────────────────
    print("[geo] Plotting step 2b ...")
    plot_step2b_shifted(args.frontier_id, wids, pre_positions, candidates,
                        out_dir)

    # ── Phase 3: UNet scoring ─────────────────────────────────────────────
    t3 = time.time()
    print("[geo] Phase 3: UNet scoring ...")
    wp_data_cache = {}
    (A0_frontiers, A0_gains, A0_connects, A0_masks,
     A0_pred_maps, n_zero) = score_candidates(
        candidates, covered_g2c, wp_positions, gain_model, device,
        wp_data_cache)
    n_scored = len(A0_frontiers)
    timings['UNet scoring'] = (time.time() - t3, f'{n_scored:,} (gain>0)')
    print(f"      {n_scored} with gain>0, {n_zero} zero")

    # ── Step 3 plot ───────────────────────────────────────────────────────
    print("[geo] Plotting step 3 ...")
    plot_step2_scored(args.frontier_id, wids, candidates, A0_frontiers,
                      A0_gains, n_zero, out_dir, filename='step3_scored.png')

    # ── Phase 4: Greedy set-cover ─────────────────────────────────────────
    t4 = time.time()
    print("[geo] Phase 4: Greedy set-cover ...")
    (sel_frontiers, sel_gains, sel_connects, sel_masks,
     sel_sources, dropped, sel_weights) = greedy_set_cover(
        A0_frontiers, A0_gains, A0_connects, A0_masks, A0_pred_maps,
        coverage_frac=args.coverage_frac,
        proximity_disq_m=args.proximity_disq_m)
    timings['Greedy set-cover'] = (time.time() - t4,
                                   f'{len(sel_frontiers)} selected')

    # ── Step 4 plot ───────────────────────────────────────────────────────
    print("[geo] Plotting step 4 ...")
    plot_step4_selected(args.frontier_id, wids, sel_frontiers, sel_gains,
                        dropped, args.coverage_frac, out_dir)

    # ── Save results ──────────────────────────────────────────────────────
    print("[geo] Saving results ...")
    # Tag source as 'geo' for geometric detections
    geo_sources = ['geo'] * len(sel_frontiers)
    save_round_results(out_dir, args.round, sel_frontiers, sel_gains,
                       sel_connects, sources=geo_sources, dropped=dropped,
                       weights=sel_weights, wp_positions=wp_positions,
                       wids=wids)

    # ── Timing summary ───────────────────────────────────────────────────
    total = sum(v[0] for v in timings.values())
    print()
    print(f"{'Phase':<25s} {'Time (s)':>10s}    {'Count':>20s}")
    print('\u2500' * 60)
    for phase, (dt, count) in timings.items():
        print(f"{phase:<25s} {dt:>10.1f}    {count:>20s}")
    print('\u2500' * 60)
    print(f"{'Total':<25s} {total:>10.1f}")
    print()


if __name__ == '__main__':
    main()
