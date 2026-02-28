#!/usr/bin/env python3
"""plot_coverage_heatmap.py
===========================
Visualize pre-refinement (A0) frontier predictions and their predicted
free-cell coverage as a heatmap.

Each cell is colored by its mean evidence: mean of -log(1 - p_i) across
all A0 frontiers that predicted it as free.  This is density-invariant
and reflects the UNet's confidence that a cell is free.

Usage
-----
python3 visualization/plot_coverage_heatmap.py \
    --frontier-id 7000 \
    --a0-json results/7000_1round/a0_frontiers_round1.json \
    --evidence-pkl results/7000_1round/a0_mean_evidence_round1.pkl \
    --out-dir results/7000_1round
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize

from visualization import plot_predictions as _bp
from models.map_utils import load_frontier_data

RESOLUTION = 0.2  # metres per cell (must match frontier_gain_model.py)


def visualise_coverage(frontier_id, a0_json_path, evidence_pkl_path, out_dir):
    """Plot A0 frontiers with per-cell mean-evidence heatmap on the atlas map."""

    stem      = Path(a0_json_path).stem
    final_png = os.path.join(out_dir, f"{stem}_coverage.png")

    print(f"  Loading frontier {frontier_id} ...")
    frontier_data = load_frontier_data(frontier_id)
    waypoint_ids  = frontier_data["waypoint_ids"]

    print(f"  Loading A0 frontiers from {a0_json_path} ...")
    with open(a0_json_path) as f:
        a0_entries = json.load(f)

    print(f"  Loading mean evidence from {evidence_pkl_path} ...")
    with open(evidence_pkl_path, 'rb') as f:
        mean_evidence = pickle.load(f)

    n_frontiers = len(a0_entries)
    n_cells     = len(mean_evidence)
    max_ev      = max(mean_evidence.values()) if mean_evidence else 0
    print(f"  A0 frontiers={n_frontiers}  cells={n_cells}  "
          f"max evidence={max_ev:.2f}")

    # ── Atlas globals ─────────────────────────────────────────────────────────
    WP_POS    = _bp._WP_POSITIONS
    ATLAS_PTS = _bp._ATLAS_POINTS_2D
    ATLAS_GI  = _bp._ATLAS_GRID_IDX
    G2C       = _bp._GRID_TO_CAT
    WP_GRAPH  = _bp._WP_GRAPH

    NEIGHBORHOOD_RADIUS = 5.0
    all_wp_xy = np.array([[WP_POS[w][0], WP_POS[w][1]]
                           for w in waypoint_ids if w in WP_POS])

    fig = plt.figure(figsize=(24, 20))
    ax  = fig.add_subplot(111)

    # ── Atlas scatter (background) ────────────────────────────────────────────
    if len(all_wp_xy) > 0:
        R  = NEIGHBORHOOD_RADIUS
        R2 = R * R
        bbox_min = all_wp_xy.min(axis=0) - R
        bbox_max = all_wp_xy.max(axis=0) + R
        box_mask = ((ATLAS_PTS[:, 0] >= bbox_min[0]) &
                    (ATLAS_PTS[:, 0] <= bbox_max[0]) &
                    (ATLAS_PTS[:, 1] >= bbox_min[1]) &
                    (ATLAS_PTS[:, 1] <= bbox_max[1]))
        box_idx  = np.where(box_mask)[0]
        box_pts  = ATLAS_PTS[box_idx]
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
        cat_alpha  = {0: 0.35,      1: 0.40,      2: 0.6,       3: 0.6}
        for cat_val in [0, 3, 1, 2]:
            m = cats == cat_val
            if np.any(m):
                ax.scatter(local_pts[m, 0], local_pts[m, 1],
                           c=cat_colors[cat_val], s=64,
                           alpha=cat_alpha[cat_val],
                           zorder=1, marker='s', linewidths=0)

    # ── Evidence heatmap layer ────────────────────────────────────────────────
    if mean_evidence:
        cells_xy = np.array([[gx * RESOLUTION, gy * RESOLUTION]
                              for gx, gy in mean_evidence.keys()])
        values   = np.array(list(mean_evidence.values()), dtype=np.float32)

        cmap = plt.cm.YlOrRd
        norm = Normalize(vmin=0, vmax=max(max_ev, 0.1))
        sc = ax.scatter(cells_xy[:, 0], cells_xy[:, 1],
                        c=values, cmap=cmap, norm=norm,
                        s=20, marker='s', alpha=0.75,
                        zorder=3, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Mean evidence: mean of $-\\log(1 - p_i)$',
                       fontsize=12)

    # ── WP graph edges ────────────────────────────────────────────────────────
    neighborhood_wps = set()
    for ref_wp in waypoint_ids:
        if ref_wp not in WP_POS:
            continue
        ox, oy = WP_POS[ref_wp][0], WP_POS[ref_wp][1]
        for wp_id, pos in WP_POS.items():
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

    # ── WP dots ──────────────────────────────────────────────────────────────
    a0_wp_set = {int(d['wp_id']) for d in a0_entries}
    for wp_id in neighborhood_wps:
        if wp_id not in WP_POS:
            continue
        wx, wy = WP_POS[wp_id][0], WP_POS[wp_id][1]
        if wp_id in a0_wp_set:
            ax.scatter(wx, wy, c='#006600', s=60, marker='o', zorder=5,
                       edgecolors='darkgreen', linewidths=1.2)
        else:
            ax.scatter(wx, wy, c='#888888', s=30, marker='o', zorder=4,
                       edgecolors='white', linewidths=0.4)
        ax.annotate(str(wp_id), (wx, wy), fontsize=6, color='#222222',
                    fontweight='bold',
                    xytext=(3, 3), textcoords='offset points')

    # ── A0 frontier markers ──────────────────────────────────────────────────
    for d in a0_entries:
        x, y = d['x'], d['y']
        ax.scatter(x, y, c='cyan', s=160, marker='*',
                   zorder=6, alpha=0.9,
                   edgecolors='darkblue', linewidths=0.8)
        fid = d.get('id')
        if fid is not None:
            ax.annotate(f'#{fid}', (x, y), xytext=(6, 6),
                        textcoords='offset points', fontsize=8,
                        color='darkblue', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground='white')])

    # ── Legend + title ────────────────────────────────────────────────────────
    ax.scatter([], [], c='cyan', s=160, marker='*', edgecolors='darkblue',
               linewidths=0.8, label=f'A0 frontiers ({n_frontiers})')
    ax.scatter([], [], c='#50c850', s=40, marker='s', label='Free')
    ax.scatter([], [], c='#222222', s=40, marker='s', label='Obstacle')
    ax.scatter([], [], c='#c8b8e8', s=40, marker='s', label='Unknown')
    ax.scatter([], [], c='#888888', s=20, marker='o', label='Nearby WPs')
    ax.scatter([], [], c='#006600', s=40, marker='o', label='Active WPs')

    ax.set_title(
        f'A0 Mean Evidence — Frontier {frontier_id}  '
        f'(A0={n_frontiers} predictions, {n_cells} cells, '
        f'max evidence={max_ev:.2f})\n'
        f'[{stem}]',
        fontsize=14)
    ax.set_xlabel('X (meters)', fontsize=13)
    ax.set_ylabel('Y (meters)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.04), loc='upper center',
              ncol=6, fontsize=9, markerscale=1.5)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(final_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {final_png}")
    return final_png


def main():
    ap = argparse.ArgumentParser(
        description='Visualize A0 (pre-refinement) mean evidence heatmap',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--frontier-id', type=int, required=True,
                    help='Frontier step ID')
    ap.add_argument('--a0-json', required=True,
                    help='Path to a0_frontiers_round*.json')
    ap.add_argument('--evidence-pkl', required=True,
                    help='Path to a0_mean_evidence_round*.pkl')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='Output directory (default: same dir as a0-json)')
    args = ap.parse_args()

    out_dir = args.out_dir or str(Path(args.a0_json).parent)
    visualise_coverage(args.frontier_id, args.a0_json,
                       args.evidence_pkl, out_dir)
    print(f"\nDone. Image saved to: {out_dir}")


if __name__ == '__main__':
    main()
