#!/usr/bin/env python3
"""vis_pseudo_labels.py
========================
Visualize the pseudo-label (refined frontier) JSON files produced by
self_supervised_refine.py on the same atlas map used by batch_plot.py.

Entries are color-coded by their ``source`` field:
  keep  – red ★   (DETR predictions that survived DROP+MERGE)
  add   – green ★  (boundary candidates added in ADD step)
  drop  – orange ✗ (removed in DROP step, low gain)
  merge – gray ✗   (removed in MERGE step, overlapping)

Old JSON files without a ``source`` field are backward-compatible and are
treated as ``keep``.

Usage
-----
# Single JSON
python3 vis_pseudo_labels.py \\
    --frontier-id 6549 \\
    --json ss_output/6549/refined_round1.json \\
    --out-dir ss_output/6549/pseudo_vis

# All rounds at once
python3 vis_pseudo_labels.py \\
    --frontier-id 6549 \\
    --json ss_output/6549/refined_round1.json \\
           ss_output/6549/refined_round2.json \\
           ss_output/6549/refined_round3.json \\
    --out-dir ss_output/6549/pseudo_vis

Output: <out-dir>/<json-stem>.png  (e.g. refined_round1.png)

Environment variable TIAMAT_DATA_DIR controls data source (default set2).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Importing plot_predictions triggers atlas loading into module-level globals
from visualization import plot_predictions as _bp
from models.map_utils import load_frontier_data

# ── Colour / marker style per source tag ─────────────────────────────────────
_SOURCE_STYLE = {
    'keep':  dict(color='red',       edge='darkred',    marker='*', zorder=7, alpha=1.0),
    'add':   dict(color='limegreen', edge='darkgreen',  marker='*', zorder=6, alpha=1.0),
    'drop':  dict(color='orange',    edge='darkorange', marker='X', zorder=5, alpha=0.65),
    'merge': dict(color='#aaaaaa',   edge='#555555',    marker='X', zorder=5, alpha=0.55),
}
_DEFAULT_SOURCE = 'keep'


def load_refined_json(json_path):
    """Load a refined_round*.json and return list of entry dicts.

    Each entry has: x, y, conf, wp_id, gain_m2, connects, source.
    Entries without a ``source`` field default to 'keep' (backward compat).
    """
    with open(json_path) as f:
        data = json.load(f)
    for d in data:
        if 'source' not in d:
            d['source'] = _DEFAULT_SOURCE
    return data


def visualise_one(frontier_id, json_path, out_dir):
    """Plot pseudo-labels from one JSON onto the atlas map, color-coded by source.

    Saves: <out_dir>/<stem>.png
    """
    stem      = Path(json_path).stem
    final_png = os.path.join(out_dir, f"{stem}.png")

    print(f"  Loading frontier {frontier_id} ...")
    frontier_data = load_frontier_data(frontier_id)
    waypoint_ids  = frontier_data["waypoint_ids"]
    n_gt = len(frontier_data.get('frontier_positions', []))

    print(f"  Loading pseudo-labels from {json_path} ...")
    entries = load_refined_json(json_path)

    # Split by source
    groups = {src: [] for src in _SOURCE_STYLE}
    for d in entries:
        src = d.get('source', _DEFAULT_SOURCE)
        if src not in groups:
            src = _DEFAULT_SOURCE
        groups[src].append(d)

    n_keep  = len(groups['keep'])
    n_add   = len(groups['add'])
    n_drop  = len(groups['drop'])
    n_merge = len(groups['merge'])
    total_live = n_keep + n_add
    print(f"  GT={n_gt}  keep={n_keep}  add={n_add}  drop={n_drop}  merge={n_merge}")

    # ── Atlas globals loaded by batch_plot at import time ─────────────────────
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

    # ── Atlas scatter ─────────────────────────────────────────────────────────
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
        cats = np.array([int(G2C.get((g[0], g[1]), 0)) for g in local_gi], dtype=np.int8)
        cat_colors = {0: '#c8b8e8', 1: '#50c850', 2: '#222222', 3: '#222222'}
        cat_alpha  = {0: 0.45,      1: 0.55,      2: 0.8,       3: 0.8}
        for cat_val in [0, 3, 1, 2]:
            m = cats == cat_val
            if np.any(m):
                ax.scatter(local_pts[m, 0], local_pts[m, 1],
                           c=cat_colors[cat_val], s=64,
                           alpha=cat_alpha[cat_val],
                           zorder=1, marker='s', linewidths=0)

    # ── WP graph edges + neighbourhood dots ───────────────────────────────────
    # Only show observing WPs and their neighbors that are also observing WPs
    neighborhood_wps = set()
    observing_wps_set = set(waypoint_ids)  # Convert to set for fast lookup
    for ref_wp in waypoint_ids:
        if ref_wp not in WP_POS:
            continue
        ox, oy = WP_POS[ref_wp][0], WP_POS[ref_wp][1]
        for wp_id in observing_wps_set:  # Only check observing WPs
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

    # WPs that have surviving (non-dropped) predictions
    active_wp_set = {int(d['wp_id']) for d in entries
                     if d.get('source', _DEFAULT_SOURCE) not in ('drop', 'merge')}
    for wp_id in neighborhood_wps:
        if wp_id not in WP_POS:
            continue
        wx, wy = WP_POS[wp_id][0], WP_POS[wp_id][1]
        if wp_id in active_wp_set:
            ax.scatter(wx, wy, c='#006600', s=60, marker='o', zorder=4,
                       edgecolors='darkgreen', linewidths=1.2)
        else:
            ax.scatter(wx, wy, c='#888888', s=30, marker='o', zorder=3,
                       edgecolors='white', linewidths=0.4)
        ax.annotate(str(wp_id), (wx, wy), fontsize=6, color='#222222',
                    fontweight='bold',
                    xytext=(3, 3), textcoords='offset points')

    # ── Frontier markers colour-coded by source ───────────────────────────────
    for src, style in _SOURCE_STYLE.items():
        grp = groups[src]
        if not grp:
            continue
        is_dropped = src in ('drop', 'merge')
        for d in grp:
            x, y  = d['x'], d['y']
            g     = d.get('gain_m2')
            conn  = d.get('connects', False)
            fid   = d.get('id')
            sz    = 80 if is_dropped else (max(30, 30 + 60 * np.sqrt(g)) if g is not None else 200)
            ax.scatter(x, y,
                       c=style['color'], s=sz, marker=style['marker'],
                       zorder=style['zorder'], alpha=style['alpha'],
                       edgecolors=style['edge'], linewidths=0.8)
            # Combined ID + gain label
            if not is_dropped:
                id_part   = f'#{fid} ' if fid is not None else ''
                gain_part = f'{g:.0f}m²' if g is not None else ''
                conn_part = '+' if conn else ''
                label_txt = id_part + gain_part + conn_part
                if label_txt:
                    ax.annotate(label_txt, (x, y), xytext=(8, 8),
                                textcoords='offset points', fontsize=9,
                                color=style['edge'], fontweight='bold',
                                path_effects=[pe.withStroke(linewidth=3,
                                                            foreground='white')])
            else:
                # Dropped entries: show only #id (smaller, above marker)
                if fid is not None:
                    ax.annotate(f'#{fid}', (x, y), xytext=(0, 7),
                                textcoords='offset points', fontsize=7,
                                color=style['edge'], ha='center',
                                path_effects=[pe.withStroke(linewidth=2,
                                                            foreground='white')])

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.scatter([], [], c='red',       s=120, marker='*', edgecolors='darkred',    linewidths=0.5,
               label=f'keep ({n_keep})')
    ax.scatter([], [], c='limegreen', s=120, marker='*', edgecolors='darkgreen',  linewidths=0.5,
               label=f'add ({n_add})')
    ax.scatter([], [], c='orange',    s=80,  marker='X', edgecolors='darkorange', linewidths=0.5,
               alpha=0.7, label=f'drop ({n_drop})')
    ax.scatter([], [], c='#aaaaaa',   s=80,  marker='X', edgecolors='#555555',    linewidths=0.5,
               alpha=0.6, label=f'merge ({n_merge})')
    ax.scatter([], [], c='#50c850',   s=40,  marker='s', label='Free')
    ax.scatter([], [], c='#222222',   s=40,  marker='s', label='Obstacle')
    ax.scatter([], [], c='#c8b8e8',   s=40,  marker='s', label='Unknown')
    ax.scatter([], [], c='#888888',   s=20,  marker='o', label='Nearby WPs')
    ax.scatter([], [], c='#006600',   s=40,  marker='o', label='Active WPs')

    ax.set_title(
        f'Pseudo-Labels — Frontier {frontier_id}  '
        f'(GT={n_gt}, live={total_live}, dropped={n_drop + n_merge})\n'
        f'keep={n_keep}  add={n_add}  drop={n_drop}  merge={n_merge}   '
        f'[{stem}]',
        fontsize=14)
    ax.set_xlabel('X (meters)', fontsize=13)
    ax.set_ylabel('Y (meters)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.04), loc='upper center',
              ncol=9, fontsize=9, markerscale=1.5)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(final_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {final_png}")
    return final_png


def main():
    ap = argparse.ArgumentParser(
        description='Visualize self-supervised pseudo-label JSON files (source-coded colors)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--frontier-id', type=int, required=True,
                    help='Frontier step ID (same as used in self_supervised_refine.py)')
    ap.add_argument('--json', nargs='+', required=True,
                    help='Path(s) to refined_round*.json file(s)')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='Output directory (default: same dir as first JSON)')
    args = ap.parse_args()

    out_dir = args.out_dir or str(Path(args.json[0]).parent)
    os.makedirs(out_dir, exist_ok=True)

    for json_path in args.json:
        print(f"\n[vis] {json_path}")
        visualise_one(args.frontier_id, json_path, out_dir)

    print(f"\nDone. Images saved to: {out_dir}")


if __name__ == '__main__':
    main()
