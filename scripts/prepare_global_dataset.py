#!/usr/bin/env python3
"""prepare_global_dataset.py
=============================
Build training dataset for the global exploration gain ranking model.

For each frontier snapshot (exploration timestep):
  1. Compute sensor footprint (union of visited WPs' full camera coverage)
  2. Build 3-channel input map (free / obstacle / unexplored)
  3. Find boundary cells (observed free adjacent to unexplored)
  4. Compute GT gain per boundary cell (connected components of uncovered free)

Sensor footprint uses depth ray marching (full ~10m camera range),
NOT the artificial 5m WP_RADIUS used by the UNet pipeline.

Output: one .npz per frontier snapshot + metadata.json.

Usage:
    python3 scripts/prepare_global_dataset.py \\
        --data-dir set2 \\
        --out-dir dataset/global_gain
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

# ── Constants ────────────────────────────────────────────────────────────────
RESOLUTION = 0.2            # m per cell
CELL_AREA = RESOLUTION ** 2  # 0.04 m²
CAMERAS = ['back', 'frontleft', 'frontright', 'left', 'right']
FALLBACK_RADIUS = 5.0       # m, for WPs without depth data
RAY_SAMPLES = 100            # samples per ray for footprint
DEPTH_STEP = 4               # depth subsampling for footprint


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_atlas(data_dir):
    """Load atlas.pkl → (atlas_points_2d, waypoint_positions, grid_to_category)."""
    with open(os.path.join(data_dir, 'atlas.pkl'), 'rb') as f:
        a = pickle.load(f)
    return a['atlas_points_2d'], a['waypoint_positions'], a['grid_to_category']


def load_frontier_data(frontier_id, data_dir):
    """Load one frontier snapshot JSON."""
    with open(os.path.join(data_dir, 'frontiers', f'{frontier_id}.json')) as f:
        return json.load(f)


def build_atlas_grid(g2c):
    """Dict {(gx,gy): cat} → (2D int8 array, gx_min, gy_min).

    cat_array[gx - gx_min, gy - gy_min] = category  (-1 = no data).
    """
    keys = np.array(list(g2c.keys()))
    gx_min, gy_min = int(keys[:, 0].min()), int(keys[:, 1].min())
    gx_max, gy_max = int(keys[:, 0].max()), int(keys[:, 1].max())
    cat = np.full((gx_max - gx_min + 1, gy_max - gy_min + 1), -1, dtype=np.int8)
    for (gx, gy), c in g2c.items():
        cat[gx - gx_min, gy - gy_min] = c
    return cat, gx_min, gy_min


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Depth projection
# ═════════════════════════════════════════════════════════════════════════════

def project_depth_to_world_xy(depth_path, intrinsic, extrinsic, step=4):
    """Project depth map → (N, 2) world XY points.

    Handles portrait cameras (frontleft / frontright depth transposition).
    """
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[0] > depth.shape[1]:
        depth = depth.T
        depth = depth[::-1, :]

    H, W = depth.shape
    K = np.asarray(intrinsic, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    T_wc = np.asarray(extrinsic, dtype=np.float64)

    u, v = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    u, v = u.astype(np.float32), v.astype(np.float32)
    d = depth[v.astype(int), u.astype(int)]

    valid = (d > 1e-3) & (d < 10.0)
    u, v, d = u[valid], v[valid], d[valid]
    if len(d) == 0:
        return np.empty((0, 2))

    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    pts_cam = np.stack([x, y, d, np.ones_like(d)], axis=1)
    pts_world = (T_wc @ pts_cam.T).T
    return pts_world[:, :2]


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Sensor footprint computation
# ═════════════════════════════════════════════════════════════════════════════

def _mark_points(grid, pts_xy, gx_min, gy_min):
    """Mark world-XY points on a boolean grid (in-place)."""
    gx = np.round(pts_xy[:, 0] / RESOLUTION).astype(int) - gx_min
    gy = np.round(pts_xy[:, 1] / RESOLUTION).astype(int) - gy_min
    ok = (gx >= 0) & (gx < grid.shape[0]) & (gy >= 0) & (gy < grid.shape[1])
    grid[gx[ok], gy[ok]] = True


def _radius_footprint(wp_xy, radius, gx_min, gy_min, grid_shape):
    """Circular fallback footprint for WPs without depth data."""
    fp = np.zeros(grid_shape, dtype=bool)
    cr = int(round(wp_xy[0] / RESOLUTION)) - gx_min
    cc = int(round(wp_xy[1] / RESOLUTION)) - gy_min
    r = int(radius / RESOLUTION)

    dr, dc = np.mgrid[-r:r + 1, -r:r + 1]
    mask = dr * dr + dc * dc <= r * r
    rows = cr + dr[mask]
    cols = cc + dc[mask]
    ok = (rows >= 0) & (rows < grid_shape[0]) & (cols >= 0) & (cols < grid_shape[1])
    fp[rows[ok], cols[ok]] = True
    return fp


def compute_wp_footprint(wp_id, wp_pos, data_dir,
                          gx_min, gy_min, grid_shape, depth_step=DEPTH_STEP):
    """Sensor footprint for one WP: depth endpoints + ray-marched cells.

    Falls back to radius-based footprint if no depth data is available.
    """
    fp = np.zeros(grid_shape, dtype=bool)
    wp_xy = np.array([wp_pos[0], wp_pos[1]])

    wp_dir = os.path.join(data_dir, 'waypoints', str(wp_id))
    meta_path = os.path.join(wp_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return _radius_footprint(wp_xy, FALLBACK_RADIUS,
                                 gx_min, gy_min, grid_shape)

    with open(meta_path) as f:
        meta = json.load(f)

    all_endpoints = []
    for cam in CAMERAS:
        if cam not in meta.get('cameras', {}):
            continue
        dpath = os.path.join(wp_dir, f'{cam}_depth.npy')
        if not os.path.exists(dpath):
            continue
        ci = meta['cameras'][cam]
        pts = project_depth_to_world_xy(
            dpath, ci['camera_intrinsic'], ci['camera_extrinsic'],
            step=depth_step)
        if len(pts) > 0:
            all_endpoints.append(pts)

    if not all_endpoints:
        return _radius_footprint(wp_xy, FALLBACK_RADIUS,
                                 gx_min, gy_min, grid_shape)

    endpoints = np.vstack(all_endpoints)        # (N, 2)

    # 1. Mark depth endpoints
    _mark_points(fp, endpoints, gx_min, gy_min)

    # 2. Ray march: sample along WP → endpoint rays
    t = np.linspace(0, 1, RAY_SAMPLES).reshape(1, -1)  # (1, T)
    ray_x = wp_xy[0] + (endpoints[:, 0:1] - wp_xy[0]) * t   # (N, T)
    ray_y = wp_xy[1] + (endpoints[:, 1:2] - wp_xy[1]) * t   # (N, T)
    ray_pts = np.stack([ray_x.ravel(), ray_y.ravel()], axis=1)
    _mark_points(fp, ray_pts, gx_min, gy_min)

    return fp


def precompute_footprints(data_dir, wp_positions, gx_min, gy_min, grid_shape,
                           cache_path, depth_step=DEPTH_STEP):
    """Compute (or load cached) per-WP sensor footprints."""
    if os.path.exists(cache_path):
        print(f"  Loading cached footprints from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    n = len(wp_positions)
    print(f"  Computing footprints for {n} waypoints ...")
    footprints = {}
    for i, (wp_id, wp_pos) in enumerate(wp_positions.items()):
        footprints[wp_id] = compute_wp_footprint(
            wp_id, wp_pos, data_dir, gx_min, gy_min, grid_shape,
            depth_step=depth_step)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"    {i + 1}/{n}")

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(footprints, f)
    print(f"  Cached → {cache_path}")
    return footprints


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Per-sample construction
# ═════════════════════════════════════════════════════════════════════════════

def build_coverage_mask(waypoint_ids, wp_footprints, grid_shape):
    """Union of sensor footprints for the visited WP set."""
    cov = np.zeros(grid_shape, dtype=bool)
    for wp_id in waypoint_ids:
        if wp_id in wp_footprints:
            cov |= wp_footprints[wp_id]
    return cov


def build_input_map(coverage, atlas_cat):
    """3-channel input: observed free / observed obstacle / unexplored.

    Only cells that are BOTH covered AND have atlas data are shown;
    everything else is marked unexplored (channel 2).
    """
    inp = np.zeros((3, *atlas_cat.shape), dtype=np.float32)
    observed = coverage & (atlas_cat >= 0)

    inp[0] = (observed & (atlas_cat == 1)).astype(np.float32)            # free
    inp[1] = (observed & ((atlas_cat == 2) | (atlas_cat == 3))).astype(np.float32)  # obstacle
    inp[2] = (~observed).astype(np.float32)                              # unexplored
    return inp


def find_boundary_cells(input_map):
    """Observed free cells with at least one 4-connected unexplored neighbour.

    Returns (K, 2) int32 array of (row, col) positions in the atlas grid.
    """
    free = input_map[0] > 0
    unexplored = input_map[2] > 0

    # Dilate unexplored by 1 cell (4-connected)
    struct = ndimage.generate_binary_structure(2, 1)
    adj_unexplored = ndimage.binary_dilation(unexplored, struct)

    boundary = free & adj_unexplored
    return np.argwhere(boundary).astype(np.int32)   # (K, 2)


def compute_gt_gains(boundary_coords, coverage, atlas_cat):
    """GT gain per boundary cell via connected components.

    Gain = total area (m²) of uncovered free cells reachable from this
    boundary cell through 4-connected uncovered free space.

    Efficient: one connected-component pass, then per-boundary lookup.
    """
    # Mask of uncovered-but-free cells in the full atlas
    uncovered_free = (~coverage) & (atlas_cat == 1)

    struct = ndimage.generate_binary_structure(2, 1)       # 4-connected
    labeled, n_comp = ndimage.label(uncovered_free, structure=struct)

    if n_comp == 0:
        return np.zeros(len(boundary_coords), dtype=np.float32)

    # Area of each component (index 0 = background)
    comp_area = np.bincount(labeled.ravel()).astype(np.float32) * CELL_AREA

    H, W = labeled.shape
    gains = np.zeros(len(boundary_coords), dtype=np.float32)

    for i, (r, c) in enumerate(boundary_coords):
        adj = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                cid = labeled[nr, nc]
                if cid > 0:
                    adj.add(cid)
        gains[i] = sum(comp_area[c] for c in adj)

    return gains


def prepare_one_sample(frontier_id, data_dir, atlas_cat, gx_min, gy_min,
                        wp_positions, wp_footprints):
    """Build one training sample. Returns dict or None if degenerate."""
    fdata = load_frontier_data(frontier_id, data_dir)
    wids = fdata['waypoint_ids']

    coverage = build_coverage_mask(wids, wp_footprints, atlas_cat.shape)
    input_map = build_input_map(coverage, atlas_cat)
    boundary = find_boundary_cells(input_map)

    if len(boundary) == 0:
        return None

    gt_gains = compute_gt_gains(boundary, coverage, atlas_cat)

    return {
        'input_map':       input_map,             # (3, H, W) float32
        'boundary_coords': boundary,              # (K, 2)    int32
        'gt_gains':        gt_gains,              # (K,)      float32
        'frontier_id':     frontier_id,
        'n_wps':           len(wids),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5.  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Prepare dataset for global exploration gain ranking model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data-dir', type=str, default='set2',
                    help='Path to TIAMAT data directory (contains atlas.pkl)')
    ap.add_argument('--out-dir', type=str, default='dataset/global_gain',
                    help='Output directory for .npz samples + metadata')
    ap.add_argument('--footprint-cache', type=str, default=None,
                    help='Footprint cache path (default: <out-dir>/wp_footprints.pkl)')
    ap.add_argument('--depth-step', type=int, default=DEPTH_STEP,
                    help='Depth sub-sampling step for footprint rays')
    ap.add_argument('--min-boundary', type=int, default=10,
                    help='Skip samples with fewer boundary cells')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Atlas ──
    print("Loading atlas ...")
    atlas_pts, wp_positions, g2c = load_atlas(args.data_dir)
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)
    print(f"  Grid : {atlas_cat.shape[0]} x {atlas_cat.shape[1]}  "
          f"({np.sum(atlas_cat >= 0):,} cells with data)")
    print(f"  WPs  : {len(wp_positions)}")

    # ── Footprints ──
    cache = args.footprint_cache or str(out_dir / 'wp_footprints.pkl')
    wp_fp = precompute_footprints(
        args.data_dir, wp_positions, gx_min, gy_min, atlas_cat.shape,
        cache_path=cache, depth_step=args.depth_step)

    # ── Frontier files ──
    fdir = os.path.join(args.data_dir, 'frontiers')
    fids = sorted(int(f.replace('.json', ''))
                  for f in os.listdir(fdir) if f.endswith('.json'))
    print(f"\nProcessing {len(fids)} frontier snapshots ...")

    meta_records = []
    n_saved = n_skip = 0

    for i, fid in enumerate(fids):
        sample = prepare_one_sample(
            fid, args.data_dir, atlas_cat, gx_min, gy_min,
            wp_positions, wp_fp)

        if sample is None or len(sample['boundary_coords']) < args.min_boundary:
            n_skip += 1
            continue

        fname = f'sample_{fid:06d}.npz'
        np.savez_compressed(
            out_dir / fname,
            input_map=sample['input_map'],
            boundary_coords=sample['boundary_coords'],
            gt_gains=sample['gt_gains'],
        )

        meta_records.append({
            'frontier_id':  fid,
            'file':         fname,
            'n_wps':        sample['n_wps'],
            'n_boundary':   len(sample['boundary_coords']),
            'gain_min':     float(sample['gt_gains'].min()),
            'gain_max':     float(sample['gt_gains'].max()),
            'gain_mean':    float(sample['gt_gains'].mean()),
        })
        n_saved += 1

        if (i + 1) % 100 == 0 or i + 1 == len(fids):
            print(f"  {i + 1}/{len(fids)}  "
                  f"(saved={n_saved}, skipped={n_skip})")

    # ── Metadata ──
    meta_path = out_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump({
            'resolution':  RESOLUTION,
            'atlas_shape': list(atlas_cat.shape),
            'gx_min':      gx_min,
            'gy_min':      gy_min,
            'n_samples':   n_saved,
            'n_skipped':   n_skip,
            'samples':     meta_records,
        }, f, indent=2)

    print(f"\nDone.  {n_saved} samples → {out_dir}/")
    print(f"  Skipped: {n_skip}  (< {args.min_boundary} boundary cells)")
    print(f"  Metadata: {meta_path}")


if __name__ == '__main__':
    main()
