#!/usr/bin/env python3
"""Frontier gain prediction network.

Train a UNet to predict how much free space each frontier will reveal.

Usage:
    python frontier_gain_model.py prepare --out gain_cache
    python frontier_gain_model.py train --data gain_cache --epochs 50 --batch-size 32
    python frontier_gain_model.py predict --frontier-id 89 --checkpoint gain_cache/model_final.pt
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

# Reuse path constants and loader from map_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_utils import (load_frontier_data, load_waypoint_meta,
                       BASE_DIR, DUMP_DIR, FRONTIERS_DIR, WP_IMAGES_DIR, CAMERAS)

# ── Grid constants ──────────────────────────────────────────────────────────
RESOLUTION = 0.2          # metres per cell
GRID_SIZE = 100            # cells per side
HALF_EXTENT = GRID_SIZE * RESOLUTION / 2.0   # 10.0 m
WP_RADIUS = 5.0           # per-WP atlas coverage radius (metres)
CELL_AREA = RESOLUTION ** 2   # 0.04 m²
FRONTIER_RC = (GRID_SIZE // 2, GRID_SIZE // 2)   # frontier center in grid
IGNORE_LABEL = 255   # label for covered cells (excluded from loss)
NUM_CLASSES = 4       # free=0, obstacle=1, unknown=2, unobserved=3


# ── Connected component utilities ───────────────────────────────────────────
def reachable_gain(gain_mask, free_mask, center=FRONTIER_RC):
    """Find gain cells reachable from *center* through free space.

    Flood-fills through (free | gain) starting from the frontier, then
    returns only the gain portion of that connected region.

    Args:
        gain_mask : (H, W) binary – predicted (or GT) gain cells.
        free_mask : (H, W) binary – known free cells from the input grid.
        center    : (row, col) of the frontier.

    Returns:
        reachable : (H, W) uint8 – gain cells reachable from frontier.
        bridge    : (H, W) uint8 – free cells that form the path between
                    frontier and the reachable gain (for visualisation).
    """
    from scipy import ndimage

    walkable = (free_mask | gain_mask).astype(np.uint8)
    labeled, n = ndimage.label(walkable)
    zeros = np.zeros_like(gain_mask, dtype=np.uint8)
    if n == 0:
        return zeros, zeros.copy()

    cr, cc = center
    if labeled[cr, cc] > 0:
        cid = labeled[cr, cc]
    else:
        # Frontier not in walkable area – pick nearest walkable cell
        coords = np.argwhere(labeled > 0)
        d2 = (coords[:, 0] - cr) ** 2 + (coords[:, 1] - cc) ** 2
        nr, nc = coords[np.argmin(d2)]
        cid = labeled[nr, nc]

    component = labeled == cid
    reachable = (component & gain_mask.astype(bool)).astype(np.uint8)
    bridge    = (component & free_mask.astype(bool) &
                 ~gain_mask.astype(bool)).astype(np.uint8)
    return reachable, bridge


def connected_to_frontier(mask, center=FRONTIER_RC):
    """Legacy wrapper – gain-only flood fill (no free-space bridging)."""
    from scipy import ndimage
    labeled, n = ndimage.label(mask.astype(np.uint8))
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    cr, cc = center
    if labeled[cr, cc] > 0:
        return (labeled == labeled[cr, cc]).astype(np.uint8)
    coords = np.argwhere(labeled > 0)
    d2 = (coords[:, 0] - cr) ** 2 + (coords[:, 1] - cc) ** 2
    nr, nc = coords[np.argmin(d2)]
    return (labeled == labeled[nr, nc]).astype(np.uint8)


# ── Atlas loading ───────────────────────────────────────────────────────────
def load_atlas():
    """Load atlas data independently (no global variables)."""
    path = os.path.join(DUMP_DIR, "atlas.pkl")
    with open(path, 'rb') as f:
        a = pickle.load(f)

    pts = a['atlas_points_2d']          # (N, 2) world XY
    g2c = a['grid_to_category']         # {(gx, gy): 0/1/2/3}
    wp_pos = a['waypoint_positions']    # {wp_id: [x, y, z]}

    return pts, wp_pos, g2c


def build_atlas_grid(grid_to_cat):
    """Convert grid_to_category dict to a 2D numpy array for fast lookup.

    Returns:
        cat_array : 2D int8 array (-1 = no atlas data, 0-3 = categories)
        gx_min, gy_min : offsets so that atlas grid (gx, gy) maps to
                         cat_array[gx - gx_min, gy - gy_min]
    """
    keys = np.array(list(grid_to_cat.keys()))
    gx_min, gy_min = int(keys[:, 0].min()), int(keys[:, 1].min())
    gx_max, gy_max = int(keys[:, 0].max()), int(keys[:, 1].max())
    cat_array = np.full((gx_max - gx_min + 1, gy_max - gy_min + 1),
                        -1, dtype=np.int8)
    for (gx, gy), cat in grid_to_cat.items():
        cat_array[gx - gx_min, gy - gy_min] = cat
    return cat_array, gx_min, gy_min


def build_coverage_grid(atlas_pts, current_mask, gx_min, gy_min, grid_shape):
    """Build 2D bool grid: True = cell covered by current WPs."""
    cov = np.zeros(grid_shape, dtype=bool)
    idx = np.where(current_mask)[0]
    if len(idx) == 0:
        return cov
    gx = (np.floor(atlas_pts[idx, 0] / RESOLUTION).astype(int) - gx_min)
    gy = (np.floor(atlas_pts[idx, 1] / RESOLUTION).astype(int) - gy_min)
    valid = ((gx >= 0) & (gx < grid_shape[0]) &
             (gy >= 0) & (gy < grid_shape[1]))
    cov[gx[valid], gy[valid]] = True
    return cov


# ── Grid building ───────────────────────────────────────────────────────────
def get_current_map_mask(waypoint_ids, wp_positions, atlas_points_2d,
                         wp_radius=WP_RADIUS):
    """Return boolean mask (N_atlas,) for atlas points within wp_radius
    of any waypoint in waypoint_ids."""
    N = len(atlas_points_2d)
    mask = np.zeros(N, dtype=bool)

    wp_xy = np.array([[wp_positions[w][0], wp_positions[w][1]]
                       for w in waypoint_ids if w in wp_positions])
    if len(wp_xy) == 0:
        return mask

    # Bounding-box pre-filter
    bbox_min = wp_xy.min(axis=0) - wp_radius
    bbox_max = wp_xy.max(axis=0) + wp_radius
    in_bbox = ((atlas_points_2d[:, 0] >= bbox_min[0]) &
               (atlas_points_2d[:, 0] <= bbox_max[0]) &
               (atlas_points_2d[:, 1] >= bbox_min[1]) &
               (atlas_points_2d[:, 1] <= bbox_max[1]))
    box_idx = np.where(in_bbox)[0]
    box_pts = atlas_points_2d[box_idx]

    # Per-WP distance check
    r2 = wp_radius * wp_radius
    near = np.zeros(len(box_pts), dtype=bool)
    for wxy in wp_xy:
        d2 = (box_pts[:, 0] - wxy[0]) ** 2 + (box_pts[:, 1] - wxy[1]) ** 2
        near |= (d2 <= r2)

    mask[box_idx[near]] = True
    return mask


def build_sample(center_xy, theta, atlas_cat, atlas_cov, atlas_gmin,
                 obs_wp_id=None, wp_data=None):
    """Build input grid (8, 100, 100) and gt mask (100, 100) for one frontier.

    Uses reverse lookup: for each BEV cell, inverse-rotate to world coords
    and query the pre-built atlas grid directly.  No rotation-induced gaps.

    Channels:
        0: free (atlas, current WP coverage)
        1: obstacle
        2: unknown
        3: unobserved (no atlas data)
        4: hit_count  — depth endpoints per cell (log1p)
        5: pass_count — ray pass-through per cell (log1p), confirms free space
        6: z_max      — max height relative to local ground
        7: z_range    — height spread (max - min), wall vs noise vs floor

    Args:
        center_xy  : (2,) frontier XY
        theta      : atan2(fy-wy, fx-wx)  WP→frontier angle
        atlas_cat  : 2D int8 array from build_atlas_grid() (-1=no data)
        atlas_cov  : 2D bool array from build_coverage_grid()
        atlas_gmin : (gx_min, gy_min) offsets for atlas arrays
        obs_wp_id  : observing WP id – used to project depth BEV (ch4-7)
        wp_data    : dict with pre-loaded WP data (alternative to obs_wp_id).
                     Keys: 'position' (3,), 'depth_pts' (N,3),
                     'camera_depth_pts' {cam:(N,3)},
                     'camera_extrinsics' {cam:4x4}
    """

    inp = np.zeros((8, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    gt  = np.full((GRID_SIZE, GRID_SIZE), 3, dtype=np.int64)  # default=unobserved

    gx_min, gy_min = atlas_gmin

    # ── Reverse lookup: BEV cell centres → world XY → atlas grid ──
    rows = np.arange(GRID_SIZE, dtype=np.float64)
    cols = np.arange(GRID_SIZE, dtype=np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')

    # BEV cell centre in rotated coords
    rx = HALF_EXTENT - (rr + 0.5) * RESOLUTION
    ry = (cc + 0.5) * RESOLUTION - HALF_EXTENT

    # Inverse rotate to world (rotate by +theta undoes the -theta forward rot)
    ct, st = np.cos(theta), np.sin(theta)
    wx = ct * rx - st * ry + center_xy[0]
    wy = st * rx + ct * ry + center_xy[1]

    # Atlas grid indices (shifted by gmin for array indexing)
    agx = (np.floor(wx / RESOLUTION).astype(int) - gx_min)
    agy = (np.floor(wy / RESOLUTION).astype(int) - gy_min)

    in_bounds = ((agx >= 0) & (agx < atlas_cat.shape[0]) &
                 (agy >= 0) & (agy < atlas_cat.shape[1]))

    # Vectorised lookup
    cat_map = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)
    cov_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    cat_map[in_bounds] = atlas_cat[agx[in_bounds], agy[in_bounds]]
    cov_map[in_bounds] = atlas_cov[agx[in_bounds], agy[in_bounds]]

    has_data = cat_map >= 0
    covered  = cov_map & has_data

    # ── Input channels (current WP coverage) ──
    inp[0][covered & (cat_map == 1)] = 1.0                          # free
    inp[1][covered & ((cat_map == 2) | (cat_map == 3))] = 1.0      # obstacle (cat3→obstacle)
    inp[2][covered & (cat_map == 0)] = 1.0                          # unknown
    inp[3] = (~covered).astype(np.float32)                     # unobserved

    # ── GT: 4-class labels for uncovered cells ──
    gt[covered] = IGNORE_LABEL
    uncov_data = (~cov_map) & has_data
    gt[uncov_data & (cat_map == 1)] = 0   # free
    gt[uncov_data & (cat_map == 2)] = 1   # obstacle
    gt[uncov_data & ((cat_map == 0) | (cat_map == 3))] = 2  # unknown
    # Cells still == 3 are truly unobserved (no atlas data)

    # ── Ch4-7: depth BEV from observing WP ──
    # (forward projection – depth points are not on the atlas grid)
    c, s = np.cos(-theta), np.sin(-theta)
    extract_r = HALF_EXTENT * 1.42  # > sqrt(2) * 10

    # Resolve depth data: either from wp_data dict or by loading from disk
    _all_depth_pts = None
    _camera_pts = None
    _wp_position = None
    _cam_extrinsics = None

    if wp_data is not None:
        _all_depth_pts = wp_data.get('depth_pts')
        _camera_pts = wp_data.get('camera_depth_pts', {})
        _wp_position = wp_data['position']
        _cam_extrinsics = wp_data.get('camera_extrinsics', {})
    elif obs_wp_id is not None:
        from map_utils import project_depth_from_waypoint, load_waypoint_meta
        try:
            _all_depth_pts, _camera_pts = project_depth_from_waypoint(
                obs_wp_id, step=1)
            meta = load_waypoint_meta(obs_wp_id)
            _wp_position = meta['agent_position']
            _cam_extrinsics = {cn: np.array(meta['cameras'][cn]['camera_extrinsic'])
                               for cn in meta['cameras']}
        except Exception:
            pass  # leave ch4-7 as zeros

    if _all_depth_pts is not None and len(_all_depth_pts) > 0:
        try:
            # Ground reference: median z of points near the WP
            wp_xy = np.array(_wp_position[:2])
            wp_dists = np.sqrt((_all_depth_pts[:, 0] - wp_xy[0]) ** 2 +
                               (_all_depth_pts[:, 1] - wp_xy[1]) ** 2)
            near_wp = wp_dists < 1.0
            if near_wp.sum() > 10:
                ground_z = np.median(_all_depth_pts[near_wp, 2])
            else:
                ground_z = _wp_position[2]

            ddx = _all_depth_pts[:, 0] - center_xy[0]
            ddy = _all_depth_pts[:, 1] - center_xy[1]
            ddz = _all_depth_pts[:, 2] - ground_z
            dmask = (np.abs(ddx) <= extract_r) & (np.abs(ddy) <= extract_r)
            ddx, ddy, ddz = ddx[dmask], ddy[dmask], ddz[dmask]
            drx = c * ddx - s * ddy
            dry = s * ddx + c * ddy
            drow = ((HALF_EXTENT - drx) / RESOLUTION).astype(np.int32)
            dcol = ((dry + HALF_EXTENT) / RESOLUTION).astype(np.int32)
            dvalid = ((drow >= 0) & (drow < GRID_SIZE) &
                      (dcol >= 0) & (dcol < GRID_SIZE))
            drow, dcol = drow[dvalid], dcol[dvalid]
            ddz_v = ddz[dvalid]

            if len(drow) > 0:
                # Ch4: hit_count
                hit_counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
                np.add.at(hit_counts, (drow, dcol), 1)
                inp[4] = np.log1p(hit_counts)

                # Ch6: z_max,  Ch7: z_range (max - min)
                z_max = np.full((GRID_SIZE, GRID_SIZE), -np.inf, dtype=np.float32)
                z_min = np.full((GRID_SIZE, GRID_SIZE),  np.inf, dtype=np.float32)
                np.maximum.at(z_max, (drow, dcol), ddz_v)
                np.minimum.at(z_min, (drow, dcol), ddz_v)
                has_hit = z_max > -np.inf
                z_max[~has_hit] = 0.0
                z_min[~has_hit] = 0.0
                inp[6] = z_max
                inp[7] = z_max - z_min

            # ── Ch5: pass_count via ray marching (per camera) ──
            pass_counts = np.zeros((GRID_SIZE, GRID_SIZE),
                                    dtype=np.float32)
            RAY_STEP = 4  # subsample rays for speed
            for cam_name, cpts in _camera_pts.items():
                if len(cpts) == 0 or cam_name not in _cam_extrinsics:
                    continue
                T_wc = np.array(_cam_extrinsics[cam_name])
                cam_world = T_wc[:3, 3]
                # Camera pos in rotated grid coords
                cam_dx = cam_world[0] - center_xy[0]
                cam_dy = cam_world[1] - center_xy[1]
                cam_rx = c * cam_dx - s * cam_dy
                cam_ry = s * cam_dx + c * cam_dy

                # Subsample for speed
                cpts_sub = cpts[::RAY_STEP]
                cdx = cpts_sub[:, 0] - center_xy[0]
                cdy = cpts_sub[:, 1] - center_xy[1]
                crx = c * cdx - s * cdy
                cry = s * cdx + c * cdy

                ray_dx = crx - cam_rx
                ray_dy = cry - cam_ry
                ray_len = np.sqrt(ray_dx ** 2 + ray_dy ** 2)
                ray_len = np.maximum(ray_len, 1e-6)

                # Sample along each ray at RESOLUTION intervals
                max_s = int(np.ceil(ray_len.max() / RESOLUTION)) + 1
                t_abs = np.arange(max_s) * RESOLUTION        # (S,)
                t_norm = (t_abs[np.newaxis, :]
                          / ray_len[:, np.newaxis])           # (N, S)
                valid_t = t_norm < 1.0  # exclude hit endpoint

                sx = cam_rx + t_norm * ray_dx[:, np.newaxis]
                sy = cam_ry + t_norm * ray_dy[:, np.newaxis]
                srow = ((HALF_EXTENT - sx) / RESOLUTION).astype(np.int32)
                scol = ((sy + HALF_EXTENT) / RESOLUTION).astype(np.int32)
                grid_ok = (valid_t & (srow >= 0) & (srow < GRID_SIZE) &
                           (scol >= 0) & (scol < GRID_SIZE))

                np.add.at(pass_counts,
                          (srow[grid_ok], scol[grid_ok]), 1)

            inp[5] = np.log1p(pass_counts)

        except Exception:
            pass  # leave ch4-7 as zeros if depth projection fails

    return inp, gt


# ── Depth projection helper ────────────────────────────────────────────────
def _project_depth_array(depth, intrinsic, extrinsic, step=1):
    """Project a raw depth map (2D ndarray) to world XYZ points.

    Handles portrait-orientation transpose (depth.shape[0] > depth.shape[1]).

    Args:
        depth     : 2D ndarray, raw depth map
        intrinsic : 3×3 camera intrinsic matrix
        extrinsic : 4×4 camera-to-world transform
        step      : pixel subsampling step (1 = full resolution)

    Returns:
        (N, 3) world XYZ points
    """
    depth = np.array(depth, dtype=np.float32)
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
        return np.empty((0, 3), dtype=np.float64)

    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    ones = np.ones_like(d)
    pts_cam = np.stack([x, y, d, ones], axis=1)
    pts_world = (T_wc @ pts_cam.T).T[:, :3]
    return pts_world


# ── Single-frontier prediction (online-ready) ─────────────────────────────
def predict_single_frontier(frontier_xy, obs_wp_id,
                            grid_to_category, wp_pos_atlas,
                            model, device):
    """Predict gain (m²) and connectivity for a single frontier.

    Args:
        frontier_xy        : (2,) frontier XY in world coords
        obs_wp_id          : int, ID of the waypoint observing this frontier
        grid_to_category   : {(gx, gy): int} current (WP-covered) atlas map
        wp_pos_atlas       : (3,) WP position from atlas['waypoint_positions'][obs_wp_id].
                             MUST come from atlas.pkl, NOT from meta['agent_position'] —
                             the two differ by up to ~0.4m, and atlas is what the
                             training prepare stage uses for theta computation.
        model              : loaded UNet in eval mode
        device             : torch device

    Returns:
        (gain_m2, connects):
            gain_m2  — predicted exploration gain in m²
            connects — True if reachable free area reaches the top BEV edge
    """
    import torch

    atlas_cat, gx_min, gy_min = build_atlas_grid(grid_to_category)
    atlas_cov = (atlas_cat >= 0)
    atlas_gmin = (gx_min, gy_min)

    frontier_xy = np.asarray(frontier_xy, dtype=np.float64)[:2]

    # Load WP data from disk
    meta = load_waypoint_meta(obs_wp_id)
    wp_position = np.asarray(meta['agent_position'], dtype=np.float64)
    folder = os.path.join(WP_IMAGES_DIR, str(obs_wp_id))
    all_pts = []
    camera_pts = {}
    cam_extrinsics = {}
    for cam_name in CAMERAS:
        if cam_name not in meta['cameras']:
            continue
        depth_path = os.path.join(folder, f"{cam_name}_depth.npy")
        if not os.path.exists(depth_path):
            continue
        cam_info = meta['cameras'][cam_name]
        intrinsic = np.array(cam_info['camera_intrinsic'])
        extrinsic = np.array(cam_info['camera_extrinsic'])
        dmap = np.load(depth_path)
        pts = _project_depth_array(dmap, intrinsic, extrinsic, step=1)
        camera_pts[cam_name] = pts
        cam_extrinsics[cam_name] = extrinsic
        if len(pts) > 0:
            all_pts.append(pts)
    depth_pts = np.vstack(all_pts) if all_pts else np.empty((0, 3))
    wp_data = {
        'position': wp_position,
        'depth_pts': depth_pts,
        'camera_depth_pts': camera_pts,
        'camera_extrinsics': cam_extrinsics,
    }

    theta_ref = np.asarray(wp_pos_atlas, dtype=np.float64)
    theta = np.arctan2(frontier_xy[1] - theta_ref[1],
                       frontier_xy[0] - theta_ref[0])

    inp, _ = build_sample(frontier_xy, theta,
                          atlas_cat, atlas_cov, atlas_gmin,
                          wp_data=wp_data)

    inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(inp_t), dim=1).cpu().numpy()[0]
    free_prob = probs[0]

    # Only count newly predicted free (unobserved → free) cells in the upper half.
    new_free = ((free_prob[:GRID_SIZE // 2, :] > 0.5)
                & (inp[3, :GRID_SIZE // 2, :] > 0))     # ch3: unobserved

    gain_m2  = float(new_free.sum() * CELL_AREA)
    connects = bool(new_free[0, :].any())

    return gain_m2, connects


# ── Prepare (pre-compute dataset) ──────────────────────────────────────────
def _load_final_frontier_positions():
    """Load frontier positions from the last exploration step (16814.json).

    These frontiers were never explored, so the atlas may not have complete
    data around them.  Training samples whose frontier position matches one
    of these should be excluded.
    """
    last_path = os.path.join(FRONTIERS_DIR, "16814.json")
    if not os.path.exists(last_path):
        return None
    with open(last_path) as f:
        d = json.load(f)
    return np.array(d['frontier_positions'])[:, :2]   # (M, 2) XY


def cmd_prepare(args):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print("Loading atlas ...")
    atlas_pts, wp_positions, g2c = load_atlas()
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_gmin = (gx_min, gy_min)
    print(f"  {len(atlas_pts):,} atlas points, {len(wp_positions)} waypoints"
          f", atlas grid {atlas_cat.shape}")

    # Load final frontier positions to filter unreliable GT
    final_fps = _load_final_frontier_positions()
    if final_fps is not None:
        print(f"  Loaded {len(final_fps)} final frontiers from 16814.json "
              f"(will exclude samples within {HALF_EXTENT}m)")
    else:
        print("  WARNING: 16814.json not found, no final-frontier filtering")

    fids = sorted(int(f.replace('.json', ''))
                  for f in os.listdir(FRONTIERS_DIR) if f.endswith('.json'))
    if args.min_id is not None:
        fids = [f for f in fids if f >= args.min_id]
    if args.max_id is not None:
        fids = [f for f in fids if f <= args.max_id]
    print(f"  {len(fids)} frontier files (id range: {fids[0]}–{fids[-1]})")

    meta_list = []
    n_skipped_final = 0
    t0 = time.time()

    for fi, fid in enumerate(fids):
        fdata = load_frontier_data(fid)
        wids    = fdata['waypoint_ids']
        fps     = fdata['frontier_positions']
        obs_wps = fdata.get('frontier_observing_wps', [])
        if not obs_wps:
            continue

        # Shared coverage grid for all frontiers in this file
        cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
        atlas_cov = build_coverage_grid(atlas_pts, cmask,
                                        gx_min, gy_min, atlas_cat.shape)

        for j, fp in enumerate(fps):
            if j >= len(obs_wps):
                break
            ow = obs_wps[j]
            if ow not in wp_positions:
                continue

            # Skip if any final (unexplored) frontier is within the
            # input grid range (HALF_EXTENT).  The atlas has no data
            # beyond those frontiers, so GT would be unreliable.
            if final_fps is not None:
                fp_xy = np.array([fp[0], fp[1]])
                dists = np.linalg.norm(final_fps - fp_xy, axis=1)
                if dists.min() < HALF_EXTENT:
                    n_skipped_final += 1
                    continue

            wp_pos = wp_positions[ow]
            theta = np.arctan2(fp[1] - wp_pos[1], fp[0] - wp_pos[0])

            inp, gt = build_sample(
                np.array([fp[0], fp[1]]), theta,
                atlas_cat, atlas_cov, atlas_gmin, obs_wp_id=ow)

            fname = f"{fid}_{j}.npz"
            np.savez_compressed(
                os.path.join(out_dir, fname),
                input=inp.astype(np.float16),
                gt=gt)
            meta_list.append({
                'file': fname,
                'frontier_id': fid,
                'frontier_idx': j,
                'obs_wp': ow,
                'gain_cells': int(gt.sum()),
            })

        if (fi + 1) % 50 == 0 or fi == len(fids) - 1:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(fids)}] {len(meta_list)} samples  "
                  f"({elapsed:.1f}s)")

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta_list, f)

    print(f"\nDone: {len(meta_list)} samples saved to {out_dir}/")
    if n_skipped_final:
        print(f"  Skipped {n_skipped_final} samples matching 16814.json "
              f"final frontiers")


# ── PyTorch imports (deferred so prepare works without torch) ──────────────
def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# ── Dataset ─────────────────────────────────────────────────────────────────
class FrontierGainDataset:
    """Lazy-load from pre-computed cache.  train/val split by frontier_id."""

    def __init__(self, cache_dir, split='train'):
        torch, _, _ = _import_torch()
        with open(os.path.join(cache_dir, 'meta.json')) as f:
            all_meta = json.load(f)

        if split == 'train':
            self.meta = [m for m in all_meta if m['frontier_id'] < 12000]
        elif split == 'val':
            self.meta = [m for m in all_meta if m['frontier_id'] >= 12000]
        else:
            self.meta = list(all_meta)

        self.cache_dir = cache_dir
        self.torch = torch

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        m = self.meta[idx]
        data = np.load(os.path.join(self.cache_dir, m['file']))
        inp = data['input'].astype(np.float32)
        gt  = data['gt'].astype(np.int64)
        return self.torch.from_numpy(inp), self.torch.from_numpy(gt)


# ── UNet ────────────────────────────────────────────────────────────────────
def _build_model():
    torch, nn, F = _import_torch()

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = DoubleConv(8, 32)
            self.enc2 = DoubleConv(32, 64)
            self.enc3 = DoubleConv(64, 128)
            self.enc4 = DoubleConv(128, 256)
            self.bottleneck = DoubleConv(256, 512)

            self.dec4 = DoubleConv(512 + 256, 256)
            self.dec3 = DoubleConv(256 + 128, 128)
            self.dec2 = DoubleConv(128 + 64, 64)
            self.dec1 = DoubleConv(64 + 32, 32)

            self.out_conv = nn.Conv2d(32, NUM_CLASSES, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            e1 = self.enc1(x)                  # 100
            e2 = self.enc2(self.pool(e1))      # 50
            e3 = self.enc3(self.pool(e2))      # 25
            e4 = self.enc4(self.pool(e3))      # 12
            b  = self.bottleneck(self.pool(e4))  # 6

            up = F.interpolate(b, e4.shape[2:], mode='bilinear',
                               align_corners=False)
            d4 = self.dec4(torch.cat([up, e4], 1))

            up = F.interpolate(d4, e3.shape[2:], mode='bilinear',
                               align_corners=False)
            d3 = self.dec3(torch.cat([up, e3], 1))

            up = F.interpolate(d3, e2.shape[2:], mode='bilinear',
                               align_corners=False)
            d2 = self.dec2(torch.cat([up, e2], 1))

            up = F.interpolate(d2, e1.shape[2:], mode='bilinear',
                               align_corners=False)
            d1 = self.dec1(torch.cat([up, e1], 1))

            return self.out_conv(d1)  # (B, 4, H, W) raw logits

    return UNet


# ── Loss / metrics ──────────────────────────────────────────────────────────
def dice_loss_free(logits, target, smooth=1.0):
    """Dice loss on the free class (ch0) only, ignoring IGNORE_LABEL cells."""
    torch, _, F = _import_torch()
    probs = F.softmax(logits, dim=1)[:, 0]        # (B, H, W) free prob
    free_gt = (target == 0).float()                 # (B, H, W)
    valid = (target != IGNORE_LABEL).float()        # (B, H, W)
    pred = (probs * valid).view(-1)
    tgt  = (free_gt * valid).view(-1)
    inter = (pred * tgt).sum()
    return 1 - (2 * inter + smooth) / (pred.sum() + tgt.sum() + smooth)


def compute_metrics(logits, target, threshold=0.5):
    """Compute IoU / precision / recall on free class (ch0)."""
    torch, _, F = _import_torch()
    probs = F.softmax(logits, dim=1)[:, 0]          # (B, H, W) free prob
    pred = (probs > threshold).float()
    tgt  = (target == 0).float()                      # free class
    valid = (target != IGNORE_LABEL).float()

    pred = pred * valid
    tgt  = tgt * valid

    tp = (pred * tgt).sum().item()
    fp = (pred * (1 - tgt)).sum().item()
    fn = ((1 - pred) * tgt).sum().item()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    iou       = tp / max(tp + fp + fn, 1)

    pred_area = pred.sum().item() * CELL_AREA
    gt_area   = tgt.sum().item() * CELL_AREA

    return dict(iou=iou, precision=precision, recall=recall,
                area_err_m2=abs(pred_area - gt_area),
                pred_area_m2=pred_area, gt_area_m2=gt_area)


# ── Training ────────────────────────────────────────────────────────────────
def _get_device():
    torch, _, _ = _import_torch()
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def cmd_train(args):
    torch, nn, _ = _import_torch()
    from torch.utils.data import DataLoader

    device = _get_device()
    print(f"Device: {device}")

    train_ds = FrontierGainDataset(args.data, 'train')
    val_ds   = FrontierGainDataset(args.data, 'val')

    # If default split gives empty val, fall back to random 80/20 split
    if len(val_ds) == 0 or len(train_ds) == 0:
        full_ds = FrontierGainDataset(args.data, 'all')
        n = len(full_ds)
        n_val = max(1, int(n * 0.2))
        n_train = n - n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds,
                                                          [n_train, n_val])
        print(f"Train: {len(train_ds)},  Val: {len(val_ds)}  (random 80/20)")
    else:
        print(f"Train: {len(train_ds)},  Val: {len(val_ds)}")

    nw = 0 if sys.platform == 'darwin' else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True)

    UNet = _build_model()
    model = UNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    ce_w, dice_w = args.bce_weight, args.dice_weight  # reuse CLI names

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for inp, gt in train_loader:
            inp = inp.to(device)
            gt  = gt.to(device)                 # (B, H, W) int64
            logits = model(inp)                 # (B, 4, H, W)
            loss = ce_w * ce_fn(logits, gt) + dice_w * dice_loss_free(logits, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inp.size(0)
        train_loss /= len(train_ds)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        val_m = dict(iou=0, precision=0, recall=0, area_err_m2=0)
        n_val = len(val_ds)
        with torch.no_grad():
            for inp, gt in val_loader:
                inp  = inp.to(device)
                gt_d = gt.to(device)              # (B, H, W) int64
                logits = model(inp)               # (B, 4, H, W)
                loss = (ce_w * ce_fn(logits, gt_d)
                        + dice_w * dice_loss_free(logits, gt_d))
                val_loss += loss.item() * inp.size(0)
                m = compute_metrics(logits, gt_d)
                for k in val_m:
                    val_m[k] += m[k] * inp.size(0)
        val_loss /= n_val
        for k in val_m:
            val_m[k] /= n_val

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"tl={train_loss:.4f}  vl={val_loss:.4f}  "
              f"IoU={val_m['iou']:.4f}  P={val_m['precision']:.4f}  "
              f"R={val_m['recall']:.4f}  "
              f"AErr={val_m['area_err_m2']:.2f}m²  lr={lr:.2e}")

        # checkpoint
        if val_m['iou'] > best_val_iou:
            best_val_iou = val_m['iou']
            torch.save(model.state_dict(),
                       os.path.join(args.data, 'model_best.pt'))
            print(f"  → saved best (IoU={best_val_iou:.4f})")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.data, f'model_epoch{epoch}.pt'))

    torch.save(model.state_dict(), os.path.join(args.data, 'model_final.pt'))
    print(f"\nDone. Best val IoU: {best_val_iou:.4f}")


# ── Predict / visualise ────────────────────────────────────────────────────
def cmd_predict(args):
    torch, _, _ = _import_torch()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = _get_device()

    UNet = _build_model()
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    atlas_pts, wp_positions, g2c = load_atlas()
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_gmin = (gx_min, gy_min)

    fdata   = load_frontier_data(args.frontier_id)
    wids    = fdata['waypoint_ids']
    fps     = fdata['frontier_positions']
    obs_wps = fdata.get('frontier_observing_wps', [])

    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cov = build_coverage_grid(atlas_pts, cmask,
                                    gx_min, gy_min, atlas_cat.shape)

    out_dir = os.path.join(BASE_DIR, f"predict_{args.frontier_id}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Single-entry mode: --wp-id + --fp-xy overrides the full loop ──────
    single_mode = (args.wp_id is not None and args.fp_xy is not None)
    if single_mode:
        ow_single = args.wp_id
        if ow_single not in wp_positions:
            print(f"ERROR: wp_id {ow_single} not found in atlas")
            return
        fps     = [args.fp_xy]        # single-element list → loop runs once
        obs_wps = [ow_single]
        print(f"Single-entry mode: wp_id={ow_single}  fp={args.fp_xy}")
    # ──────────────────────────────────────────────────────────────────────

    print(f"Frontier {args.frontier_id}: {len(fps)} frontiers")
    print(f"{'Idx':>4}  {'Pred(m²)':>9}  {'GT(m²)':>9}")
    print("-" * 26)

    for j, fp in enumerate(fps):
        if j >= len(obs_wps):
            break
        ow = obs_wps[j]
        if ow not in wp_positions:
            continue

        wp_pos = wp_positions[ow]
        theta = np.arctan2(fp[1] - wp_pos[1], fp[0] - wp_pos[0])
        inp, gt = build_sample(
            np.array([fp[0], fp[1]]), theta,
            atlas_cat, atlas_cov, atlas_gmin, obs_wp_id=ow)

        inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4,H,W)
        free_prob = probs[0]                           # P(free)
        # GT free mask (class 0)
        gt_free = (gt == 0).astype(np.uint8)

        # GT reachable gain (flood-fill, for IoU reference)
        free_mask = inp[0] > 0
        gt_reach, gt_bridge = reachable_gain(gt_free, free_mask)
        gt_area = gt_reach.sum() * CELL_AREA

        # ── Two-part gain (upper half) — mirrors predict_gain_with_mask ──
        from scipy import ndimage as _ndi_vis
        _H     = GRID_SIZE // 2
        _cu    = (_H - 1, GRID_SIZE // 2)    # frontier row in upper-half slice
        _obs_u = ((inp[0, :_H, :] > 0)
                  | (inp[2, :_H, :] > 0))    # ch0: free + ch2: unknown, upper half
        _new_u = ((free_prob[:_H, :] > 0.5)
                  & (inp[3, :_H, :] > 0))    # ch3: unobserved, upper half

        _new_conn = np.zeros((_H, GRID_SIZE), dtype=bool)  # Part 2 disabled
        _gp1 = float(_new_u.sum() * CELL_AREA)
        _gp2 = 0.0
        _gm  = _gp1

        print(f"{j:4d}  {_gm:9.2f}  {gt_area:9.2f}")

        # ── visualise ──
        fig, axes = plt.subplots(1, 7, figsize=(35, 5))

        # Build input composite image (reused for overlay)
        vis = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32) * 0.7
        vis[inp[0] > 0] = [0.2, 0.8, 0.2]   # free  → green
        vis[inp[1] > 0] = [0.1, 0.1, 0.1]   # obstacle → dark
        vis[inp[2] > 0] = [0.7, 0.5, 0.9]   # unknown  → purple

        # Frontier is at grid centre after rotation
        fc, fr = GRID_SIZE // 2, GRID_SIZE // 2

        # Depth channels
        hit_ch   = inp[4]   # hit_count (log1p)
        pass_ch  = inp[5]   # pass_count (log1p)
        zp90_ch  = inp[6]   # z_p90

        # (0) Input + depth overlay: green=pass(free), gray=hit(by height)
        vis_depth = vis.copy()
        has_pass = pass_ch > 0
        has_hit  = hit_ch > 0
        pass_only = has_pass & ~has_hit
        if pass_only.any():
            a = 0.35
            vis_depth[pass_only] = ((1 - a) * vis_depth[pass_only]
                                    + a * np.array([0.3, 0.9, 0.3]))
        if has_hit.any():
            h_clipped = np.clip(zp90_ch, -0.1, 1.5)
            h_norm = (h_clipped - (-0.1)) / (1.5 - (-0.1))
            gray = 0.8 - 0.7 * h_norm
            gray_rgb = np.stack([gray, gray, gray], axis=-1)
            alpha = 0.35 + 0.35 * h_norm
            a = alpha[has_hit, np.newaxis]
            vis_depth[has_hit] = ((1 - a) * vis_depth[has_hit]
                                  + a * gray_rgb[has_hit])
        n_hit  = int(has_hit.sum())
        n_pass = int(has_pass.sum())
        axes[0].imshow(np.clip(vis_depth, 0, 1))
        axes[0].set_title(f'Input+Depth (hit={n_hit} pass={n_pass})')
        axes[0].scatter(fc, fr, c='red', s=120, marker='*',
                        edgecolors='yellow', linewidths=1.5, zorder=5)

        # Shared data
        pred_cls = probs.argmax(axis=0)  # (H, W) predicted class
        uncov = (gt != IGNORE_LABEL)     # cells not in current coverage
        gt_show   = gt_reach.astype(np.float32)

        # Class colors: free=green, obstacle=dark, unknown=purple, unobserved=gray
        CLS_COLORS = {
            0: np.array([0.2, 0.8, 0.2]),   # free
            1: np.array([0.1, 0.1, 0.1]),   # obstacle
            2: np.array([0.7, 0.5, 0.9]),   # unknown
            3: np.array([0.6, 0.6, 0.6]),   # unobserved
        }

        # (1) Pred 4-class composite
        cls_vis = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32) * 0.85
        for c_id, c_rgb in CLS_COLORS.items():
            cls_vis[uncov & (pred_cls == c_id)] = c_rgb
        axes[1].imshow(np.clip(cls_vis, 0, 1))
        axes[1].scatter(fc, fr, c='red', s=80, marker='*',
                        edgecolors='white', linewidths=1, zorder=5)
        n_free = int((uncov & (pred_cls == 0)).sum())
        n_obst = int((uncov & (pred_cls == 1)).sum())
        n_unob = int((uncov & (pred_cls == 3)).sum())
        axes[1].set_title(f'Pred 4cls: F={n_free} O={n_obst} U={n_unob}')

        # (2) GT 4-class
        gt_vis = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32) * 0.85
        for c_id, c_rgb in CLS_COLORS.items():
            gt_vis[gt == c_id] = c_rgb
        axes[2].imshow(np.clip(gt_vis, 0, 1))
        axes[2].scatter(fc, fr, c='red', s=80, marker='*',
                        edgecolors='white', linewidths=1, zorder=5)
        n_gt_free = int((gt == 0).sum())
        n_gt_obst = int((gt == 1).sum())
        n_gt_unob = int((gt == 3).sum())
        axes[2].set_title(f'GT 4cls: F={n_gt_free} O={n_gt_obst} U={n_gt_unob}')

        # (3) Input + Pred 4cls overlay (blend predicted classes onto input)
        ov_pred = vis.copy()
        blend_a = 0.55
        for c_id, c_rgb in CLS_COLORS.items():
            m = uncov & (pred_cls == c_id)
            if m.any():
                ov_pred[m] = (1 - blend_a) * ov_pred[m] + blend_a * c_rgb
        axes[3].imshow(np.clip(ov_pred, 0, 1))
        axes[3].scatter(fc, fr, c='cyan', s=80, marker='*',
                        edgecolors='white', linewidths=1, zorder=5)
        axes[3].set_title(f'Input + Pred 4cls')

        # (4) Input + GT 4cls overlay
        ov_gt = vis.copy()
        for c_id, c_rgb in CLS_COLORS.items():
            m = (gt == c_id)
            if m.any():
                ov_gt[m] = (1 - blend_a) * ov_gt[m] + blend_a * c_rgb
        axes[4].imshow(np.clip(ov_gt, 0, 1))
        axes[4].scatter(fc, fr, c='cyan', s=80, marker='*',
                        edgecolors='white', linewidths=1, zorder=5)
        axes[4].set_title(f'Input + GT 4cls')

        # (5) Pred gain – new logic (upper half only)
        #     hot  = Part 1: all new_free (unobserved → predicted free)
        #     cyan = Part 2: obs_free newly connected via new_free
        #     gray = lower half (not counted)
        vis5 = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
        vis5[_H:, :] = [0.25, 0.25, 0.25]              # lower half: dim gray
        _fp_u   = free_prob[:_H, :]
        _hot_in = np.zeros((_H, GRID_SIZE), dtype=np.float32)
        _hot_in[_new_u] = _fp_u[_new_u]                 # confidence as brightness
        _hot5   = plt.cm.hot(_hot_in)[:, :, :3].astype(np.float32)
        vis5[:_H][_new_u]    = _hot5[_new_u]            # Part 1: hot
        vis5[:_H][_new_conn] = [0.0, 0.85, 0.85]        # Part 2: cyan
        axes[5].imshow(vis5)
        axes[5].scatter(fc, fr, c='cyan', s=80, marker='*',
                        edgecolors='white', linewidths=1, zorder=5)
        axes[5].set_title(f'Pred gain {_gm:.0f}m²  '
                          f'(new={_gp1:.0f} + link={_gp2:.0f})')

        # (6) GT free gain (reachable only)
        axes[6].imshow(gt_show, cmap='hot', vmin=0, vmax=1)
        axes[6].scatter(fc, fr, c='cyan', s=80, marker='*',
                        edgecolors='white', linewidths=1, zorder=5)
        axes[6].set_title(f'GT free ({gt_area:.1f} m²)')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        tag   = f'wp{ow}' if single_mode else str(j)
        fname = f'F_{tag}.png'
        plt.suptitle(
            f'Frontier {args.frontier_id} / F_{tag}  wp={ow}  '
            f'pred={_gm:.0f}m²  gt={gt_area:.0f}m²')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname),
                    dpi=100, bbox_inches='tight')
        plt.close()
        if single_mode:
            print(f"Saved: {os.path.join(out_dir, fname)}")

    print(f"\nVisualizations saved to {out_dir}/")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Frontier gain prediction network')
    sub = parser.add_subparsers(dest='command')

    # prepare
    p = sub.add_parser('prepare', help='Pre-compute dataset')
    p.add_argument('--out', default='gain_cache',
                   help='Output cache directory')
    p.add_argument('--min-id', type=int, default=None,
                   help='Min frontier file ID (inclusive)')
    p.add_argument('--max-id', type=int, default=None,
                   help='Max frontier file ID (inclusive)')

    # train
    p = sub.add_parser('train', help='Train the model')
    p.add_argument('--data', default='gain_cache')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bce-weight', type=float, default=1.0)
    p.add_argument('--dice-weight', type=float, default=1.0)
    p.add_argument('--save-every', type=int, default=10)

    # predict
    p = sub.add_parser('predict', help='Predict and visualise')
    p.add_argument('--frontier-id', type=int, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--wp-id', type=int, default=None,
                   help='Single observing waypoint id (use with --fp-xy)')
    p.add_argument('--fp-xy', type=float, nargs=2, default=None,
                   metavar=('X', 'Y'),
                   help='Frontier world position X Y (use with --wp-id)')

    args = parser.parse_args()
    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
