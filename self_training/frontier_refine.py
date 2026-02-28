#!/usr/bin/env python3
"""frontier_refine.py
======================
Generate pseudo-label supervision signals from DETR-predicted frontiers.

Given raw predictions for a frontier step, applies three operations:
  DROP  : remove predictions whose gain < REMOVE_GAIN_THRESH
  MERGE : drop the lower-gain member of any overlapping pair (IoU > OVERLAP_THRESH)
  ADD   : scan waypoints in the map for free-unknown boundary cells not
          already covered by any remaining prediction

Produces a refined_round*.json that is used as the DETR fine-tuning target.

Standalone usage (run DETR inference internally):
  python3 frontier_refine.py \\
      --frontier-id 6549 \\
      --detr-checkpoint /path/to/model_best.pt \\
      --gain-checkpoint gain_cache/model_final.pt \\
      --out-dir ss_output/6549_v4 \\
      --round 1

Standalone usage (supply pre-computed predictions as JSON):
  python3 frontier_refine.py \\
      --frontier-id 6549 \\
      --pred-json raw_preds.json \\
      --gain-checkpoint gain_cache/model_final.pt \\
      --out-dir ss_output/6549_v4 \\
      --round 1

Environment variable TIAMAT_DATA_DIR controls data source.
"""

import os, sys, json, math, argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

# ── shared imports ──────────────────────────────────────────────────────────
from models.frontier_gain_model import (
    load_atlas, get_current_map_mask,
    build_atlas_grid, build_coverage_grid,
    build_sample, reachable_gain,
    RESOLUTION, GRID_SIZE, HALF_EXTENT, CELL_AREA, WP_RADIUS,
    _build_model as _build_gain_model, _get_device,
    load_waypoint_meta, CAMERAS, WP_IMAGES_DIR,
    _project_depth_array,
)
from models.frontier_detr_model import (
    world_to_bev_normalized, bev_normalized_to_world,
    _build_detr_model, MAX_QUERIES,
)
from models.map_utils import load_frontier_data

# ── BEV geometry constants ───────────────────────────────────────────────────
FRONTIER_RC = (GRID_SIZE // 2 - 1, GRID_SIZE // 2)   # frontier centre in BEV
UPPER_HALF  = GRID_SIZE // 2                           # rows 0..49

# ── Refinement thresholds (overridable via CLI) ──────────────────────────────
REMOVE_GAIN_THRESH = 1.0    # m²  – drop a prediction if gain < this
OVERLAP_THRESH     = 0.50   # IoU – drop lower-gain if overlap > this
COVER_RADIUS       = 3.0    # m   – "covered" = within this of any existing frontier
BOUNDARY_RADIUS    = 5.0    # m   – search for boundary cells within this of WP
SAMPLE_STRIDE      = 0.6    # m   – min spacing between boundary candidate points
ADD_GAIN_THRESH    = 0.0    # m²  – minimum gain to ADD (0 = no filter)
CONF_THRESH        = 0.3    # DETR confidence threshold

# Flood-fill radius for atlas-based gain estimate
ATLAS_FLOOD_RADIUS = 15.0   # metres

# ── v2 Marginal-Coverage-Based Refinement constants ──────────────────────────
DELTA_DROP     = 1.0    # m²  marginal coverage below which a prediction is DROPped
DELTA_KEEP     = 1.0    # m²  minimum new coverage to KEEP in greedy MERGE
DELTA_ADD      = 5.0    # m²  minimum marginal coverage to ADD a candidate
MAX_ADD_PER_WP = 2      # max candidates added per observing WP in ADD step
ETA            = 0.0    # distance penalty coefficient in ADD step scoring
KAPPA          = 5.0    # gain smoothing constant for weight computation
Q_MIN          = 0.2    # minimum training weight

# ── v3 Set-Cover Refinement constants ─────────────────────────────────────────
COVERAGE_FRAC      = 0.65  # stop when this fraction of (cell, label) universe is covered
SHIFT_RADIUS_CELLS = 5     # centroid radius in cells (1.0 m at 0.2 m/cell)
WALL_MARGIN_CELLS  = 3     # erosion margin matching visual wall width
PROXIMITY_DISQ_M   = 2.0   # disqualify candidates within this distance of selected


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Gain helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_wp_data(wp_id):
    """Load depth + pose data for a WP from disk."""
    meta   = load_waypoint_meta(wp_id)
    folder = os.path.join(WP_IMAGES_DIR, str(wp_id))
    all_pts, cam_pts, cam_ext = [], {}, {}
    for cam_name in CAMERAS:
        if cam_name not in meta['cameras']:
            continue
        depth_path = os.path.join(folder, f"{cam_name}_depth.npy")
        if not os.path.exists(depth_path):
            continue
        cam_info  = meta['cameras'][cam_name]
        intrinsic = np.array(cam_info['camera_intrinsic'])
        extrinsic = np.array(cam_info['camera_extrinsic'])
        dmap      = np.load(depth_path)
        pts       = _project_depth_array(dmap, intrinsic, extrinsic, step=1)
        cam_pts[cam_name] = pts
        cam_ext[cam_name] = extrinsic
        if len(pts) > 0:
            all_pts.append(pts)
    return {
        'position':          np.asarray(meta['agent_position'], dtype=np.float64),
        'depth_pts':         np.vstack(all_pts) if all_pts else np.empty((0, 3)),
        'camera_depth_pts':  cam_pts,
        'camera_extrinsics': cam_ext,
    }


def bev_upper_mask_to_world_cells(reach_mask, frontier_xy, theta):
    """Convert BEV mask (any rows) to frozenset of atlas (gx,gy) grid indices."""
    rows, cols = np.where(reach_mask)
    if len(rows) == 0:
        return frozenset()
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    fx, fy = float(frontier_xy[0]), float(frontier_xy[1])
    cells = set()
    for r, c in zip(rows, cols):
        rx = HALF_EXTENT - (r + 0.5) * RESOLUTION
        ry = (c + 0.5) * RESOLUTION - HALF_EXTENT
        wx = fx + rx * cos_t - ry * sin_t
        wy = fy + rx * sin_t + ry * cos_t
        cells.add((int(round(wx / RESOLUTION)), int(round(wy / RESOLUTION))))
    return frozenset(cells)


def predict_gain_with_mask(frontier_xy, obs_wp_id, covered_g2c,
                           wp_pos_atlas, gain_model, device,
                           wp_data_cache=None, extended=False):
    """Run gain model; return (gain_m2, connects, world_cells[, evidence, preds]).

    Coverage uses the full covered_g2c mask (atlas_cov = atlas_cat >= 0),
    consistent with cmd_prepare training and cmd_predict visualization.

    Gain formula mirrors cmd_predict axes[5] (two-part upper-half):
      Part 1: all predicted-free unobserved cells in the upper BEV half
      Part 2: observed-free cells in upper half newly connected via Part 1

    world_cells is a frozenset of (gx, gy) atlas indices covering the
    gain region (used for MERGE IoU).

    When *extended=True*, additionally returns:
      cell_evidence : {(gx,gy): -log(1-p)}  for free-predicted unobserved cells
      cell_preds    : {(gx,gy): (argmax_class, p_free, p_obstacle)}  for ALL
                      unobserved cells in the full BEV.
    """
    frontier_xy = np.asarray(frontier_xy, dtype=np.float64)[:2]
    wp_pos      = np.asarray(wp_pos_atlas, dtype=np.float64)
    theta       = math.atan2(frontier_xy[1] - wp_pos[1],
                             frontier_xy[0] - wp_pos[0])

    atlas_cat, gx_min, gy_min = build_atlas_grid(covered_g2c)
    # Full coverage: all cells in covered_g2c are marked covered.
    # This is identical to cmd_prepare / cmd_predict which use
    # build_coverage_grid over all visited WPs — the covered_g2c dict
    # already contains exactly those cells.
    atlas_cov  = (atlas_cat >= 0)
    atlas_gmin = (gx_min, gy_min)

    if wp_data_cache is not None and obs_wp_id in wp_data_cache:
        wp_data = wp_data_cache[obs_wp_id]
    else:
        try:
            wp_data = _load_wp_data(obs_wp_id)
        except Exception:
            return (0.0, False, frozenset(), {}, {}) if extended \
                else (0.0, False, frozenset())
        if wp_data_cache is not None:
            wp_data_cache[obs_wp_id] = wp_data

    try:
        inp, _ = build_sample(frontier_xy, theta,
                              atlas_cat, atlas_cov, atlas_gmin,
                              wp_data=wp_data)
    except Exception:
        return (0.0, False, frozenset(), {}, {}) if extended \
            else (0.0, False, frozenset())

    inp_t = torch.from_numpy(inp.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(gain_model(inp_t), dim=1).cpu().numpy()[0]
    free_prob = probs[0]

    # Gain value: upper half only (direction facing away from WP).
    _H            = UPPER_HALF
    new_free_upper = ((free_prob[:_H, :] > 0.5)
                      & (inp[3, :_H, :] > 0))  # ch3: unobserved → predicted free

    gain_m2  = float(new_free_upper.sum() * CELL_AREA)
    connects = bool(new_free_upper[0, :].any())

    # Overlap mask: full BEV (all rows) so that MERGE/ADD deduplication
    # compares coverage in the same atlas coordinate system regardless of
    # which WP observed the frontier.
    new_free_full = ((free_prob > 0.5)
                     & (inp[3] > 0))            # ch3: unobserved → predicted free
    cells = bev_upper_mask_to_world_cells(new_free_full, frontier_xy, theta)

    if not extended:
        return gain_m2, connects, cells

    # ── Extended returns for v3 set-cover ──────────────────────────────────
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    fx, fy = float(frontier_xy[0]), float(frontier_xy[1])

    # cell_evidence: -log(1-p) for free-predicted unobserved cells
    rows, cols = np.where(new_free_full)
    cell_evidence = {}
    for r, c in zip(rows, cols):
        rx = HALF_EXTENT - (r + 0.5) * RESOLUTION
        ry = (c + 0.5) * RESOLUTION - HALF_EXTENT
        wx = fx + rx * cos_t - ry * sin_t
        wy = fy + rx * sin_t + ry * cos_t
        key = (int(round(wx / RESOLUTION)), int(round(wy / RESOLUTION)))
        p = float(free_prob[r, c])
        ev = -math.log(max(1.0 - p, 1e-7))
        if key not in cell_evidence or ev > cell_evidence[key]:
            cell_evidence[key] = ev

    # cell_preds: (argmax_class, p_free, p_obstacle) for ALL unobserved cells
    argmax_map = np.argmax(probs, axis=0)  # (H, W)
    unobs_mask = inp[3] > 0
    rows_u, cols_u = np.where(unobs_mask)
    cell_preds = {}
    for r, c in zip(rows_u, cols_u):
        rx = HALF_EXTENT - (r + 0.5) * RESOLUTION
        ry = (c + 0.5) * RESOLUTION - HALF_EXTENT
        wx = fx + rx * cos_t - ry * sin_t
        wy = fy + rx * sin_t + ry * cos_t
        key = (int(round(wx / RESOLUTION)), int(round(wy / RESOLUTION)))
        cls = int(argmax_map[r, c])
        pf = float(probs[0, r, c])
        po = float(probs[1, r, c])
        mp = max(pf, po)
        if key not in cell_preds or mp > max(cell_preds[key][1], cell_preds[key][2]):
            cell_preds[key] = (cls, pf, po)

    return gain_m2, connects, cells, cell_evidence, cell_preds


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Free-unknown boundary finder
# ═══════════════════════════════════════════════════════════════════════════

def find_free_unknown_boundary(wp_id, wp_positions, g2c,
                               radius=BOUNDARY_RADIUS,
                               stride=SAMPLE_STRIDE,
                               include_absent=True):
    """Return world-space positions of free-unknown boundary cells near wp_id.

    A "boundary cell" is a free atlas cell (cat==1) whose 4-connected
    neighbourhood contains at least one:
      - cat==0  (unknown / unmapped but recorded in atlas)
      - absent  (not in atlas at all)  – only when include_absent=True

    Note: cat==3 (depth-hit obstacle) is intentionally EXCLUDED; those
    neighbours are obstacle-like and atlas_reachable_gain returns 0 for them.

    Parameters
    ----------
    include_absent : bool
        When True (default), cells adjacent to absent atlas entries are also
        included.  Set to False to restrict to cat0-only boundaries; this
        avoids spurious detections at the outer edge of the atlas coverage
        area where absent cells are "void" rather than unexplored terrain.
    """
    if wp_id not in wp_positions:
        return []
    wx, wy = float(wp_positions[wp_id][0]), float(wp_positions[wp_id][1])
    gx_c   = int(round(wx / RESOLUTION))
    gy_c   = int(round(wy / RESOLUTION))
    r_cells      = int(radius / RESOLUTION) + 2
    stride_cells = max(1, int(stride / RESOLUTION))

    boundary = []
    for dgx in range(-r_cells, r_cells + 1, stride_cells):
        for dgy in range(-r_cells, r_cells + 1, stride_cells):
            gx, gy = gx_c + dgx, gy_c + dgy
            dx = (gx - gx_c) * RESOLUTION
            dy = (gy - gy_c) * RESOLUTION
            if dx * dx + dy * dy > radius * radius:
                continue
            if g2c.get((gx, gy)) != 1:          # must be free
                continue
            is_boundary = False
            for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n_cat = g2c.get((gx + ddx, gy + ddy))
                if n_cat == 0:                   # explicitly unknown
                    is_boundary = True
                    break
                if include_absent and n_cat is None:  # absent from atlas
                    is_boundary = True
                    break
            if is_boundary:
                boundary.append(np.array([gx * RESOLUTION, gy * RESOLUTION]))
    return boundary


def find_boundary_components(wp_id, wp_positions, g2c,
                             radius=BOUNDARY_RADIUS):
    """Find free-unknown boundary cells near wp_id grouped by connectivity.

    A boundary cell is a free atlas cell (cat==1) with at least one
    4-connected neighbour that is unknown (cat==0) or absent from atlas.

    Connected components of boundary cells are found by BFS.  For each
    component one representative (the cell closest to the WP position) is
    returned as a world-space [x, y] array.

    Parameters
    ----------
    wp_id       : int
    wp_positions: {wp_id: (x,y,z)}
    g2c         : {(gx,gy): cat}  atlas (typically covered_g2c)
    radius      : float  search radius in metres

    Returns
    -------
    list of np.ndarray([x, y])  — one per connected component
    """
    if wp_id not in wp_positions:
        return []
    wp_pos = wp_positions[wp_id]
    wx, wy = float(wp_pos[0]), float(wp_pos[1])
    gx_c   = int(round(wx / RESOLUTION))
    gy_c   = int(round(wy / RESOLUTION))
    r_cells = int(radius / RESOLUTION) + 2

    # Step 1: collect all free-unknown boundary cells within radius
    boundary_set = set()
    for dgx in range(-r_cells, r_cells + 1):
        for dgy in range(-r_cells, r_cells + 1):
            gx, gy = gx_c + dgx, gy_c + dgy
            dx = (gx - gx_c) * RESOLUTION
            dy = (gy - gy_c) * RESOLUTION
            if dx * dx + dy * dy > radius * radius:
                continue
            if g2c.get((gx, gy)) != 1:   # must be free
                continue
            for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n_cat = g2c.get((gx + ddx, gy + ddy))
                if n_cat == 0 or n_cat is None:
                    boundary_set.add((gx, gy))
                    break

    if not boundary_set:
        return []

    # Step 2: BFS to find connected components (4-connected within boundary_set)
    visited = set()
    representatives = []
    for start in boundary_set:
        if start in visited:
            continue
        component = []
        queue = deque([start])
        visited.add(start)
        while queue:
            gx, gy = queue.popleft()
            component.append((gx, gy))
            for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (gx + ddx, gy + ddy)
                if nb in boundary_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        # Step 3: pick cell closest to WP as representative
        best = min(component,
                   key=lambda c: (c[0] - gx_c) ** 2 + (c[1] - gy_c) ** 2)
        representatives.append(
            np.array([best[0] * RESOLUTION, best[1] * RESOLUTION]))

    return representatives


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Atlas-based reachable gain
# ═══════════════════════════════════════════════════════════════════════════

def atlas_reachable_gain(pt, g2c, flood_radius=ATLAS_FLOOD_RADIUS):
    """Atlas flood-fill gain from a candidate frontier point.

    Flood-fills through free (cat==1) cells starting from *pt*, then counts
    all adjacent unknown (cat==0) and absent-from-atlas cells as gain.

    Returns
    -------
    (gain_m2, connects, cell_frozenset)
        gain_m2        : float, m² of reachable unknown/absent cells
        connects       : bool, True if gain region touches the flood boundary
        cell_frozenset : frozenset of (gx,gy) atlas indices (for MERGE IoU)
    """
    gx_s    = int(round(float(pt[0]) / RESOLUTION))
    gy_s    = int(round(float(pt[1]) / RESOLUTION))
    r_cells = int(flood_radius / RESOLUTION)

    free_visited = set()
    gain_cells   = set()
    queue        = deque()

    start_cat = g2c.get((gx_s, gy_s))
    if start_cat == 1:
        queue.append((gx_s, gy_s))
    else:
        for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if g2c.get((gx_s + ddx, gy_s + ddy)) == 1:
                queue.append((gx_s + ddx, gy_s + ddy))

    while queue:
        gx, gy = queue.popleft()
        if (gx, gy) in free_visited:
            continue
        if abs(gx - gx_s) > r_cells or abs(gy - gy_s) > r_cells:
            continue
        free_visited.add((gx, gy))

        for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = gx + ddx, gy + ddy
            n_cat  = g2c.get((nx, ny))
            if n_cat == 1:
                if (nx, ny) not in free_visited:
                    queue.append((nx, ny))
            elif n_cat == 0 or n_cat is None:   # unknown or absent → gain
                gain_cells.add((nx, ny))
            # cat 2/3 → obstacle, skip

    gain_m2  = len(gain_cells) * CELL_AREA
    connects = any(
        abs(gx - gx_s) >= r_cells - 1 or abs(gy - gy_s) >= r_cells - 1
        for gx, gy in gain_cells
    )
    return gain_m2, connects, frozenset(gain_cells)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Overlap / coverage helpers
# ═══════════════════════════════════════════════════════════════════════════

def cells_iou(a: frozenset, b: frozenset) -> float:
    """IoU of two cell sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def is_covered(pt, existing_xy, radius=None):
    """True if pt is within COVER_RADIUS of any point in existing_xy."""
    r = radius if radius is not None else COVER_RADIUS
    r2 = r * r
    px, py = float(pt[0]), float(pt[1])
    for fx, fy in existing_xy:
        if (px - fx) ** 2 + (py - fy) ** 2 <= r2:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Refinement: DROP → MERGE → ADD
# ═══════════════════════════════════════════════════════════════════════════

def refine_frontier_set(wids, pred_frontiers, pred_gains,
                        pred_connects, pred_masks,
                        covered_g2c, wp_positions,
                        gain_model, device,
                        wp_data_cache=None,
                        g2c=None,
                        add_wps=None,
                        include_absent=True):
    """Apply DROP-low-gain → MERGE-overlapping → ADD-missing.

    Parameters
    ----------
    wids          : list[int]   observing WP IDs for this frontier step
    pred_frontiers: list[[x,y,conf,wp_id]]
    pred_gains    : list[float]
    pred_connects : list[bool]
    pred_masks    : list[frozenset]  reachable atlas cells per prediction
    covered_g2c   : {(gx,gy): cat}  WP-visible subset of atlas
    wp_positions  : {wp_id: (x,y,z)}  ALL waypoints in the map
    gain_model    : UNet (eval mode, for scoring DETR predictions)
    device        : torch.device
    wp_data_cache : optional {wp_id: wp_data}
    g2c           : full atlas {(gx,gy): cat}; if None, falls back to covered_g2c
    add_wps       : list of WP IDs to search in ADD step; if None, uses all
                    wp_positions keys
    include_absent: passed to find_free_unknown_boundary; set False to restrict
                    ADD candidates to cat0-only (avoids outer-map-edge noise)

    Returns
    -------
    (frontiers, gains, connects, masks, sources, dropped)
      sources : list[str]  'keep' or 'add' for each entry in frontiers
      dropped : list[dict] entries removed in DROP/MERGE steps, each with
                           {x, y, conf, wp_id, gain_m2, connects, source}
                           where source is 'drop' or 'merge'
    """
    frontiers = list(pred_frontiers)
    gains     = list(pred_gains)
    connects  = list(pred_connects)
    masks     = list(pred_masks)
    dropped   = []   # accumulates removed entries with source tag

    # ── STEP 1: DROP low-gain ────────────────────────────────────────────────
    keep_mask = [g >= REMOVE_GAIN_THRESH for g in gains]
    for i, (f, g, c) in enumerate(zip(frontiers, gains, connects)):
        if not keep_mask[i]:
            dropped.append({'x': float(f[0]), 'y': float(f[1]),
                            'conf': float(f[2]), 'wp_id': int(f[3]),
                            'gain_m2': float(g), 'connects': bool(c),
                            'source': 'drop'})
    keep = [i for i, ok in enumerate(keep_mask) if ok]
    frontiers = [frontiers[i] for i in keep]
    gains     = [gains[i]     for i in keep]
    connects  = [connects[i]  for i in keep]
    masks     = [masks[i]     for i in keep]
    print(f"    [refine] after low-gain DROP: {len(frontiers)} frontiers")

    # ── STEP 2: MERGE overlapping ────────────────────────────────────────────
    n = len(frontiers)
    drop_set = set()
    for i in range(n):
        if i in drop_set:
            continue
        for j in range(i + 1, n):
            if j in drop_set:
                continue
            if cells_iou(masks[i], masks[j]) > OVERLAP_THRESH:
                drop_set.add(i if gains[i] < gains[j] else j)
    for i in drop_set:
        f, g, c = frontiers[i], gains[i], connects[i]
        dropped.append({'x': float(f[0]), 'y': float(f[1]),
                        'conf': float(f[2]), 'wp_id': int(f[3]),
                        'gain_m2': float(g), 'connects': bool(c),
                        'source': 'merge'})
    keep = [i for i in range(n) if i not in drop_set]
    frontiers = [frontiers[i] for i in keep]
    gains     = [gains[i]     for i in keep]
    connects  = [connects[i]  for i in keep]
    masks     = [masks[i]     for i in keep]
    print(f"    [refine] after overlap MERGE: {len(frontiers)} frontiers")

    # sources for surviving DETR predictions
    sources = ['keep'] * len(frontiers)

    # ── STEP 3: ADD missing frontiers ────────────────────────────────────────
    # Both boundary detection AND gain estimation use covered_g2c to simulate
    # the robot's actual knowledge at this frontier (not the full global atlas).
    # atlas_reachable_gain treats absent cells (n_cat is None) as explorable
    # gain, so cells outside coverage boundary are correctly counted as unknown.
    gain_g2c = covered_g2c
    wp_list = add_wps if add_wps is not None else list(wp_positions.keys())

    # ── DEBUG: watch a specific WP ──────────────────────────────────────────
    _DEBUG_WP = int(os.environ.get('DEBUG_WP', '0'))

    existing_xy = [(float(f[0]), float(f[1])) for f in frontiers]
    added = 0
    for wp_id in wp_list:
        if wp_id not in wp_positions:
            continue
        candidates = find_free_unknown_boundary(
            wp_id, wp_positions, covered_g2c,
            include_absent=include_absent)
        _dbg = (_DEBUG_WP and wp_id == _DEBUG_WP)
        if _dbg:
            print(f"  [DEBUG WP {wp_id}] boundary candidates: {len(candidates)}")
        for pt in candidates:
            if is_covered(pt, existing_xy):
                if _dbg:
                    print(f"    [DEBUG] pt=({pt[0]:.1f},{pt[1]:.1f}) → FILTERED by is_covered")
                continue
            # Atlas flood-fill as fast gate filter AND fallback gain estimator.
            g_atlas, c_atlas, m_atlas = atlas_reachable_gain(pt, gain_g2c)
            if _dbg:
                print(f"    [DEBUG] pt=({pt[0]:.1f},{pt[1]:.1f}) atlas_gain={g_atlas:.2f}m² → "
                      f"{'pass' if g_atlas > ADD_GAIN_THRESH else 'FILTERED'}")
            if g_atlas > ADD_GAIN_THRESH:
                # Score with UNet gain model for accurate gain_m2.
                # For outer-boundary ADD candidates the "beyond" cells are absent
                # from covered_g2c (atlas_cat=-1 → inp[3]=0 → pred_bin=0).
                # Fall back to atlas_reachable_gain in that case so those
                # candidates still receive a meaningful non-zero gain estimate.
                g, c, m = predict_gain_with_mask(
                    pt, wp_id, covered_g2c,
                    wp_positions[wp_id], gain_model, device, wp_data_cache)
                if g == 0.0:
                    g, c, m = g_atlas, c_atlas, m_atlas
                if _dbg:
                    print(f"      → UNet gain={g:.2f}m²")
                frontiers.append([float(pt[0]), float(pt[1]), 0.5, wp_id])
                gains.append(g)
                connects.append(c)
                masks.append(m)
                sources.append('add')
                existing_xy.append((float(pt[0]), float(pt[1])))
                added += 1
    print(f"    [refine] ADD: +{added} new frontiers → total {len(frontiers)}")

    return frontiers, gains, connects, masks, sources, dropped


def refine_frontier_set_v2(wids, pred_world,
                           covered_g2c, wp_positions,
                           gain_model, device,
                           wp_data_cache=None,
                           add_wps=None):
    """Marginal-Coverage-Based frontier refinement (v2).

    Unified DROP → MERGE → ADD pipeline driven by marginal coverage Δ(p|A):
      Δ(p|A) = number of gain cells covered exclusively by p
                                (i.e. not covered by any other frontier in A)

    Steps
    -----
    1. Score DETR predictions A0 with UNet; discard entries with gain==0.
    2. DROP: remove p where Δ(p|A0)*CELL_AREA < DELTA_DROP.
    3. MERGE: greedy sort by gain desc; keep if it adds ≥ DELTA_KEEP to union.
    4. ADD: for each WP in add_wps, find boundary components, score with UNet,
       add at most MAX_ADD_PER_WP candidates with marginal ≥ DELTA_ADD.
    5. Weights: q(p) = clip(Δ_final / (Δ_final + KAPPA), Q_MIN, 1.0).

    Parameters
    ----------
    wids        : list[int]  observing WP IDs for this frontier step
    pred_world  : list[[x,y,conf,wp_id]]  raw DETR predictions (unscored)
    covered_g2c : {(gx,gy): cat}  WP-visible atlas subset
    wp_positions: {wp_id: (x,y,z)}
    gain_model  : UNet in eval mode
    device      : torch.device
    wp_data_cache: optional {wp_id: wp_data}
    add_wps     : list[int] or None  WPs to search in ADD step; None → wids only

    Returns
    -------
    (frontiers, gains, connects, masks, sources, dropped, weights)
      frontiers : list[[x,y,conf,wp_id]]
      gains     : list[float]
      connects  : list[bool]
      masks     : list[frozenset]  reachable atlas cells per frontier
      sources   : list[str]  'keep' or 'add'
      dropped   : list[dict]  entries removed in DROP/MERGE (source='drop'/'merge')
      weights   : list[float]  per-frontier training weight q(p)
    """
    # ── Step 1: Score A0 (DETR predictions) with UNet ────────────────────────
    print("  [v2.1] Scoring DETR predictions ...")
    A0_frontiers, A0_gains, A0_connects, A0_masks = [], [], [], []
    for entry in pred_world:
        x, y, conf, wp_id = entry[0], entry[1], entry[2], int(entry[3])
        if wp_id not in wp_positions:
            continue
        g, c, m = predict_gain_with_mask(
            np.array([x, y]), wp_id, covered_g2c,
            wp_positions[wp_id], gain_model, device, wp_data_cache)
        if g == 0.0:
            continue   # discard zero-gain predictions (no atlas fallback in v2)
        A0_frontiers.append([float(x), float(y), float(conf), int(wp_id)])
        A0_gains.append(g)
        A0_connects.append(c)
        A0_masks.append(m)
    print(f"      {len(pred_world)} raw → {len(A0_frontiers)} with gain>0")

    dropped = []

    # ── Step 2: DROP by marginal coverage ────────────────────────────────────
    print("  [v2.2] DROP step ...")
    frontiers, gains, connects, masks = [], [], [], []
    if A0_frontiers:
        # Build cell_count: how many frontiers in A0 cover each cell
        cell_count: dict = {}
        for m in A0_masks:
            for cell in m:
                cell_count[cell] = cell_count.get(cell, 0) + 1

        for f, g, c, m in zip(A0_frontiers, A0_gains, A0_connects, A0_masks):
            marginal_m2 = sum(
                1 for cell in m if cell_count.get(cell, 0) == 1
            ) * CELL_AREA
            if marginal_m2 >= DELTA_DROP:
                frontiers.append(f); gains.append(g)
                connects.append(c);  masks.append(m)
            else:
                dropped.append({'x': float(f[0]), 'y': float(f[1]),
                                'conf': float(f[2]), 'wp_id': int(f[3]),
                                'gain_m2': float(g), 'connects': bool(c),
                                'source': 'drop'})
    print(f"      after DROP: {len(frontiers)} frontiers")

    # ── Step 3: MERGE by greedy marginal coverage ─────────────────────────────
    print("  [v2.3] MERGE step ...")
    U: set = set()   # union of gain cells accepted so far
    if frontiers:
        order = sorted(range(len(frontiers)), key=lambda i: -gains[i])
        keep_idx = []
        for i in order:
            new_cells = masks[i] - U
            if len(new_cells) * CELL_AREA >= DELTA_KEEP:
                keep_idx.append(i)
                U.update(masks[i])
            else:
                f, g, c = frontiers[i], gains[i], connects[i]
                dropped.append({'x': float(f[0]), 'y': float(f[1]),
                                'conf': float(f[2]), 'wp_id': int(f[3]),
                                'gain_m2': float(g), 'connects': bool(c),
                                'source': 'merge'})
        keep_idx.sort()
        frontiers = [frontiers[i] for i in keep_idx]
        gains     = [gains[i]     for i in keep_idx]
        connects  = [connects[i]  for i in keep_idx]
        masks     = [masks[i]     for i in keep_idx]
    print(f"      after MERGE: {len(frontiers)} frontiers")

    sources = ['keep'] * len(frontiers)

    # ── Step 4: ADD missing frontiers ────────────────────────────────────────
    print("  [v2.4] ADD step ...")
    wp_list = add_wps if add_wps is not None else wids
    added   = 0

    for wp_id in wp_list:
        if wp_id not in wp_positions:
            continue
        wp_pos = wp_positions[wp_id]

        # One representative per boundary connected component
        candidates = find_boundary_components(wp_id, wp_positions, covered_g2c)
        if not candidates:
            continue

        # Score all candidates for this WP
        scored = []
        for pt in candidates:
            g, c, m = predict_gain_with_mask(
                pt, wp_id, covered_g2c,
                wp_pos, gain_model, device, wp_data_cache)
            if g == 0.0:
                continue
            new_cells  = m - U
            delta_m2   = len(new_cells) * CELL_AREA
            if delta_m2 < DELTA_ADD:
                continue
            dist  = math.sqrt((float(pt[0]) - float(wp_pos[0])) ** 2 +
                              (float(pt[1]) - float(wp_pos[1])) ** 2)
            score = delta_m2 - ETA * dist
            scored.append((score, delta_m2, pt, g, c, m))

        # Sort by score descending; greedily add up to MAX_ADD_PER_WP
        scored.sort(key=lambda t: -t[0])
        n_added_wp = 0
        for _score, _dm2, pt, g, c, m in scored:
            if n_added_wp >= MAX_ADD_PER_WP:
                break
            # Re-check marginal against updated U
            new_cells = m - U
            if len(new_cells) * CELL_AREA < DELTA_ADD:
                continue
            frontiers.append([float(pt[0]), float(pt[1]), 0.5, wp_id])
            gains.append(g)
            connects.append(c)
            masks.append(m)
            sources.append('add')
            U.update(m)
            added      += 1
            n_added_wp += 1

    print(f"      ADD: +{added} new frontiers → total {len(frontiers)}")

    # ── Step 5: Compute per-frontier training weights ─────────────────────────
    print("  [v2.5] Computing weights ...")
    final_cell_count: dict = {}
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
# 5b. v3: Set-cover greedy refinement
# ═══════════════════════════════════════════════════════════════════════════

def _eroded_green(covered_g2c, margin=WALL_MARGIN_CELLS):
    """Return green cells at least *margin* cells from any non-green cell.

    This produces the "visually green" region: the visualization renders
    obstacle cells wider than single atlas cells, so green cells near walls
    appear inside the wall.  Eroding by *margin* cells compensates.
    """
    green_cells = {k for k, v in covered_g2c.items() if v == 1}
    if margin <= 0:
        return green_cells
    # Non-green cells bordering the green region
    border_nongreen = set()
    for gx, gy in green_cells:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (gx + dx, gy + dy)
            if nb not in green_cells:
                border_nongreen.add(nb)
    # Dilate non-green border by margin
    too_close = set()
    m2 = margin * margin
    for bx, by in border_nongreen:
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                if dx * dx + dy * dy <= m2:
                    too_close.add((bx + dx, by + dy))
    return green_cells - too_close


def _shift_to_green(wx, wy, covered_g2c, safe_green,
                    radius_cells=SHIFT_RADIUS_CELLS):
    """Shift a frontier to the centroid of nearby *safe* green cells.

    1. BFS (unrestricted traversal) to find nearest safe-green cell.
    2. Centroid of safe-green cells within *radius_cells* of that anchor.
    3. Return shifted world coordinates.

    Falls back to original position if no safe-green cell within 50 cells.
    """
    gx = int(round(wx / RESOLUTION))
    gy = int(round(wy / RESOLUTION))

    # Find anchor: nearest safe-green cell
    if (gx, gy) in safe_green:
        anchor = (gx, gy)
    else:
        visited = {(gx, gy)}
        queue = deque([(gx, gy, 0)])
        anchor = None
        while queue:
            cx, cy, d = queue.popleft()
            if d > 50:  # max 10 m search
                break
            if (cx, cy) in safe_green:
                anchor = (cx, cy)
                break
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, d + 1))
        if anchor is None:
            return wx, wy

    # Centroid of safe-green cells within radius
    ax, ay = anchor
    sx, sy, cnt = 0, 0, 0
    r2 = radius_cells * radius_cells
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > r2:
                continue
            nx, ny = ax + dx, ay + dy
            if (nx, ny) in safe_green:
                sx += nx
                sy += ny
                cnt += 1
    if cnt == 0:
        cx, cy = anchor
    else:
        cx, cy = sx / cnt, sy / cnt
    return cx * RESOLUTION, cy * RESOLUTION


def refine_frontier_set_v3(wids, pred_world,
                           covered_g2c, wp_positions,
                           gain_model, device,
                           wp_data_cache=None,
                           add_wps=None,
                           coverage_frac=COVERAGE_FRAC,
                           shift_radius_cells=SHIFT_RADIUS_CELLS,
                           wall_margin_cells=WALL_MARGIN_CELLS,
                           proximity_disq_m=PROXIMITY_DISQ_M):
    """Set-cover greedy frontier selection (v3).

    Each frontier produces hard opinions: (cell, label) pairs where
    label = argmax in {free, obstacle}.  Greedy set cover picks the
    frontier covering the most uncovered pairs.  Stop when
    |covered| / |universe| >= coverage_frac.

    Pipeline: Shift → Score → Opinions → Set-cover → Weights.

    Returns
    -------
    (frontiers, gains, connects, masks, sources, dropped, weights,
     A0_frontiers, a0_cell_count, A0_masks, a0_mean_evidence, A0_pred_maps)
    """

    # ── Step 0: Shift frontiers to green region ─────────────────────────────
    print("  [v3.0] Shifting frontiers to green region ...")
    safe_green = _eroded_green(covered_g2c, margin=wall_margin_cells)
    print(f"      Eroded green: {len(safe_green)} safe cells "
          f"(margin={wall_margin_cells})")
    shifted = 0
    for i, entry in enumerate(pred_world):
        ox, oy = entry[0], entry[1]
        nx, ny = _shift_to_green(ox, oy, covered_g2c, safe_green,
                                 radius_cells=shift_radius_cells)
        if nx != ox or ny != oy:
            pred_world[i] = [nx, ny, entry[2], entry[3]]
            shifted += 1
    print(f"      Shifted {shifted}/{len(pred_world)} frontiers")

    # ── Step 1: Score DETR predictions with UNet ────────────────────────────
    print("  [v3.1] Scoring DETR predictions ...")
    A0_frontiers, A0_gains, A0_connects, A0_masks, A0_pred_maps = \
        [], [], [], [], []
    evidence_sum: dict = {}
    evidence_cnt: dict = {}
    for entry in pred_world:
        x, y, conf, wp_id = entry[0], entry[1], entry[2], int(entry[3])
        if wp_id not in wp_positions:
            continue
        g, c, m, ev, cp = predict_gain_with_mask(
            np.array([x, y]), wp_id, covered_g2c,
            wp_positions[wp_id], gain_model, device,
            wp_data_cache, extended=True)
        if g == 0.0:
            continue
        A0_frontiers.append([float(x), float(y), float(conf), int(wp_id)])
        A0_gains.append(g)
        A0_connects.append(c)
        A0_masks.append(m)
        A0_pred_maps.append(cp)
        for cell, val in ev.items():
            evidence_sum[cell] = evidence_sum.get(cell, 0.0) + val
            evidence_cnt[cell] = evidence_cnt.get(cell, 0) + 1
    a0_mean_evidence = {cell: evidence_sum[cell] / evidence_cnt[cell]
                        for cell in evidence_sum}
    print(f"      {len(pred_world)} raw → {len(A0_frontiers)} with gain>0")

    # Cell coverage count (for compatibility / diagnostics)
    a0_cell_count: dict = {}
    for m in A0_masks:
        for cell in m:
            a0_cell_count[cell] = a0_cell_count.get(cell, 0) + 1

    # ── Step 2: Build per-frontier opinion sets ─────────────────────────────
    print("  [v3.2] Building opinion sets ...")
    opinion_sets = []   # list of set[(cell, label)] per frontier
    universe = set()    # all (cell, label) pairs across all frontiers
    for cp in A0_pred_maps:
        opinions = set()
        for cell, (cls, pf, po) in cp.items():
            if cls in (0, 1):  # free or obstacle
                opinions.add((cell, cls))
        opinion_sets.append(opinions)
        universe |= opinions
    n_universe = len(universe)
    print(f"      {n_universe} (cell, label) pairs in universe")

    if n_universe == 0:
        print("      No structural opinions — returning empty set")
        return ([], [], [], [], [], [], [],
                A0_frontiers, a0_cell_count, A0_masks, a0_mean_evidence,
                A0_pred_maps)

    # ── Step 3: Greedy set cover ────────────────────────────────────────────
    disq_r2 = proximity_disq_m * proximity_disq_m
    print(f"  [v3.3] Greedy set cover "
          f"(proximity={proximity_disq_m}m, target={coverage_frac}) ...")

    covered = set()
    selected_idx = []
    candidates = set(range(len(A0_frontiers)))

    while candidates:
        best_idx = -1
        best_gain = 0
        for i in candidates:
            gain = len(opinion_sets[i] - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx < 0 or best_gain == 0:
            break

        selected_idx.append(best_idx)
        candidates.discard(best_idx)
        covered |= opinion_sets[best_idx]

        # Disqualify candidates too close to ANY selected frontier
        too_close = set()
        for i in candidates:
            cx, cy = A0_frontiers[i][0], A0_frontiers[i][1]
            for si in selected_idx:
                dx = cx - A0_frontiers[si][0]
                dy = cy - A0_frontiers[si][1]
                if dx * dx + dy * dy <= disq_r2:
                    too_close.add(i)
                    break
        candidates -= too_close

        frac = len(covered) / n_universe
        print(f"        pick #{len(selected_idx)}: A0[{best_idx}]  "
              f"+{best_gain} opinions  coverage={frac:.3f}  "
              f"disqualified={len(too_close)}")
        if frac >= coverage_frac:
            break

    print(f"      Selected {len(selected_idx)}/{len(A0_frontiers)} frontiers  "
          f"(coverage={len(covered)}/{n_universe} "
          f"= {len(covered)/n_universe:.3f})")

    # Build dropped list
    selected_set = set(selected_idx)
    dropped = []
    for i in range(len(A0_frontiers)):
        if i not in selected_set:
            f = A0_frontiers[i]
            dropped.append({'x': float(f[0]), 'y': float(f[1]),
                            'conf': float(f[2]), 'wp_id': int(f[3]),
                            'gain_m2': float(A0_gains[i]),
                            'connects': bool(A0_connects[i]),
                            'source': 'drop'})

    # Build output lists
    frontiers = [A0_frontiers[i] for i in selected_idx]
    gains     = [A0_gains[i]     for i in selected_idx]
    connects  = [A0_connects[i]  for i in selected_idx]
    masks     = [A0_masks[i]     for i in selected_idx]
    sources   = ['keep'] * len(frontiers)

    # ── Step 4: Compute weights (same formula as v2) ────────────────────────
    print("  [v3.4] Computing weights ...")
    final_cell_count: dict = {}
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

    return (frontiers, gains, connects, masks, sources, dropped, weights,
            A0_frontiers, a0_cell_count, A0_masks, a0_mean_evidence,
            A0_pred_maps)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  DETR inference
# ═══════════════════════════════════════════════════════════════════════════

def run_detr_inference(detr_model, device, wids, covered_g2c, wp_positions,
                       wp_data_cache=None):
    """Run DETR on all observing WPs; return list of [x, y, conf, wp_id]."""
    atlas_cat, gx_min, gy_min = build_atlas_grid(covered_g2c)
    atlas_cov  = (atlas_cat >= 0)
    atlas_gmin = (gx_min, gy_min)

    pred_world = []
    for wp_id in wids:
        if wp_id not in wp_positions:
            continue
        wp_pos    = np.asarray(wp_positions[wp_id], dtype=np.float64)
        center_xy = wp_pos[:2]
        theta     = 0.0

        if wp_data_cache is not None and wp_id in wp_data_cache:
            wp_data = wp_data_cache[wp_id]
        else:
            try:
                wp_data = _load_wp_data(wp_id)
            except Exception:
                continue
            if wp_data_cache is not None:
                wp_data_cache[wp_id] = wp_data

        try:
            inp, _ = build_sample(center_xy, theta,
                                  atlas_cat, atlas_cov, atlas_gmin,
                                  wp_data=wp_data)
        except Exception:
            continue

        inp_t = torch.from_numpy(inp.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_xy_t, pred_conf_t = detr_model(inp_t)
        pred_xy   = pred_xy_t[0].cpu().numpy()
        pred_conf = torch.sigmoid(pred_conf_t[0]).cpu().numpy()

        mask = pred_conf > CONF_THRESH
        if not mask.any():
            continue

        world_xy = bev_normalized_to_world(pred_xy[mask], center_xy, theta)
        confs    = pred_conf[mask]
        for i in range(len(world_xy)):
            pred_world.append([float(world_xy[i, 0]), float(world_xy[i, 1]),
                                float(confs[i]), wp_id])
    return pred_world


# ═══════════════════════════════════════════════════════════════════════════
# 7.  I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_round_results(out_dir, rnd, frontiers, gains, connects,
                       sources=None, dropped=None, weights=None,
                       wp_positions=None, wids=None):
    """Save refined frontier list as JSON.

    Parameters
    ----------
    sources : list[str] or None  'keep'/'add' per frontier; if None all tagged 'keep'
    dropped : list[dict] or None  entries removed by DROP/MERGE with source tag
    weights : list[float] or None  per-frontier training weight; if None defaults to 1.0
    wp_positions : dict or None  {wp_id: [x, y, z]} for reassigning to nearest observing WP
    wids : list[int] or None  observing WP IDs; used with wp_positions for reassignment
    """
    if sources is None:
        sources = ['keep'] * len(frontiers)
    if weights is None:
        weights = [1.0] * len(frontiers)

    # ── Reassign wp_id to nearest observing WP ────────────────────────────────
    if wp_positions is not None and wids is not None:
        for frontier in frontiers:
            fx, fy = frontier[0], frontier[1]
            min_dist = float('inf')
            closest_wp = None
            for wp_id in wids:
                if wp_id not in wp_positions:
                    continue
                wp_pos = wp_positions[wp_id]
                dist = math.sqrt((fx - wp_pos[0])**2 + (fy - wp_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_wp = wp_id
            if closest_wp is not None:
                frontier[3] = closest_wp  # Update wp_id to nearest observing WP

    data = [{'id':      i,
             'x':       float(e[0]),   'y':        float(e[1]),
             'conf':    float(e[2]),   'wp_id':    int(e[3]),
             'gain_m2': float(g),      'connects': bool(c),
             'source':  str(s),        'weight':   float(w)}
            for i, (e, g, c, s, w) in enumerate(
                zip(frontiers, gains, connects, sources, weights))]
    n_live = len(data)
    if dropped:
        for j, d in enumerate(dropped):
            d['id'] = n_live + j
        data.extend(dropped)   # already dicts with 'source' field
    path = os.path.join(out_dir, f"refined_round{rnd}.json")
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)
    n_live = len(frontiers)
    n_drop = len(dropped) if dropped else 0
    print(f"    [saved] {n_live} kept+added, {n_drop} dropped → {path}")
    return path


def save_a0_data(out_dir, rnd, a0_frontiers, a0_cell_count,
                 a0_masks=None, a0_mean_evidence=None, a0_pred_maps=None):
    """Save pre-refinement A0 diagnostic data (v3).

    Files produced:
        a0_frontiers_round{rnd}.json   — scored DETR predictions before selection
        a0_cell_count_round{rnd}.pkl   — {(gx,gy): int} coverage count
        a0_masks_round{rnd}.pkl        — list of frozenset, per-frontier gain cells
        a0_mean_evidence_round{rnd}.pkl — {(gx,gy): float} mean -log(1-p)
        a0_pred_maps_round{rnd}.pkl    — list of {(gx,gy): (cls,pf,po)} full preds
    """
    import pickle

    data = [{'id': i, 'x': float(f[0]), 'y': float(f[1]),
             'conf': float(f[2]), 'wp_id': int(f[3])}
            for i, f in enumerate(a0_frontiers)]
    json_path = os.path.join(out_dir, f"a0_frontiers_round{rnd}.json")
    with open(json_path, 'w') as fh:
        json.dump(data, fh, indent=2)

    pkl_path = os.path.join(out_dir, f"a0_cell_count_round{rnd}.pkl")
    with open(pkl_path, 'wb') as fh:
        pickle.dump(a0_cell_count, fh)

    if a0_masks is not None:
        with open(os.path.join(out_dir, f"a0_masks_round{rnd}.pkl"), 'wb') as fh:
            pickle.dump(a0_masks, fh)

    if a0_mean_evidence is not None:
        with open(os.path.join(out_dir, f"a0_mean_evidence_round{rnd}.pkl"), 'wb') as fh:
            pickle.dump(a0_mean_evidence, fh)

    if a0_pred_maps is not None:
        with open(os.path.join(out_dir, f"a0_pred_maps_round{rnd}.pkl"), 'wb') as fh:
            pickle.dump(a0_pred_maps, fh)

    print(f"    [saved] A0: {len(a0_frontiers)} frontiers, "
          f"{len(a0_cell_count)} cells → {json_path}")


def print_round_summary(rnd, frontiers, gains, connects):
    mean_g = float(np.mean(gains)) if gains else 0.0
    n_conn = sum(connects)
    print(f"    [round {rnd}] {len(frontiers)} frontiers  "
          f"mean_gain={mean_g:.1f} m²  connects={n_conn}/{len(frontiers)}")


# ═══════════════════════════════════════════════════════════════════════════
# 8.  High-level: generate supervision signal for one round
# ═══════════════════════════════════════════════════════════════════════════

def generate_supervision_signal(frontier_id, detr_model, gain_model,
                                device, g2c, covered_g2c, wp_positions,
                                wids, rnd, out_dir,
                                wp_data_cache=None,
                                add_wps=None,
                                include_absent=True,
                                refine_method='v2'):
    """Full pipeline: DETR inference → refinement → save JSON.

    Parameters
    ----------
    add_wps        : list[int] or None  WPs to search in ADD step; None = wids
    include_absent : bool  (unused in v2/v3, kept for API compatibility)
    refine_method  : 'v2' | 'v3'  refinement algorithm to use

    Returns
    -------
    (ref_frontiers, ref_gains, ref_connects, ref_masks,
     ref_sources, ref_dropped, ref_weights, json_path)
    """
    # 1. DETR inference
    print("  [1] DETR inference ...")
    pred_world = run_detr_inference(
        detr_model, device, wids, covered_g2c, wp_positions,
        wp_data_cache=wp_data_cache)
    print(f"      {len(pred_world)} raw predictions")

    # 2. Refinement (dispatch by method)
    if refine_method == 'v3':
        print("  [2] Refining frontier set (v3 — set cover) ...")
        (ref_frontiers, ref_gains, ref_connects, ref_masks,
         ref_sources, ref_dropped, ref_weights,
         a0_frontiers, a0_cell_count, a0_masks,
         a0_mean_evidence, a0_pred_maps) = refine_frontier_set_v3(
            wids, pred_world,
            covered_g2c, wp_positions, gain_model, device,
            wp_data_cache=wp_data_cache,
            add_wps=add_wps)
    else:
        print("  [2] Refining frontier set (v2 — DROP+MERGE+ADD) ...")
        (ref_frontiers, ref_gains, ref_connects, ref_masks,
         ref_sources, ref_dropped, ref_weights) = refine_frontier_set_v2(
            wids, pred_world,
            covered_g2c, wp_positions, gain_model, device,
            wp_data_cache=wp_data_cache,
            add_wps=add_wps)

    print_round_summary(rnd, ref_frontiers, ref_gains, ref_connects)

    # 3. Save
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = save_round_results(out_dir, rnd, ref_frontiers, ref_gains, ref_connects,
                              sources=ref_sources, dropped=ref_dropped,
                              weights=ref_weights,
                              wp_positions=wp_positions, wids=wids)

    # 4. Save A0 diagnostic data (v3 only)
    if refine_method == 'v3':
        save_a0_data(out_dir, rnd, a0_frontiers, a0_cell_count,
                     a0_masks, a0_mean_evidence, a0_pred_maps)

    return (ref_frontiers, ref_gains, ref_connects, ref_masks,
            ref_sources, ref_dropped, ref_weights, path)


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global DELTA_DROP, DELTA_KEEP, DELTA_ADD, MAX_ADD_PER_WP
    global ETA, KAPPA, Q_MIN, CONF_THRESH, BOUNDARY_RADIUS
    global COVERAGE_FRAC, SHIFT_RADIUS_CELLS, WALL_MARGIN_CELLS, PROXIMITY_DISQ_M

    ap = argparse.ArgumentParser(
        description='Generate self-supervised frontier pseudo-labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--frontier-id',      type=int,   required=True)
    ap.add_argument('--detr-checkpoint',  type=str,   default=None,
                    help='DETR checkpoint (mutually exclusive with --pred-json)')
    ap.add_argument('--pred-json',        type=str,   default=None,
                    help='Pre-computed predictions JSON [{x,y,conf,wp_id},...]')
    ap.add_argument('--gain-checkpoint',  required=True)
    ap.add_argument('--out-dir',          type=str,   default=None)
    ap.add_argument('--round',            type=int,   default=1,
                    help='Round index used in output filename')
    ap.add_argument('--refine-method',    type=str,   default='v2',
                    choices=['v2', 'v3'],
                    help='"v2"=DROP+MERGE+ADD; "v3"=set-cover greedy selection')
    # v2 thresholds
    ap.add_argument('--delta-drop',       type=float, default=DELTA_DROP,
                    help='Marginal coverage (m²) below which prediction is DROPped')
    ap.add_argument('--delta-keep',       type=float, default=DELTA_KEEP,
                    help='Min new coverage (m²) to KEEP in greedy MERGE')
    ap.add_argument('--delta-add',        type=float, default=DELTA_ADD,
                    help='Min marginal coverage (m²) to ADD a candidate')
    ap.add_argument('--max-add-per-wp',   type=int,   default=MAX_ADD_PER_WP,
                    help='Max candidates added per observing WP')
    ap.add_argument('--kappa',            type=float, default=KAPPA,
                    help='Weight smoothing constant')
    ap.add_argument('--q-min',            type=float, default=Q_MIN,
                    help='Minimum training weight')
    ap.add_argument('--conf-thresh',      type=float, default=CONF_THRESH,
                    help='DETR confidence threshold')
    ap.add_argument('--boundary-radius',  type=float, default=BOUNDARY_RADIUS,
                    help='Search radius (m) for boundary components in ADD step')
    ap.add_argument('--add-wps',          type=str,   default='obs',
                    choices=['all', 'obs'],
                    help='"all" = all WPs in map; "obs" = observing WPs only')
    # v3 thresholds
    ap.add_argument('--coverage-frac',    type=float, default=COVERAGE_FRAC,
                    help='v3: stop when this fraction of opinion universe is covered')
    ap.add_argument('--proximity-disq-m', type=float, default=PROXIMITY_DISQ_M,
                    help='v3: disqualify candidates within this distance (m)')
    args = ap.parse_args()

    if args.detr_checkpoint is None and args.pred_json is None:
        ap.error('Provide either --detr-checkpoint or --pred-json')
    if args.detr_checkpoint is not None and args.pred_json is not None:
        ap.error('--detr-checkpoint and --pred-json are mutually exclusive')

    # Override globals
    DELTA_DROP      = args.delta_drop
    DELTA_KEEP      = args.delta_keep
    DELTA_ADD       = args.delta_add
    MAX_ADD_PER_WP  = args.max_add_per_wp
    KAPPA           = args.kappa
    Q_MIN           = args.q_min
    CONF_THRESH     = args.conf_thresh
    BOUNDARY_RADIUS = args.boundary_radius
    COVERAGE_FRAC   = args.coverage_frac
    PROXIMITY_DISQ_M = args.proximity_disq_m

    device = _get_device()
    print(f"[refine] Device: {device}")

    # Atlas + frontier data
    print("[refine] Loading atlas ...")
    atlas_pts, wp_positions, g2c = load_atlas()

    print(f"[refine] Loading frontier {args.frontier_id} ...")
    frontier_data = load_frontier_data(args.frontier_id)
    wids = frontier_data['waypoint_ids']
    print(f"         obs_WPs={len(wids)}  total_WPs={len(wp_positions)}")

    # covered_g2c
    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cat_f, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_cov_grid = build_coverage_grid(
        atlas_pts, cmask, gx_min, gy_min, atlas_cat_f.shape)
    covered_g2c = {k: v for k, v in g2c.items()
                   if atlas_cov_grid[k[0] - gx_min, k[1] - gy_min]}
    print(f"         covered atlas cells: {len(covered_g2c):,}")

    out_dir = args.out_dir or f"ss_output/{args.frontier_id}"

    # Gain model
    print("[refine] Loading gain model ...")
    UNet       = _build_gain_model()
    gain_model = UNet().to(device)
    gain_model.load_state_dict(
        torch.load(args.gain_checkpoint, map_location=device, weights_only=True))
    gain_model.eval()

    # Predictions
    wp_data_cache = {}
    if args.detr_checkpoint is not None:
        print("[refine] Loading DETR model ...")
        FrontierDETR = _build_detr_model()
        detr_model   = FrontierDETR().to(device)
        detr_model.load_state_dict(
            torch.load(args.detr_checkpoint, map_location=device, weights_only=True))
        detr_model.eval()
        print("[refine] Running DETR inference ...")
        pred_world = run_detr_inference(
            detr_model, device, wids, covered_g2c, wp_positions,
            wp_data_cache=wp_data_cache)
        print(f"         {len(pred_world)} raw predictions")
    else:
        print(f"[refine] Loading predictions from {args.pred_json} ...")
        with open(args.pred_json) as f:
            raw = json.load(f)
        pred_world = [[d['x'], d['y'], d['conf'], d['wp_id']] for d in raw]

    # Determine WPs to search in ADD step
    add_wps = list(wp_positions.keys()) if args.add_wps == 'all' else wids

    # Refine
    if args.refine_method == 'v3':
        print("[refine] Refining (v3 — set cover) ...")
        (ref_frontiers, ref_gains, ref_connects, ref_masks,
         ref_sources, ref_dropped, ref_weights,
         a0_frontiers, a0_cell_count, a0_masks,
         a0_mean_evidence, a0_pred_maps) = refine_frontier_set_v3(
            wids, pred_world,
            covered_g2c, wp_positions, gain_model, device,
            wp_data_cache=wp_data_cache,
            add_wps=add_wps)
    else:
        print("[refine] Refining (v2 — DROP+MERGE+ADD) ...")
        (ref_frontiers, ref_gains, ref_connects, ref_masks,
         ref_sources, ref_dropped, ref_weights) = refine_frontier_set_v2(
            wids, pred_world,
            covered_g2c, wp_positions, gain_model, device,
            wp_data_cache=wp_data_cache,
            add_wps=add_wps)

    print_round_summary(args.round, ref_frontiers, ref_gains, ref_connects)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_round_results(out_dir, args.round, ref_frontiers, ref_gains, ref_connects,
                       sources=ref_sources, dropped=ref_dropped, weights=ref_weights,
                       wp_positions=wp_positions, wids=wids)

    if args.refine_method == 'v3':
        save_a0_data(out_dir, args.round, a0_frontiers, a0_cell_count,
                     a0_masks, a0_mean_evidence, a0_pred_maps)

    print(f"\n[refine] Done.  Output: {out_dir}/refined_round{args.round}.json")


if __name__ == '__main__':
    main()
