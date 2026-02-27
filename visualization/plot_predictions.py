#!/usr/bin/env python3
"""
Batch process frontier files and generate combined views
"""
import os
import sys

from models import map_utils
from models.map_utils import (
    load_frontier_data, load_waypoint_meta, load_waypoint_position,
    project_depth_from_waypoint, find_frontier_camera, project_point_to_image,
    get_frontier_camera_projection, CAMERAS, BASE_DIR, WP_IMAGES_DIR,
    build_equirectangular_panorama, project_pixel_to_panorama,
    IMAGE_SCALE, FRONTIERS_DIR,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import numpy as np

# Output directory for batch results
BATCH_OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# Load atlas (extract only what we need, then free memory)
import pickle
with open(os.path.join(map_utils.DUMP_DIR, "atlas.pkl"), 'rb') as _f:
    _atlas = pickle.load(_f)
_WP_GRAPH = _atlas['connectivity_graph']
_WP_EDGES = list(_WP_GRAPH.edges())
_WP_POSITIONS = _atlas['waypoint_positions']       # dict: wp_id → [x, y, z]
_ATLAS_POINTS_2D = _atlas['atlas_points_2d']       # (N, 2) world XY
_ATLAS_GRID_IDX = _atlas['atlas_grid_indices']      # (N, 2) grid coords
_GRID_TO_CAT = _atlas['grid_to_category']           # {(gx,gy): 0/1/2/3}
del _atlas


def _orient_for_display(image, u, v, cam_name):
    """Rotate image and pixel coords so all cameras appear upright.

    - frontleft / frontright: 90° CW
    - right: 180°
    """
    if cam_name in ('frontleft', 'frontright'):
        # 90° CW: np.rot90(img, k=-1)  shape (H,W) -> (W,H)
        H = image.shape[0]
        image = np.rot90(image, k=-1)
        u, v = H - 1 - v, u
    elif cam_name == 'right':
        # 180°: np.rot90(img, k=2)  shape unchanged
        H, W = image.shape[:2]
        image = np.rot90(image, k=2)
        u, v = W - 1 - u, H - 1 - v
    return image, u, v


def plot_single_map_to_folder(frontier_id, obs_wp, frontier_indices, frontier_positions,
                               waypoint_ids, output_folder, depth_step=4):
    """
    Plot a combined figure and save to specified folder

    Args:
        frontier_id: Frontier file ID
        obs_wp: Observing waypoint ID
        frontier_indices: List of frontier indices
        frontier_positions: All frontier positions
        waypoint_ids: All waypoint IDs
        output_folder: Output folder path
        depth_step: Depth sampling step
    """
    print(f"  Creating combined view for WP_{obs_wp} with frontiers: {['F_'+str(i) for i in frontier_indices]}")

    # Get all frontier projections
    frontier_projections = []
    for i in frontier_indices:
        fp = frontier_positions[i]
        proj = get_frontier_camera_projection(fp, i, obs_wp)
        if proj is not None:
            frontier_projections.append(proj)
            print(f"    F_{i} -> {proj['camera']}.png at ({proj['pixel'][0]:.1f}, {proj['pixel'][1]:.1f})")
        else:
            print(f"    Warning: F_{i} is not visible in any camera")

    # Determine layout: top = map | RGB | depth,  bottom = panorama strip
    n_cameras = len(frontier_projections)
    if n_cameras == 0:
        fig = plt.figure(figsize=(28, 16))
        gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.15)
        ax_map = fig.add_subplot(gs_outer[0])
        axes_rgb = []
        axes_depth = []
        gs_pano = gs_outer[1]
    else:
        n_rows = max(n_cameras, 1)
        main_h = max(8, 5 * n_rows)
        pano_h = 4
        fig = plt.figure(figsize=(28, main_h + pano_h))
        gs_outer = GridSpec(2, 1, figure=fig,
                            height_ratios=[main_h, pano_h], hspace=0.12)
        # Top: 3-column grid for map | RGB | depth
        gs_top = gs_outer[0].subgridspec(n_rows, 3, width_ratios=[1.2, 1, 1])
        ax_map = fig.add_subplot(gs_top[:, 0])  # map spans all rows
        axes_rgb = []
        axes_depth = []
        for j in range(n_cameras):
            axes_rgb.append(fig.add_subplot(gs_top[j, 1]))
            axes_depth.append(fig.add_subplot(gs_top[j, 2]))
        gs_pano = gs_outer[1]

    # Project depth
    all_points, camera_points = project_depth_from_waypoint(
        obs_wp, step=depth_step
    )

    cam_colors = {
        'back': '#888888',
        'frontleft': '#FF6600',
        'frontright': '#00CCCC',
        'left': '#CC00CC',
        'right': '#00AA00'
    }

    for cam_name, points in camera_points.items():
        if len(points) > 0:
            x_world = points[:, 0]
            y_world = points[:, 1]
            color = cam_colors.get(cam_name, 'gray')
            ax_map.scatter(x_world, y_world, c=color, s=2, alpha=0.5, label=cam_name)
            mean_x, mean_y = np.mean(x_world), np.mean(y_world)
            ax_map.annotate(cam_name, (mean_x, mean_y),
                           fontsize=9, fontweight='bold', color=color,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor=color, alpha=0.8))

    # Plot only the observing waypoint
    wp_positions = {}
    try:
        x, y, heading = load_waypoint_position(obs_wp)
        wp_positions[obs_wp] = (x, y, heading)

        ax_map.scatter(x, y, c='green', s=200, marker='o', zorder=5,
                      edgecolors='darkgreen', linewidths=2)
        ax_map.annotate(f'WP_{obs_wp}', (x, y), xytext=(5, 5),
                       textcoords='offset points', fontsize=10, fontweight='bold')

        arrow_length = 0.5
        dx = arrow_length * np.cos(heading)
        dy = arrow_length * np.sin(heading)
        ax_map.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1,
                    fc='green', ec='darkgreen', zorder=6)
    except Exception as e:
        pass

    # Plot frontiers
    for i in frontier_indices:
        fp = frontier_positions[i]
        fx, fy = fp[0], fp[1]
        ax_map.scatter(fx, fy, c='red', s=300, marker='*', zorder=4,
                      edgecolors='darkred', linewidths=1)
        ax_map.annotate(f'F_{i}', (fx, fy), xytext=(8, 8),
                       textcoords='offset points', fontsize=9, color='darkred')
        if obs_wp in wp_positions:
            wx, wy, _ = wp_positions[obs_wp]
            ax_map.plot([fx, wx], [fy, wy], 'g--', alpha=0.5, linewidth=1.5)

    # Set map properties
    ax_map.set_xlabel('X (meters) - Forward', fontsize=12)
    ax_map.set_ylabel('Y (meters) - Left', fontsize=12)
    frontier_str = ', '.join([f'F_{i}' for i in frontier_indices])
    ax_map.set_title(f'Top-Down Map (WP_{obs_wp}) | Frontiers: {frontier_str}', fontsize=14)
    ax_map.grid(True, alpha=0.3)
    ax_map.set_aspect('equal')
    ax_map.legend(loc='upper right', markerscale=2)

    # Compute agent_z for height diff
    meta = load_waypoint_meta(obs_wp)
    agent_z = meta["agent_position"][2]

    # Pre-compute frontier z from depth for all frontiers (used by both RGB and panorama)
    wp_folder = os.path.join(WP_IMAGES_DIR, str(obs_wp))
    frontier_z_map = {}  # frontier_idx -> world z
    for i in frontier_indices:
        fp = frontier_positions[i]
        dxy = all_points[:, :2] - np.array([fp[0], fp[1]])
        dists = np.linalg.norm(dxy, axis=1)
        mask = dists <= 0.3
        if np.any(mask):
            fz = np.percentile(all_points[mask, 2], 95)
        else:
            fz = all_points[np.argmin(dists), 2]
        dz = fz - agent_z
        frontier_z_map[i] = (agent_z + dz, dz)

    # Plot RGB and depth images
    for j, proj in enumerate(frontier_projections):
        ax_rgb = axes_rgb[j]
        ax_dep = axes_depth[j]
        u, v = proj['pixel']
        f_idx = proj['frontier_idx']
        cam_name = proj['camera']

        # Frontier height from pre-computed map
        fp = frontier_positions[f_idx]
        _, dz = frontier_z_map[f_idx]
        dz_str = f"  dz={dz:+.2f}m"

        # --- RGB image (rotate for upright display) ---
        disp_img, u_disp, v_disp = _orient_for_display(proj['image'], u, v, cam_name)
        ax_rgb.imshow(disp_img)
        ax_rgb.scatter(u_disp, v_disp, c='red', s=200, marker='*',
                      edgecolors='yellow', linewidths=2, zorder=5)
        ax_rgb.annotate(f"F_{f_idx}{dz_str}", (u_disp, v_disp), xytext=(10, -10),
                       textcoords='offset points', fontsize=12, fontweight='bold',
                       color='red', bbox=dict(boxstyle='round,pad=0.3',
                                             facecolor='yellow', alpha=0.9))
        ax_rgb.set_title(f"F_{f_idx} on {cam_name}.png (RGB)", fontsize=12)

        # --- Depth image ---
        depth_path = os.path.join(wp_folder, f"{cam_name}_depth.npy")
        if os.path.exists(depth_path):
            depth = np.load(depth_path)

            # Read depth value at frontier pixel (before rotation)
            u_d, v_d = u / IMAGE_SCALE, v / IMAGE_SCALE
            H_d, W_d = depth.shape
            r_d, c_d = int(round(v_d)), int(round(u_d))
            r_d = np.clip(r_d, 0, H_d - 1)
            c_d = np.clip(c_d, 0, W_d - 1)
            depth_val = depth[r_d, c_d]

            # Rotate for upright display (same as RGB)
            depth_vis = depth.astype(np.float32).copy()
            depth_vis[(depth_vis <= 0) | (depth_vis > 10)] = np.nan
            depth_vis, u_d_disp, v_d_disp = _orient_for_display(depth_vis, u_d, v_d, cam_name)

            im = ax_dep.imshow(depth_vis, cmap='turbo')
            fig.colorbar(im, ax=ax_dep, fraction=0.046, pad=0.04, label='depth (m)')

            # Marker
            ax_dep.scatter(u_d_disp, v_d_disp, c='red', s=200, marker='*',
                          edgecolors='yellow', linewidths=2, zorder=5)

            # Annotation with depth value
            depth_str = f"F_{f_idx}  d={depth_val:.2f}m"
            ax_dep.annotate(depth_str, (u_d_disp, v_d_disp), xytext=(10, -10),
                           textcoords='offset points', fontsize=12, fontweight='bold',
                           color='white', bbox=dict(boxstyle='round,pad=0.3',
                                                   facecolor='black', alpha=0.8))
        ax_dep.set_title(f"F_{f_idx} on {cam_name}_depth", fontsize=12)

    # --- Equirectangular panorama strip ---
    PANO_W, PANO_H, PANO_FOV = 2400, 480, (-45, 45)
    panorama = build_equirectangular_panorama(
        meta, wp_folder, PANO_W, PANO_H, PANO_FOV)

    ax_pano = fig.add_subplot(gs_pano)
    ax_pano.imshow(panorama)

    # Camera name labels at each camera's viewing-centre azimuth
    heading = meta.get('agent_heading', 0)
    for cam_name in CAMERAS:
        if cam_name not in meta['cameras']:
            continue
        T_wc = np.array(meta['cameras'][cam_name]['camera_extrinsic'])
        view_dir = T_wc[:3, 2]  # camera Z-axis in world frame
        theta = np.arctan2(view_dir[1], view_dir[0])
        px = (PANO_W * (heading + np.pi - theta) / (2.0 * np.pi)) % PANO_W
        ax_pano.text(px, 12, cam_name, ha='center', va='top',
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # Mark frontiers on panorama via camera pixel → panorama projection
    # (exact inverse of the panorama build pipeline)
    proj_lookup = {p['frontier_idx']: p for p in frontier_projections}
    for i in frontier_indices:
        if i not in proj_lookup:
            continue  # skip frontiers not visible in any camera
        proj = proj_lookup[i]
        pano_pt = project_pixel_to_panorama(
            proj['camera'], proj['pixel'][0], proj['pixel'][1],
            meta, PANO_W, PANO_H, PANO_FOV)
        if pano_pt is not None:
            px, py = pano_pt
            ax_pano.scatter(px, py, c='red', s=200, marker='*',
                            edgecolors='yellow', linewidths=1.5, zorder=5)
            ax_pano.annotate(f"F_{i}", (px, py), xytext=(6, -8),
                             textcoords='offset points', fontsize=10, fontweight='bold',
                             color='red', bbox=dict(boxstyle='round,pad=0.2',
                                                    facecolor='yellow', alpha=0.85))

    ax_pano.set_title('360° Panorama (equirectangular projection)', fontsize=13)
    ax_pano.set_xticks([])
    ax_pano.set_yticks([])

    plt.tight_layout()

    # Save to folder
    output_path = os.path.join(output_folder, f"combined_wp{obs_wp}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def predict_frontier_gains(frontier_data, checkpoint_path):
    """Predict gain (m²) for each frontier using the trained UNet.

    Returns:
        (gains, connects): gains is list[float] predicted gain in m²,
            connects is list[bool] whether reachable free reaches the top edge.
            Returns (None, None) on error.
    """
    try:
        import torch
        from models.frontier_gain_model import (
            load_atlas, get_current_map_mask,
            build_atlas_grid, build_coverage_grid,
            predict_single_frontier,
            _build_model, _get_device,
        )
        from models.map_utils import load_waypoint_meta, CAMERAS, WP_IMAGES_DIR
    except ImportError as e:
        print(f"    [gain] skip prediction: {e}")
        return None, None

    device = _get_device()
    UNet = _build_model()
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    atlas_pts, wp_positions, g2c = load_atlas()
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)

    wids = frontier_data['waypoint_ids']
    fps = frontier_data['frontier_positions']
    obs_wps = frontier_data.get('frontier_observing_wps', [])

    # Build WP-covered g2c (simulates online atlas)
    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cov_grid = build_coverage_grid(atlas_pts, cmask,
                                         gx_min, gy_min, atlas_cat.shape)
    covered_g2c = {k: v for k, v in g2c.items()
                   if atlas_cov_grid[k[0] - gx_min, k[1] - gy_min]}

    gains = []
    connects = []
    for j, fp in enumerate(fps):
        if j >= len(obs_wps) or obs_wps[j] not in wp_positions:
            gains.append(0.0)
            connects.append(False)
            continue
        ow = obs_wps[j]

        g, c = predict_single_frontier(
            frontier_xy=np.array([fp[0], fp[1]]),
            obs_wp_id=ow,
            grid_to_category=covered_g2c,
            wp_pos_atlas=wp_positions[ow],
            model=model,
            device=device,
        )
        gains.append(g)
        connects.append(c)

    return gains, connects


def predict_frontier_positions(frontier_data, detr_checkpoint):
    """Run DETR model to predict frontier positions for each WP.

    Returns:
        (pred_world, gt_world):
            pred_world  – list of [x, y, conf, wp_id] predicted positions
            gt_world    – list of [x, y] GT frontier positions (from frontier_data)
        Returns (None, None) on error.
    """
    try:
        import torch
        from models.frontier_detr_model import (
            _build_detr_model, _get_device,
            bev_normalized_to_world,
        )
        from models.frontier_gain_model import (
            load_atlas, get_current_map_mask,
            build_atlas_grid, build_coverage_grid, build_sample,
        )
    except ImportError as e:
        print(f"    [detr] skip prediction: {e}")
        return None, None

    device = _get_device()
    FrontierDETR = _build_detr_model()
    model = FrontierDETR().to(device)
    model.load_state_dict(
        torch.load(detr_checkpoint, map_location=device, weights_only=True))
    model.eval()

    atlas_pts, wp_positions, g2c = load_atlas()
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_gmin = (gx_min, gy_min)

    wids = frontier_data['waypoint_ids']
    fps  = np.array(frontier_data['frontier_positions'])
    obs_wps = frontier_data.get('frontier_observing_wps', [])

    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cov = build_coverage_grid(atlas_pts, cmask,
                                    gx_min, gy_min, atlas_cat.shape)

    # Build wp → frontier-index map (for GT collection)
    wp_to_frontiers = {}
    for j, ow in enumerate(obs_wps):
        if j >= len(fps):
            break
        wp_to_frontiers.setdefault(ow, []).append(j)

    conf_thresh = 0.3
    pred_world = []
    gt_world   = []

    for wp_id in wids:
        if wp_id not in wp_positions:
            continue
        wp_pos    = wp_positions[wp_id]
        center_xy = np.array([wp_pos[0], wp_pos[1]])
        theta     = 0.0

        inp, _ = build_sample(center_xy, theta,
                               atlas_cat, atlas_cov, atlas_gmin,
                               obs_wp_id=wp_id)

        inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_xy, pred_conf = model(inp_t)

        pred_xy_np = pred_xy[0].cpu().numpy()
        conf_np    = torch.sigmoid(pred_conf[0]).cpu().numpy()

        mask = conf_np > conf_thresh
        if mask.any():
            preds_bev = pred_xy_np[mask]
            confs     = conf_np[mask]
            world_xy  = bev_normalized_to_world(preds_bev, center_xy, theta)
            for i in range(len(world_xy)):
                pred_world.append([world_xy[i, 0], world_xy[i, 1],
                                   float(confs[i]), wp_id])

        for j in wp_to_frontiers.get(wp_id, []):
            gt_world.append(fps[j, :2].tolist())

    return pred_world, gt_world


def predict_gains_for_pred_frontiers(frontier_data, pred_world, gain_checkpoint):
    """Predict gain (m²) for each DETR-predicted frontier using the UNet.

    Args:
        frontier_data: dict from frontier JSON
        pred_world:    list of [x, y, conf, wp_id] from predict_frontier_positions
        gain_checkpoint: path to UNet checkpoint

    Returns:
        (gains, connects): parallel lists to pred_world.
            gains    – list[float] predicted gain in m²
            connects – list[bool] whether reachable free connects outward
        Returns (None, None) on error.
    """
    try:
        import torch
        from models.frontier_gain_model import (
            load_atlas, get_current_map_mask,
            build_atlas_grid, build_coverage_grid,
            predict_single_frontier,
            _build_model, _get_device,
        )
    except ImportError as e:
        print(f"    [gain] skip prediction: {e}")
        return None, None

    device = _get_device()
    UNet = _build_model()
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(gain_checkpoint, map_location=device, weights_only=True))
    model.eval()

    atlas_pts, wp_positions, g2c = load_atlas()
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)

    wids = frontier_data['waypoint_ids']
    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cov_grid = build_coverage_grid(atlas_pts, cmask,
                                         gx_min, gy_min, atlas_cat.shape)
    covered_g2c = {k: v for k, v in g2c.items()
                   if atlas_cov_grid[k[0] - gx_min, k[1] - gy_min]}

    gains = []
    connects = []
    for entry in pred_world:
        x, y, _conf, wp_id = entry[0], entry[1], entry[2], int(entry[3])
        if wp_id not in wp_positions:
            gains.append(0.0)
            connects.append(False)
            continue
        try:
            g, c = predict_single_frontier(
                frontier_xy=np.array([x, y]),
                obs_wp_id=wp_id,
                grid_to_category=covered_g2c,
                wp_pos_atlas=wp_positions[wp_id],
                model=model,
                device=device,
            )
        except (FileNotFoundError, KeyError):
            g, c = 0.0, False
        gains.append(g)
        connects.append(c)

    return gains, connects


def plot_summary_map_generation(frontier_id, frontier_data, output_folder,
                                pred_world=None, gt_world=None):
    """Summary map for frontier generation: DETR predictions vs GT.

    Atlas background = only cells covered by WPs in this frontier step.
    GT frontiers shown as green squares, predicted as red circles.
    """
    waypoint_ids           = frontier_data["waypoint_ids"]
    frontier_positions     = frontier_data["frontier_positions"]
    frontier_observing_wps = frontier_data.get("frontier_observing_wps", [])
    unique_obs_wps         = list(dict.fromkeys(frontier_observing_wps))
    n_frontiers            = len(frontier_positions)

    print(f"  [Summary] Generating frontier generation map for frontier {frontier_id} "
          f"({n_frontiers} frontiers, {len(unique_obs_wps)} obs WPs) ...")

    fig = plt.figure(figsize=(24, 20))
    ax  = fig.add_subplot(111)

    # ── Atlas scatter layers (same as plot_summary_map) ──────────────
    NEIGHBORHOOD_RADIUS = 5.0
    all_wp_xy = np.array([[_WP_POSITIONS[w][0], _WP_POSITIONS[w][1]]
                           for w in waypoint_ids if w in _WP_POSITIONS])
    if len(all_wp_xy) > 0:
        R = NEIGHBORHOOD_RADIUS
        bbox_min = all_wp_xy.min(axis=0) - R
        bbox_max = all_wp_xy.max(axis=0) + R
        box_mask = ((_ATLAS_POINTS_2D[:, 0] >= bbox_min[0]) &
                    (_ATLAS_POINTS_2D[:, 0] <= bbox_max[0]) &
                    (_ATLAS_POINTS_2D[:, 1] >= bbox_min[1]) &
                    (_ATLAS_POINTS_2D[:, 1] <= bbox_max[1]))
        box_idx  = np.where(box_mask)[0]
        box_pts  = _ATLAS_POINTS_2D[box_idx]
        near_sub = np.zeros(len(box_pts), dtype=bool)
        R2 = R * R
        for wxy in all_wp_xy:
            d2 = (box_pts[:, 0] - wxy[0]) ** 2 + (box_pts[:, 1] - wxy[1]) ** 2
            near_sub |= (d2 <= R2)
        near_mask          = np.zeros(len(_ATLAS_POINTS_2D), dtype=bool)
        near_mask[box_idx[near_sub]] = True
        local_pts = _ATLAS_POINTS_2D[near_mask]
        local_gi  = _ATLAS_GRID_IDX[near_mask]
        cats = np.array([int(_GRID_TO_CAT.get((g[0], g[1]), 0))
                         for g in local_gi], dtype=np.int8)
        cat_colors = {0: '#c8b8e8', 1: '#50c850', 2: '#222222', 3: '#222222'}
        cat_alpha  = {0: 0.45,      1: 0.55,      2: 0.8,       3: 0.8}
        for cat_val in [0, 3, 1, 2]:
            m = cats == cat_val
            if np.any(m):
                ax.scatter(local_pts[m, 0], local_pts[m, 1],
                           c=cat_colors[cat_val], s=64,
                           alpha=cat_alpha[cat_val],
                           zorder=1, marker='s', linewidths=0)

    # Neighborhood WPs + edges
    neighborhood_wps = set()
    for ref_wp in waypoint_ids:
        if ref_wp not in _WP_POSITIONS:
            continue
        ox, oy = _WP_POSITIONS[ref_wp][0], _WP_POSITIONS[ref_wp][1]
        for wp_id, pos in _WP_POSITIONS.items():
            dx, dy = pos[0] - ox, pos[1] - oy
            if dx * dx + dy * dy <= NEIGHBORHOOD_RADIUS ** 2:
                neighborhood_wps.add(wp_id)

    for u_node, v_node in _WP_GRAPH.edges():
        if u_node in neighborhood_wps and v_node in neighborhood_wps:
            if u_node in _WP_POSITIONS and v_node in _WP_POSITIONS:
                ux, uy = _WP_POSITIONS[u_node][0], _WP_POSITIONS[u_node][1]
                vx, vy = _WP_POSITIONS[v_node][0], _WP_POSITIONS[v_node][1]
                ax.plot([ux, vx], [uy, vy],
                        color='#bbbbbb', lw=1.0, alpha=0.6, zorder=2)

    obs_wp_set = set(unique_obs_wps)
    for wp_id in neighborhood_wps:
        if wp_id not in _WP_POSITIONS:
            continue
        wx, wy = _WP_POSITIONS[wp_id][0], _WP_POSITIONS[wp_id][1]
        if wp_id in obs_wp_set:
            ax.scatter(wx, wy, c='#006600', s=60, marker='o', zorder=4,
                       edgecolors='darkgreen', linewidths=1.2)
        else:
            ax.scatter(wx, wy, c='#888888', s=30, marker='o', zorder=3,
                       edgecolors='white', linewidths=0.4)
        ax.annotate(str(wp_id), (wx, wy), fontsize=6, color='#222222',
                    fontweight='bold',
                    xytext=(3, 3), textcoords='offset points')

    # ── GT frontiers ─────────────────────────────────────────────────
    import matplotlib.patheffects as pe
    if gt_world:
        gt_arr    = np.array(gt_world)
        gt_unique = np.unique(np.round(gt_arr, 3), axis=0)
        ax.scatter(gt_unique[:, 0], gt_unique[:, 1],
                   c='lime', s=80, marker='s',
                   edgecolors='darkgreen', linewidths=1.2,
                   zorder=5, label=f'GT frontiers ({len(gt_unique)})')
    else:
        # Fallback: use raw frontier positions from data
        gt_xy = np.array([[fp[0], fp[1]] for fp in frontier_positions])
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1],
                   c='lime', s=80, marker='s',
                   edgecolors='darkgreen', linewidths=1.2,
                   zorder=5, label=f'GT frontiers ({n_frontiers})')

    # ── Predicted frontiers ───────────────────────────────────────────
    if pred_world:
        pred_arr = np.array(pred_world)
        ax.scatter(pred_arr[:, 0], pred_arr[:, 1],
                   c='red', s=120, marker='*',
                   alpha=np.clip(pred_arr[:, 2], 0.25, 1.0).tolist(),
                   edgecolors='darkred', linewidths=0.5,
                   zorder=6, label=f'Predicted ({len(pred_arr)})')

    # ── Legend & axes ─────────────────────────────────────────────────
    ax.scatter([], [], c='#50c850', s=40, marker='s', label='Free')
    ax.scatter([], [], c='#222222', s=40, marker='s', label='Obstacle')
    ax.scatter([], [], c='#c8b8e8', s=40, marker='s', label='Unknown')
    ax.scatter([], [], c='#888888', s=20, marker='o', label='Nearby WPs')
    ax.scatter([], [], c='#006600', s=40, marker='o', label='Observing WPs')

    gt_count  = len(np.unique(np.round(np.array(gt_world), 3), axis=0)) if gt_world else n_frontiers
    pred_count = len(pred_world) if pred_world else 0
    ax.set_title(
        f'Frontier Generation — Frontier {frontier_id}  '
        f'({n_frontiers} frontiers, {len(unique_obs_wps)} obs WPs, '
        f'{len(waypoint_ids)} total WPs)\n'
        f'GT={gt_count}  Pred={pred_count}',
        fontsize=15)
    ax.set_xlabel('X (meters)', fontsize=13)
    ax.set_ylabel('Y (meters)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.04), loc='upper center',
              ncol=7, fontsize=9, markerscale=2)

    output_path = os.path.join(output_folder, 'summary_map_frontier_generation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved summary: {output_path}")


def plot_summary_map_gain_generation(frontier_id, frontier_data, output_folder,
                                     pred_world=None, pred_gains=None,
                                     pred_connects=None):
    """Summary map: DETR-predicted frontiers with UNet gain annotations.

    Same atlas background as plot_summary_map_generation.
    No GT frontiers — only predicted stars sized/colored by gain.
    Saves: summary_map_gain_generation.png
    """
    import matplotlib.patheffects as pe

    waypoint_ids           = frontier_data["waypoint_ids"]
    frontier_positions     = frontier_data["frontier_positions"]
    frontier_observing_wps = frontier_data.get("frontier_observing_wps", [])
    unique_obs_wps         = list(dict.fromkeys(frontier_observing_wps))
    n_frontiers            = len(frontier_positions)
    pred_count             = len(pred_world) if pred_world else 0

    print(f"  [Summary] Generating frontier gain-generation map for frontier {frontier_id} "
          f"({n_frontiers} GT frontiers, {pred_count} predicted) ...")

    fig = plt.figure(figsize=(24, 20))
    ax  = fig.add_subplot(111)

    # ── Atlas scatter (identical to plot_summary_map_generation) ────────
    NEIGHBORHOOD_RADIUS = 5.0
    all_wp_xy = np.array([[_WP_POSITIONS[w][0], _WP_POSITIONS[w][1]]
                           for w in waypoint_ids if w in _WP_POSITIONS])
    if len(all_wp_xy) > 0:
        R = NEIGHBORHOOD_RADIUS
        bbox_min = all_wp_xy.min(axis=0) - R
        bbox_max = all_wp_xy.max(axis=0) + R
        box_mask = ((_ATLAS_POINTS_2D[:, 0] >= bbox_min[0]) &
                    (_ATLAS_POINTS_2D[:, 0] <= bbox_max[0]) &
                    (_ATLAS_POINTS_2D[:, 1] >= bbox_min[1]) &
                    (_ATLAS_POINTS_2D[:, 1] <= bbox_max[1]))
        box_idx  = np.where(box_mask)[0]
        box_pts  = _ATLAS_POINTS_2D[box_idx]
        near_sub = np.zeros(len(box_pts), dtype=bool)
        R2 = R * R
        for wxy in all_wp_xy:
            d2 = (box_pts[:, 0] - wxy[0]) ** 2 + (box_pts[:, 1] - wxy[1]) ** 2
            near_sub |= (d2 <= R2)
        near_mask = np.zeros(len(_ATLAS_POINTS_2D), dtype=bool)
        near_mask[box_idx[near_sub]] = True
        local_pts = _ATLAS_POINTS_2D[near_mask]
        local_gi  = _ATLAS_GRID_IDX[near_mask]
        cats = np.array([int(_GRID_TO_CAT.get((g[0], g[1]), 0))
                         for g in local_gi], dtype=np.int8)
        cat_colors = {0: '#c8b8e8', 1: '#50c850', 2: '#222222', 3: '#222222'}
        cat_alpha  = {0: 0.45,      1: 0.55,      2: 0.8,       3: 0.8}
        for cat_val in [0, 3, 1, 2]:
            m = cats == cat_val
            if np.any(m):
                ax.scatter(local_pts[m, 0], local_pts[m, 1],
                           c=cat_colors[cat_val], s=64,
                           alpha=cat_alpha[cat_val],
                           zorder=1, marker='s', linewidths=0)

    # WP graph edges + dots
    neighborhood_wps = set()
    for ref_wp in waypoint_ids:
        if ref_wp not in _WP_POSITIONS:
            continue
        ox, oy = _WP_POSITIONS[ref_wp][0], _WP_POSITIONS[ref_wp][1]
        for wp_id, pos in _WP_POSITIONS.items():
            dx, dy = pos[0] - ox, pos[1] - oy
            if dx * dx + dy * dy <= NEIGHBORHOOD_RADIUS ** 2:
                neighborhood_wps.add(wp_id)

    for u_node, v_node in _WP_GRAPH.edges():
        if u_node in neighborhood_wps and v_node in neighborhood_wps:
            if u_node in _WP_POSITIONS and v_node in _WP_POSITIONS:
                ux, uy = _WP_POSITIONS[u_node][0], _WP_POSITIONS[u_node][1]
                vx, vy = _WP_POSITIONS[v_node][0], _WP_POSITIONS[v_node][1]
                ax.plot([ux, vx], [uy, vy],
                        color='#bbbbbb', lw=1.0, alpha=0.6, zorder=2)

    # WPs that produced at least one predicted frontier
    active_wp_set = {int(entry[3]) for entry in pred_world} if pred_world else set()
    for wp_id in neighborhood_wps:
        if wp_id not in _WP_POSITIONS:
            continue
        wx, wy = _WP_POSITIONS[wp_id][0], _WP_POSITIONS[wp_id][1]
        if wp_id in active_wp_set:
            ax.scatter(wx, wy, c='#006600', s=60, marker='o', zorder=4,
                       edgecolors='darkgreen', linewidths=1.2)
        else:
            ax.scatter(wx, wy, c='#888888', s=30, marker='o', zorder=3,
                       edgecolors='white', linewidths=0.4)
        ax.annotate(str(wp_id), (wx, wy), fontsize=6, color='#222222',
                    fontweight='bold',
                    xytext=(3, 3), textcoords='offset points')

    # ── Predicted frontiers with gain ────────────────────────────────────
    if pred_world:
        for k, entry in enumerate(pred_world):
            x, y = entry[0], entry[1]
            g = pred_gains[k] if pred_gains is not None else None
            c = pred_connects[k] if pred_connects is not None else False

            # Star size: same formula as plot_summary_map
            star_size = max(30, 30 + 60 * np.sqrt(g)) if g is not None else 200

            ax.scatter(x, y, c='red', s=star_size, marker='*',
                       zorder=6, edgecolors='darkred', linewidths=1.0)

            label_txt = f'{g:.0f}m²' if g is not None else ''
            if c:
                label_txt += '+'
            if label_txt:
                ax.annotate(label_txt, (x, y), xytext=(8, 8),
                            textcoords='offset points', fontsize=9,
                            color='darkred', fontweight='bold',
                            path_effects=[pe.withStroke(linewidth=3,
                                                        foreground='white')])

    # ── Legend & axes ─────────────────────────────────────────────────────
    ax.scatter([], [], c='red', s=120, marker='*',
               edgecolors='darkred', linewidths=0.5,
               label=f'Predicted ({pred_count})')
    ax.scatter([], [], c='#50c850', s=40, marker='s', label='Free')
    ax.scatter([], [], c='#222222', s=40, marker='s', label='Obstacle')
    ax.scatter([], [], c='#c8b8e8', s=40, marker='s', label='Unknown')
    ax.scatter([], [], c='#888888', s=20, marker='o', label='Nearby WPs')
    ax.scatter([], [], c='#006600', s=40, marker='o', label='Active WPs (predicted)')

    ax.set_title(
        f'Frontier Gain Generation — Frontier {frontier_id}  '
        f'({n_frontiers} frontiers, {len(unique_obs_wps)} obs WPs, '
        f'{len(waypoint_ids)} total WPs)\n'
        f'Pred={pred_count}',
        fontsize=15)
    ax.set_xlabel('X (meters)', fontsize=13)
    ax.set_ylabel('Y (meters)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.04), loc='upper center',
              ncol=6, fontsize=9, markerscale=2)

    output_path = os.path.join(output_folder, 'summary_map_gain_generation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved summary: {output_path}")


def plot_summary_map(frontier_id, frontier_data, output_folder,
                     map_only=False, predicted_gains=None,
                     frontier_connects=None):
    """
    Center-map summary, optionally surrounded by RGB frontier projections.

    Args:
        map_only: If True, only draw the center map (no RGB strips).
        predicted_gains: list[float] per-frontier predicted gain (m²), or None.
        frontier_connects: list[bool] whether frontier connects outward, or None.
    """
    waypoint_ids = frontier_data["waypoint_ids"]
    frontier_positions = frontier_data["frontier_positions"]
    frontier_observing_wps = frontier_data.get("frontier_observing_wps", [])
    unique_obs_wps = list(dict.fromkeys(frontier_observing_wps))
    n_frontiers = len(frontier_positions)

    mode_tag = "map-only" if map_only else "full"
    print(f"  [Summary] Generating summary map ({mode_tag}) for frontier {frontier_id} "
          f"({n_frontiers} frontiers, {len(unique_obs_wps)} obs WPs) ...")

    # --- Collect all frontier projections (skip when map_only) ---
    projections = []
    valid_indices = list(range(n_frontiers))  # default: all indices valid
    if not map_only:
        for i, fp in enumerate(frontier_positions):
            if i < len(frontier_observing_wps):
                try:
                    proj = get_frontier_camera_projection(
                        fp, i, frontier_observing_wps[i])
                except Exception:
                    proj = None
            else:
                proj = None
            projections.append(proj)

        valid_indices = [i for i, p in enumerate(projections) if p is not None]
        if len(valid_indices) == 0:
            print("    No valid frontier projections, skipping summary map")
            return

    # --- Load WP positions from atlas (fast, covers all WPs) ---
    wp_pos_cache = {}
    for wp_id in waypoint_ids:
        if wp_id in _WP_POSITIONS:
            pos = _WP_POSITIONS[wp_id]
            wp_pos_cache[wp_id] = (pos[0], pos[1], 0)  # (x, y, heading=0)

    # --- Figure layout ---
    if map_only:
        fig = plt.figure(figsize=(24, 20))
        ax_map = fig.add_subplot(111)
    else:
        # --- Assign frontiers to edges by angle from centroid ---
        f_xy = np.array([[frontier_positions[i][0], frontier_positions[i][1]]
                         for i in valid_indices])
        cx, cy = f_xy.mean(axis=0)

        edges = {'right': [], 'top': [], 'left': [], 'bottom': []}
        for fi in valid_indices:
            fp = frontier_positions[fi]
            ang = np.degrees(np.arctan2(fp[1] - cy, fp[0] - cx)) % 360
            if ang < 45 or ang >= 315:
                edges['right'].append(fi)
            elif 45 <= ang < 135:
                edges['top'].append(fi)
            elif 135 <= ang < 225:
                edges['left'].append(fi)
            else:
                edges['bottom'].append(fi)

        edges['top'].sort(key=lambda i: frontier_positions[i][0])
        edges['bottom'].sort(key=lambda i: frontier_positions[i][0])
        edges['left'].sort(key=lambda i: -frontier_positions[i][1])
        edges['right'].sort(key=lambda i: -frontier_positions[i][1])

        n_t = len(edges['top'])
        n_b = len(edges['bottom'])
        n_l = len(edges['left'])
        n_r = len(edges['right'])

        MAX_H, MAX_V = 8, 8
        IMG = 3.0

        t_rows = int(np.ceil(n_t / MAX_H)) if n_t else 0
        t_cols = min(n_t, MAX_H) if n_t else 1
        b_rows = int(np.ceil(n_b / MAX_H)) if n_b else 0
        b_cols = min(n_b, MAX_H) if n_b else 1
        l_rows = min(n_l, MAX_V) if n_l else 1
        l_cols = int(np.ceil(n_l / MAX_V)) if n_l else 0
        r_rows = min(n_r, MAX_V) if n_r else 1
        r_cols = int(np.ceil(n_r / MAX_V)) if n_r else 0

        top_h = t_rows * IMG if n_t else 0.3
        bot_h = b_rows * IMG if n_b else 0.3
        left_w = l_cols * IMG * 1.3 if n_l else 0.3
        right_w = r_cols * IMG * 1.3 if n_r else 0.3
        map_w = 24
        side_max = max(l_rows if n_l else 0, r_rows if n_r else 0)
        map_h = max(20, side_max * IMG)

        fig = plt.figure(figsize=(left_w + map_w + right_w,
                                   top_h + map_h + bot_h))
        gs = GridSpec(3, 3, figure=fig,
                      width_ratios=[left_w, map_w, right_w],
                      height_ratios=[top_h, map_h, bot_h],
                      hspace=0.06, wspace=0.04)

        ax_map = fig.add_subplot(gs[1, 1])

    # ===== CENTER: top-down map ===== #

    NEIGHBORHOOD_RADIUS = 5.0  # meters, XY distance

    # Layer 0 — Atlas spatial map (free / obstacle / unknown) within neighborhood
    # Build mask: atlas points within NEIGHBORHOOD_RADIUS of any WP in the JSON
    all_wp_xy = np.array([[_WP_POSITIONS[w][0], _WP_POSITIONS[w][1]]
                           for w in waypoint_ids if w in _WP_POSITIONS])
    if len(all_wp_xy) > 0:
        # Bounding-box pre-filter then per-WP distance check
        R = NEIGHBORHOOD_RADIUS
        bbox_min = all_wp_xy.min(axis=0) - R
        bbox_max = all_wp_xy.max(axis=0) + R
        box_mask = ((_ATLAS_POINTS_2D[:, 0] >= bbox_min[0]) &
                    (_ATLAS_POINTS_2D[:, 0] <= bbox_max[0]) &
                    (_ATLAS_POINTS_2D[:, 1] >= bbox_min[1]) &
                    (_ATLAS_POINTS_2D[:, 1] <= bbox_max[1]))
        box_idx = np.where(box_mask)[0]
        box_pts = _ATLAS_POINTS_2D[box_idx]
        # Check distance to nearest WP for each candidate point
        near_sub = np.zeros(len(box_pts), dtype=bool)
        R2 = R * R
        for wxy in all_wp_xy:
            d2 = (box_pts[:, 0] - wxy[0]) ** 2 + (box_pts[:, 1] - wxy[1]) ** 2
            near_sub |= (d2 <= R2)
        near_mask = np.zeros(len(_ATLAS_POINTS_2D), dtype=bool)
        near_mask[box_idx[near_sub]] = True
        local_pts = _ATLAS_POINTS_2D[near_mask]
        local_gi = _ATLAS_GRID_IDX[near_mask]
        # Look up global category for each point
        cats = np.array([int(_GRID_TO_CAT.get((g[0], g[1]), 0))
                         for g in local_gi], dtype=np.int8)
        # cat: 0=unknown, 1=free, 2=obstacle, 3=unknown-ish
        cat_colors = {0: '#c8b8e8', 1: '#50c850', 2: '#222222', 3: '#222222'}
        cat_alpha  = {0: 0.45, 1: 0.55, 2: 0.8, 3: 0.8}
        for cat_val in [0, 3, 1, 2]:  # unknown first, obstacle on top
            m = cats == cat_val
            if np.any(m):
                ax_map.scatter(local_pts[m, 0], local_pts[m, 1],
                               c=cat_colors[cat_val], s=64,
                               alpha=cat_alpha[cat_val],
                               zorder=1, marker='s', linewidths=0)

    # Layer 1 — Local atlas neighborhood WPs (within 5m of each JSON WP)
    neighborhood_wps = set()
    for ref_wp in waypoint_ids:
        if ref_wp not in _WP_POSITIONS:
            continue
        ox, oy = _WP_POSITIONS[ref_wp][0], _WP_POSITIONS[ref_wp][1]
        for wp_id, pos in _WP_POSITIONS.items():
            dx, dy = pos[0] - ox, pos[1] - oy
            if dx * dx + dy * dy <= NEIGHBORHOOD_RADIUS ** 2:
                neighborhood_wps.add(wp_id)

    # Layer 2 — Edges between neighborhood WPs
    for u_node, v_node in _WP_GRAPH.edges():
        if u_node in neighborhood_wps and v_node in neighborhood_wps:
            if u_node in _WP_POSITIONS and v_node in _WP_POSITIONS:
                ux, uy = _WP_POSITIONS[u_node][0], _WP_POSITIONS[u_node][1]
                vx, vy = _WP_POSITIONS[v_node][0], _WP_POSITIONS[v_node][1]
                ax_map.plot([ux, vx], [uy, vy],
                            color='#bbbbbb', lw=1.0, alpha=0.6, zorder=2)

    # Layer 3 — Neighborhood WP dots + labels
    obs_wp_set = set(unique_obs_wps)
    for wp_id in neighborhood_wps:
        if wp_id not in _WP_POSITIONS:
            continue
        wx, wy = _WP_POSITIONS[wp_id][0], _WP_POSITIONS[wp_id][1]
        if wp_id in obs_wp_set:
            ax_map.scatter(wx, wy, c='#006600', s=60, marker='o', zorder=4,
                           edgecolors='darkgreen', linewidths=1.2)
        else:
            ax_map.scatter(wx, wy, c='#888888', s=30, marker='o', zorder=3,
                           edgecolors='white', linewidths=0.4)
        ax_map.annotate(str(wp_id), (wx, wy), fontsize=6, color='#222222',
                        fontweight='bold',
                        xytext=(3, 3), textcoords='offset points')

    # Layer 4 — Frontier stars + dashed lines to observing WP
    #   When predicted_gains is provided, marker size encodes the gain value.
    for i, fp in enumerate(frontier_positions):
        # Marker size: scaled by predicted gain (sqrt for visible contrast)
        #   0m²→30,  5m²→120,  20m²→250,  50m²→420,  100m²→620,  200m²→900
        if predicted_gains is not None:
            g = predicted_gains[i]
            star_size = max(30, 30 + 60 * np.sqrt(g))
        else:
            g = None
            star_size = 400

        ax_map.scatter(fp[0], fp[1], c='red', s=star_size, marker='*',
                       zorder=5, edgecolors='darkred', linewidths=1)

        # Label: F_i  (+gain annotation, + suffix if connects outward)
        label_txt = f'F_{i}'
        if g is not None:
            label_txt += f'  {g:.0f}m²'
            if frontier_connects is not None and frontier_connects[i]:
                label_txt += '+'
        import matplotlib.patheffects as pe
        ax_map.annotate(label_txt, (fp[0], fp[1]), xytext=(8, 8),
                        textcoords='offset points', fontsize=9,
                        color='darkred', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=3,
                                                    foreground='white')])

        if i < len(frontier_observing_wps):
            ow = frontier_observing_wps[i]
            if ow in wp_pos_cache:
                owx, owy, _ = wp_pos_cache[ow]
                ax_map.plot([fp[0], owx], [fp[1], owy],
                            'r--', alpha=0.4, lw=1)

    ax_map.set_xlabel('X (meters)', fontsize=14)
    ax_map.set_ylabel('Y (meters)', fontsize=14)
    ax_map.set_title(f'Summary — Frontier {frontier_id}  '
                     f'({n_frontiers} frontiers, {len(unique_obs_wps)} obs WPs, '
                     f'{len(waypoint_ids)} total WPs)', fontsize=16)
    ax_map.grid(True, alpha=0.3)
    ax_map.set_aspect('equal')

    # Legend outside the map (below)
    ax_map.scatter([], [], c='#50c850', s=40, marker='s', label='Free')
    ax_map.scatter([], [], c='#222222', s=40, marker='s', label='Obstacle')
    ax_map.scatter([], [], c='#c8b8e8', s=40, marker='s', label='Unknown')
    ax_map.plot([], [], color='#bbbbbb', lw=1, label='WP edges')
    ax_map.scatter([], [], c='#888888', s=20, marker='o', label='Nearby WPs')
    ax_map.scatter([], [], c='#006600', s=40, marker='o', label='Observing WPs')
    ax_map.scatter([], [], c='red', s=200, marker='*', label='Frontiers')
    ax_map.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center',
                  ncol=7, fontsize=9, markerscale=2)

    # ===== FOUR-EDGE RGB STRIPS + CONNECTION LINES ===== #
    if not map_only:
        def draw_rgb(ax, fi):
            """Draw full RGB image with frontier star on an axes."""
            p = projections[fi]
            img, u, v = _orient_for_display(p['image'], p['pixel'][0], p['pixel'][1], p['camera'])
            ax.imshow(img)
            ax.scatter(u, v, c='red', s=200, marker='*',
                       edgecolors='yellow', linewidths=2, zorder=5)
            ax.annotate(f"F_{fi}", (u, v), xytext=(10, -10),
                        textcoords='offset points', fontsize=10,
                        fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='yellow', alpha=0.9))
            ax.set_title(f'F_{fi} ({p["camera"]})', fontsize=9,
                         fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        rgb_info = {}  # fi -> (ax, edge_name)

        if n_t:
            gs_t = gs[0, :].subgridspec(t_rows, t_cols,
                                          hspace=0.25, wspace=0.08)
            for j, fi in enumerate(edges['top']):
                ax = fig.add_subplot(gs_t[j // t_cols, j % t_cols])
                draw_rgb(ax, fi)
                rgb_info[fi] = (ax, 'top')

        if n_b:
            gs_b = gs[2, :].subgridspec(b_rows, b_cols,
                                          hspace=0.25, wspace=0.08)
            for j, fi in enumerate(edges['bottom']):
                ax = fig.add_subplot(gs_b[j // b_cols, j % b_cols])
                draw_rgb(ax, fi)
                rgb_info[fi] = (ax, 'bottom')

        if n_l:
            gs_l = gs[1, 0].subgridspec(l_rows, l_cols,
                                          hspace=0.25, wspace=0.08)
            for j, fi in enumerate(edges['left']):
                row, col = j % l_rows, j // l_rows
                ax = fig.add_subplot(gs_l[row, col])
                draw_rgb(ax, fi)
                rgb_info[fi] = (ax, 'left')

        if n_r:
            gs_r = gs[1, 2].subgridspec(r_rows, r_cols,
                                          hspace=0.25, wspace=0.08)
            for j, fi in enumerate(edges['right']):
                row, col = j % r_rows, j // r_rows
                ax = fig.add_subplot(gs_r[row, col])
                draw_rgb(ax, fi)
                rgb_info[fi] = (ax, 'right')

        anchor_pt = {
            'top':    (0.5, 0),
            'bottom': (0.5, 1),
            'left':   (1,   0.5),
            'right':  (0,   0.5),
        }
        for fi, (ax_rgb, edge_name) in rgb_info.items():
            fp = frontier_positions[fi]
            con = ConnectionPatch(
                xyA=(fp[0], fp[1]), coordsA=ax_map.transData,
                xyB=anchor_pt[edge_name], coordsB=ax_rgb.transAxes,
                arrowstyle='->', color='red', lw=0.8, alpha=0.25,
            )
            fig.add_artist(con)

    out_name = "summary_map_frontier_gain.png" if map_only else "summary_map.png"
    output_path = os.path.join(output_folder, out_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved summary: {output_path}")


def process_frontier_file(frontier_id, depth_step=4,
                          mode='full', checkpoint=None, detr_checkpoint=None,
                          out_dir=None):
    """
    Process a single frontier file and save results to its folder.

    Args:
        frontier_id: Frontier file ID
        depth_step: Depth sampling step
        mode: 'full'                        – per-WP combined views + summary_map (with RGB)
              'summary'                     – summary_map only (with RGB thumbnails)
              'summary_map_frontier_gain'   – center map only (no RGB), gain predictions
              'summary_map_frontier_generation' – DETR frontier generation map
        checkpoint: Path to gain model checkpoint (enables gain prediction)
        detr_checkpoint: Path to DETR model checkpoint (enables frontier generation prediction)

    Returns:
        True if successful, False otherwise
    """
    filepath = os.path.join(FRONTIERS_DIR, f"{frontier_id}.json")
    if not os.path.exists(filepath):
        print(f"[{frontier_id}] File not found: {filepath}")
        return False

    try:
        frontier_data = load_frontier_data(frontier_id)
        frontier_positions = frontier_data["frontier_positions"]
        frontier_observing_wps = frontier_data.get("frontier_observing_wps", [])

        if not frontier_observing_wps:
            print(f"[{frontier_id}] No frontier_observing_wps found")
            return False

        output_folder = out_dir if out_dir else os.path.join(BATCH_OUTPUT_DIR, str(frontier_id))
        os.makedirs(output_folder, exist_ok=True)

        wp_to_frontiers = {}
        for i, obs_wp in enumerate(frontier_observing_wps):
            if obs_wp not in wp_to_frontiers:
                wp_to_frontiers[obs_wp] = []
            wp_to_frontiers[obs_wp].append(i)

        print(f"[{frontier_id}] Processing {len(frontier_positions)} frontiers, "
              f"{len(wp_to_frontiers)} waypoint(s)")

        if mode == 'full':
            for obs_wp, frontier_indices in wp_to_frontiers.items():
                plot_single_map_to_folder(
                    frontier_id=frontier_id,
                    obs_wp=obs_wp,
                    frontier_indices=frontier_indices,
                    frontier_positions=frontier_positions,
                    waypoint_ids=frontier_data["waypoint_ids"],
                    output_folder=output_folder,
                    depth_step=depth_step,
                )

        if mode == 'summary_map_frontier_generation':
            # DETR frontier generation map (no gain)
            print(f"  [detr] Predicting frontier positions ...")
            pred_world, gt_world = (None, None)
            if detr_checkpoint:
                pred_world, gt_world = predict_frontier_positions(
                    frontier_data, detr_checkpoint)
            plot_summary_map_generation(frontier_id, frontier_data, output_folder,
                                        pred_world=pred_world, gt_world=gt_world)
        elif mode == 'summary_map_gain_generation':
            # DETR positions + UNet gain on predicted frontiers (no GT)
            print(f"  [detr] Predicting frontier positions ...")
            pred_world, _gt_world = (None, None)
            if detr_checkpoint:
                pred_world, _gt_world = predict_frontier_positions(
                    frontier_data, detr_checkpoint)
            pred_gains, pred_connects = (None, None)
            if pred_world and checkpoint:
                print(f"  [gain] Predicting gains for predicted frontiers ...")
                pred_gains, pred_connects = predict_gains_for_pred_frontiers(
                    frontier_data, pred_world, checkpoint)
            plot_summary_map_gain_generation(frontier_id, frontier_data, output_folder,
                                             pred_world=pred_world,
                                             pred_gains=pred_gains,
                                             pred_connects=pred_connects)
        else:
            # Predict gains if checkpoint provided
            pgains = None
            pconns = None
            if checkpoint:
                print(f"  [gain] Predicting frontier gains ...")
                pgains, pconns = predict_frontier_gains(frontier_data, checkpoint)

            # Generate summary map
            map_only = (mode == 'summary_map_frontier_gain')
            plot_summary_map(frontier_id, frontier_data, output_folder,
                             map_only=map_only, predicted_gains=pgains,
                             frontier_connects=pconns)

        return True

    except Exception as e:
        print(f"[{frontier_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_all_frontier_ids():
    """
    Get all frontier IDs from the frontiers folder

    Returns:
        List of frontier IDs sorted numerically
    """
    frontier_ids = []
    for filename in os.listdir(FRONTIERS_DIR):
        if filename.endswith('.json'):
            try:
                frontier_id = int(filename.replace('.json', ''))
                frontier_ids.append(frontier_id)
            except ValueError:
                pass
    return sorted(frontier_ids)


def batch_process(frontier_ids=None, depth_step=4,
                  mode='full', checkpoint=None, detr_checkpoint=None,
                  out_dir=None):
    """
    Batch process frontier files.

    Args:
        frontier_ids: List of frontier IDs to process. If None, process all files.
        depth_step: Depth sampling step
        mode: 'full'                        – per-WP combined views + summary_map (with RGB)
              'summary'                     – summary_map only (with RGB thumbnails)
              'summary_map_frontier_gain'   – center map only (no RGB), gain predictions
              'summary_map_frontier_generation' – DETR frontier generation map
        checkpoint: Path to gain model checkpoint (enables gain prediction)
        detr_checkpoint: Path to DETR model checkpoint (enables frontier generation prediction)
    """
    if frontier_ids is None:
        frontier_ids = get_all_frontier_ids()

    print(f"Batch processing {len(frontier_ids)} frontier files  (mode={mode})")
    print(f"Output directory: {BATCH_OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

    successful = []
    failed = []

    for i, frontier_id in enumerate(frontier_ids):
        print(f"\n[{i+1}/{len(frontier_ids)}] Processing frontier {frontier_id}...")
        result = process_frontier_file(
            frontier_id, depth_step, mode,
            checkpoint=checkpoint, detr_checkpoint=detr_checkpoint,
            out_dir=out_dir)
        if result:
            successful.append(frontier_id)
        else:
            failed.append(frontier_id)

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed IDs: {failed}")

    print(f"\nResults saved to: {BATCH_OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Batch process frontier files')
    parser.add_argument('--ids', type=int, nargs='+', default=None,
                       help='Specific frontier IDs to process (default: all)')
    parser.add_argument('--mode',
                       choices=['full', 'summary',
                                'summary_map_frontier_gain',
                                'summary_map_frontier_generation',
                                'summary_map_gain_generation'],
                       default='full',
                       help='full: per-WP views + summary_map(RGB), '
                            'summary: summary_map(RGB) only, '
                            'summary_map_frontier_gain: center map only (no RGB, gain predictions), '
                            'summary_map_frontier_generation: DETR frontier generation map (GT vs pred), '
                            'summary_map_gain_generation: DETR pred + UNet gain (no GT)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Gain model checkpoint (enables predicted gain)')
    parser.add_argument('--detr-checkpoint', type=str, default=None,
                       help='DETR model checkpoint (enables frontier generation prediction)')
    parser.add_argument('--depth-step', '-s', type=int, default=4,
                       help='Depth sampling step (default: 4, only for full mode)')
    parser.add_argument('--out-dir', type=str, default=None,
                       help='Override output directory (default: batch_output/<frontier_id>)')
    args = parser.parse_args()

    batch_process(frontier_ids=args.ids, depth_step=args.depth_step,
                  mode=args.mode, checkpoint=args.checkpoint,
                  detr_checkpoint=args.detr_checkpoint,
                  out_dir=args.out_dir)
