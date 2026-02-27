#!/usr/bin/env python3
"""
Plot top-down map view showing waypoints, frontiers, and depth point cloud projections
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data directory: support environment variable override or use absolute path
DUMP_DIR = os.getenv('TIAMAT_DATA_DIR', '/scratch/mcity_project_root/mcity_project/zhcao/DARPA_TIAMAT/set2')
FRONTIERS_DIR = os.path.join(DUMP_DIR, "frontiers")
WP_IMAGES_DIR = os.path.join(DUMP_DIR, "waypoints")

# Camera list
CAMERAS = ['back', 'frontleft', 'frontright', 'left', 'right']

# Note: Y-flip is now calculated dynamically based on camera extrinsics
# The old static CAMERA_Y_FLIP dict is no longer used

# Scale factor of RGB image relative to intrinsics (RGB is 640x480, intrinsics for 128x96)
IMAGE_SCALE = 5.0
# Intrinsic image resolution
INTR_W = 128
INTR_H = 96


def load_frontier_data(frontier_id):
    """Load frontier data from JSON file"""
    filepath = os.path.join(FRONTIERS_DIR, f"{frontier_id}.json")
    with open(filepath, 'r') as f:
        return json.load(f)


def load_waypoint_meta(wp_id):
    """Load complete waypoint meta information"""
    meta_path = os.path.join(WP_IMAGES_DIR, str(wp_id), "meta.json")
    with open(meta_path, 'r') as f:
        return json.load(f)


def load_waypoint_position(wp_id):
    """Load waypoint position from wp_images folder (for top-down view)"""
    meta = load_waypoint_meta(wp_id)
    pos = meta["agent_position"]
    heading = meta.get("agent_heading", 0)
    # ROS coordinate system: X forward, Y left, Z up
    # Top-down view uses X and Y
    return pos[0], pos[1], heading  # x, y, heading


def depth_to_world_points(depth_path, intrinsic, extrinsic, step=4, cam_name=None):
    """
    Project depth map to world coordinates

    Depth values are Z-depth (perpendicular distance from camera plane)
    """
    depth = np.load(depth_path)

    # Check if transpose is needed (frontleft and frontright depth maps are 320x240 instead of 240x320)
    if depth.shape[0] > depth.shape[1]:
        depth = depth.T
        # Portrait cameras (frontleft, frontright): depth is vertically flipped after transpose
        depth = depth[::-1, :]

    H, W = depth.shape

    # Intrinsics
    K = np.array(intrinsic)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Extrinsic matrix (camera-to-world)
    T_wc = np.array(extrinsic)

    # Generate pixel grid
    u, v = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Get corresponding depth values (Euclidean distance)
    d = depth[v.astype(int), u.astype(int)].astype(np.float32)

    # Filter invalid depths
    valid = (d > 1e-3) & (d < 10.0)
    u = u[valid]
    v = v[valid]
    d = d[valid]

    if len(d) == 0:
        return np.array([]).reshape(0, 3)

    # Z-depth unprojection: depth values are perpendicular (Z-buffer) distance
    # Standard camera coordinate system: X right, Y down, Z forward
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    z = d

    ones = np.ones_like(z)
    points_cam = np.stack([x, y, z, ones], axis=1)  # (N, 4)

    # camera -> world (extrinsic is camera-to-world)
    points_world_h = (T_wc @ points_cam.T).T
    points_world = points_world_h[:, :3]

    return points_world


def project_depth_from_waypoint(wp_id, step=4):
    """
    Project depth maps from all cameras of specified waypoint to world coordinates
    """
    meta = load_waypoint_meta(wp_id)
    folder = os.path.join(WP_IMAGES_DIR, str(wp_id))

    all_points = []
    camera_points = {}

    for cam_name in CAMERAS:
        if cam_name not in meta['cameras']:
            continue

        cam_info = meta['cameras'][cam_name]
        intrinsic = cam_info['camera_intrinsic']
        extrinsic = cam_info['camera_extrinsic']

        depth_path = os.path.join(folder, f"{cam_name}_depth.npy")
        if not os.path.exists(depth_path):
            continue

        points = depth_to_world_points(depth_path, intrinsic, extrinsic, step, cam_name=cam_name)
        camera_points[cam_name] = points
        all_points.append(points)

    if all_points:
        all_points = np.vstack(all_points)
    else:
        all_points = np.array([]).reshape(0, 3)

    return all_points, camera_points


def get_frontier_z_from_depth(frontier_xy, wp_id, search_radius=0.5):
    """
    Get the z coordinate for a frontier from nearby depth points.

    Args:
        frontier_xy: Frontier XY coordinates [x, y]
        wp_id: Waypoint ID
        search_radius: Search radius in meters

    Returns:
        z: Median z value of nearby depth points, or agent_z if no points found
    """
    meta = load_waypoint_meta(wp_id)
    all_points, _ = project_depth_from_waypoint(wp_id, step=2)

    if len(all_points) == 0:
        return meta['agent_position'][2]

    # Find points within search radius
    distances = np.sqrt((all_points[:, 0] - frontier_xy[0])**2 +
                        (all_points[:, 1] - frontier_xy[1])**2)
    nearby_mask = distances < search_radius
    nearby_points = all_points[nearby_mask]

    if len(nearby_points) == 0:
        # Fall back to agent z if no nearby depth points
        return meta['agent_position'][2]

    # Return median z of nearby points
    return np.median(nearby_points[:, 2])


PORTRAIT_CAMERAS = set()  # no portrait cameras in set2

def project_point_to_image(point_world, intrinsic, extrinsic, cam_name=None):
    """
    Project world coordinate point to image coordinates

    Args:
        point_world: World coordinates [x, y, z] or [x, y, z, 1]
        intrinsic: Camera intrinsic matrix 3x3
        extrinsic: Camera extrinsic matrix 4x4 (camera-to-world)
        cam_name: Camera name (used to determine portrait/landscape orientation)

    Returns:
        (u, v): Image coordinates, or None if point is behind camera
    """
    # Ensure homogeneous coordinates
    if len(point_world) == 3:
        point_world = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    else:
        point_world = np.array(point_world)

    K = np.array(intrinsic)
    T_wc = np.array(extrinsic)

    # World to camera
    T_cw = np.linalg.inv(T_wc)
    point_cam = T_cw @ point_world

    # Check if point is in front of camera
    if point_cam[2] <= 0:
        return None

    # Intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Project to image (intrinsics are for 320x240 landscape)
    u_small = fx * point_cam[0] / point_cam[2] + cx
    v_small = fy * point_cam[1] / point_cam[2] + cy

    if cam_name in PORTRAIT_CAMERAS:
        u = (INTR_H - 1 - v_small) * IMAGE_SCALE
        v = u_small * IMAGE_SCALE
    else:
        u = u_small * IMAGE_SCALE
        v = v_small * IMAGE_SCALE

    return u, v


def find_frontier_camera(frontier_pos, meta, frontier_z=None, min_depth=0.5):
    """
    Find a camera that can see the frontier

    Args:
        frontier_pos: Frontier world coordinates [x, y] or [x, y, z]
        meta: Waypoint meta information
        frontier_z: Z coordinate for frontier (if None, uses agent_z as fallback)
        min_depth: Minimum depth (point_cam[2]) threshold to consider camera valid.
                   Points too close to camera cause projection explosion.

    Returns:
        cam_name: Name of camera that can see the frontier, or None if not visible
    """
    if len(frontier_pos) == 2:
        z = frontier_z if frontier_z is not None else meta['agent_position'][2]
        point_world = [frontier_pos[0], frontier_pos[1], z]
    else:
        point_world = frontier_pos

    # Ensure homogeneous coordinates
    if len(point_world) == 3:
        point_world_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    else:
        point_world_h = np.array(point_world)

    # Track best camera (largest depth value for stability)
    best_camera = None
    best_depth = 0

    for cam_name in CAMERAS:
        if cam_name not in meta['cameras']:
            continue

        cam_info = meta['cameras'][cam_name]
        K = np.array(cam_info['camera_intrinsic'])
        T_wc = np.array(cam_info['camera_extrinsic'])

        # World to camera
        T_cw = np.linalg.inv(T_wc)
        point_cam = T_cw @ point_world_h

        # Check if point is sufficiently in front of camera
        if point_cam[2] < min_depth:
            continue

        # Project to image
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u_small = fx * point_cam[0] / point_cam[2] + cx
        v_small = fy * point_cam[1] / point_cam[2] + cy

        if cam_name in PORTRAIT_CAMERAS:
            u = (INTR_H - 1 - v_small) * IMAGE_SCALE
            v = u_small * IMAGE_SCALE
            img_w = int(INTR_H * IMAGE_SCALE)
            img_h = int(INTR_W * IMAGE_SCALE)
        else:
            u = u_small * IMAGE_SCALE
            v = v_small * IMAGE_SCALE
            img_w = int(INTR_W * IMAGE_SCALE)
            img_h = int(INTR_H * IMAGE_SCALE)

        # Check if within image bounds
        if 0 <= u < img_w and 0 <= v < img_h:
            # Prefer camera with larger depth (more stable projection)
            if point_cam[2] > best_depth:
                best_depth = point_cam[2]
                best_camera = cam_name

    return best_camera


def project_frontiers_to_images(frontier_positions, wp_id, output_dir=None,
                                 frontier_indices=None):
    """
    Project all frontiers onto corresponding camera images

    Args:
        frontier_positions: List of frontier positions
        wp_id: Waypoint ID
        output_dir: Output directory, defaults to BASE_DIR
        frontier_indices: List of frontier indices for naming. If None, starts from 0

    Returns:
        results: Dictionary containing projection results for each frontier
    """
    if output_dir is None:
        output_dir = BASE_DIR

    if frontier_indices is None:
        frontier_indices = list(range(len(frontier_positions)))

    meta = load_waypoint_meta(wp_id)
    folder = os.path.join(WP_IMAGES_DIR, str(wp_id))

    results = {}

    for i, fp in enumerate(frontier_positions):
        frontier_name = f"F_{frontier_indices[i]}"

        # Find camera that can see this frontier
        cam_name = find_frontier_camera(fp, meta)

        if cam_name is None:
            print(f"Warning: {frontier_name} is not visible in any camera")
            results[frontier_name] = {'camera': None, 'pixel': None}
            continue

        # Project to image
        cam_info = meta['cameras'][cam_name]
        point_world = [fp[0], fp[1], 0.0] if len(fp) == 2 else fp

        result = project_point_to_image(
            point_world,
            cam_info['camera_intrinsic'],
            cam_info['camera_extrinsic'],
            cam_name
        )

        if result is None:
            results[frontier_name] = {'camera': cam_name, 'pixel': None}
            continue

        u, v = result
        results[frontier_name] = {'camera': cam_name, 'pixel': (u, v)}

        # Load and plot image
        img_path = os.path.join(folder, f"{cam_name}.png")
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        from PIL import Image
        img = Image.open(img_path)
        img_array = np.array(img)

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(img_array)

        # Plot frontier point
        ax.scatter(u, v, c='red', s=300, marker='*',
                  edgecolors='yellow', linewidths=2, zorder=5)
        ax.annotate(frontier_name, (u, v), xytext=(15, -15),
                   textcoords='offset points', fontsize=16, fontweight='bold',
                   color='red', bbox=dict(boxstyle='round,pad=0.3',
                                         facecolor='yellow', alpha=0.9))

        ax.set_title(f'{frontier_name} Projected onto {cam_name}.png (WP_{wp_id})',
                    fontsize=14)
        ax.set_xlabel('u (pixels)')
        ax.set_ylabel('v (pixels)')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{frontier_name}_on_{cam_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"{frontier_name} -> {cam_name}.png at pixel ({u:.1f}, {v:.1f})")
        print(f"  Saved: {output_path}")

    return results


def get_frontier_camera_projection(frontier_pos, frontier_idx, wp_id):
    """
    Get frontier projection onto camera image

    Args:
        frontier_pos: Frontier position [x, y] or [x, y, z]
        frontier_idx: Frontier index for naming
        wp_id: Waypoint ID

    Returns:
        dict with 'camera', 'pixel', 'image' keys, or None if not visible
    """
    from PIL import Image

    meta = load_waypoint_meta(wp_id)
    folder = os.path.join(WP_IMAGES_DIR, str(wp_id))

    # Get z coordinate from nearby depth points
    if len(frontier_pos) == 2:
        frontier_z = get_frontier_z_from_depth(frontier_pos, wp_id)
        point_world = [frontier_pos[0], frontier_pos[1], frontier_z]
    else:
        point_world = frontier_pos
        frontier_z = frontier_pos[2]

    # Find camera that can see this frontier
    cam_name = find_frontier_camera(frontier_pos, meta, frontier_z=frontier_z)
    if cam_name is None:
        return None

    # Project to image
    cam_info = meta['cameras'][cam_name]

    result = project_point_to_image(
        point_world,
        cam_info['camera_intrinsic'],
        cam_info['camera_extrinsic'],
        cam_name
    )

    if result is None:
        return None

    u, v = result

    # Load image
    img_path = os.path.join(folder, f"{cam_name}.png")
    if not os.path.exists(img_path):
        return None

    img = Image.open(img_path)
    img_array = np.array(img)

    return {
        'camera': cam_name,
        'pixel': (u, v),
        'image': img_array,
        'frontier_idx': frontier_idx
    }


def plot_single_map(frontier_id, obs_wp, frontier_indices, frontier_positions,
                    waypoint_ids, depth_step=4):
    """
    Plot a combined figure with top-down map and frontier camera projections

    Args:
        frontier_id: Frontier file ID
        obs_wp: Observing waypoint ID (used for depth projection)
        frontier_indices: List of frontier indices to plot on this map
        frontier_positions: All frontier positions
        waypoint_ids: All waypoint IDs
        depth_step: Depth sampling step
    """
    print(f"\n{'='*60}")
    print(f"Creating combined view for WP_{obs_wp} with frontiers: {['F_'+str(i) for i in frontier_indices]}")
    print(f"{'='*60}")

    # First, get all frontier projections
    print("Projecting frontiers to camera images...")
    frontier_projections = []
    for i in frontier_indices:
        fp = frontier_positions[i]
        proj = get_frontier_camera_projection(fp, i, obs_wp)
        if proj is not None:
            frontier_projections.append(proj)
            print(f"  F_{i} -> {proj['camera']}.png at pixel ({proj['pixel'][0]:.1f}, {proj['pixel'][1]:.1f})")
        else:
            print(f"  Warning: F_{i} is not visible in any camera")

    # Determine layout based on number of camera images
    n_cameras = len(frontier_projections)
    if n_cameras == 0:
        # Only map, no camera images
        fig, ax_map = plt.subplots(figsize=(14, 12))
        axes_cam = []
    elif n_cameras == 1:
        # Map on left, 1 camera image on right
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        ax_map = axes[0]
        axes_cam = [axes[1]]
    elif n_cameras == 2:
        # Map on left (tall), 2 camera images stacked on right
        fig = plt.figure(figsize=(20, 12))
        ax_map = fig.add_subplot(1, 2, 1)
        axes_cam = [fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 4)]
    else:
        # Map on top, camera images in a row below
        fig = plt.figure(figsize=(20, 16))
        ax_map = fig.add_subplot(2, 1, 1)
        axes_cam = []
        for j in range(n_cameras):
            axes_cam.append(fig.add_subplot(2, n_cameras, n_cameras + j + 1))

    # Project depth from observing waypoint
    print(f"Projecting depth from WP_{obs_wp}...")
    all_points, camera_points = project_depth_from_waypoint(
        obs_wp, step=depth_step
    )

    # Use distinct colors for each camera
    cam_colors = {
        'back': '#888888',       # Gray
        'frontleft': '#FF6600',  # Orange
        'frontright': '#00CCCC', # Cyan
        'left': '#CC00CC',       # Magenta
        'right': '#00AA00'       # Green
    }

    for cam_name, points in camera_points.items():
        if len(points) > 0:
            x_world = points[:, 0]
            y_world = points[:, 1]

            color = cam_colors.get(cam_name, 'gray')
            ax_map.scatter(x_world, y_world, c=color, s=2, alpha=0.5,
                          label=cam_name)

            mean_x, mean_y = np.mean(x_world), np.mean(y_world)
            ax_map.annotate(cam_name, (mean_x, mean_y),
                           fontsize=9, fontweight='bold', color=color,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor=color, alpha=0.8))

    print(f"Total projected points: {len(all_points)}")

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
        print(f"Warning: Could not load waypoint {obs_wp}: {e}")

    # Plot frontiers on map
    for i in frontier_indices:
        fp = frontier_positions[i]
        fx, fy = fp[0], fp[1]

        ax_map.scatter(fx, fy, c='red', s=300, marker='*', zorder=4,
                      edgecolors='darkred', linewidths=1)

        label = f'F_{i}'
        ax_map.annotate(label, (fx, fy), xytext=(8, 8),
                       textcoords='offset points', fontsize=9, color='darkred')

        if obs_wp in wp_positions:
            wx, wy, _ = wp_positions[obs_wp]
            ax_map.plot([fx, wx], [fy, wy], 'g--', alpha=0.5, linewidth=1.5)

    # Set map properties
    ax_map.set_xlabel('X (meters) - Forward', fontsize=12)
    ax_map.set_ylabel('Y (meters) - Left', fontsize=12)

    frontier_str = ', '.join([f'F_{i}' for i in frontier_indices])
    title = f'Top-Down Map (WP_{obs_wp}) | Frontiers: {frontier_str}'
    ax_map.set_title(title, fontsize=14)
    ax_map.grid(True, alpha=0.3)
    ax_map.set_aspect('equal')
    ax_map.legend(loc='upper right', markerscale=2)

    # Plot camera images with frontier projections
    for j, (ax_cam, proj) in enumerate(zip(axes_cam, frontier_projections)):
        ax_cam.imshow(proj['image'])

        u, v = proj['pixel']
        ax_cam.scatter(u, v, c='red', s=200, marker='*',
                      edgecolors='yellow', linewidths=2, zorder=5)
        ax_cam.annotate(f"F_{proj['frontier_idx']}", (u, v), xytext=(10, -10),
                       textcoords='offset points', fontsize=12, fontweight='bold',
                       color='red', bbox=dict(boxstyle='round,pad=0.3',
                                             facecolor='yellow', alpha=0.9))

        ax_cam.set_title(f"F_{proj['frontier_idx']} on {proj['camera']}.png", fontsize=12)
        ax_cam.set_xlabel('u (pixels)')
        ax_cam.set_ylabel('v (pixels)')

    plt.tight_layout()

    # Save combined figure
    output_path = os.path.join(BASE_DIR, f"combined_view_{frontier_id}_wp{obs_wp}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Combined view saved to: {output_path}")

    plt.show()


def plot_map(frontier_id=92, depth_step=4):
    """
    Plot combined views (map + camera projections) grouped by observing waypoints

    For each unique observing waypoint, creates a combined figure with:
    - Top-down map with depth projection from that waypoint
    - Camera images with frontiers projected onto them

    Coordinate system: X is forward/backward, Y is left/right, Z is height
    Top-down view displays X-Y plane

    Args:
        frontier_id: Frontier file ID
        depth_step: Depth sampling step
    """
    # Load frontier data
    frontier_data = load_frontier_data(frontier_id)

    waypoint_ids = frontier_data["waypoint_ids"]
    frontier_positions = frontier_data["frontier_positions"]
    frontier_observing_wps = frontier_data.get("frontier_observing_wps", [])

    if not frontier_observing_wps:
        print("Error: No frontier_observing_wps found in data")
        return

    # Group frontiers by observing waypoint
    wp_to_frontiers = {}
    for i, obs_wp in enumerate(frontier_observing_wps):
        if obs_wp not in wp_to_frontiers:
            wp_to_frontiers[obs_wp] = []
        wp_to_frontiers[obs_wp].append(i)

    print(f"Frontier file: {frontier_id}.json")
    print(f"Total frontiers: {len(frontier_positions)}")
    print(f"Unique observing waypoints: {list(wp_to_frontiers.keys())}")
    for wp, indices in wp_to_frontiers.items():
        print(f"  WP_{wp}: {['F_'+str(i) for i in indices]}")

    # Create a combined view for each observing waypoint
    for obs_wp, frontier_indices in wp_to_frontiers.items():
        plot_single_map(
            frontier_id=frontier_id,
            obs_wp=obs_wp,
            frontier_indices=frontier_indices,
            frontier_positions=frontier_positions,
            waypoint_ids=waypoint_ids,
            depth_step=depth_step
        )


def build_equirectangular_panorama(meta, wp_folder,
                                    pano_width=2400, pano_height=480,
                                    fov_v_deg=(-45, 45)):
    """
    Build a true equirectangular panorama by projecting all camera images
    onto a common cylindrical surface.  Overlapping regions are blended
    using edge-distance weights for seamless transitions.

    Convention (robot frame: +X forward, +Y left, +Z up):
        - Panorama x=0 → behind-left (heading + pi)
        - Panorama x=W/2 → forward (heading)
        - Panorama x=W → behind-right (heading - pi)
        - Panorama y=0 → top (max elevation)
        - Panorama y=H → bottom (min elevation)

    Args:
        meta: waypoint meta dict (cameras, agent_heading)
        wp_folder: path to visit folder containing camera PNGs
        pano_width: output panorama width in pixels
        pano_height: output panorama height in pixels
        fov_v_deg: (min_elevation, max_elevation) in degrees

    Returns:
        panorama: np.ndarray (pano_height, pano_width, 3) uint8
    """
    from PIL import Image

    heading = meta.get('agent_heading', 0)

    # Build (theta, phi) grids for every output pixel
    x = np.arange(pano_width)
    y = np.arange(pano_height)
    xg, yg = np.meshgrid(x, y)

    # Azimuth: heading+pi  →  heading-pi  (left-to-right)
    theta = heading + np.pi - 2.0 * np.pi * xg / pano_width

    # Elevation: top=max_elev  →  bottom=min_elev
    phi_min = np.radians(fov_v_deg[0])
    phi_max = np.radians(fov_v_deg[1])
    phi = phi_max - (phi_max - phi_min) * yg / pano_height

    # World-frame ray directions
    cos_phi = np.cos(phi)
    rays_world = np.stack([
        cos_phi * np.cos(theta),
        cos_phi * np.sin(theta),
        np.sin(phi),
    ], axis=-1)  # (H, W, 3)

    # Accumulators for weighted blending
    color_accum = np.zeros((pano_height, pano_width, 3), dtype=np.float64)
    weight_accum = np.zeros((pano_height, pano_width), dtype=np.float64)

    FEATHER = 40  # edge-feathering margin in intrinsic pixels

    for cam_name in CAMERAS:
        if cam_name not in meta['cameras']:
            continue

        cam_info = meta['cameras'][cam_name]
        K = np.array(cam_info['camera_intrinsic'])
        T_wc = np.array(cam_info['camera_extrinsic'])
        R_cw = np.linalg.inv(T_wc[:3, :3])

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Rotate world rays into camera frame  (H, W, 3)
        rays_cam = np.einsum('ij,hwj->hwi', R_cw, rays_world)

        # Project onto intrinsic plane (320x240 space)
        z_cam = rays_cam[..., 2]
        front = z_cam > 0.01
        with np.errstate(divide='ignore', invalid='ignore'):
            u_s = fx * rays_cam[..., 0] / z_cam + cx
            v_s = fy * rays_cam[..., 1] / z_cam + cy

        # Valid if in front of camera and within intrinsic image bounds
        u_max = INTR_W - 1
        v_max = INTR_H - 1
        valid = front & (u_s >= 0) & (u_s <= u_max) & (v_s >= 0) & (v_s <= v_max)

        # Edge-distance weight for smooth blending in overlap zones
        d_edge = np.minimum(
            np.minimum(u_s, u_max - u_s),
            np.minimum(v_s, v_max - v_s),
        )
        w = np.where(valid, np.clip(d_edge / FEATHER, 0.01, 1.0), 0.0)

        # Map intrinsic coords → RGB pixel coords (camera-type dependent)
        if cam_name in PORTRAIT_CAMERAS:
            u_rgb = (v_max - v_s) * IMAGE_SCALE
            v_rgb = u_s * IMAGE_SCALE
            img_w = int(INTR_H * IMAGE_SCALE)
            img_h = int(INTR_W * IMAGE_SCALE)
        else:
            u_rgb = u_s * IMAGE_SCALE
            v_rgb = v_s * IMAGE_SCALE
            img_w = int(INTR_W * IMAGE_SCALE)
            img_h = int(INTR_H * IMAGE_SCALE)

        # Load camera RGB
        img_path = os.path.join(wp_folder, f"{cam_name}.png")
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path))

        # Nearest-neighbour sampling (clip guarantees safe indexing)
        u_int = np.clip(np.round(u_rgb).astype(int), 0, img_w - 1)
        v_int = np.clip(np.round(v_rgb).astype(int), 0, img_h - 1)
        sampled = img[v_int, u_int]  # (H, W, 3)

        # Accumulate weighted colour
        color_accum += sampled.astype(np.float64) * w[..., np.newaxis]
        weight_accum += w

    # Normalise
    has_data = weight_accum > 0
    panorama = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
    panorama[has_data] = np.clip(
        color_accum[has_data] / weight_accum[has_data, np.newaxis],
        0, 255,
    ).astype(np.uint8)

    return panorama


def project_pixel_to_panorama(cam_name, u_rgb, v_rgb, meta,
                               pano_width=2400, pano_height=480,
                               fov_v_deg=(-45, 45)):
    """
    Map a camera RGB pixel to equirectangular panorama coordinates.

    Reverses the exact projection pipeline used by
    build_equirectangular_panorama, so the result is pixel-perfect consistent.

    Steps:
        RGB pixel → intrinsic coords → camera-frame ray → world ray
        → (θ, φ) → panorama pixel

    Args:
        cam_name: camera name
        u_rgb, v_rgb: pixel coordinates in the camera's RGB image
        meta: waypoint meta dict
        pano_width, pano_height, fov_v_deg: must match build_equirectangular_panorama

    Returns:
        (px, py) panorama pixel coordinates, or None if outside vertical FOV
    """
    cam_info = meta['cameras'][cam_name]
    K = np.array(cam_info['camera_intrinsic'])
    T_wc = np.array(cam_info['camera_extrinsic'])
    heading = meta.get('agent_heading', 0)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Step 1: RGB pixel → intrinsic coords (reverse of camera-type mapping)
    if cam_name in PORTRAIT_CAMERAS:
        u_s = v_rgb / IMAGE_SCALE
        v_s = (INTR_H - 1) - u_rgb / IMAGE_SCALE
    else:
        u_s = u_rgb / IMAGE_SCALE
        v_s = v_rgb / IMAGE_SCALE

    # Step 2: intrinsic → camera-frame ray direction
    ray_cam = np.array([
        (u_s - cx) / fx,
        (v_s - cy) / fy,
        1.0,
    ])

    # Step 3: camera → world ray direction
    R_wc = T_wc[:3, :3]
    ray_world = R_wc @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    # Step 4: world ray → (θ, φ)
    theta = np.arctan2(ray_world[1], ray_world[0])
    phi = np.arctan2(ray_world[2], np.sqrt(ray_world[0]**2 + ray_world[1]**2))

    # Step 5: (θ, φ) → panorama pixel
    px = (pano_width * (heading + np.pi - theta) / (2.0 * np.pi)) % pano_width

    phi_min = np.radians(fov_v_deg[0])
    phi_max = np.radians(fov_v_deg[1])
    py = pano_height * (phi_max - phi) / (phi_max - phi_min)

    if py < 0 or py >= pano_height:
        return None
    return px, py


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot combined map and frontier projections')
    parser.add_argument('--frontier', '-f', type=int, default=92,
                       help='Frontier ID to visualize (default: 92)')
    parser.add_argument('--depth-step', '-s', type=int, default=4,
                       help='Depth sampling step (default: 4)')
    args = parser.parse_args()

    plot_map(args.frontier, depth_step=args.depth_step)
