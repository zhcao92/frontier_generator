#!/usr/bin/env python3
"""self_supervised_refine.py
============================
Self-supervised DETR fine-tuning loop.

Each round:
  1. Generate pseudo-label supervision signal  (via frontier_refine.py)
  2. Fine-tune DETR on those pseudo-labels

Usage
-----
python3 self_supervised_refine.py \\
    --frontier-id 6549 \\
    --detr-checkpoint /scratch/.../detr_cache/model_best.pt \\
    --gain-checkpoint gain_cache/model_final.pt \\
    --rounds 3 \\
    --out-dir ss_output/6549

Environment variable TIAMAT_DATA_DIR controls data source (default set2).
"""

import os, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

# ── shared imports ──────────────────────────────────────────────────────────
from models.frontier_gain_model import (
    load_atlas, get_current_map_mask,
    build_atlas_grid, build_coverage_grid,
    build_sample, CELL_AREA,
    _build_model as _build_gain_model, _get_device,
)
from models.frontier_detr_model import (
    world_to_bev_normalized,
    _build_detr_model, MAX_QUERIES, LAMBDA_L1, LAMBDA_CONF,
)
from models.map_utils import load_frontier_data

# ── supervision-signal generation (extracted to frontier_refine.py) ─────────
from self_training.frontier_refine import (
    generate_supervision_signal,
    _load_wp_data,
)

# ── fine-tuning hyperparameters ──────────────────────────────────────────────
FEW_EPOCHS  = 10  # Changed from 50 to 10 for quick testing
FINETUNE_LR = 1e-5
N_ROUNDS    = 3


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DETR fine-tuning
# ═══════════════════════════════════════════════════════════════════════════

def _hungarian_loss(pred_xy, pred_conf, gt_norm):
    """Permutation-invariant set prediction loss for one sample."""
    Q      = pred_xy.shape[0]
    T      = gt_norm.shape[0]
    device = pred_xy.device

    bce_loss_fn = nn.BCEWithLogitsLoss()

    if T == 0:
        return bce_loss_fn(pred_conf, torch.zeros(Q, device=device))

    dist      = torch.cdist(pred_xy.float(), gt_norm.float(), p=1)
    conf_cost = -torch.sigmoid(pred_conf).unsqueeze(1).expand(Q, T)
    cost_np   = (LAMBDA_L1 * dist + LAMBDA_CONF * conf_cost).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)

    row_t = torch.tensor(row_ind, device=device)
    col_t = torch.tensor(col_ind, device=device)

    l1_loss  = LAMBDA_L1  * nn.L1Loss()(pred_xy[row_t], gt_norm[col_t])
    conf_lbl = torch.zeros(Q, device=device)
    conf_lbl[row_t] = 1.0
    bce_loss = LAMBDA_CONF * bce_loss_fn(pred_conf, conf_lbl)
    return l1_loss + bce_loss


def finetune_detr_one_round(detr_model, device, wids, ref_frontiers,
                            covered_g2c, wp_positions,
                            epochs=FEW_EPOCHS, lr=FINETUNE_LR,
                            wp_data_cache=None):
    """Fine-tune DETR for `epochs` epochs on pseudo-labels.

    ref_frontiers : [[x, y, conf, wp_id], ...]  Refined frontiers with wp_id
                    already assigned to nearest observing WP.
                    During training, each obs WP only uses frontiers assigned to it.
    """
    atlas_cat, gx_min, gy_min = build_atlas_grid(covered_g2c)
    atlas_cov  = (atlas_cat >= 0)
    atlas_gmin = (gx_min, gy_min)

    # Group frontiers by their assigned wp_id (already nearest observing WP from JSON).
    frontier_to_wp = {}  # {frontier_idx: wp_id}
    all_fp_xy = []
    for i, entry in enumerate(ref_frontiers):
        wp_id = int(entry[3])
        frontier_to_wp[i] = wp_id
        all_fp_xy.append([entry[0], entry[1]])

    all_fp_xy = np.array(all_fp_xy, dtype=np.float64) if all_fp_xy else None

    samples = []
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

        # Only use frontiers assigned to this WP (nearest-distance assignment).
        assigned_indices = [i for i, assigned_wp in frontier_to_wp.items() if assigned_wp == wp_id]
        if assigned_indices:
            assigned_fp_xy = all_fp_xy[assigned_indices]
            gt_norm = world_to_bev_normalized(assigned_fp_xy, center_xy, theta)
            if len(gt_norm) > MAX_QUERIES:
                gt_norm = gt_norm[:MAX_QUERIES]
        else:
            gt_norm = np.zeros((0, 2), dtype=np.float32)

        samples.append((inp.astype(np.float32), gt_norm.astype(np.float32)))

    if not samples:
        print("    [finetune] no valid samples – skipping")
        return detr_model

    optimizer = optim.Adam(detr_model.parameters(), lr=lr)
    detr_model.train()
    for epoch in range(epochs):
        np.random.shuffle(samples)
        total_loss = 0.0
        for inp_np, gt_np in samples:
            inp_t  = torch.tensor(inp_np[None], dtype=torch.float32,
                                  device=device)
            pred_xy_t, pred_conf_t = detr_model(inp_t)
            loss = _hungarian_loss(pred_xy_t[0], pred_conf_t[0],
                                   torch.tensor(gt_np, dtype=torch.float32,
                                                device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"      epoch {epoch+1:02d}/{epochs}  "
              f"loss={total_loss/len(samples):.4f}")
    detr_model.eval()
    return detr_model


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Main loop
# ═══════════════════════════════════════════════════════════════════════════

def self_supervised_loop(frontier_id, detr_checkpoint, gain_checkpoint,
                         n_rounds=N_ROUNDS, out_dir=None,
                         no_finetune=False,
                         add_wps_mode='all',
                         include_absent=True):
    """N rounds of: supervision signal generation → DETR fine-tune.

    Parameters
    ----------
    no_finetune    : bool  skip DETR fine-tuning
    add_wps_mode   : 'all' | 'obs'  WPs to search
    include_absent : bool  include absent-atlas cells in boundary detection
    """
    if out_dir is None:
        out_dir = f"ss_output/{frontier_id}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"[SS] Device: {device}")

    # ── Atlas + frontier data ─────────────────────────────────────────────
    print("[SS] Loading atlas ...")
    atlas_pts, wp_positions, g2c = load_atlas()

    print(f"[SS] Loading frontier {frontier_id} ...")
    frontier_data = load_frontier_data(frontier_id)
    wids          = frontier_data['waypoint_ids']
    gt_frontiers  = frontier_data['frontier_positions']
    print(f"     GT={len(gt_frontiers)}  obs_WPs={len(wids)}")

    # ── covered_g2c ───────────────────────────────────────────────────────
    cmask         = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cat_f, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_cov_grid = build_coverage_grid(
        atlas_pts, cmask, gx_min, gy_min, atlas_cat_f.shape)
    covered_g2c = {k: v for k, v in g2c.items()
                   if atlas_cov_grid[k[0] - gx_min, k[1] - gy_min]}
    print(f"     covered atlas cells: {len(covered_g2c):,}")

    # ── Load models ───────────────────────────────────────────────────────
    print("[SS] Loading DETR model ...")
    FrontierDETR = _build_detr_model()
    detr_model   = FrontierDETR().to(device)
    detr_model.load_state_dict(
        torch.load(detr_checkpoint, map_location=device, weights_only=True))
    detr_model.eval()

    print("[SS] Loading gain model ...")
    UNet       = _build_gain_model()
    gain_model = UNet().to(device)
    gain_model.load_state_dict(
        torch.load(gain_checkpoint, map_location=device, weights_only=True))
    gain_model.eval()

    # Determine ADD WP list
    add_wps = list(wp_positions.keys()) if add_wps_mode == 'all' else wids

    wp_data_cache = {}
    final_ckpt    = detr_checkpoint

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'='*60}")
        print(f"[SS] Round {rnd}/{n_rounds}")
        print(f"{'='*60}")

        # ── 1. Supervision signal ─────────────────────────────────────────
        (ref_frontiers, ref_gains, ref_connects, ref_masks,
         _src, _dropped, _weights, _) = \
            generate_supervision_signal(
                frontier_id, detr_model, gain_model,
                device, g2c, covered_g2c, wp_positions,
                wids, rnd, out_dir,
                wp_data_cache=wp_data_cache,
                add_wps=add_wps,
                include_absent=include_absent)

        # ref_frontiers now has wp_id already set to nearest observing WP.
        # Training will use these assignments directly.
        print(f"  [2] pseudo-label count: {len(ref_frontiers)}")

        # ── 2. Fine-tune DETR ─────────────────────────────────────────────
        if no_finetune:
            print("  [2] Fine-tuning SKIPPED (--no-finetune)")
            final_ckpt = detr_checkpoint
        else:
            print(f"  [2] Fine-tuning DETR ({FEW_EPOCHS} epochs, lr={FINETUNE_LR}) ...")
            detr_model = finetune_detr_one_round(
                detr_model, device, wids, ref_frontiers,
                covered_g2c, wp_positions,
                epochs=FEW_EPOCHS, lr=FINETUNE_LR,
                wp_data_cache=wp_data_cache)

            final_ckpt = os.path.join(out_dir, f"detr_ss_round{rnd}.pt")
            torch.save(detr_model.state_dict(), final_ckpt)
            print(f"  [saved] checkpoint → {final_ckpt}")

    print(f"\n[SS] Done.  Final model: {final_ckpt}")
    return detr_model, final_ckpt


# ═══════════════════════════════════════════════════════════════════════════
# 3.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global FEW_EPOCHS, FINETUNE_LR

    ap = argparse.ArgumentParser(
        description='Self-supervised DETR fine-tuning loop',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--frontier-id',     type=int,   required=True)
    ap.add_argument('--detr-checkpoint', required=True)
    ap.add_argument('--gain-checkpoint', required=True)
    ap.add_argument('--rounds',          type=int,   default=N_ROUNDS)
    ap.add_argument('--out-dir',         type=str,   default=None)
    ap.add_argument('--epochs',          type=int,   default=FEW_EPOCHS)
    ap.add_argument('--lr',              type=float, default=FINETUNE_LR)
    ap.add_argument('--no-finetune',     action='store_true',
                    help='Skip DETR fine-tuning (debug ADD logic)')
    ap.add_argument('--add-wps',         type=str,   default='all',
                    choices=['all', 'obs'],
                    help='"all"=all map WPs; "obs"=observing WPs only')
    ap.add_argument('--no-absent',       action='store_true',
                    help='Restrict boundaries to cat0-only (skip absent cells)')
    args = ap.parse_args()

    FEW_EPOCHS  = args.epochs
    FINETUNE_LR = args.lr

    self_supervised_loop(
        frontier_id    = args.frontier_id,
        detr_checkpoint= args.detr_checkpoint,
        gain_checkpoint= args.gain_checkpoint,
        n_rounds       = args.rounds,
        out_dir        = args.out_dir,
        no_finetune    = args.no_finetune,
        add_wps_mode   = args.add_wps,
        include_absent = not args.no_absent,
    )


if __name__ == '__main__':
    main()
