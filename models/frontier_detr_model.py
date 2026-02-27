#!/usr/bin/env python3
"""DETR-based frontier position prediction model.

Given a waypoint's local 8-channel BEV grid (atlas + depth), predict where
frontiers exist as a set of (x, y) positions using a DETR architecture.

Usage:
    python frontier_detr_model.py prepare --out detr_cache --min-id 50 --max-id 2000
    python frontier_detr_model.py train --data detr_cache --epochs 200 --batch-size 32
    python frontier_detr_model.py predict --frontier-id 1515 --checkpoint detr_cache/model_best.pt
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# Reuse path constants and loader from map_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_utils import load_frontier_data, BASE_DIR, DUMP_DIR, FRONTIERS_DIR

# Reuse grid building and BEV construction from frontier_gain_model
from models.frontier_gain_model import (
    build_sample, build_atlas_grid, build_coverage_grid,
    get_current_map_mask, load_atlas,
    RESOLUTION, GRID_SIZE, HALF_EXTENT,
)

# ── DETR constants ───────────────────────────────────────────────────────
MAX_QUERIES = 10      # max frontiers predicted per WP
LAMBDA_L1 = 5.0
LAMBDA_CONF = 1.0
BEV_EXTENT = GRID_SIZE * RESOLUTION   # 20.0 m


# ── Coordinate conversion ───────────────────────────────────────────────
def world_to_bev_normalized(frontier_xy, center_xy, theta):
    """Convert world XY coordinates to normalized BEV [0,1]² coordinates.

    Uses the same mapping as build_sample():
        rx = HALF_EXTENT - row * RESOLUTION   (row increases downward)
        ry = col * RESOLUTION - HALF_EXTENT   (col increases rightward)

    Args:
        frontier_xy : (K, 2) world XY
        center_xy   : (2,) BEV center in world coords
        theta       : rotation angle (same as passed to build_sample)

    Returns:
        (K, 2) normalized [0,1]² BEV coordinates (row_norm, col_norm)
    """
    dx = frontier_xy[:, 0] - center_xy[0]
    dy = frontier_xy[:, 1] - center_xy[1]
    c, s = np.cos(-theta), np.sin(-theta)
    rx = c * dx - s * dy
    ry = s * dx + c * dy
    row_norm = (HALF_EXTENT - rx) / BEV_EXTENT
    col_norm = (ry + HALF_EXTENT) / BEV_EXTENT
    return np.stack([row_norm, col_norm], axis=1)


def bev_normalized_to_world(bev_norm, center_xy, theta):
    """Convert normalized BEV [0,1]² back to world XY.

    Inverse of world_to_bev_normalized.

    Args:
        bev_norm  : (K, 2) normalized (row_norm, col_norm)
        center_xy : (2,) BEV center in world coords
        theta     : rotation angle (same as used during BEV construction)

    Returns:
        (K, 2) world XY
    """
    rx = HALF_EXTENT - bev_norm[:, 0] * BEV_EXTENT
    ry = bev_norm[:, 1] * BEV_EXTENT - HALF_EXTENT
    # Inverse rotation: rotate by +theta
    ct, st = np.cos(theta), np.sin(theta)
    wx = ct * rx - st * ry + center_xy[0]
    wy = st * rx + ct * ry + center_xy[1]
    return np.stack([wx, wy], axis=1)


# ── Prepare (pre-compute dataset) ──────────────────────────────────────
def cmd_prepare_detr(args):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print("Loading atlas ...")
    atlas_pts, wp_positions, g2c = load_atlas()
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_gmin = (gx_min, gy_min)
    print(f"  {len(atlas_pts):,} atlas points, {len(wp_positions)} waypoints"
          f", atlas grid {atlas_cat.shape}")

    fids = sorted(int(f.replace('.json', ''))
                  for f in os.listdir(FRONTIERS_DIR) if f.endswith('.json'))
    if args.min_id is not None:
        fids = [f for f in fids if f >= args.min_id]
    if args.max_id is not None:
        fids = [f for f in fids if f <= args.max_id]
    print(f"  {len(fids)} frontier files (id range: {fids[0]}–{fids[-1]})")

    meta_list = []
    t0 = time.time()

    for fi, fid in enumerate(fids):
        fdata = load_frontier_data(fid)
        wids = fdata['waypoint_ids']
        fps = np.array(fdata['frontier_positions'])   # (M, 3)
        obs_wps = fdata.get('frontier_observing_wps', [])

        if len(wids) == 0:
            continue

        # Shared coverage grid for all WPs in this file
        cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
        atlas_cov = build_coverage_grid(atlas_pts, cmask,
                                        gx_min, gy_min, atlas_cat.shape)

        # Build index: wp_id → list of frontier indices it observes
        wp_to_frontiers = {}
        for j, ow in enumerate(obs_wps):
            if j >= len(fps):
                break
            wp_to_frontiers.setdefault(ow, []).append(j)

        # Process each WP as one training sample
        for wp_id in wids:
            if wp_id not in wp_positions:
                continue

            wp_pos = wp_positions[wp_id]
            center_xy = np.array([wp_pos[0], wp_pos[1]])

            # Random rotation for data augmentation
            theta = np.random.uniform(0, 2 * np.pi)

            # Build 8-channel BEV centered on this WP
            inp, _ = build_sample(center_xy, theta,
                                  atlas_cat, atlas_cov, atlas_gmin,
                                  obs_wp_id=wp_id)

            # Find frontiers observed by this WP
            frontier_indices = wp_to_frontiers.get(wp_id, [])
            if frontier_indices:
                fp_xy = fps[frontier_indices, :2]   # (K, 2) world XY
                targets_norm = world_to_bev_normalized(fp_xy, center_xy, theta)

                # Keep only targets within [0, 1]²
                valid = ((targets_norm[:, 0] >= 0) & (targets_norm[:, 0] <= 1) &
                         (targets_norm[:, 1] >= 0) & (targets_norm[:, 1] <= 1))
                targets_norm = targets_norm[valid].astype(np.float32)
            else:
                targets_norm = np.empty((0, 2), dtype=np.float32)

            n_targets = len(targets_norm)

            fname = f"{fid}_{wp_id}.npz"
            np.savez_compressed(
                os.path.join(out_dir, fname),
                input=inp.astype(np.float16),
                targets=targets_norm,
                n_targets=np.int32(n_targets))

            meta_list.append({
                'file': fname,
                'frontier_id': fid,
                'wp_id': wp_id,
                'n_targets': n_targets,
            })

        if (fi + 1) % 50 == 0 or fi == len(fids) - 1:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(fids)}] {len(meta_list)} samples  "
                  f"({elapsed:.1f}s)")

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta_list, f)

    n_pos = sum(1 for m in meta_list if m['n_targets'] > 0)
    n_neg = sum(1 for m in meta_list if m['n_targets'] == 0)
    print(f"\nDone: {len(meta_list)} samples saved to {out_dir}/")
    print(f"  Positive (has frontiers): {n_pos}")
    print(f"  Negative (no frontiers): {n_neg}")


# ── PyTorch imports (deferred so prepare works without torch) ───────────
def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def _get_device():
    torch, _, _ = _import_torch()
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS on macOS can crash with transformer models; use CPU for reliability
    return torch.device('cpu')


# ── Dataset ──────────────────────────────────────────────────────────────
class FrontierDETRDataset:
    """Lazy-loading dataset for DETR training.  Train/val split by frontier_id."""

    def __init__(self, cache_dir, split='train', split_id=1500):
        torch, _, _ = _import_torch()
        with open(os.path.join(cache_dir, 'meta.json')) as f:
            all_meta = json.load(f)

        if split == 'train':
            self.meta = [m for m in all_meta if m['frontier_id'] < split_id]
        elif split == 'val':
            self.meta = [m for m in all_meta if m['frontier_id'] >= split_id]
        else:
            self.meta = list(all_meta)

        self.cache_dir = cache_dir
        self.torch = torch

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        m = self.meta[idx]
        data = np.load(os.path.join(self.cache_dir, m['file']))
        inp = data['input'].astype(np.float32)          # (8, 100, 100)
        targets = data['targets'].astype(np.float32)    # (K, 2)
        n_targets = int(data['n_targets'])

        # Pad targets to MAX_QUERIES
        padded = np.zeros((MAX_QUERIES, 2), dtype=np.float32)
        n = min(n_targets, MAX_QUERIES)
        if n > 0:
            padded[:n] = targets[:n]

        return (self.torch.from_numpy(inp),
                self.torch.from_numpy(padded),
                n)


# ── Model ────────────────────────────────────────────────────────────────
def _build_detr_model():
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

    class PositionalEncoding2D(nn.Module):
        """Fixed 2D sinusoidal positional encoding for spatial feature maps."""

        def __init__(self, d_model, h, w):
            super().__init__()
            pe = torch.zeros(d_model, h, w)
            half = d_model // 2
            # Row encoding
            pos_h = torch.arange(h).unsqueeze(1).float()  # (h, 1)
            div = torch.exp(torch.arange(0, half, 2).float()
                            * -(math.log(10000.0) / half))
            pe[0:half:2, :, :] = torch.sin(
                pos_h * div.unsqueeze(0)).T.unsqueeze(2).expand(-1, -1, w)
            pe[1:half:2, :, :] = torch.cos(
                pos_h * div.unsqueeze(0)).T.unsqueeze(2).expand(-1, -1, w)
            # Col encoding
            pos_w = torch.arange(w).unsqueeze(1).float()  # (w, 1)
            pe[half::2, :, :] = torch.sin(
                pos_w * div.unsqueeze(0)).T.unsqueeze(1).expand(-1, h, -1)
            pe[half + 1::2, :, :] = torch.cos(
                pos_w * div.unsqueeze(0)).T.unsqueeze(1).expand(-1, h, -1)
            self.register_buffer('pe', pe)  # (d_model, h, w)

        def forward(self, x):
            """x: (B, d_model, H, W)"""
            return x + self.pe.unsqueeze(0)

    class FrontierDETR(nn.Module):
        def __init__(self, d_model=256, nhead=8, num_encoder_layers=3,
                     num_decoder_layers=3, num_queries=MAX_QUERIES):
            super().__init__()
            self.d_model = d_model
            self.num_queries = num_queries

            # CNN backbone (UNet encoder)
            self.enc1 = DoubleConv(8, 32)
            self.enc2 = DoubleConv(32, 64)
            self.enc3 = DoubleConv(64, 128)
            self.enc4 = DoubleConv(128, d_model)
            self.pool = nn.MaxPool2d(2)

            # 2D positional encoding for 12×12 feature map
            self.pos_enc = PositionalEncoding2D(d_model, 12, 12)

            # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=False)
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_encoder_layers)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=False)
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=num_decoder_layers)

            # Learned object queries
            self.query_embed = nn.Parameter(
                torch.randn(num_queries, d_model) * 0.02)

            # Prediction heads
            self.xy_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 2),
            )
            self.conf_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 1),
            )

        def forward(self, x):
            """
            Args:
                x: (B, 8, 100, 100) BEV input
            Returns:
                pred_xy:   (B, num_queries, 2) normalized [0,1]² coordinates
                pred_conf: (B, num_queries) confidence logits
            """
            B = x.size(0)

            # CNN backbone
            e1 = self.enc1(x)               # (B, 32, 100, 100)
            e2 = self.enc2(self.pool(e1))   # (B, 64, 50, 50)
            e3 = self.enc3(self.pool(e2))   # (B, 128, 25, 25)
            e4 = self.enc4(self.pool(e3))   # (B, 256, 12, 12)

            # Add positional encoding
            e4 = self.pos_enc(e4)           # (B, 256, 12, 12)

            # Flatten spatial dims → sequence: (144, B, 256)
            feat = e4.flatten(2).permute(2, 0, 1)

            # Transformer encoder
            memory = self.transformer_encoder(feat)  # (144, B, 256)

            # Transformer decoder with object queries
            queries = self.query_embed.unsqueeze(1).expand(
                -1, B, -1)                           # (Q, B, 256)
            hs = self.transformer_decoder(queries, memory)  # (Q, B, 256)

            # Prediction heads
            hs = hs.permute(1, 0, 2)                # (B, Q, 256)
            pred_xy = torch.sigmoid(self.xy_head(hs))   # (B, Q, 2)
            pred_conf = self.conf_head(hs).squeeze(-1)   # (B, Q)

            return pred_xy, pred_conf

    return FrontierDETR


# ── Hungarian matching loss ──────────────────────────────────────────────
def hungarian_loss(pred_xy, pred_conf, target_xy, n_targets,
                   lambda_l1=LAMBDA_L1, lambda_conf=LAMBDA_CONF):
    """Compute Hungarian matching loss for a batch.

    Args:
        pred_xy    : (B, Q, 2) predicted normalized coordinates
        pred_conf  : (B, Q) confidence logits
        target_xy  : (B, Q, 2) padded target coordinates
        n_targets  : tensor/list of int, actual target count per sample

    Returns:
        total_loss, loss_dict
    """
    torch, _, F = _import_torch()
    from scipy.optimize import linear_sum_assignment

    B, Q, _ = pred_xy.shape
    device = pred_xy.device

    total_l1 = torch.tensor(0.0, device=device)
    total_conf = torch.tensor(0.0, device=device)
    n_matched = 0

    for b in range(B):
        nt = int(n_targets[b]) if not isinstance(n_targets, int) else n_targets
        nt = min(nt, Q)

        if nt == 0:
            # All queries should predict "no frontier"
            total_conf = total_conf + F.binary_cross_entropy_with_logits(
                pred_conf[b],
                torch.zeros(Q, device=device),
                reduction='sum')
            continue

        gt = target_xy[b, :nt]   # (nt, 2)
        pred = pred_xy[b]         # (Q, 2)

        # Cost matrix
        l1_cost = torch.cdist(pred.unsqueeze(0),
                              gt.unsqueeze(0), p=1).squeeze(0)  # (Q, nt)
        conf_prob = torch.sigmoid(pred_conf[b])                 # (Q,)
        conf_cost = -torch.log(conf_prob + 1e-8).unsqueeze(1).expand(-1, nt)

        cost = lambda_l1 * l1_cost + lambda_conf * conf_cost
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # Matched pairs: L1 loss
        total_l1 = total_l1 + F.l1_loss(
            pred[row_ind], gt[col_ind], reduction='sum')

        # Confidence targets: 1 for matched, 0 for unmatched
        conf_target = torch.zeros(Q, device=device)
        conf_target[row_ind] = 1.0
        total_conf = total_conf + F.binary_cross_entropy_with_logits(
            pred_conf[b], conf_target, reduction='sum')

        n_matched += nt

    # Normalise
    n_total = max(n_matched, 1)
    loss_l1 = lambda_l1 * total_l1 / n_total
    loss_conf = lambda_conf * total_conf / B
    total_loss = loss_l1 + loss_conf

    return total_loss, {
        'loss_l1': loss_l1.item(),
        'loss_conf': loss_conf.item(),
        'loss_total': total_loss.item(),
        'n_matched': n_matched,
    }


# ── Metrics ──────────────────────────────────────────────────────────────
def compute_detr_metrics(pred_xy, pred_conf, target_xy, n_targets,
                         conf_thresh=0.5,
                         dist_thresholds_m=(0.5, 1.0)):
    """Compute AP at different distance thresholds and precision/recall.

    Distances are computed in metres (BEV_EXTENT converts normalised → metres).
    """
    torch, _, _ = _import_torch()
    from scipy.optimize import linear_sum_assignment

    B = pred_xy.shape[0]

    n_gt_total = 0
    n_pred_total = 0
    n_tp = {d: 0 for d in dist_thresholds_m}
    l1_sum_m = 0.0
    l1_count = 0

    for b in range(B):
        nt = min(int(n_targets[b]), pred_xy.shape[1])
        conf = torch.sigmoid(pred_conf[b])
        mask = conf > conf_thresh
        n_pred = int(mask.sum().item())
        n_pred_total += n_pred

        if nt == 0:
            continue
        n_gt_total += nt

        if n_pred == 0:
            continue

        gt = target_xy[b, :nt]       # (nt, 2)
        pred = pred_xy[b][mask]       # (n_pred, 2)

        # Distance matrix in metres
        dist_m = torch.cdist(
            pred.unsqueeze(0), gt.unsqueeze(0), p=2
        ).squeeze(0) * BEV_EXTENT    # (n_pred, nt)

        # For each GT, nearest prediction distance
        min_dist, _ = dist_m.min(dim=0)   # (nt,)
        for d in dist_thresholds_m:
            n_tp[d] += int((min_dist < d).sum().item())

        # Mean L1 via Hungarian matching
        cost_np = dist_m.detach().cpu().numpy()
        if cost_np.shape[0] > 0 and cost_np.shape[1] > 0:
            ri, ci = linear_sum_assignment(cost_np)
            for r, c in zip(ri, ci):
                l1_sum_m += cost_np[r, c]
                l1_count += 1

    metrics = {}
    for d in dist_thresholds_m:
        metrics[f'ap_{d}m'] = n_tp[d] / max(n_gt_total, 1)
    metrics['precision'] = (n_tp[1.0] / max(n_pred_total, 1)
                            if 1.0 in n_tp else 0.0)
    metrics['recall'] = (n_tp[1.0] / max(n_gt_total, 1)
                         if 1.0 in n_tp else 0.0)
    metrics['mean_l1_m'] = l1_sum_m / max(l1_count, 1)
    metrics['n_gt'] = n_gt_total
    metrics['n_pred'] = n_pred_total
    return metrics


# ── Training ─────────────────────────────────────────────────────────────
def cmd_train_detr(args):
    torch, nn, _ = _import_torch()
    from torch.utils.data import DataLoader

    device = _get_device()
    print(f"Device: {device}")

    train_ds = FrontierDETRDataset(args.data, 'train')
    val_ds = FrontierDETRDataset(args.data, 'val')
    print(f"Train: {len(train_ds)},  Val: {len(val_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: Empty train or val set. Check data and split_id.")
        return

    def collate_fn(batch):
        inps, targets, n_tgts = zip(*batch)
        return (torch.stack(inps),
                torch.stack(targets),
                torch.tensor(n_tgts, dtype=torch.long))

    nw = 0 if sys.platform == 'darwin' else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=nw, pin_memory=True,
                            collate_fn=collate_fn)

    FrontierDETR = _build_detr_model()
    model = FrontierDETR().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Separate backbone and transformer params for different lr
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith(('enc1', 'enc2', 'enc3', 'enc4')):
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_ap = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_n = 0
        for inp, targets, n_tgts in train_loader:
            inp = inp.to(device)
            targets = targets.to(device)

            pred_xy, pred_conf = model(inp)
            loss, _ = hungarian_loss(pred_xy, pred_conf, targets, n_tgts)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            train_loss += loss.item() * inp.size(0)
            train_n += inp.size(0)
        train_loss /= max(train_n, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_n = 0
        all_pred_xy, all_pred_conf = [], []
        all_target_xy, all_n_targets = [], []

        with torch.no_grad():
            for inp, targets, n_tgts in val_loader:
                inp = inp.to(device)
                targets = targets.to(device)

                pred_xy, pred_conf = model(inp)
                loss, _ = hungarian_loss(pred_xy, pred_conf, targets, n_tgts)
                val_loss += loss.item() * inp.size(0)
                val_n += inp.size(0)

                all_pred_xy.append(pred_xy.cpu())
                all_pred_conf.append(pred_conf.cpu())
                all_target_xy.append(targets.cpu())
                all_n_targets.append(n_tgts)

        val_loss /= max(val_n, 1)

        # Compute metrics on full val set
        vm = compute_detr_metrics(
            torch.cat(all_pred_xy), torch.cat(all_pred_conf),
            torch.cat(all_target_xy), torch.cat(all_n_targets))

        scheduler.step()
        lr = optimizer.param_groups[1]['lr']   # transformer lr

        ap1 = vm['ap_1.0m']
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"tl={train_loss:.4f}  vl={val_loss:.4f}  "
              f"AP@0.5={vm['ap_0.5m']:.4f}  AP@1.0={ap1:.4f}  "
              f"P={vm['precision']:.4f}  R={vm['recall']:.4f}  "
              f"L1={vm['mean_l1_m']:.2f}m  lr={lr:.2e}")

        if ap1 > best_val_ap:
            best_val_ap = ap1
            torch.save(model.state_dict(),
                       os.path.join(args.data, 'model_best.pt'))
            print(f"  -> saved best (AP@1.0m={best_val_ap:.4f})")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.data, f'model_epoch{epoch}.pt'))

    torch.save(model.state_dict(), os.path.join(args.data, 'model_final.pt'))
    print(f"\nDone. Best val AP@1.0m: {best_val_ap:.4f}")


# ── Predict / Visualise ─────────────────────────────────────────────────
def _bev_vis(inp):
    """Build RGB composite of BEV input channels (free/obstacle/unknown + depth)."""
    vis = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32) * 0.75
    vis[inp[0] > 0] = [0.2, 0.8, 0.2]   # free     → green
    vis[inp[1] > 0] = [0.1, 0.1, 0.1]   # obstacle → dark
    vis[inp[2] > 0] = [0.7, 0.5, 0.9]   # unknown  → purple
    # depth overlay
    pass_ch = inp[5]
    hit_ch  = inp[4]
    z_ch    = inp[6]
    pass_only = (pass_ch > 0) & (hit_ch == 0)
    if pass_only.any():
        a = 0.35
        vis[pass_only] = (1 - a) * vis[pass_only] + a * np.array([0.3, 0.9, 0.3])
    if hit_ch.any():
        h_clipped = np.clip(z_ch, -0.1, 1.5)
        h_norm = (h_clipped - (-0.1)) / (1.5 - (-0.1))
        gray = 0.8 - 0.7 * h_norm
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        m = hit_ch > 0
        alpha = (0.35 + 0.35 * h_norm)[m, np.newaxis]
        vis[m] = (1 - alpha) * vis[m] + alpha * gray_rgb[m]
    return np.clip(vis, 0, 1)


def cmd_predict_detr(args):
    torch, _, _ = _import_torch()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = _get_device()

    FrontierDETR = _build_detr_model()
    model = FrontierDETR().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    atlas_pts, wp_positions, g2c = load_atlas()
    atlas_cat, gx_min, gy_min = build_atlas_grid(g2c)
    atlas_gmin = (gx_min, gy_min)

    fdata = load_frontier_data(args.frontier_id)
    wids = fdata['waypoint_ids']
    fps = np.array(fdata['frontier_positions'])
    obs_wps = fdata.get('frontier_observing_wps', [])

    cmask = get_current_map_mask(wids, wp_positions, atlas_pts)
    atlas_cov = build_coverage_grid(atlas_pts, cmask,
                                    gx_min, gy_min, atlas_cat.shape)

    # Build wp → frontiers index
    wp_to_frontiers = {}
    for j, ow in enumerate(obs_wps):
        if j >= len(fps):
            break
        wp_to_frontiers.setdefault(ow, []).append(j)

    out_dir = os.path.join(BASE_DIR, f"predict_detr_{args.frontier_id}")
    os.makedirs(out_dir, exist_ok=True)

    conf_thresh = 0.3

    # Collect all predictions and GT in world coords (for summary map)
    all_pred_world = []   # [(x, y, conf), ...]
    all_gt_world   = []   # [(x, y), ...]

    print(f"Frontier {args.frontier_id}: {len(wids)} WPs, {len(fps)} frontiers")
    print(f"{'WP':>6}  {'#GT':>4}  {'#Pred':>5}  Predicted BEV positions (row,col,conf)")
    print("-" * 70)

    for wp_idx, wp_id in enumerate(wids):
        if wp_id not in wp_positions:
            continue
        wp_pos    = wp_positions[wp_id]
        center_xy = np.array([wp_pos[0], wp_pos[1]])
        theta     = 0.0   # no rotation at inference

        inp, _ = build_sample(center_xy, theta,
                              atlas_cat, atlas_cov, atlas_gmin,
                              obs_wp_id=wp_id)

        inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_xy, pred_conf = model(inp_t)

        pred_xy_np = pred_xy[0].cpu().numpy()               # (Q, 2)
        conf_np    = torch.sigmoid(pred_conf[0]).cpu().numpy()  # (Q,)

        # Confidence-filtered predictions
        mask      = conf_np > conf_thresh
        preds_bev = pred_xy_np[mask]   # (n, 2) normalised
        confs     = conf_np[mask]

        # GT frontiers for this WP
        gt_indices = wp_to_frontiers.get(wp_id, [])
        gt_fp_xy   = fps[gt_indices, :2] if gt_indices else np.empty((0, 2))
        gt_bev     = (world_to_bev_normalized(gt_fp_xy, center_xy, theta)
                      if len(gt_fp_xy) else np.empty((0, 2)))

        # Convert predictions to world coords
        if len(preds_bev):
            world_xy = bev_normalized_to_world(preds_bev, center_xy, theta)
            for i in range(len(world_xy)):
                all_pred_world.append([world_xy[i, 0], world_xy[i, 1], confs[i]])

        for xy in gt_fp_xy:
            all_gt_world.append(xy.tolist())

        # Console summary
        pred_str = '  '.join(
            f'({r:.2f},{c:.2f},{conf:.2f})'
            for (r, c), conf in zip(preds_bev, confs)
        ) if len(preds_bev) else '—'
        print(f"{wp_id:>6}  {len(gt_fp_xy):>4}  {len(preds_bev):>5}  {pred_str}")

        # ── Per-WP BEV figure (3 panels) ──────────────────────────────
        vis = _bev_vis(inp)
        WC  = GRID_SIZE // 2   # WP centre pixel (row=50, col=50)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 0 – input + depth
        axes[0].imshow(vis)
        axes[0].scatter(WC, WC, c='orange', s=150, marker='*',
                        edgecolors='white', linewidths=1.5, zorder=5)
        n_hit  = int((inp[4] > 0).sum())
        n_pass = int((inp[5] > 0).sum())
        axes[0].set_title(f'Input+Depth  hit={n_hit} pass={n_pass}')

        # Panel 1 – input + GT frontiers
        axes[1].imshow(vis)
        axes[1].scatter(WC, WC, c='orange', s=150, marker='*',
                        edgecolors='white', linewidths=1.5, zorder=5)
        if len(gt_bev):
            # bev_norm: (row_norm, col_norm); imshow x=col, y=row
            gt_valid = ((gt_bev[:, 0] >= 0) & (gt_bev[:, 0] <= 1) &
                        (gt_bev[:, 1] >= 0) & (gt_bev[:, 1] <= 1))
            gv = gt_bev[gt_valid] * GRID_SIZE
            axes[1].scatter(gv[:, 1], gv[:, 0], c='lime', s=120,
                            marker='s', edgecolors='darkgreen',
                            linewidths=1.5, zorder=6)
        axes[1].set_title(f'+ GT frontiers ({len(gt_fp_xy)})')

        # Panel 2 – input + predictions
        axes[2].imshow(vis)
        axes[2].scatter(WC, WC, c='orange', s=150, marker='*',
                        edgecolors='white', linewidths=1.5, zorder=5)
        if len(preds_bev):
            pv = preds_bev * GRID_SIZE
            sizes = 60 + 120 * confs
            axes[2].scatter(pv[:, 1], pv[:, 0], c='red', s=sizes,
                            marker='o', edgecolors='darkred',
                            linewidths=1, alpha=0.85, zorder=6)
            for k in range(len(preds_bev)):
                axes[2].annotate(f'{confs[k]:.2f}',
                                 (pv[k, 1], pv[k, 0]),
                                 textcoords='offset points', xytext=(4, 4),
                                 fontsize=7, color='white')
        axes[2].set_title(f'+ Predictions ({len(preds_bev)}, conf>{conf_thresh})')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(
            f'DETR — Frontier {args.frontier_id} / WP {wp_id}  '
            f'(#{wp_idx+1}/{len(wids)})',
            fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'WP_{wp_id}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()

    # ── Summary frontier generation map ───────────────────────────────
    # Build covered-only atlas background (only cells within WP_RADIUS of wids)
    # atlas_cov[i,j] = True means that grid cell was covered
    # Only show covered cells; uncovered → light background
    atlas_vis = np.ones((*atlas_cat.shape, 3), dtype=np.float32) * 0.94
    covered = atlas_cov.astype(bool)
    cat_colors = {
        0: np.array([0.2, 0.8, 0.2]),   # free     → green
        1: np.array([0.1, 0.1, 0.1]),   # obstacle → dark
        2: np.array([0.7, 0.5, 0.9]),   # unknown  → purple
        3: np.array([0.85, 0.85, 0.85]),# unobserved → light gray
    }
    for cat_id, color in cat_colors.items():
        atlas_vis[(atlas_cat == cat_id) & covered] = color

    # Transpose: atlas_cat[gx, gy] → need rows=gy, cols=gx for correct axes
    atlas_vis_T = atlas_vis.transpose(1, 0, 2)  # (n_gy, n_gx, 3)

    n_gx, n_gy = atlas_cat.shape
    extent = [gx_min * RESOLUTION,
              (gx_min + n_gx) * RESOLUTION,
              gy_min * RESOLUTION,
              (gy_min + n_gy) * RESOLUTION]

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(atlas_vis_T, extent=extent, origin='lower', aspect='equal')

    # WP positions
    wp_xy = np.array([[wp_positions[w][0], wp_positions[w][1]]
                      for w in wids if w in wp_positions])
    if len(wp_xy):
        ax.scatter(wp_xy[:, 0], wp_xy[:, 1], c='steelblue', s=15,
                   alpha=0.6, zorder=3, label=f'Waypoints ({len(wp_xy)})')

    # GT frontiers
    if all_gt_world:
        gt_arr    = np.array(all_gt_world)
        gt_unique = np.unique(np.round(gt_arr, 3), axis=0)
        ax.scatter(gt_unique[:, 0], gt_unique[:, 1],
                   c='lime', s=60, marker='s',
                   edgecolors='darkgreen', linewidths=1,
                   zorder=5, label=f'GT frontiers ({len(gt_unique)})')

    # Predicted frontiers
    if all_pred_world:
        pred_arr = np.array(all_pred_world)
        ax.scatter(pred_arr[:, 0], pred_arr[:, 1],
                   c='red', s=40, marker='o',
                   alpha=np.clip(pred_arr[:, 2], 0.25, 1.0).tolist(),
                   edgecolors='darkred', linewidths=0.5,
                   zorder=6, label=f'Predicted ({len(pred_arr)})')

    gt_u = len(set(tuple(g) for g in all_gt_world))
    ax.set_title(
        f'DETR Frontier Generation — File {args.frontier_id}  '
        f'({len(wids)} WPs, GT={gt_u}, Pred={len(all_pred_world)})',
        fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'summary_frontier_generation_map.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print()
    print(f"GT frontiers (unique): {gt_u}")
    print(f"Predictions total:     {len(all_pred_world)}")
    print(f"Saved to {out_dir}/")


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='DETR frontier position prediction')
    sub = parser.add_subparsers(dest='command')

    # prepare
    p = sub.add_parser('prepare', help='Pre-compute DETR dataset')
    p.add_argument('--out', default='detr_cache',
                   help='Output cache directory')
    p.add_argument('--min-id', type=int, default=50,
                   help='Min frontier file ID (inclusive)')
    p.add_argument('--max-id', type=int, default=2000,
                   help='Max frontier file ID (inclusive)')

    # train
    p = sub.add_parser('train', help='Train DETR model')
    p.add_argument('--data', default='detr_cache')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=50)

    # predict
    p = sub.add_parser('predict', help='Predict and visualise')
    p.add_argument('--frontier-id', type=int, required=True)
    p.add_argument('--checkpoint', required=True)

    args = parser.parse_args()
    if args.command == 'prepare':
        cmd_prepare_detr(args)
    elif args.command == 'train':
        cmd_train_detr(args)
    elif args.command == 'predict':
        cmd_predict_detr(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
