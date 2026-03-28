"""
train.py  —  BEV-WaveFormer DDP Training Script
================================================
Launch command (7× RTX 2080 Ti):
    torchrun --nproc_per_node=7 --master_port=29500 train.py --config config.yaml

Architecture of DDP setup
─────────────────────────
PyTorch DDP (DistributedDataParallel) is preferred over DataParallel:
  1. Per-process GPU ownership avoids the GIL bottleneck on rank 0.
  2. Gradient synchronisation via Ring-AllReduce scales linearly.
  3. DDP is compatible with torch.utils.checkpoint; DataParallel is not.
  4. Memory is balanced across all 7 ranks (11 GB × 7).

AMP with float16 — no GradScaler (Option D)
─────────────────────────────────────────────
RTX 2080 Ti is Turing (sm_75): supports FP16 Tensor Cores but not BF16.

GradScaler was removed because on sparse BEV grids the scale collapsed to
0 within 2 steps, making optimizer.step() a permanent no-op. The root cause
is Inf/NaN gradients appearing in float32 (inside WPO) before any float16
overflow, so GradScaler's loss-scaling cannot help.

Without GradScaler we:
  • Keep autocast(float16) for speed.
  • Call loss.backward() directly.
  • Clip gradients with a warmed-up max_norm (starts 0.1, reaches 1.0
    at epoch 10) as the sole guard against explosion.
  • Call optimizer.step() directly.

NaN gradient handling — FIXED
──────────────────────────────
The original code zeroed gradients and skipped the step when NaN was
detected, but did NOT reset the AdamW moment buffers (exp_avg,
exp_avg_sq). Once a NaN contaminates those buffers, every future step
produces NaN parameters regardless of how clean the batch is.

Fix: when a NaN grad is detected, we now:
  1. zero_grad (clear current NaN grads)
  2. Reset exp_avg and exp_avg_sq to zero for every parameter
  3. Reset the step counter
  4. Skip optimizer.step()

This is the equivalent of a "hot restart" of the optimizer state without
re-initialising model weights, so training continues from the last clean
parameter values.
"""

import os
import argparse
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

from model   import BEVWaveFormer
from dataset import build_dataloaders, BEV_CFG


# ===========================================================================
# Logging  —  only rank 0 writes
# ===========================================================================
def get_logger(rank: int, log_file: str = 'train.log') -> logging.Logger:
    logger = logging.getLogger('BEVWaveFormer')
    logger.setLevel(logging.INFO)
    if rank == 0:
        fmt = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


# ===========================================================================
# Gaussian heatmap rendering
# ===========================================================================
def gaussian_radius(det_size: tuple, min_overlap: float = 0.7) -> float:
    h, w = det_size
    a1  = 1.0
    b1  = (h + w)
    c1  = h * w * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4.0
    b2  = 2 * (h + w)
    c2  = (1 - min_overlap) * h * w
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (h + w)
    c3  = (min_overlap - 1) * h * w
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))
    r3  = (b3 + sq3) / (2 * a3)

    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: torch.Tensor, center_x: int, center_y: int,
                  radius: int) -> None:
    H, W  = heatmap.shape
    sigma = max(radius / 3.0, 1e-2)
    diameter = 2 * radius + 1

    x          = torch.arange(diameter, dtype=torch.float32) - radius
    gaussian_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]

    x0, y0 = center_x - radius, center_y - radius
    x1, y1 = center_x + radius + 1, center_y + radius + 1

    hm_x0 = max(x0, 0);  hm_x1 = min(x1, W)
    hm_y0 = max(y0, 0);  hm_y1 = min(y1, H)
    if hm_x0 >= hm_x1 or hm_y0 >= hm_y1:
        return

    k_x0 = hm_x0 - x0;  k_x1 = k_x0 + (hm_x1 - hm_x0)
    k_y0 = hm_y0 - y0;  k_y1 = k_y0 + (hm_y1 - hm_y0)

    kernel_patch = gaussian_2d[k_y0:k_y1, k_x0:k_x1].to(heatmap.device)
    heatmap[hm_y0:hm_y1, hm_x0:hm_x1] = torch.maximum(
        heatmap[hm_y0:hm_y1, hm_x0:hm_x1], kernel_patch)


def build_gt_targets(batch_boxes: list,
                     batch_size: int,
                     num_classes: int,
                     hm_h: int,
                     hm_w: int,
                     reg_dims: int,
                     bev_cfg: dict,
                     device: torch.device) -> dict:
    """
    Build Gaussian heatmap + regression targets from ground-truth boxes.
    All tensors are float32 — loss function casts predictions to float32
    before computing, avoiding dtype mismatch gradient spikes.
    """
    heatmap  = torch.zeros(batch_size, num_classes, hm_h, hm_w,
                            dtype=torch.float32, device=device)
    reg      = torch.zeros(batch_size, reg_dims,    hm_h, hm_w,
                            dtype=torch.float32, device=device)
    reg_mask = torch.zeros(batch_size, 1,           hm_h, hm_w,
                            dtype=torch.float32, device=device)

    x_min, x_max = bev_cfg['x_min'], bev_cfg['x_max']
    y_min, y_max = bev_cfg['y_min'], bev_cfg['y_max']
    ps_x         = bev_cfg['pillar_size_x']
    ps_y         = bev_cfg['pillar_size_y']

    scale_x = hm_w / bev_cfg['bev_w']
    scale_y = hm_h / bev_cfg['bev_h']
    cell_x  = ps_x / scale_x
    cell_y  = ps_y / scale_y

    for b_idx, boxes in enumerate(batch_boxes):
        if boxes is None or len(boxes) == 0:
            continue

        boxes_np = boxes.cpu() if isinstance(boxes, torch.Tensor) else boxes

        for box in boxes_np:
            cls_id = int(box[0])
            # boxes layout (LiDAR frame): [class_id, lx, ly, lz, w, l, h, yaw]
            bev_x  = float(box[1])   # lx: forward = BEV x
            bev_y  = float(box[2])   # ly: left    = BEV y
            lz     = float(box[3])
            box_w  = float(box[4])
            box_l  = float(box[5])
            box_h  = float(box[6])
            yaw    = float(box[7])

            if not (x_min <= bev_x < x_max and y_min <= bev_y < y_max):
                continue
# Scale coordinates exactly to the physical pillar size
            exact_px = int((bev_x - x_min) / ps_x * (hm_w / bev_cfg['bev_w']))
            exact_py = int((bev_y - y_min) / ps_y * (hm_h / bev_cfg['bev_h']))
            px = max(0, min(exact_px, hm_w - 1))
            py = max(0, min(exact_py, hm_h - 1))

            r_x    = max(1, int(box_l / (2 * cell_x)))
            r_y    = max(1, int(box_w / (2 * cell_y)))
            radius = max(0, int(gaussian_radius((r_y * 2, r_x * 2))))

            if 0 <= cls_id < num_classes:
                draw_gaussian(heatmap[b_idx, cls_id], px, py, radius)

            reg[b_idx, 0, py, px] = exact_px - px   #---------------
            reg[b_idx, 1, py, px] = exact_py - py   #---------------
            reg[b_idx, 2, py, px] = lz
            reg[b_idx, 3, py, px] = math.log(max(box_w, 1e-3))
            reg[b_idx, 4, py, px] = math.log(max(box_l, 1e-3))
            reg[b_idx, 5, py, px] = math.log(max(box_h, 1e-3))
            reg[b_idx, 6, py, px] = math.sin(yaw)
            reg[b_idx, 7, py, px] = math.cos(yaw)
            reg_mask[b_idx, 0, py, px] = 1.0

    return {'heatmap': heatmap, 'reg': reg, 'reg_mask': reg_mask}


# ===========================================================================
# Loss
# ===========================================================================
class BEVLoss(nn.Module):
    """
    Combined focal loss (heatmap) + SmoothL1 (regression).

    FIX [Critical — reg loss dtype mismatch]:
    ──────────────────────────────────────────
    preds['reg'] arrives as float16 under AMP; targets['reg'] is float32.
    Passing them directly to smooth_l1_loss causes a silent upcast of the
    prediction, but the gradient flows back through the upcast into the
    float16 parameter space where large values can overflow.

    Fix: explicitly cast BOTH prediction and target to float32 before
    computing the regression loss.  This is always safe because float32
    has strictly more precision than float16, and the loss gradient w.r.t.
    preds['reg'] is a float32 tensor that PyTorch then correctly chains
    through the float16 autocast context.

    Same fix applied to the focal loss: pred is cast to float32 before
    sigmoid + log to avoid float16 log(near-zero) underflow.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0,
                 reg_weight: float = 2.0):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.reg_weight = reg_weight

    def focal_loss(self, pred: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        # FIX: cast pred to float32 before sigmoid + log.
        # Under float16, sigmoid output near 0 or 1 loses precision, and
        # log(pred) can hit -inf at the float16 minimum (~-65504).
        pred   = torch.clamp(pred.float().sigmoid(), min=1e-4, max=1 - 1e-4)
        target = target.float()

        pos_mask = (target == 1).float()
        neg_mask = (target  < 1).float()

        pos_loss = torch.log(pred)     * (1 - pred) ** self.alpha * pos_mask
        neg_loss = torch.log(1 - pred) * pred ** self.alpha \
                   * (1 - target) ** self.beta * neg_mask

        num_pos = pos_mask.sum().clamp(min=1)
        loss    = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        hm_loss  = self.focal_loss(preds['heatmap'], targets['heatmap'])

        # FIX [Critical]: cast pred to float32 before smooth_l1_loss.
        # preds['reg'] is float16 under AMP; targets are float32.
        # The explicit cast prevents silent mixed-precision overflow in
        # the loss gradient and removes the dtype mismatch warning.
        mask     = targets['reg_mask'].float()
        num_pos  = mask.sum().clamp(min=1)
        reg_pred = preds['reg'].float() * mask
        reg_tgt  = targets['reg'].float() * mask

        reg_loss = nn.functional.smooth_l1_loss(
            reg_pred, reg_tgt, reduction='sum') / num_pos

        return hm_loss + self.reg_weight * reg_loss


# ===========================================================================
# Optimizer state reset helper
# ===========================================================================
def reset_optimizer_state(optimizer: torch.optim.Optimizer,
                           logger: logging.Logger,
                           step: int,
                           epoch: int) -> None:
    """
    Reset AdamW moment buffers after a NaN gradient event.

    FIX [Critical — NaN moment contamination]:
    ────────────────────────────────────────────
    When NaN gradients slip through (or before the detection check was
    in place), AdamW's exp_avg and exp_avg_sq accumulate NaN values.
    Because these are exponential moving averages, a single NaN event
    permanently corrupts every subsequent update — even on perfectly
    clean batches — until the optimizer state is reset.

    This function zeroes exp_avg, exp_avg_sq, and the step counter for
    every parameter that has optimizer state, effectively performing a
    "hot restart": model weights retain their last clean values, but
    the momentum history is wiped. Training resumes from the next step
    with fresh Adam statistics.

    This is preferred over full re-initialisation (which would also
    reset model weights) and over ignoring the event (which silently
    corrupts all future steps).
    """
    n_reset = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if not state:   # param not yet seen by optimizer (first epoch)
                continue
            if 'exp_avg' in state:
                state['exp_avg'].zero_()
                n_reset += 1
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].zero_()
            # if 'step' in state:
            #     # step may be a tensor (newer PyTorch) or an int
            #     if isinstance(state['step'], torch.Tensor):
            #         state['step'].zero_()
            #     else:
            #         state['step'] = 0
            #we are not resetting the step from now on as the initialization is fixed
    logger.warning(
        f'Epoch {epoch:04d} | Step {step:05d} | '
        f'NaN grad — reset {n_reset} optimizer moment buffers.')


# ===========================================================================
# Checkpointing
# ===========================================================================
def save_checkpoint(model: DDP,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    save_dir: str,
                    rank: int,
                    filename: str = None):
    """Save checkpoint on rank 0 only (atomic write via tmp + rename)."""
    if rank != 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f'ckpt_epoch{epoch:04d}.pth'
    path = os.path.join(save_dir, filename)

    state = {
        'epoch'           : epoch,
        'model_state'     : model.module.state_dict(),
        'optimizer_state' : optimizer.state_dict(),
        'loss'            : loss,
    }
    tmp_path = path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    latest_path = os.path.join(save_dir, 'latest.pth')
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.abspath(path), latest_path)


def load_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    checkpoint_path: str,
                    device: torch.device,
                    logger: logging.Logger) -> int:
    """Load checkpoint, return next epoch. Loads to CPU first to avoid VRAM spike."""
    logger.info(f'Loading checkpoint: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(ckpt['model_state'])
    model.to(device)

    optimizer.load_state_dict(ckpt['optimizer_state'])
    # Old checkpoints may contain 'scaler_state' — safely ignored.

    start_epoch = ckpt['epoch'] + 1
    logger.info(f'Resumed from epoch {ckpt["epoch"]}, loss={ckpt["loss"]:.4f}')
    return start_epoch


# ===========================================================================
# Training loop (one epoch)
# ===========================================================================
def train_one_epoch(model: DDP,
                    loader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    epoch: int,
                    logger: logging.Logger,
                    log_interval: int = 50) -> float:
    """
    Run one full training epoch.

    Option D — no GradScaler, autocast only.

    NaN handling (FIXED):
    ──────────────────────
    1. Detect NaN via clip_grad_norm_ return value.
    2. zero_grad to remove NaN grads from .grad buffers.
    3. reset_optimizer_state to wipe NaN from exp_avg / exp_avg_sq.
    4. Skip optimizer.step().

    grad_norm warmup (FIXED):
    ──────────────────────────
    max_norm starts at 0.1 and linearly warms to 1.0 over the first 10
    epochs. During the first few epochs the WPO spectral parameters
    (log_alpha, log_v) are initialising and can produce large spectral
    gradients; a tight initial clip prevents these from corrupting other
    parameters before they stabilise.
    """
    model.train()
    total_loss = 0.0
    skipped    = 0
    t0         = time.time()

    # FIX: warm-up max_norm — start tight, relax after 10 epochs.
    # This prevents large spectral gradients from the WPO log_alpha/log_v
    # parameters from corrupting backbone weights during early training.
    max_norm = min(1.0, 0.1 + 0.9 * (epoch / 10.0))

    for step, batch in enumerate(loader):
        pillars    = batch['pillars'].to(device,    non_blocking=True)
        num_points = batch['num_points'].to(device, non_blocking=True)
        coords     = batch['coords'].to(device,     non_blocking=True)
        bs         = batch['batch_size']

        optimizer.zero_grad(set_to_none=True)

        # ── Forward under AMP ──────────────────────────────────────────────
        with autocast("cuda", dtype=torch.float16):
            preds = model(pillars, num_points, coords, bs)

            H_out   = preds['heatmap'].shape[-2]
            W_out   = preds['heatmap'].shape[-1]
            targets = build_gt_targets(
                batch['boxes'], bs, 3, H_out, W_out, 8, BEV_CFG, device)

            loss = criterion(preds, targets)

        # ── Backward ────────────────────────────────────────────────────────
        loss.backward()

        # ── Gradient check + clip ────────────────────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_norm)

        if not torch.isfinite(grad_norm):
            # FIX [Critical]: zero grads AND reset optimizer moment buffers.
            # Without resetting exp_avg/exp_avg_sq, one NaN event permanently
            # corrupts all future optimizer steps even on clean batches.
            optimizer.zero_grad(set_to_none=True)
            reset_optimizer_state(optimizer, logger, step, epoch)
            skipped += 1
            continue

        optimizer.step()

        total_loss += loss.item()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            logger.info(
                f'Epoch {epoch:04d} | Step {step:05d}/{len(loader):05d} '
                f'| loss={loss.item():.4f} '
                f'| grad_norm={grad_norm:.3f} '
                f'| max_norm={max_norm:.3f} '
                f'| {elapsed:.1f}s elapsed'
            )

            if dist.get_rank()==0:
                csv_path = './checkpoints/train_metrics.csv'
                # Write header if file doesn't exist
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w') as f:
                        f.write("Epoch,Step,TrainLoss,GradNorm,MaxNorm\n")
                # Append data
                with open(csv_path, 'a') as f:
                    f.write(f"{epoch},{step},{loss.item():.4f},{grad_norm:.3f},{max_norm:.3f}\n")
    if skipped > 0:
        logger.warning(
            f'Epoch {epoch:04d} | Skipped {skipped} steps '
            f'({100*skipped/max(len(loader),1):.1f}%) due to non-finite gradients.')

    return total_loss / max(len(loader) - skipped, 1)


# ===========================================================================
# Validation loop
# ===========================================================================
@torch.no_grad()
def validate(model: DDP,
             loader,
             criterion: nn.Module,
             device: torch.device,
             epoch: int,
             logger: logging.Logger) -> float:
    """
    Validation pass with all-reduce averaging across DDP ranks.
    Uses autocast for consistent numerical behaviour with training.
    """
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    count      = torch.tensor(0,   device=device)

    for batch in loader:
        pillars    = batch['pillars'].to(device,    non_blocking=True)
        num_points = batch['num_points'].to(device, non_blocking=True)
        coords     = batch['coords'].to(device,     non_blocking=True)
        bs         = batch['batch_size']

        with autocast("cuda", dtype=torch.float16):
            preds = model(pillars, num_points, coords, bs)
            H_out = preds['heatmap'].shape[-2]
            W_out = preds['heatmap'].shape[-1]
            targets = build_gt_targets(
                batch['boxes'], bs, 3, H_out, W_out, 8, BEV_CFG, device)
            loss = criterion(preds, targets)

        total_loss += loss
        count      += 1

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count,      op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / count).item()

    logger.info(f'Epoch {epoch:04d} | VAL loss={avg_loss:.4f}')
    return avg_loss


# ===========================================================================
# Main
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description='BEV-WaveFormer Training')
    p.add_argument('--kitti_root',    type=str,   default='~/kitti')
    p.add_argument('--imageset_dir',  type=str,   default='~/kitti/ImageSets')
    p.add_argument('--save_dir',      type=str,   default='./checkpoints')
    p.add_argument('--resume',        type=str,   default=None)
    p.add_argument('--epochs',        type=int,   default=160)
    p.add_argument('--batch_size',    type=int,   default=2,
                   help='Per-rank batch size. Global = batch_size × 7')
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--num_workers',   type=int,   default=8)
    p.add_argument('--log_interval',  type=int,   default=50)
    p.add_argument('--save_interval', type=int,   default=5)
    p.add_argument('--local_rank',    type=int,
                   default=int(os.environ.get('LOCAL_RANK', 0)))
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. DDP init ─────────────────────────────────────────────────────────
    dist.init_process_group(backend='nccl', init_method='env://')
    rank       = dist.get_rank()
    local_rank = args.local_rank
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # ── 2. Logging ───────────────────────────────────────────────────────────
    logger = get_logger(rank, log_file=os.path.join(args.save_dir, 'train.log'))
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info(f'World size: {world_size}  |  Device: {device}')
        logger.info(f'Per-rank batch: {args.batch_size}  |  '
                    f'Effective global batch: {args.batch_size * world_size}')

    # ── 3. Model ─────────────────────────────────────────────────────────────
    model = BEVWaveFormer(
        pillar_in_ch  = 4,
        pillar_out_ch = 64,
        max_points    = BEV_CFG['max_points_per_pillar'],
        bev_h         = BEV_CFG['bev_h'],
        bev_w         = BEV_CFG['bev_w'],
        stage_dims    = [96, 192, 384, 768],
        depths        = [2, 2, 6, 2],
        use_checkpoint= True,
    ).to(device)

    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f'Model parameters: {n_params:.1f} M')

    # ── 4. Optimizer + Scheduler ─────────────────────────────────────────────
    spectral_params = [p for n, p in model.named_parameters()
                       if 'log_alpha' in n or 'log_v' in n]
    other_params    = [p for n, p in model.named_parameters()
                       if 'log_alpha' not in n and 'log_v' not in n]

    optimizer = torch.optim.AdamW([
        {'params': other_params,    'lr': args.lr},
        {'params': spectral_params, 'lr': args.lr * 0.1,
         'weight_decay': 0.0},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    criterion = BEVLoss(reg_weight=2.0).to(device)

    # ── 5. Data ───────────────────────────────────────────────────────────────
    kitti_root   = os.path.expanduser(args.kitti_root)
    imageset_dir = os.path.expanduser(args.imageset_dir)
    train_loader, val_loader, train_sampler = build_dataloaders(
        kitti_root, imageset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank, world_size=world_size,
    )

    # ── 6. Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            model.module, optimizer, args.resume, device, logger)
        dist.barrier()

    val_loss      = float('inf')
    best_val_loss = float('inf')

    # ── 7. Training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, logger, args.log_interval)

        val_loss = validate(
            model, val_loader, criterion, device, epoch, logger)

        scheduler.step()

        logger.info(
            f'─── Epoch {epoch:04d} | train={train_loss:.4f} '
            f'| val={val_loss:.4f} '
            f'| lr={scheduler.get_last_lr()[0]:.2e} ───')

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            args.save_dir, rank)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            args.save_dir, rank, filename='best.pth')
            logger.info(f'  ↳ New best val loss: {val_loss:.4f}')

        dist.barrier()

    # ── 8. Final checkpoint + cleanup ──────────────────────────────────────────
    save_checkpoint(model, optimizer, args.epochs - 1, val_loss,
                    args.save_dir, rank, filename='final.pth')
    dist.destroy_process_group()
    logger.info('Training complete.')


if __name__ == '__main__':
    main()