"""
validate_checkpoint.py  —  Multi-GPU DDP validation on a saved checkpoint
==========================================================================
Uses all available GPUs via torchrun, same memory footprint as training.

Launch:
    torchrun --nproc_per_node=7 --master_port=29501 validate_checkpoint.py \
        --checkpoint ./checkpoints/best.pth \
        --kitti_root ~/kitti \
        --imageset_dir ~/kitti/ImageSets \
        --batch_size 2

Use a different --master_port than training (29500) to avoid conflicts.
Per-rank batch_size=2 is identical to training — guaranteed no OOM.
Increase to 4 if VRAM allows (no gradients = ~half the memory of training).

Metrics reported (rank 0 only, aggregated across all ranks via all_reduce):
    - Val focal loss + reg loss + total loss  (same formula as BEVLoss)
    - Per-class: GT object count, avg heatmap response at GT centres,
      Recall@0.3 and Recall@0.5  (detection proxy without NMS)
"""

import os
import math
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from torch.utils.data import DataLoader

from model   import BEVWaveFormer
from dataset import KITTIPillarDataset, kitti_collate_fn, BEV_CFG

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']


# ---------------------------------------------------------------------------
# Logging — rank 0 only
# ---------------------------------------------------------------------------
def get_logger(rank: int) -> logging.Logger:
    logger = logging.getLogger('val')
    logger.setLevel(logging.INFO)
    if rank == 0:
        fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        ch  = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


# ---------------------------------------------------------------------------
# Loss helpers (self-contained, no train.py import)
# ---------------------------------------------------------------------------
def focal_loss_val(pred: torch.Tensor, target: torch.Tensor,
                   alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pred     = torch.clamp(pred.float().sigmoid(), min=1e-4, max=1 - 1e-4)
    target   = target.float()
    pos_mask = (target == 1).float()
    neg_mask = (target  < 1).float()
    pos_loss = torch.log(pred)     * (1 - pred) ** alpha * pos_mask
    neg_loss = torch.log(1 - pred) * pred ** alpha \
               * (1 - target) ** beta * neg_mask
    num_pos  = pos_mask.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


# ---------------------------------------------------------------------------
# GT target builder (self-contained)
# ---------------------------------------------------------------------------
def _gaussian_radius(h, w, min_overlap=0.7):
    a1=1.0; b1=h+w; c1=h*w*(1-min_overlap)/(1+min_overlap)
    sq1=math.sqrt(max(b1**2-4*a1*c1,0)); r1=(b1-sq1)/(2*a1)
    a2=4.0; b2=2*(h+w); c2=(1-min_overlap)*h*w
    sq2=math.sqrt(max(b2**2-4*a2*c2,0)); r2=(b2-sq2)/(2*a2)
    a3=4*min_overlap; b3=-2*min_overlap*(h+w); c3=(min_overlap-1)*h*w
    sq3=math.sqrt(max(b3**2-4*a3*c3,0)); r3=(b3+sq3)/(2*a3)
    return max(0, int(min(r1,r2,r3)))

def _draw_gaussian(heatmap, cx, cy, radius):
    H,W=heatmap.shape; sigma=max(radius/3.0,1e-2); d=2*radius+1
    x=torch.arange(d,dtype=torch.float32)-radius
    g2d=(torch.exp(-x**2/(2*sigma**2))[:,None])*(torch.exp(-x**2/(2*sigma**2))[None,:])
    x0,y0=cx-radius,cy-radius; x1,y1=cx+radius+1,cy+radius+1
    hx0,hx1=max(x0,0),min(x1,W); hy0,hy1=max(y0,0),min(y1,H)
    if hx0>=hx1 or hy0>=hy1: return
    patch=g2d[hy0-y0:hy0-y0+(hy1-hy0), hx0-x0:hx0-x0+(hx1-hx0)].to(heatmap.device)
    heatmap[hy0:hy1,hx0:hx1]=torch.maximum(heatmap[hy0:hy1,hx0:hx1],patch)

def build_targets(batch_boxes, batch_size, num_classes, hm_h, hm_w, bev_cfg, device):
    heatmap  = torch.zeros(batch_size, num_classes, hm_h, hm_w, device=device)
    reg      = torch.zeros(batch_size, 8,           hm_h, hm_w, device=device)
    reg_mask = torch.zeros(batch_size, 1,           hm_h, hm_w, device=device)
    x_min,x_max=bev_cfg['x_min'],bev_cfg['x_max']
    y_min,y_max=bev_cfg['y_min'],bev_cfg['y_max']
    ps_x,ps_y=bev_cfg['pillar_size_x'],bev_cfg['pillar_size_y']
    cell_x=(ps_x/(hm_w/bev_cfg['bev_w'])); cell_y=(ps_y/(hm_h/bev_cfg['bev_h']))
    for b,boxes in enumerate(batch_boxes):
        if boxes is None or len(boxes)==0: continue
        for box in (boxes.cpu() if isinstance(boxes,torch.Tensor) else boxes):
            cls_id=int(box[0]); bev_x,bev_y=float(box[1]),float(box[2])
            lz=float(box[3]); bw,bl,bh,yaw=float(box[4]),float(box[5]),float(box[6]),float(box[7])
            if not (x_min<=bev_x<x_max and y_min<=bev_y<y_max): continue

            exact_px = (bev_x - x_min) / ps_x * (hm_w / bev_cfg['bev_w'])
            exact_py = (bev_y - y_min) / ps_y * (hm_h / bev_cfg['bev_h'])

            px = max(0, min(int(exact_px), hm_w - 1))
            py = max(0, min(int(exact_py), hm_h - 1))

            r=_gaussian_radius(max(1,int(bw/(2*cell_y)))*2,max(1,int(bl/(2*cell_x)))*2)
            if 0<=cls_id<num_classes: _draw_gaussian(heatmap[b,cls_id],px,py,r)
            reg[b,0,py,px]=exact_px - px; reg[b,1,py,px]=exact_py - py; reg[b,2,py,px]=lz
            reg[b,3,py,px]=math.log(max(bw,1e-3)); reg[b,4,py,px]=math.log(max(bl,1e-3))
            reg[b,5,py,px]=math.log(max(bh,1e-3)); reg[b,6,py,px]=math.sin(yaw)
            reg[b,7,py,px]=math.cos(yaw); reg_mask[b,0,py,px]=1.0
    return heatmap, reg, reg_mask


# ---------------------------------------------------------------------------
# Metrics accumulator  (per-rank; aggregated via all_reduce at the end)
# ---------------------------------------------------------------------------
class ValMetrics:
    """
    Accumulates stats locally on each rank.
    Call .aggregate(device) after the loop to all_reduce across ranks.
    """
    def __init__(self, num_classes: int):
        self.C = num_classes
        # Use float tensors so we can all_reduce them easily
        self.focal_sum    = 0.0
        self.reg_sum      = 0.0
        self.n_batches    = 0
        self.gt_response  = [0.0] * num_classes
        self.gt_count     = [0]   * num_classes
        self.recall_03    = [0]   * num_classes
        self.recall_05    = [0]   * num_classes

    @torch.no_grad()
    def update(self, pred_hm, gt_hm, pred_reg, gt_reg, reg_mask):
        self.focal_sum += focal_loss_val(pred_hm, gt_hm).item()
        mask    = reg_mask.float()
        num_pos = mask.sum().clamp(min=1)
        self.reg_sum += (F.smooth_l1_loss(
            pred_reg.float()*mask, gt_reg.float()*mask,
            reduction='sum') / num_pos).item()
        self.n_batches += 1

        pred_prob = pred_hm.float().sigmoid()
        for c in range(self.C):
            centres = (gt_hm[:, c] == 1.0)
            if centres.any():
                resp = pred_prob[:, c][centres]
                self.gt_response[c] += resp.sum().item()
                self.gt_count[c]    += centres.sum().item()
                self.recall_03[c]   += (resp >= 0.3).sum().item()
                self.recall_05[c]   += (resp >= 0.5).sum().item()

    def aggregate(self, device):
        """All-reduce all counters across DDP ranks. Call on all ranks."""
        def _reduce(val):
            t = torch.tensor(float(val), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item()

        self.focal_sum  = _reduce(self.focal_sum)
        self.reg_sum    = _reduce(self.reg_sum)
        self.n_batches  = int(_reduce(self.n_batches))
        for c in range(self.C):
            self.gt_response[c] = _reduce(self.gt_response[c])
            self.gt_count[c]    = int(_reduce(self.gt_count[c]))
            self.recall_03[c]   = int(_reduce(self.recall_03[c]))
            self.recall_05[c]   = int(_reduce(self.recall_05[c]))

    def report(self, logger) -> dict:
        avg_focal  = self.focal_sum / max(self.n_batches, 1)
        avg_reg    = self.reg_sum   / max(self.n_batches, 1)
        total_loss = avg_focal + 2.0 * avg_reg

        logger.info('=' * 62)
        logger.info(f'  Val focal loss : {avg_focal:.4f}')
        logger.info(f'  Val reg   loss : {avg_reg:.4f}')
        logger.info(f'  Val total loss : {total_loss:.4f}  (focal + 2×reg)')
        logger.info('-' * 62)
        logger.info(f'  {"Class":<14} {"GT objs":>8} {"AvgResp":>9} '
                    f'{"Recall@.3":>10} {"Recall@.5":>10}')
        logger.info('-' * 62)

        results = {}
        for c, name in enumerate(CLASS_NAMES):
            n        = max(self.gt_count[c], 1)
            avg_resp = self.gt_response[c] / n
            rec03    = self.recall_03[c]   / n
            rec05    = self.recall_05[c]   / n
            logger.info(f'  {name:<14} {self.gt_count[c]:>8d} '
                        f'{avg_resp:>9.4f} {rec03:>10.4f} {rec05:>10.4f}')
            results[name] = dict(gt_count=self.gt_count[c],
                                  avg_response=avg_resp,
                                  recall_at_03=rec03,
                                  recall_at_05=rec05)
        logger.info('=' * 62)
        results.update(total_loss=total_loss,
                       focal_loss=avg_focal,
                       reg_loss=avg_reg)

    # --- NEW: Save to structured CSV for research plotting ---
        if dist.get_rank()==0:
            csv_path = './checkpoints/val_metrics.csv'
            
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a') as f:
                if write_header:
                    f.write("Epoch,TotalLoss,FocalLoss,RegLoss,Car_Recall3,Car_Recall5,Ped_Recall3,Cyc_Recall3\n")
                
                # Extract metrics safely
                c_rec3 = results.get('Car', {}).get('recall_at_03', 0.0)
                c_rec5 = results.get('Car', {}).get('recall_at_05', 0.0)
                p_rec3 = results.get('Pedestrian', {}).get('recall_at_03', 0.0)
                cy_rec3 = results.get('Cyclist', {}).get('recall_at_03', 0.0)
                
                # Note: We must pass 'epoch' to report() to log it, or just use a counter. 
                # (If epoch isn't available inside report(), modify the report() signature to accept epoch)
                f.write(f"{epoch},{total_loss:.4f},{avg_focal:.4f},{avg_reg:.4f},{c_rec3:.4f},{c_rec5:.4f},{p_rec3:.4f},{cy_rec3:.4f}\n")
                

        return results


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   type=str, required=True)
    p.add_argument('--kitti_root',   type=str, default='~/kitti')
    p.add_argument('--imageset_dir', type=str, default='~/kitti/ImageSets')
    p.add_argument('--batch_size',   type=int, default=2,
                   help='Per-rank batch size. Start at 2 (same as training). '
                        'Try 4 if VRAM allows — no grads = ~half training memory.')
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--split',        type=str, default='val',
                   choices=['val', 'train'])
    p.add_argument('--local_rank',   type=int,
                   default=int(os.environ.get('LOCAL_RANK', 0)))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ── DDP init ─────────────────────────────────────────────────────────────
    dist.init_process_group(backend='nccl', init_method='env://')
    rank       = dist.get_rank()
    local_rank = args.local_rank
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    logger = get_logger(rank)

    if rank == 0:
        logger.info(f'Validation with {world_size} GPUs')
        logger.info(f'Checkpoint : {args.checkpoint}')
        logger.info(f'Split      : {args.split}')
        logger.info(f'Batch/rank : {args.batch_size}  '
                    f'(global {args.batch_size * world_size})')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BEVWaveFormer(
        pillar_in_ch  = 4,
        pillar_out_ch = 64,
        max_points    = BEV_CFG['max_points_per_pillar'],
        bev_h         = BEV_CFG['bev_h'],
        bev_w         = BEV_CFG['bev_w'],
        stage_dims    = [96, 192, 384, 768],
        depths        = [2, 2, 6, 2],
        use_checkpoint= False,   # inference: no activation checkpointing needed
    )

    # Load checkpoint on CPU first — avoids a double-VRAM spike on GPU
    ckpt  = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        if missing:     logger.warning(f'Missing keys  : {missing}')
        if unexpected:  logger.warning(f'Unexpected keys: {unexpected}')
        logger.info(f'Checkpoint epoch={ckpt.get("epoch","?")}, '
                    f'saved_loss={ckpt.get("loss", float("nan")):.4f}')

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)
    model.eval()

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f'Model: {n_params:.1f}M parameters')

    # ── Data ──────────────────────────────────────────────────────────────────
    kitti_root   = os.path.expanduser(args.kitti_root)
    imageset_dir = os.path.expanduser(args.imageset_dir)
    split_txt    = os.path.join(imageset_dir, f'{args.split}.txt')

    ds      = KITTIPillarDataset(kitti_root, args.split, split_txt, BEV_CFG)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                  shuffle=False, drop_last=False)
    loader  = DataLoader(ds,
                         batch_size=args.batch_size,
                         sampler=sampler,
                         num_workers=args.num_workers,
                         collate_fn=kitti_collate_fn,
                         pin_memory=True)

    if rank == 0:
        logger.info(f'Val set: {len(ds)} frames → '
                    f'{len(loader)} batches/rank')

    # ── Validation loop ───────────────────────────────────────────────────────
    metrics = ValMetrics(num_classes=3)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pillars    = batch['pillars'].to(device,    non_blocking=True)
            num_points = batch['num_points'].to(device, non_blocking=True)
            coords     = batch['coords'].to(device,     non_blocking=True)
            bs         = batch['batch_size']

            with autocast('cuda', dtype=torch.float16):
                preds = model(pillars, num_points, coords, bs)

            H_out = preds['heatmap'].shape[-2]
            W_out = preds['heatmap'].shape[-1]
            gt_hm, gt_reg, reg_mask = build_targets(
                batch['boxes'], bs, 3, H_out, W_out, BEV_CFG, device)

            metrics.update(preds['heatmap'], gt_hm,
                           preds['reg'],     gt_reg, reg_mask)

            if rank == 0 and (i + 1) % 20 == 0:
                logger.info(f'  {i+1}/{len(loader)} batches...')

    # ── Aggregate across all ranks and report ─────────────────────────────────
    metrics.aggregate(device)

    if rank == 0:
        metrics.report(logger, epoch)
        logger.info('Done.')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()