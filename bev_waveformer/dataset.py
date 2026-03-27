"""
dataset.py  —  KITTI 3D Object Detection Dataset Loader
=========================================================

KITTI 3D DOWNLOAD INSTRUCTIONS
───────────────────────────────
  1. Velodyne LiDAR point clouds (29 GB):
     https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
  2. Left color images (12 GB, optional):
     https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
  3. Camera calibration matrices (16 MB):
     https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
  4. Training labels (5 MB):
     https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
  5. ImageSets:
     https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt
     https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt

wget download commands:
  mkdir -p ~/kitti && cd ~/kitti
  wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
  wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
  wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
  unzip data_object_velodyne.zip && unzip data_object_calib.zip && unzip data_object_label_2.zip

Expected directory structure:
  ~/kitti/
  ├── training/
  │   ├── velodyne/        ← .bin files (N×4 float32: x,y,z,intensity)
  │   ├── label_2/         ← .txt files
  │   └── calib/           ← .txt files
  └── testing/
      └── velodyne/

Total training frames: 7481 (7:1 train/val split → ~6502 / ~979)

FIXES in this revision
──────────────────────
  [Moderate — BN stability] points_to_pillars now guarantees a minimum of
  MIN_PILLARS=8 non-empty pillar entries per sample by duplicating the
  real pillars cyclically when too few are found.

  Rationale: PillarFeatureNet uses BatchNorm1d over (P×T) samples.
  With P=1 and T=32 points of which most are zero-padded, BN sees 32
  samples of near-zero variance → 1/σ blows up → NaN.  The BN eps fix
  in model.py (eps=1e-3) is the first line of defence; this min_pillars
  guarantee is the second, ensuring BN always has enough non-trivial
  samples to compute stable statistics.

  Duplicated pillars carry the same coords as their source, so BEVScatter's
  deduplication logic removes them before scatter — no repeated features
  are written into the BEV grid.  The guarantee is purely for BN health.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# KITTI class map
# ---------------------------------------------------------------------------
KITTI_CLASSES = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

# ---------------------------------------------------------------------------
# BEV grid configuration
# ---------------------------------------------------------------------------
BEV_CFG = dict(
    x_min=-39.68, x_max=39.68,
    y_min=-39.68, y_max=39.68,
    z_min=-3.0,   z_max=1.0,
    pillar_size_x=0.16,
    pillar_size_y=0.16,
    max_points_per_pillar=32,
    max_pillars=16000,
    bev_h=512,
    bev_w=512,
)

# FIX [Moderate — BN stability]:
# Minimum number of distinct pillars returned per sample.  If fewer real
# pillars are found (e.g. near-empty scan from fog/sensor fault), we
# cyclically duplicate existing pillars to reach this count.  The
# duplicates share coords with their source, so BEVScatter's dedup
# removes them before scatter — no spurious features enter the BEV grid.
# This ensures BatchNorm1d in PillarFeatureNet always has enough samples
# to compute non-degenerate statistics (variance > eps=1e-3).
MIN_PILLARS = 8


def points_to_pillars(points: np.ndarray, cfg: dict):
    """
    Convert raw LiDAR point cloud to pillar tensors.

    Vectorised NumPy implementation (replaces the original Python for-loop
    that ran at ~19 s/step).

    Args:
        points : (N, 4)  float32   x, y, z, intensity
        cfg    : BEV_CFG dict

    Returns:
        pillars      : (P, max_pts, 4)  float32  (zero-padded)
        num_points   : (P,)             int32
        coords       : (P, 2)           int32    (row, col)
        num_pillars  : int
    """
    x_min, x_max = cfg['x_min'], cfg['x_max']
    y_min, y_max = cfg['y_min'], cfg['y_max']
    z_min, z_max = cfg['z_min'], cfg['z_max']
    ps_x,  ps_y  = cfg['pillar_size_x'], cfg['pillar_size_y']
    max_pts       = cfg['max_points_per_pillar']
    max_pillars   = cfg['max_pillars']
    H, W          = cfg['bev_h'], cfg['bev_w']

    # --- Filter to valid range ---
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]

    if len(points) == 0:
        # Edge case: completely empty scan
        # Return MIN_PILLARS copies of the zero pillar so BN has samples
        empty_pillars = np.zeros((MIN_PILLARS, max_pts, 4), dtype=np.float32)
        empty_numpts  = np.ones((MIN_PILLARS,),              dtype=np.int32)
        empty_coords  = np.zeros((MIN_PILLARS, 2),           dtype=np.int32)
        return empty_pillars, empty_numpts, empty_coords, MIN_PILLARS

    # --- Assign each point to a (col, row) pillar ---
    col_idx   = ((points[:, 0] - x_min) / ps_x).astype(np.int32).clip(0, W - 1)
    row_idx   = ((points[:, 1] - y_min) / ps_y).astype(np.int32).clip(0, H - 1)
    pillar_id = row_idx * W + col_idx

    sort_order  = np.argsort(pillar_id, kind='stable')
    sorted_pts  = points[sort_order]
    sorted_ids  = pillar_id[sort_order]

    unique_ids, first_occ, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True)

    if len(unique_ids) > max_pillars:
        top_k      = np.argpartition(counts, -max_pillars)[-max_pillars:]
        unique_ids = unique_ids[top_k]
        first_occ  = first_occ[top_k]
        counts     = counts[top_k]

    P      = len(unique_ids)
    n_take = np.minimum(counts, max_pts)

    col_range   = np.arange(max_pts, dtype=np.int32)[None, :]
    gather_idx  = first_occ[:, None] + np.minimum(
        col_range, (n_take - 1)[:, None])
    pillars_raw = sorted_pts[gather_idx]

    valid_mask  = col_range < n_take[:, None]
    pillars     = pillars_raw * valid_mask[:, :, None].astype(np.float32)
    num_points  = n_take.astype(np.int32)
    coords      = np.stack(
        [unique_ids // W, unique_ids % W], axis=1).astype(np.int32)

    # FIX [Moderate — BN stability]:
    # If fewer than MIN_PILLARS real pillars were found, cyclically
    # duplicate existing pillars to reach the minimum.  Duplicates share
    # coords with their source, so BEVScatter's deduplication removes
    # them before writing into the BEV grid — no ghost features appear.
    if P < MIN_PILLARS:
        n_dup      = MIN_PILLARS - P
        dup_idx    = np.arange(n_dup) % P           # cyclic repeat of real pillars
        pillars    = np.concatenate([pillars,    pillars[dup_idx]],    axis=0)
        num_points = np.concatenate([num_points, num_points[dup_idx]], axis=0)
        coords     = np.concatenate([coords,     coords[dup_idx]],     axis=0)
        P          = MIN_PILLARS

    return pillars, num_points, coords, P


def parse_kitti_calib(calib_path: str) -> np.ndarray:
    """
    Parse KITTI calibration file and return Tr_velo_to_cam (3×4).
    We need this to convert GT boxes from camera frame → LiDAR frame.

    The calibration file contains lines like:
        Tr_velo_to_cam: r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2
    which is a 3×4 matrix [R | t] mapping LiDAR → camera coords.
    We invert it to get camera → LiDAR.

    Returns:
        velo_to_cam : (3, 4)  float64   [R | t]  LiDAR → cam
    """
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('Tr_velo_to_cam'):
                vals = list(map(float, line.strip().split(':')[1].split()))
                return np.array(vals, dtype=np.float64).reshape(3, 4)
    raise ValueError(f'Tr_velo_to_cam not found in {calib_path}')


def cam_to_lidar(loc_cam: np.ndarray, ry: float,
                 velo_to_cam: np.ndarray):
    """
    Convert a single KITTI GT box from camera frame to LiDAR frame.

    KITTI labels store box centres in camera frame (x right, y down, z forward).
    LiDAR frame is (x forward, y left, z up).

    The transform is:   P_lidar = R^T  (P_cam  -  t)
    where [R | t] = velo_to_cam  (LiDAR → cam).

    Yaw conversion:
        KITTI ry is rotation around camera Y axis (pointing down).
        LiDAR yaw is rotation around LiDAR Z axis (pointing up).
        yaw_lidar = -ry - π/2   (standard KITTI → LiDAR convention)

    Args:
        loc_cam     : (3,) [x_cam, y_cam, z_cam]
        ry          : rotation around camera Y (radians)
        velo_to_cam : (3, 4) calibration matrix

    Returns:
        lx, ly, lz  : LiDAR-frame centre coordinates
        yaw_lidar   : yaw in LiDAR frame (radians)
    """
    R = velo_to_cam[:, :3]   # (3, 3)
    t = velo_to_cam[:, 3]    # (3,)

    # Invert: P_lidar = R^T (P_cam - t)
    p_lidar = R.T @ (loc_cam - t)          # (3,)
    lx, ly, lz = p_lidar[0], p_lidar[1], p_lidar[2]

    # Standard KITTI yaw → LiDAR yaw conversion
    yaw_lidar = -ry - np.pi / 2
    # Normalise to (-π, π]
    yaw_lidar = (yaw_lidar + np.pi) % (2 * np.pi) - np.pi

    return float(lx), float(ly), float(lz), float(yaw_lidar)


def parse_kitti_label(label_path: str, calib_path: str):
    """
    Parse one KITTI label file and convert boxes to LiDAR frame.

    KITTI labels are in camera frame. We load the calibration to apply
    Tr_velo_to_cam^{-1} so all box centres are in the same coordinate
    system as the LiDAR point cloud (and therefore the BEV grid).

    Without this transform, GT Gaussians are rendered at wrong BEV pixels
    and the model can never learn to produce high responses at real object
    locations — exactly what caused Recall@0.3 = 0.0000 above.

    Returns list of dicts with keys:
        class_id, lx, ly, lz   (LiDAR frame centre)
        w, l, h                 (box dimensions, unchanged)
        yaw                     (LiDAR frame yaw)
    """
    velo_to_cam = parse_kitti_calib(calib_path)

    objs = []
    with open(label_path, 'r') as f:
        for line in f:
            parts   = line.strip().split()
            cls_str = parts[0]
            if cls_str not in KITTI_CLASSES:
                continue
            # KITTI label format:
            # type trunc occ alpha  bbox(4)  dim(h,w,l)  loc(x,y,z)  ry
            h, w, l    = float(parts[8]),  float(parts[9]),  float(parts[10])
            x, y, z    = float(parts[11]), float(parts[12]), float(parts[13])
            ry         = float(parts[14])

            lx, ly, lz, yaw_lidar = cam_to_lidar(
                np.array([x, y, z], dtype=np.float64), ry, velo_to_cam)

            objs.append(dict(
                class_id = KITTI_CLASSES[cls_str],
                lx=lx, ly=ly, lz=lz,   # LiDAR frame
                w=w, l=l, h=h,
                yaw=yaw_lidar,
            ))
    return objs


class KITTIPillarDataset(Dataset):
    """
    PyTorch Dataset for KITTI 3D Object Detection (LiDAR pillar format).

    Each __getitem__ returns:
        pillars      : (P, max_pts, 4)  float32
        num_points   : (P,)             int32
        coords       : (P, 2)           int32    row, col
        num_pillars  : int
        boxes        : (M, 8)  float32  [class_id, x, y, z, w, l, h, yaw]
        frame_id     : str
    """
    def __init__(self, root: str, split: str = 'train',
                 imageset_path: Optional[str] = None,
                 cfg: dict = BEV_CFG):
        self.root  = Path(root)
        self.cfg   = cfg
        self.split = split

        if imageset_path is None:
            imageset_path = self.root.parent / 'ImageSets' / f'{split}.txt'
        with open(imageset_path, 'r') as f:
            self.frame_ids = [l.strip() for l in f if l.strip()]

        self.lidar_dir = self.root / 'training' / 'velodyne'
        self.label_dir = self.root / 'training' / 'label_2'
        self.calib_dir = self.root / 'training' / 'calib'

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> dict:
        fid      = self.frame_ids[idx]
        bin_path = self.lidar_dir / f'{fid}.bin'
        points   = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)

        

        boxes = np.zeros((0, 8), dtype=np.float32)
        # Load labels for both train AND val with proper cam→LiDAR transform.
        label_path = self.label_dir / f'{fid}.txt'
        calib_path = self.calib_dir / f'{fid}.txt'
        if label_path.exists() and calib_path.exists():
            objs = parse_kitti_label(str(label_path), str(calib_path))
            if objs:
                # boxes: [class_id, lx, ly, lz, w, l, h, yaw]
                # lx/ly/lz are in LiDAR frame — matches BEV grid axes.
                boxes = np.array([[
                    o['class_id'], o['lx'], o['ly'], o['lz'],
                    o['w'], o['l'], o['h'], o['yaw']
                ] for o in objs], dtype=np.float32)

            # --- V2: 3D Data Augmentation (Training ONLY) ---
        if self.split == 'train' and len(boxes) > 0:
            # 1. Random Flip Y (Left/Right)
            if np.random.rand() < 0.5:
                points[:, 1] = -points[:, 1]
                boxes[:, 2]  = -boxes[:, 2]
                boxes[:, 7]  = -boxes[:, 7]  # Negate yaw
            if np.random.rand() < 0.5:
                points[:, 0] = -points[:, 0]
                boxes[:, 1]  = -boxes[:, 1]
                boxes[:, 7]  = np.pi - boxes[:, 7]  # Reflect yaw
            # 3. Random Global Rotation (-45 to +45 degrees)
            if np.random.rand() < 0.5:
                rot_angle = np.random.uniform(-np.pi/4, np.pi/4)
                cos_val, sin_val = np.cos(rot_angle), np.sin(rot_angle)
                rot_mat = np.array([[cos_val, -sin_val], [sin_val, cos_val]], dtype=np.float32)

                # Apply rotation matrix to X and Y coordinates
                points[:, :2] = points[:, :2] @ rot_mat
                boxes[:, 1:3] = boxes[:, 1:3] @ rot_mat
                boxes[:, 7]  += rot_angle
            
        pillars, num_pts, coords, n_pil = points_to_pillars(points, self.cfg)
        
        return {
            'pillars'    : torch.from_numpy(pillars),
            'num_points' : torch.from_numpy(num_pts),
            'coords'     : torch.from_numpy(coords),
            'num_pillars': n_pil,
            'boxes'      : torch.from_numpy(boxes),
            'frame_id'   : fid,
        }


def kitti_collate_fn(batch: list) -> dict:
    """
    Concatenate pillars across batch, prepend batch_idx to coords.
    Per-rank batch_size=2; global batch = batch_size × num_ranks.
    """
    batch_pillars = []
    batch_numpts  = []
    batch_coords  = []
    batch_boxes   = []
    frame_ids     = []

    for b_idx, sample in enumerate(batch):
        P = sample['num_pillars']
        batch_pillars.append(sample['pillars'])
        batch_numpts.append(sample['num_points'])
        b_col = torch.full((P, 1), b_idx, dtype=torch.int32)
        batch_coords.append(torch.cat([b_col, sample['coords']], dim=1))
        batch_boxes.append(sample['boxes'])
        frame_ids.append(sample['frame_id'])

    return {
        'pillars'    : torch.cat(batch_pillars, dim=0),
        'num_points' : torch.cat(batch_numpts,  dim=0),
        'coords'     : torch.cat(batch_coords,  dim=0),
        'boxes'      : batch_boxes,
        'batch_size' : len(batch),
        'frame_ids'  : frame_ids,
    }


def build_dataloaders(kitti_root: str,
                      imageset_dir: str,
                      batch_size: int,
                      num_workers: int,
                      rank: int,
                      world_size: int,
                      cfg: dict = BEV_CFG):
    """Build train/val DataLoaders with DistributedSampler."""
    train_ds = KITTIPillarDataset(kitti_root, 'train',
                                   os.path.join(imageset_dir, 'train.txt'), cfg)
    val_ds   = KITTIPillarDataset(kitti_root, 'val',
                                   os.path.join(imageset_dir, 'val.txt'),   cfg)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                        rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               sampler=train_sampler,
                               num_workers=num_workers,
                               collate_fn=kitti_collate_fn,
                               pin_memory=True,
                               persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                               sampler=val_sampler,
                               num_workers=num_workers,
                               collate_fn=kitti_collate_fn,
                               pin_memory=True,
                               persistent_workers=(num_workers > 0))

    return train_loader, val_loader, train_sampler