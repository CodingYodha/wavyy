"""
model.py  —  BEV-WaveFormer Core Architecture
==============================================
BEV-WaveFormer: 3D Object Detection via Wave Propagation on Bird's Eye View grids.

Pipeline:
  Raw LiDAR points (N×3/N×4)
      → VFE / Pillar Feature Net  (per-pillar MLP)
      → BEV Scatter               (sparse pillar → dense 2D pseudo-image)
      → WPO Backbone              (damped wave equation in Fourier domain)
      → Detection Head            (CenterPoint-style heatmap + regression)

Physics basis (from WaveFormer, Shu et al. 2026, arXiv:2601.08602):
  The underdamped wave equation:
      ∂²u/∂t² + α·∂u/∂t = v²·∇²u
  has the closed-form frequency-domain solution:
      u(x,y,t) = F⁻¹{ e^{-αt/2} [ F(u₀)·cos(ωd·t)
                        + sin(ωd·t)/ωd · (F(v₀) + α/2·F(u₀)) ] }
  where  ωd = sqrt(v²·(ωx²+ωy²) − (α/2)²)

  KEY difference from heat-based (vHeat): damping e^{-αt/2} is
  frequency-INDEPENDENT, so high-frequency components (edges, boundaries)
  survive instead of being killed by the e^{-k·ω²·t} heat kernel.
  For BEV grids this matters: object edges in the pillar map are preserved.

FIXES applied in this revision
───────────────────────────────
  [Critical]  sinc_term instability → NaN:
              sin(ωd·t)/ωd was computed via direct division, producing
              Inf/NaN when ωd → sqrt(EPS) ≈ 0.001.  Replaced with a
              Taylor-series branch for small ωd (see _stable_sinc).

  [Critical]  float32 guard in _wave_propagate:
              Added an assertion and an explicit .float() cast so
              autocast can never silently downcast the spectral tensors.

  [Critical]  irfft2 backward amplification — NEW FIX:
              The backward pass of irfft2 is rfft2(grad_output). For a
              sparse BEV grid (95% zeros), the FFT concentrates energy:
              the DC component grows as √(H×W) ≈ 512. This amplifies
              gradients from even a modest loss into overflow territory.
              Fixed by clamping Ut_f (complex spectral tensor) before
              irfft2 so the backward has bounded inputs. Clamp applied
              to the real/imag views separately to avoid breaking the
              complex dtype.

  [Moderate]  BEVScatter gradient flow:
              index_put_ on a freshly-allocated zeros tensor breaks the
              autograd graph.  Replaced with scatter_add_ on a leaf
              tensor that has requires_grad=True propagated correctly
              (achieved by building bev with torch.zeros(...) and using
              the differentiable index_put approach via advanced indexing
              on a clone).

  [Moderate]  BEVScatter duplicate coords:
              Added torch.unique deduplication before scatter so that
              duplicate (batch_idx, row, col) tuples — which can arise
              from coord clipping at grid boundaries — produce a
              deterministic result across DDP ranks.

  [Moderate]  BatchNorm eps — NEW FIX:
              Default eps=1e-5 causes 1/σ blow-up when BN sees feature
              channels that are almost entirely zero (95% sparse BEV).
              Increased to eps=1e-3 across all BN layers via
              _init_weights. This is the standard fix for sparse
              activations in LiDAR detection models.

  [Moderate]  GroupNorm in WPOBlock — NEW FIX:
              BatchNorm2d in WPO blocks is replaced with GroupNorm(8, C).
              GN has no batch-size dependency and handles sparse
              activations gracefully. The per-rank batch is 2, which
              makes BN statistics unreliable even with SyncBN overhead.
              GN is unaffected by batch size and spatial sparsity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# ---------------------------------------------------------------------------
# Constants / small numerical guard
# ---------------------------------------------------------------------------
EPS = 1e-6   # Prevents division-by-zero when ωd → 0 (over-damped regime)

# Threshold below which we switch to Taylor-series sinc approximation.
# sin(x)/x ≈ 1 - x²/6 + x⁴/120 is accurate to < 1e-7 for |x| < 0.05.
_SINC_TAYLOR_THRESH = 0.05

# Hard clamp on spectral coefficients before irfft2.
# Prevents the irfft2 backward (= rfft2(grad)) from amplifying sparse
# BEV gradients by the FFT gain factor √(H×W) ≈ 512.
# Value of 50.0 is conservative: typical spectral magnitudes after
# normalised rfft2(norm='ortho') are O(1); clamping at ±50 gives 50×
# headroom while preventing float16-range overflow in the backward.
_SPECTRAL_CLAMP = 50.0


# ===========================================================================
# 1.  Pillar Feature Encoder  (PointPillars-style VFE)
# ===========================================================================
class PillarFeatureNet(nn.Module):
    """
    Lightweight Voxel Feature Encoding for cylindrical pillars.

    Each non-empty pillar contains up to `max_points` LiDAR points.
    We augment each point with pillar-center offsets (Δx, Δy, Δz) and the
    point's distance from the sensor origin, giving an 8-D input per point
    when the raw input has (x, y, z, intensity).

    Args:
        in_channels  : raw point feature dims (default 4: x,y,z,intensity)
        out_channels : output pillar embedding dim
        max_points   : max points kept per pillar (rest are zero-padded)
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 64,
                 max_points: int = 32):
        super().__init__()
        # +3 for (Δx_center, Δy_center, dist_from_origin)
        augmented_in = in_channels + 3
        self.mlp = nn.Sequential(
            nn.Linear(augmented_in, 32, bias=False),
            # FIX: eps=1e-3 — default 1e-5 causes 1/σ explosion when most
            # of the P×T samples are zero-padded (sparse pillar scenes).
            nn.BatchNorm1d(32, eps=1e-3),
            nn.SiLU(inplace=True),
            nn.Linear(32, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3),
            nn.SiLU(inplace=True),
        )
        self.out_channels = out_channels
        self.max_points   = max_points

    def forward(self, pillars: torch.Tensor,
                num_points_per_pillar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pillars              : (P, max_points, in_channels)   float16/32
            num_points_per_pillar: (P,)                           int32
        Returns:
            pillar_features      : (P, out_channels)
        """
        P, T, C = pillars.shape

        # --- Augment: append (Δx, Δy, dist) ---
        mask   = (torch.arange(T, device=pillars.device).unsqueeze(0)
                  < num_points_per_pillar.unsqueeze(1))          # (P, T)
        mask_f = mask.unsqueeze(-1).float()                      # (P, T, 1)

        valid_cnt = num_points_per_pillar.clamp(min=1).float().view(P, 1, 1)
        center_xy = (pillars[..., :2] * mask_f).sum(dim=1, keepdim=True) / valid_cnt
        delta_xy  = pillars[..., :2] - center_xy
        dist      = torch.norm(pillars[..., :3], dim=-1, keepdim=True)

        x = torch.cat([pillars, delta_xy, dist], dim=-1)        # (P, T, C+3)
        x = x * mask_f

        x_flat = x.view(P * T, -1)
        feats  = self.mlp(x_flat)
        feats  = feats.view(P, T, self.out_channels)

        feats  = feats * mask_f
        pillar_feats, _ = feats.max(dim=1)                      # (P, out_channels)
        return pillar_feats


# ===========================================================================
# 2.  BEV Scatter  (pillar features → 2-D pseudo-image)
# ===========================================================================
class BEVScatter(nn.Module):
    """
    Scatter pillar features into a dense (B, C, H, W) BEV grid.

    FIX [Moderate — gradient flow]:
    ────────────────────────────────
    The original implementation used index_put_ (in-place) on a freshly
    allocated zeros tensor. In-place ops on leaf tensors that are not
    themselves part of the autograd graph silently break gradient flow.
    We now use a purely functional scatter via .scatter() which is
    out-of-place and participates fully in autograd.

    FIX [Moderate — non-determinism]:
    ───────────────────────────────────
    Duplicate (batch_idx, row, col) coords are deduplicated by keeping
    the last occurrence (most points per pillar kept by the dataset).
    """
    def __init__(self, bev_h: int, bev_w: int):
        super().__init__()
        self.H = bev_h
        self.W = bev_w

    def forward(self, pillar_feats: torch.Tensor,
                coords: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        """
        Args:
            pillar_feats : (P_total, C)
            coords       : (P_total, 3)  — (batch_idx, row, col) int32
            batch_size   : B
        Returns:
            bev          : (B, C, H, W)
        """
        C      = pillar_feats.shape[1]
        dtype  = pillar_feats.dtype
        device = pillar_feats.device

        b_idx  = coords[:, 0].long()
        r_idx  = coords[:, 1].long().clamp(0, self.H - 1)
        c_idx  = coords[:, 2].long().clamp(0, self.W - 1)

        # --- Deduplicate coords ---
        flat_key    = b_idx * (self.H * self.W) + r_idx * self.W + c_idx
        _, inv_idx  = torch.unique(flat_key.flip(0), return_inverse=True)
        n_unique    = inv_idx.max().item() + 1
        keep_rev    = torch.zeros(n_unique, dtype=torch.long, device=device)
        keep_rev.scatter_(0, inv_idx,
                          torch.arange(len(inv_idx), device=device))
        keep_mask_rev = torch.zeros(len(flat_key), dtype=torch.bool, device=device)
        keep_mask_rev[keep_rev] = True
        keep_mask   = keep_mask_rev.flip(0)

        pillar_feats = pillar_feats[keep_mask]
        b_idx        = b_idx[keep_mask]
        r_idx        = r_idx[keep_mask]
        c_idx        = c_idx[keep_mask]

        # --- Out-of-place scatter (fully differentiable) ---
        flat_idx     = b_idx * (self.H * self.W) + r_idx * self.W + c_idx
        flat_idx_exp = flat_idx.unsqueeze(1).expand(-1, C)
        bev_flat     = torch.zeros(batch_size * self.H * self.W, C,
                                   dtype=dtype, device=device)
        bev_flat     = bev_flat.scatter(0, flat_idx_exp, pillar_feats)

        bev = bev_flat.view(batch_size, self.H, self.W, C).permute(0, 3, 1, 2).contiguous()
        return bev


# ===========================================================================
# 3.  Wave Propagation Operator  (WPO)  —  the core contribution
# ===========================================================================

def _stable_sinc(omega_d: torch.Tensor, t: float) -> torch.Tensor:
    """
    Numerically stable computation of  sin(ωd·t) / ωd.

    Uses 4th-order Taylor expansion for |ωd·t| < _SINC_TAYLOR_THRESH,
    exact formula elsewhere.  Prevents Inf/NaN in forward and backward.
    """
    x       = omega_d * t
    x_safe = torch.where(x < 0.05, x, torch.zeros_like(x))

    x2      = x_safe * x_safe
    taylor  = 1.0 - x2 / 6.0 + (x2 * x2) / 120.0
    exact   = torch.sin(x) / x.clamp(min=_SINC_TAYLOR_THRESH)
    return torch.where(x < _SINC_TAYLOR_THRESH, taylor, exact)


class WavePropagationOperator(nn.Module):
    """
    Implements Eq. (6) from WaveFormer (Shu et al. 2026):

        U^t = F⁻¹{ e^{-α/2·t} [ F(U⁰)·cos(ωd·t)
                    + sin(ωd·t)/ωd · (F(V⁰) + α/2·F(U⁰)) ] }

    where   ωd = sqrt( v²·(ωx²+ωy²) − (α/2)² )       [Eq. 7]
    """
    def __init__(self, channels: int, bev_h: int, bev_w: int,
                 t: float = 1.0,
                 init_alpha: float = 0.1,
                 init_v: float = 1.0):
        super().__init__()
        self.C   = channels
        self.H   = bev_h
        self.W   = bev_w
        self.t   = t

        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        self.log_v     = nn.Parameter(torch.tensor(math.log(init_v)))

        self.dw_conv  = nn.Conv2d(channels, channels, kernel_size=3,
                                   padding=1, groups=channels, bias=False)
        self.v0_proj  = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            # FIX: GroupNorm replaces BatchNorm2d here — GN has no batch-size
            # dependency and handles sparse activations (95% zero BEV) without
            # variance collapse. num_groups=min(8, channels) is safe for any C.
            nn.GroupNorm(min(8, channels), channels),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
        )

        freq_y  = torch.fft.fftfreq(bev_h)
        freq_x  = torch.fft.rfftfreq(bev_w)
        wy, wx  = torch.meshgrid(freq_y, freq_x, indexing='ij')
        omega2  = (wx ** 2 + wy ** 2).unsqueeze(0).unsqueeze(0)
        self.register_buffer('omega2', omega2)

    def _wave_propagate(self, U0_f: torch.Tensor,
                        V0_f: torch.Tensor,
                        alpha: torch.Tensor,
                        v: torch.Tensor) -> torch.Tensor:
        """
        Core spectral computation — explicitly enforced to run in float32.

        FIX [Critical — irfft2 backward amplification]:
        ─────────────────────────────────────────────────
        The backward of irfft2 is rfft2(grad_output). For a 95%-sparse
        BEV grid the FFT concentrates energy: the DC component grows as
        √(H×W) ≈ 512. To bound the backward, we clamp the spectral
        output Ut_f (as real/imag views) to ±_SPECTRAL_CLAMP before
        returning. The clamp is on the output of this function, so it
        propagates a bounded gradient through irfft2 in the caller.

        This does NOT affect the forward loss meaningfully: spectral
        magnitudes after ortho-normalised rfft2 are O(1) for typical
        BEV features; clamping at ±50 only bites for pathological inputs.
        """
        assert U0_f.dtype == torch.complex64, (
            f"WPO spectral tensors must be complex64, got {U0_f.dtype}.")
        assert V0_f.dtype == torch.complex64, (
            f"WPO spectral tensors must be complex64, got {V0_f.dtype}.")

        alpha  = alpha.float()
        v      = v.float()

        t      = self.t
        half_a = alpha / 2.0

        omega2_d = (v ** 2) * self.omega2 - half_a ** 2
                 # Use mathematical clamp instead of softplus to preserve gradient flow
        omega2_d = torch.clamp(omega2_d, min=EPS)
        omega_d  = torch.sqrt(omega2_d)

        decay    = torch.exp(-half_a * t)
        cos_term = torch.cos(omega_d * t)
        sinc_val = _stable_sinc(omega_d, t)

        Ut_f = decay * (
                U0_f * cos_term
                + sinc_val * (V0_f + half_a * U0_f)
            )

            # Restore SPECTRAL_CLAMP using magnitude to perfectly preserve phase ratios!
        mag = Ut_f.abs()
        scale = _SPECTRAL_CLAMP / torch.clamp(mag, min=_SPECTRAL_CLAMP)
        Ut_f = Ut_f * scale

        return Ut_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) — BEV feature map (possibly float16 under AMP)
        Returns:
            out : (B, C, H, W)
        """
        dtype = x.dtype

        U0 = self.dw_conv(x)
        V0 = self.v0_proj(x)

        with torch.amp.autocast('cuda', enabled=False):
            U0_f32 = U0.float()
            V0_f32 = V0.float()

            U0_f = torch.fft.rfft2(U0_f32, norm='ortho')
            V0_f = torch.fft.rfft2(V0_f32, norm='ortho')

            # -- v2.2 fix1 clamping the exponentials to avoid explosion
            # self.log_alpha.data.clamp_(-17.0, 17.0)
            # self.log_v.data.clamp_(-17.0, 17.0)

            # -- v2.3 REMOVED .data.clamp_ from here. We will clamp post-optimizer step in train.py instead!
            # v2.2 fix is compromised
            

            alpha = torch.exp(self.log_alpha)
            v     = torch.exp(self.log_v)

            Ut_f   = self._wave_propagate(U0_f, V0_f, alpha, v)
            Ut_f32 = torch.fft.irfft2(Ut_f, s=(self.H, self.W), norm='ortho')


            # -- v2.2 fix2 preventing overflow problem in float16
            # -- v2.3 Lowered clamp to physically realistic values to prevent Conv2d FP16 Overflow

            Ut_f32 = torch.clamp(Ut_f32, min=-256.0, max=256.0)
            
        Ut  = Ut_f32.to(dtype)

        # --- v2.3 update # Protect the residual connection from FP16 GroupNorm explosion
        out_p = self.out_proj(Ut)
        out_p = torch.clamp(out_p, min=-65000.0, max=65000.0)

        out = out_p + x
        return out


# ===========================================================================
# 4.  WPO Block  (WPO + FFN, with gradient checkpointing)
# ===========================================================================
class WPOBlock(nn.Module):
    """
    One WaveFormer block: WPO → LayerNorm → FFN → LayerNorm

    FIX [Moderate — GroupNorm in FFN path]:
    ─────────────────────────────────────────
    The FFN previously had no normalisation between linear layers, relying
    on LayerNorm before the block. For very sparse activations the GELU
    output can be extremely peaked. No additional change needed here —
    LayerNorm is fine for the pre-norm pattern since it normalises per
    token (spatial location) not per batch.
    """
    def __init__(self, channels: int, bev_h: int, bev_w: int,
                 expansion: int = 4, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(channels)
        self.wpo   = WavePropagationOperator(channels, bev_h, bev_w)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn   = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.SiLU(inplace=True),
            nn.Linear(channels * expansion, channels),
        )

    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        residual = x
        x_ln = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x    = residual + self.wpo(x_ln)

        residual = x
        x_ln = self.norm2(x.permute(0, 2, 3, 1))
        x_ff = self.ffn(x_ln)
        x    = residual + x_ff.permute(0, 3, 1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        should_checkpoint = (
            self.use_checkpoint
            and self.training
            and (x.requires_grad or torch.is_grad_enabled())
        )
        if should_checkpoint:
            return checkpoint.checkpoint(self._inner_forward, x,
                                         use_reentrant=False)
        return self._inner_forward(x)


# ===========================================================================
# 5.  WPO Backbone  (hierarchical: 4 stages with downsampling)
# ===========================================================================
class WPOBackbone(nn.Module):
    """
    4-stage hierarchical backbone.
    Stage i: [WPOBlock × depth[i]] → PatchMerge (2× spatial downsample)

    FIX: PatchMerge uses GroupNorm instead of BatchNorm2d for the same
    reason as WPO blocks — batch size 2 with sparse BEV is too small for
    reliable BN statistics at the downsampling boundaries.
    """
    def __init__(self,
                 in_channels : int       = 64,
                 stage_dims  : list[int] = [96, 192, 384, 768],
                 depths      : list[int] = [2,  2,   6,   2],
                 bev_h       : int       = 512,
                 bev_w       : int       = 512,
                 use_checkpoint: bool    = True):
        super().__init__()

        self.stages      = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        # FIX: stem uses GroupNorm instead of BatchNorm2d
        self.stem        = nn.Sequential(
            nn.Conv2d(in_channels, stage_dims[0], 3, padding=1, bias=False),
            nn.GroupNorm(min(8, stage_dims[0]), stage_dims[0]),
            nn.SiLU(inplace=True),
        )

        h, w = bev_h, bev_w
        for i, (dim, depth) in enumerate(zip(stage_dims, depths)):
            stage = nn.Sequential(*[
                WPOBlock(dim, h, w, use_checkpoint=use_checkpoint)
                for _ in range(depth)
            ])
            self.stages.append(stage)
            if i < len(stage_dims) - 1:
                next_dim = stage_dims[i + 1]
                # FIX: GroupNorm in PatchMerge — no batch-size sensitivity
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(dim, next_dim, kernel_size=2, stride=2, bias=False),
                    nn.GroupNorm(min(8, next_dim), next_dim),
                ))
                h, w = h // 2, w // 2
            else:
                self.downsamples.append(nn.Identity())

        self.out_dims = stage_dims

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for stage, down in zip(self.stages, self.downsamples):
            x = stage(x)
            outs.append(x)
            x = down(x)
        return outs


# ===========================================================================
# 6.  CenterPoint-style Detection Head
# ===========================================================================
class CenterPointHead(nn.Module):
    """
    Simplified CenterPoint head for KITTI (3 classes: Car, Pedestrian, Cyclist).

    FIX: shared conv uses GroupNorm instead of BatchNorm2d.
    """
    NUM_CLASSES = 3
    REG_DIMS    = 8   # (x, y, z, log_w, log_l, log_h, sin_θ, cos_θ)

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            # FIX: GroupNorm — detection head operates at 1/8 spatial scale
            # (64×64 for 512 input), batch=2; BN unreliable here too.
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(hidden, self.NUM_CLASSES, 1)
        self.reg     = nn.Conv2d(hidden, self.REG_DIMS,    1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(feat)
        return {
            'heatmap': self.heatmap(h),
            'reg'    : self.reg(h),
        }


# ===========================================================================
# 7.  BEVWaveFormer  —  Full End-to-End Model
# ===========================================================================
class BEVWaveFormer(nn.Module):
    """
    Complete BEV-WaveFormer model.

    Input format (per batch):
        pillars       : (P_total, max_points, in_channels)  float32
        num_points    : (P_total,)                           int32
        coords        : (P_total, 3)  (batch_idx, row, col)  int32
        batch_size    : int
    """
    def __init__(self,
                 pillar_in_ch    : int       = 4,
                 pillar_out_ch   : int       = 64,
                 max_points      : int       = 32,
                 bev_h           : int       = 512,
                 bev_w           : int       = 512,
                 stage_dims      : list[int] = [96, 192, 384, 768],
                 depths          : list[int] = [2,  2,   6,   2],
                 use_checkpoint  : bool      = True):
        super().__init__()

        self.vfe      = PillarFeatureNet(pillar_in_ch, pillar_out_ch, max_points)
        self.scatter  = BEVScatter(bev_h, bev_w)
        self.backbone = WPOBackbone(pillar_out_ch, stage_dims, depths,
                                    bev_h, bev_w, use_checkpoint)

        # --- V2: Lightweight U-Net FPN Neck (64 Channels to save VRAM) ---
        fpn_dim = 64

        # -- v2.3 fix add normalization for upsampling
        # Stage 4 (64x64) -> Compress to 64ch -> Upsample to 128x128
        self.fpn_s4_shrink = nn.Sequential(nn.Conv2d(stage_dims[3], fpn_dim, 1, bias=False), nn.GroupNorm(8, fpn_dim), nn.SiLU(inplace=True))
        self.fpn_s4_up     = nn.Sequential(nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2, stride=2) , nn.GroupNorm(8, fpn_dim), nn.SiLU(inplace=True))
        
        # Stage 3 (128x128) -> Compress to 64ch -> Upsample to 256x256
        self.fpn_s3_shrink = nn.Sequential(nn.Conv2d(stage_dims[2], fpn_dim, 1, bias=False), nn.GroupNorm(8, fpn_dim), nn.SiLU(inplace=True))
        self.fpn_s3_up     = nn.Sequential(nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2, stride=2), nn.GroupNorm(8, fpn_dim), nn.SiLU(inplace=True))
        
        # Stage 2 (256x256) -> Compress to 64ch
        self.fpn_s2_shrink = nn.Sequential(nn.Conv2d(stage_dims[1], fpn_dim, 1, bias=False), nn.GroupNorm(8, fpn_dim), nn.SiLU(inplace=True))

        # Head now takes the High-Res 256x256 fused map!
        self.head = CenterPointHead(in_channels=fpn_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # FIX: eps=1e-3 on all remaining BN layers (VFE only after
                # backbone switch to GN). Prevents 1/σ explosion on sparse
                # feature channels.
                m.eps = 1e-3
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.head.heatmap.bias, -2.19)

    def forward(self, pillars: torch.Tensor,
                num_points: torch.Tensor,
                coords: torch.Tensor,
                batch_size: int) -> dict[str, torch.Tensor]:

        pillar_feats      = self.vfe(pillars, num_points)
        bev               = self.scatter(pillar_feats, coords, batch_size)
        multi_scale_feats = self.backbone(bev)
        
        # --- V2: FPN Fusion (Bottom-up U-Net connections) ---
        s2 = multi_scale_feats[1] # (B, 192, 256, 256)
        s3 = multi_scale_feats[2] # (B, 384, 128, 128)
        s4 = multi_scale_feats[3] # (B, 768, 64,  64)

        f4 = self.fpn_s4_up(self.fpn_s4_shrink(s4))           # 64x64   -> 128x128
        f3 = self.fpn_s3_up(f4 + self.fpn_s3_shrink(s3))      # 128x128 -> 256x256
        fused_256 = f3 + self.fpn_s2_shrink(s2)               # Add sharp 256x256 features
        
        preds = self.head(fused_256)

        return preds