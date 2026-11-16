"""Relief recovery by inverse rendering + gradient descent (pure PyTorch, runs on the local GPU).

A coin face is a bas-relief seen from the front, so instead of deforming a generic mesh
(the old pytorch3d approach) we optimise a height field h(x, y) directly:

    unknowns : height h (multi-scale pyramid), albedo a (low-frequency, RGB),
               lighting L (2nd-order spherical harmonics, 9 coeffs) + one specular lobe
    renderer : normals n = normalize(-dh/dx, -dh/dy, 1)   (orthographic camera)
               I = a * SH(n; L) + ks * max(0, n.H)^p
    loss     : robust photometric error inside the coin
               + smoothness (Laplacian) + "relief is raised" prior + border anchored at 0

Low-frequency albedo forces fine image detail to be explained by geometry (shading),
which is what turns the photo into relief. The global relief scale is ambiguous
(bas-relief ambiguity), so the result is rescaled afterwards to a physical height.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

# real spherical harmonics constants
_C0, _C1, _C2, _C3, _C4 = 0.282095, 0.488603, 1.092548, 0.315392, 0.546274


@dataclass
class ReliefParams:
    iters: int = 600
    lr: float = 0.02
    # "pyramid" (default): optimise the height directly (multi-scale). Sharpest on real photos,
    #     relies on the luminance prior (w_lum) for plateau heights.
    # "slopes": optimise (dh/dx, dh/dy) and integrate them globally (Frankot-Chellappa). Recovers
    #     plateaus from edges alone (validated on synthetic data) but streakier on real photos.
    param: str = "pyramid"
    pyramid_levels: int = 5          # height pyramid: R/16 ... R  (param="pyramid")
    slope_lr: float = 0.01
    albedo_res: int = 24             # albedo grid (low frequency on purpose)
    w_smooth: float = 0.1            # Laplacian smoothness of h (change of slope)
    w_raised: float = 1.0            # penalty on h < 0 (designs are raised above the field)
    w_border: float = 5.0            # h ~ 0 on the rim ring and outside the coin
    w_mag: float = 0.01              # weak shrinkage of h
    w_grad: float = 0.3              # slope penalty: coins are low reliefs (fixes the free direction of SfS)
    w_lum: float = 0.05              # low-frequency prior: brighter areas (worn / polished) sit higher
    lum_scale: float = 0.015         # height (half-width units) per std of band-passed luminance
    init_light_dir: tuple[float, float, float] = (-0.45, 0.45, 0.77)  # studio light from top-left
    specular: bool = True
    seed: int = 0


@dataclass
class ReliefResult:
    height: np.ndarray       # (R, R) float32, unit = coin half-width (x, y in [-1, 1]); 0 = field
    albedo: np.ndarray       # (R, R, 3)
    shading: np.ndarray      # (R, R) diffuse shading
    render: np.ndarray       # (R, R, 3) re-rendered image
    normals: np.ndarray      # (R, R, 3)
    loss_history: list[float] = field(default_factory=list)
    photometric_rmse: float = 0.0
    seconds: float = 0.0
    light_sh: list[float] = field(default_factory=list)


def _sh_basis(n: torch.Tensor) -> torch.Tensor:
    """n: (..., 3) unit normals -> (..., 9) real SH basis."""
    x, y, z = n.unbind(-1)
    return torch.stack([
        torch.full_like(x, _C0),
        _C1 * y, _C1 * z, _C1 * x,
        _C2 * x * y, _C2 * y * z, _C3 * (3 * z * z - 1), _C2 * x * z, _C4 * (x * x - y * y),
    ], dim=-1)


def _normals(h: torch.Tensor, dx: float) -> torch.Tensor:
    """h: (R, R) with row 0 = top (+y). Returns (R, R, 3) unit normals (x right, y up, z to viewer)."""
    hp = F.pad(h[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    gx = (hp[1:-1, 2:] - hp[1:-1, :-2]) / (2 * dx)
    gy = -(hp[2:, 1:-1] - hp[:-2, 1:-1]) / (2 * dx)  # rows go down, y goes up
    n = torch.stack([-gx, -gy, torch.ones_like(gx)], dim=-1)
    return n / n.norm(dim=-1, keepdim=True)


def integrate_slopes(gx: torch.Tensor, gy_row: torch.Tensor, dx: float) -> torch.Tensor:
    """Least-squares integration of a gradient field (Frankot-Chellappa, spectral, differentiable).

    gx = dh/dx (columns, to the right), gy_row = dh/d(row) (rows, downward), in height units per
    coordinate unit. Mirror-extended to 2R x 2R so that the FFT's periodic boundary is harmless.
    """
    R0, R1 = gx.shape
    P = torch.cat([torch.cat([gx, -gx.flip(1)], 1), torch.cat([gx.flip(0), -gx.flip(0).flip(1)], 1)], 0)
    Q = torch.cat([torch.cat([gy_row, gy_row.flip(1)], 1), torch.cat([-gy_row.flip(0), -gy_row.flip(0).flip(1)], 1)], 0)
    N0, N1 = P.shape
    wy = 2 * math.pi * torch.fft.fftfreq(N0, d=1.0, device=gx.device)[:, None]
    wx = 2 * math.pi * torch.fft.fftfreq(N1, d=1.0, device=gx.device)[None, :]
    denom = wx ** 2 + wy ** 2
    denom[0, 0] = 1.0
    Hf = (-1j * (wx * torch.fft.fft2(P) + wy * torch.fft.fft2(Q))) / denom
    Hf[0, 0] = 0
    h = torch.fft.ifft2(Hf).real * dx  # gradients were per coordinate unit, pixels are dx wide
    return h[:R0, :R1]


def _laplacian(h: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=h.dtype, device=h.device)[None, None]
    return F.conv2d(F.pad(h[None, None], (1, 1, 1, 1), mode="replicate"), k)[0, 0]


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps) - eps


def reconstruct_relief(rgb: np.ndarray, mask: np.ndarray, params: ReliefParams | None = None,
                       device: str | None = None, log=None) -> ReliefResult:
    """rgb: (R, R, 3) float [0,1]; mask: (R, R) bool coin pixels."""
    p = params or ReliefParams()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(p.seed)
    t0 = time.time()

    R = rgb.shape[0]
    dx = 2.0 / R
    target = torch.from_numpy(rgb).float().to(dev)
    m = torch.from_numpy(mask.astype(np.float32)).to(dev)
    # erode the mask a little: the silhouette edge is not a valid shading signal
    m_in = -F.max_pool2d(-m[None, None], kernel_size=5, stride=1, padding=2)[0, 0]
    # rim ring: just inside the edge (relief anchored at 0 there)
    m_ring = (m - (-F.max_pool2d(-m[None, None], kernel_size=max(3, R // 40) | 1, stride=1,
                                 padding=(max(3, R // 40) | 1) // 2)[0, 0])).clamp(0, 1)

    # band-passed luminance (difference of Gaussians), normalised inside the coin
    lum = target.mean(-1)
    def _blur(x, s):
        k = int(s * 3) * 2 + 1
        g = torch.exp(-((torch.arange(k, device=dev) - k // 2) ** 2) / (2 * s * s))
        g = (g / g.sum()).to(x.dtype)
        x = F.conv2d(F.pad(x[None, None], (k // 2, k // 2, 0, 0), mode="replicate"), g[None, None, None, :])
        x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g[None, None, :, None])
        return x[0, 0]
    wsum = lambda x, s: _blur(x * m, s) / _blur(m, s).clamp(min=1e-3)  # mask-normalised blur
    dog = wsum(lum, 2.0) - wsum(lum, R / 10)
    dog = (dog - (dog * m).sum() / m.sum()) * m
    dog = dog / torch.sqrt((dog ** 2).sum() / m.sum()).clamp(min=1e-6)

    # --- parameters
    if p.param == "slopes":
        slopes = torch.zeros(2, R, R, device=dev, requires_grad=True)  # (dh/dx, dh/drow)
        h_params, h_lr = [slopes], p.slope_lr
    else:
        sizes = [max(4, R // (2 ** k)) for k in reversed(range(p.pyramid_levels))]
        pyr = [torch.zeros(1, 1, s, s, device=dev, requires_grad=True) for s in sizes]
        h_params, h_lr = pyr, p.lr

    mean_rgb = (target * m[..., None]).sum((0, 1)) / m.sum().clamp(min=1)
    a0 = torch.logit((mean_rgb / 0.75).clamp(0.02, 0.98))
    albedo_lr = (a0[:, None, None].expand(3, p.albedo_res, p.albedo_res).clone()[None]).requires_grad_(True)

    d = torch.tensor(p.init_light_dir, device=dev)
    d = d / d.norm()
    sh0 = torch.zeros(9, device=dev)
    sh0[0] = 0.45 / _C0
    sh0[1], sh0[2], sh0[3] = 0.45 * d[1] / _C1, 0.45 * d[2] / _C1, 0.45 * d[0] / _C1
    sh = sh0.clone().requires_grad_(True)

    spec_dir = d.clone().requires_grad_(True)
    spec_k = torch.tensor(-3.0, device=dev, requires_grad=True)       # sigmoid -> strength
    spec_p = torch.tensor(math.log(20.0), device=dev, requires_grad=True)  # exp -> shininess

    groups = [
        {"params": h_params, "lr": h_lr},
        {"params": [albedo_lr], "lr": 0.05},
        {"params": [sh], "lr": 0.01},
    ]
    if p.specular:
        groups.append({"params": [spec_dir, spec_k, spec_p], "lr": 0.01})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p.iters, eta_min=h_lr * 0.05)

    view = torch.tensor([0.0, 0.0, 1.0], device=dev)

    def height() -> torch.Tensor:
        if p.param == "slopes":
            h = integrate_slopes(slopes[0] * m, slopes[1] * m, dx)
            ring_mean = (h * m_ring).sum() / m_ring.sum().clamp(min=1)
            return (h - ring_mean) * m  # rim at 0, nothing outside the coin
        h = sum(F.interpolate(q, size=(R, R), mode="bicubic", align_corners=True) for q in pyr)
        return h[0, 0] * m

    def render(h: torch.Tensor):
        n = _normals(h, dx)
        shading = (_sh_basis(n) @ sh).clamp(min=0)
        albedo = torch.sigmoid(F.interpolate(albedo_lr, size=(R, R), mode="bicubic", align_corners=True))[0]
        albedo = albedo.permute(1, 2, 0)
        img = albedo * shading[..., None]
        if p.specular:
            L = spec_dir / spec_dir.norm()
            H = (L + view) / (L + view).norm()
            spec = (n @ H).clamp(min=1e-6) ** torch.exp(spec_p)
            img = img + torch.sigmoid(spec_k) * spec[..., None]
        return img, albedo, shading, n

    hist: list[float] = []
    n_in = m_in.sum().clamp(min=1)
    for it in range(p.iters):
        opt.zero_grad(set_to_none=True)
        h = height()
        img, albedo, shading, n = render(h)
        photo = (_charbonnier(img - target).sum(-1) * m_in).sum() / n_in
        lap = _laplacian(h)
        smooth = ((lap / dx) ** 2 * m_in).sum() / n_in  # squared change of slope between neighbours
        raised = ((F.relu(-h) / dx) ** 2 * m).sum() / n_in * dx ** 2
        border = ((h / dx) ** 2 * m_ring).sum() / m_ring.sum().clamp(min=1) * dx ** 2
        mag = (h ** 2 * m).sum() / n_in
        grad = (((n[..., 0] / n[..., 2]) ** 2 + (n[..., 1] / n[..., 2]) ** 2) * m_in).sum() / n_in
        hc = h - (h * m_in).sum() / n_in
        lum_prior = (((hc - p.lum_scale * dog) / p.lum_scale) ** 2 * m_in).sum() / n_in
        loss = (photo + p.w_lum * lum_prior + p.w_smooth * smooth + p.w_raised * raised + p.w_border * border + p.w_mag * mag
                + p.w_grad * grad)
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
        if log and (it % 100 == 0 or it == p.iters - 1):
            log(f"    it {it:4d} loss {hist[-1]:.5f} photo {float(photo.detach()):.5f}")

    with torch.no_grad():
        h = height()
        img, albedo, shading, n = render(h)
        rmse = float(torch.sqrt((((img - target) ** 2).mean(-1) * m_in).sum() / n_in))
    return ReliefResult(
        height=h.cpu().numpy().astype(np.float32),
        albedo=albedo.cpu().numpy().astype(np.float32),
        shading=shading.cpu().numpy().astype(np.float32),
        render=img.clamp(0, 1).cpu().numpy().astype(np.float32),
        normals=n.cpu().numpy().astype(np.float32),
        loss_history=hist,
        photometric_rmse=rmse,
        seconds=time.time() - t0,
        light_sh=[float(v) for v in sh.detach().cpu()],
    )


def normalise_height(h: np.ndarray, mask: np.ndarray, target_max: float) -> np.ndarray:
    """Shift the field to 0 and scale so the 99th percentile of the relief equals target_max."""
    vals = h[mask]
    if vals.size == 0:
        return h
    base = np.percentile(vals, 5)
    out = (h - base) * mask
    top = np.percentile(out[mask], 99.5)
    if top > 1e-8:
        out = out * (target_max / top)
    return np.clip(out, 0, target_max * 1.5).astype(np.float32)
