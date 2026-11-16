"""Synthetic checks: render a known relief, recover it, build the mesh."""

import numpy as np
import pytest
import torch

from coin3d.mesh import CoinPhysical, Face, build_coin_mesh
from coin3d.relief import ReliefParams, normalise_height, reconstruct_relief
from coin3d.segment import crop_coin

R = 128


def _synthetic_coin():
    """Coin-like scene: flat field, raised designs with steep sides (bust, letters, bead ring)."""
    from scipy.ndimage import gaussian_filter
    y, x = np.mgrid[1:-1:R * 1j, -1:1:R * 1j]
    mask = x ** 2 + y ** 2 < 0.85 ** 2
    design = ((x + 0.05) ** 2 / 0.12 + (y - 0.05) ** 2 / 0.2 < 1).astype(float)      # bust
    for k in range(5):                                                                   # "letters"
        design[(np.abs(x - (-0.4 + 0.2 * k)) < 0.03) & (y > -0.72) & (y < -0.55)] = 1
    ang = np.arctan2(y, x)
    design[(np.abs(np.sqrt(x ** 2 + y ** 2) - 0.75) < 0.03) & (np.cos(ang * 24) > 0.3)] = 1  # beads
    h = gaussian_filter(design, 1.5) * 0.04 * mask
    gy, gx = np.gradient(h, 2 / R)
    n = np.stack([-gx, gy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([-0.45, 0.45, 0.77]); L /= np.linalg.norm(L)
    shade = np.clip(n @ L, 0, 1) * 0.8 + 0.15
    rgb = np.where(mask[..., None], shade[..., None] * np.array([0.85, 0.7, 0.35]), 1.0).astype(np.float32)
    return rgb, mask, h


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_inverse_renderer_recovers_raised_shape():
    """Pure shading (no wear prior): checks sign conventions, light model and slope integration."""
    rgb, mask, h_true = _synthetic_coin()
    res = reconstruct_relief(rgb, mask, ReliefParams(iters=400, param="slopes", w_lum=0, w_grad=0.05), device=DEV)
    h = normalise_height(res.height, mask, 0.04)
    corr = np.corrcoef(h[mask], h_true[mask])[0, 1]
    assert corr > 0.6, corr  # same shape, and not inverted (convex/concave ambiguity)
    assert res.photometric_rmse < 0.01


def test_default_params_fit_the_photo():
    rgb, mask, _ = _synthetic_coin()
    res = reconstruct_relief(rgb, mask, ReliefParams(iters=200), device=DEV)
    assert res.photometric_rmse < 0.03
    assert np.isfinite(res.height).all()


def test_integrate_slopes_roundtrip():
    from coin3d.relief import integrate_slopes
    _, _, h = _synthetic_coin()
    gr, gc = np.gradient(h, 2 / R)
    rec = integrate_slopes(torch.tensor(gc), torch.tensor(gr), 2 / R).numpy()
    assert np.corrcoef(rec.ravel(), h.ravel())[0, 1] > 0.99


def test_crop_coin_on_white_background():
    img = np.ones((300, 400, 3), np.float32)
    yy, xx = np.mgrid[:300, :400]
    disk = (xx - 250) ** 2 + (yy - 140) ** 2 < 90 ** 2
    img[disk] = [0.6, 0.5, 0.3]
    c = crop_coin(img, 128)
    assert 0.6 < c.coverage < 0.8  # disk of the square crop ~ pi/4
    assert c.mask[64, 64] and not c.mask[2, 2]


def test_mesh_orientation_and_scale():
    rgb, mask, h = _synthetic_coin()
    face = Face(rgb=rgb, mask=mask, height=h * 10)
    mesh = build_coin_mesh(face, face, CoinPhysical(diameter_mm=20, thickness_mm=2), metal="Or",
                           rings=16, n_theta=64)
    ext = mesh.bounds[1] - mesh.bounds[0]
    assert abs(max(ext[:2]) - 0.020) < 0.001          # metres
    c, fn = mesh.triangles_center, mesh.face_normals
    assert fn[c[:, 2] > 0.0011, 2].mean() > 0.8      # obverse faces +Z
    assert fn[c[:, 2] < -0.0011, 2].mean() < -0.8    # reverse faces -Z
    side = np.abs(c[:, 2]) < 0.0005
    radial = np.sum(fn[side, :2] * c[side, :2], 1) / np.linalg.norm(c[side, :2], axis=1)
    assert radial.mean() > 0.9                        # wall faces outward


def test_physical_estimate_from_weight():
    p = CoinPhysical.estimate(19.5, 7.3, "Or")  # aureus-like
    assert 1.0 < p.thickness_mm < 1.6
    assert CoinPhysical.estimate(None, None, None).diameter_mm == 20.0
