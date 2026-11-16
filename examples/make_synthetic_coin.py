"""Render a synthetic coin "photo" (obverse + reverse) whose true relief is known.

    python examples/make_synthetic_coin.py examples/synthetic
    python -m coin3d examples/synthetic/obverse.png examples/synthetic/reverse.png -o out --diameter 20 --metal bronze

The relief is drawn here (bust, legend, bead ring, wreath), shaded with a top-left studio light,
placed on a white background, then reconstructed by coin3d without being told the true heights.
`true_obverse_height.npy` lets you compare the recovered height map with the ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

N = 900            # photo size in pixels
COIN_R = 0.78      # coin radius, fraction of the half-width


def _grid():
    y, x = np.mgrid[1:-1:N * 1j, -1:1:N * 1j]
    return x, y, np.hypot(x, y)


def _common(x, y, r):
    d = np.zeros_like(x)
    ang = np.arctan2(y, x)
    d[(np.abs(r - 0.72) < 0.018) & (np.cos(ang * 60) > 0.2)] = 0.8          # bead ring
    d[np.abs(r - COIN_R + 0.015) < 0.012] = 0.6                               # raised rim
    return d


def obverse_relief():
    x, y, r = _grid()
    d = _common(x, y, r)
    head = ((x + 0.02) / 0.24) ** 2 + ((y - 0.12) / 0.28) ** 2 < 1
    neck = (np.abs(x + 0.04) < 0.12) & (y < -0.05) & (y > -0.38)
    bust = ((x / 0.42) ** 2 + ((y + 0.48) / 0.2) ** 2 < 1) & (y > -0.55)
    nose = ((x - 0.24) / 0.06) ** 2 + ((y - 0.08) / 0.05) ** 2 < 1
    d = np.maximum(d, (head | neck | bust | nose) * 1.0)
    hair = head & (np.sin((x * 3 + y * 5) * 14) > 0.4) & (x < 0.08)            # hair strands, a bit higher
    d[hair] = 1.25
    eye = ((x - 0.12) / 0.04) ** 2 + ((y - 0.16) / 0.02) ** 2 < 1
    d[eye] = 0.7
    ang = np.arctan2(y, x)
    for k in range(13):                                                        # legend: blocky "letters", top arc
        a = np.deg2rad(205 - k * 19)
        ca = np.arctan2(np.sin(ang - a), np.cos(ang - a))
        d[(np.abs(r - 0.6) < 0.05) & (np.abs(ca) < 0.07) & ~((np.abs(r - 0.6) < 0.02) & (k % 3 == 0))] = 0.9
    return d


def reverse_relief():
    x, y, r = _grid()
    d = _common(x, y, r)
    ang = np.arctan2(y, x)
    for k in range(18):                                                        # wreath of leaves
        a = np.deg2rad(-80 + k * 20) if k < 9 else np.deg2rad(260 - (k - 9) * 20)
        cx, cy = 0.45 * np.cos(a), 0.45 * np.sin(a)
        u = (x - cx) * np.cos(a) + (y - cy) * np.sin(a)
        v = -(x - cx) * np.sin(a) + (y - cy) * np.cos(a)
        d[(u / 0.06) ** 2 + (v / 0.13) ** 2 < 1] = 0.8
    d[(np.abs(x) < 0.05) & (np.abs(y) < 0.25)] = 1.0                           # "I" in the centre
    d[(np.abs(y - 0.22) < 0.04) & (np.abs(x) < 0.16)] = 1.0
    d[(np.abs(y + 0.22) < 0.04) & (np.abs(x) < 0.16)] = 1.0
    return d


def render(design: np.ndarray, seed: int):
    x, y, r = _grid()
    mask = r < COIN_R
    h = gaussian_filter(design, 2.5) * 0.035 * mask
    rng = np.random.default_rng(seed)
    h += gaussian_filter(rng.normal(size=h.shape), 6) * 0.0015 * mask             # wear / surface noise
    gy, gx = np.gradient(h, 2 / N)
    n = np.stack([-gx, gy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([-0.45, 0.45, 0.77]); L /= np.linalg.norm(L)
    H = L + np.array([0, 0, 1.0]); H /= np.linalg.norm(H)
    diffuse = np.clip(n @ L, 0, 1)
    spec = np.clip(n @ H, 0, 1) ** 40
    patina = np.clip(1 + 4 * gaussian_filter(rng.normal(size=h.shape), 40), 0.9, 1.1)  # mild colour variation
    patina = patina * (0.85 + 0.3 * h / h.max())                                 # worn high points are brighter
    base = np.array([0.72, 0.52, 0.30])                                          # bronze
    shade = (diffuse * 0.75 + 0.2)[..., None] * base * patina[..., None] + 0.25 * spec[..., None]
    img = np.where(mask[..., None], np.clip(shade, 0, 1), 0.97)
    return (img * 255).astype(np.uint8), h


def main(out: str = "examples/synthetic") -> None:
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    for name, design, seed in (("obverse", obverse_relief(), 1), ("reverse", reverse_relief(), 2)):
        img, h = render(design, seed)
        Image.fromarray(img).save(o / f"{name}.png")
        np.save(o / f"true_{name}_height.npy", h.astype(np.float32))
    print(f"wrote {o}/obverse.png, reverse.png (+ true heights)")


if __name__ == "__main__":
    main(*sys.argv[1:])
