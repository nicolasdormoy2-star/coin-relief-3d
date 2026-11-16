"""Debug sheet: photo | re-render | albedo | shading | height | geometry relit from 2 new directions."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _relit(h: np.ndarray, mask: np.ndarray, light, dx: float) -> np.ndarray:
    gy, gx = np.gradient(h, dx)
    gy = -gy
    n = np.stack([-gx, -gy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.asarray(light, np.float32)
    L /= np.linalg.norm(L)
    s = np.clip(n @ L, 0, 1) * 0.85 + 0.15
    return np.where(mask, s, 1.0)


def _heat(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = h[mask]
    lo, hi = (np.percentile(v, 1), np.percentile(v, 99)) if v.size else (0, 1)
    t = np.clip((h - lo) / max(hi - lo, 1e-8), 0, 1)
    rgb = np.stack([t, t ** 2, 0.3 + 0.4 * (1 - t)], -1)
    return np.where(mask[..., None], rgb, 1.0)


def debug_sheet(photo, mask, res, height_norm, path, title: str = "") -> None:
    R = photo.shape[0]
    dx = 2.0 / R
    gray = lambda a: np.repeat(a[..., None], 3, -1)
    tiles = [
        ("photo", photo),
        ("re-render", res.render),
        ("albedo", res.albedo),
        ("shading", gray(np.clip(res.shading / max(res.shading.max(), 1e-6), 0, 1))),
        ("height", _heat(height_norm, mask)),
        ("relit, side light", gray(_relit(height_norm * 4, mask, (0.7, 0.2, 0.7), dx))),
        ("relit, grazing light", gray(_relit(height_norm * 4, mask, (0.0, 0.8, 0.45), dx))),
    ]
    W = len(tiles) * R
    sheet = Image.new("RGB", (W, R + 22), "white")
    d = ImageDraw.Draw(sheet)
    for i, (name, arr) in enumerate(tiles):
        sheet.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)), (i * R, 22))
        d.text((i * R + 4, 4), name, fill="black")
    if title:
        d.text((W - 8 * len(title) - 4, 4), title, fill="gray")
    sheet.save(path)
