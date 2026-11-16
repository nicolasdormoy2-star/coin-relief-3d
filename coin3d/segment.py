"""Coin segmentation: find the coin on a (roughly uniform) studio background, crop a centred square."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class CoinCrop:
    rgb: np.ndarray        # (R, R, 3) float32 in [0, 1], background whitened
    mask: np.ndarray       # (R, R) bool, coin pixels
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) of the square crop in the source image
    background: np.ndarray  # (3,) float32 estimated background colour

    @property
    def coverage(self) -> float:
        return float(self.mask.mean())


def _border_pixels(img: np.ndarray, frac: float = 0.03) -> np.ndarray:
    h, w = img.shape[:2]
    b = max(2, int(min(h, w) * frac))
    return np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                           img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = np.pad(m, 1).copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return mask | holes


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == k


def coin_mask(img: np.ndarray, work_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask at source resolution, background rgb). img: float32 [0,1] (H, W, 3)."""
    h, w = img.shape[:2]
    s = work_size / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_RGB2LAB)
    border = _border_pixels(lab)
    bg_lab = np.median(border, axis=0)
    # distance to background, normalised by the background's own spread (robust to noisy/gradient backgrounds)
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    spread = np.percentile(np.linalg.norm(border - bg_lab, axis=1), 95)
    d8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu / 255 * dist.max(), spread * 1.5, 6.0)
    mask = dist > thr
    k = max(3, int(min(small.shape[:2]) * 0.012)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(bool)
    mask = _fill_holes(_largest_component(mask))
    # back to source resolution with a smooth edge
    big = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=max(1.0, 1.0 / s))
    bg_rgb = np.median(_border_pixels(img), axis=0).astype(np.float32)
    return big > 0.5, bg_rgb


def crop_coin(img: np.ndarray, size: int = 512, pad: float = 0.03) -> CoinCrop:
    """Segment the coin and return a centred square crop resized to (size, size)."""
    mask, bg = coin_mask(img)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("no coin found")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * (1 + pad)
    X0, Y0, X1, Y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    # pad source so the square never leaves the image
    P = max(0, -X0, -Y0, X1 - img.shape[1], Y1 - img.shape[0])
    src = np.pad(img, ((P, P), (P, P), (0, 0)), mode="edge")
    msk = np.pad(mask, P)
    sq = src[Y0 + P:Y1 + P, X0 + P:X1 + P]
    sm = msk[Y0 + P:Y1 + P, X0 + P:X1 + P].astype(np.float32)
    rgb = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(sm, (size, size), interpolation=cv2.INTER_LINEAR) > 0.5
    rgb = np.where(m[..., None], rgb, 1.0).astype(np.float32)
    return CoinCrop(rgb=rgb, mask=m, bbox=(X0, Y0, X1, Y1), background=bg)


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
