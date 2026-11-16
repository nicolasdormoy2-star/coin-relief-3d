"""Remove a light text watermark (bottom-centre, light, desaturated) by inpainting.

The watermark would otherwise be read as relief by the shape-from-shading step.
"""

from __future__ import annotations

import cv2
import numpy as np

# search window, as fractions of the source photo (x0, y0, x1, y1)
DEFAULT_WINDOW = (0.33, 0.80, 0.67, 0.97)


def watermark_mask(img: np.ndarray, coin: np.ndarray | None = None, window=DEFAULT_WINDOW) -> np.ndarray:
    """Boolean mask of watermark pixels lying on the coin. img: float32 RGB [0,1]; coin: bool mask."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(window[0] * w), int(window[1] * h), int(window[2] * w), int(window[3] * h))
    roi = img[y0:y1, x0:x1]
    mn, mx = roi.min(axis=2), roi.max(axis=2)
    k = max(21, int(h * 0.02)) | 1
    local = cv2.medianBlur((mn * 255).astype(np.uint8), k).astype(np.float32) / 255
    # text strokes: near-white, desaturated, clearly brighter than their surroundings
    cand = ((mn - local) > 0.18) & ((mx - mn) < 0.09) & (mn > 0.62)
    if coin is not None:
        inner = cv2.erode(coin.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        cand &= inner[y0:y1, x0:x1]
    # 1) strict detection -> locate the text block (densest horizontal band of glyph-like blobs)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros(cand.shape, np.uint8)
    if n > 1:
        ok = (stats[1:, cv2.CC_STAT_AREA] >= 3) & (stats[1:, cv2.CC_STAT_HEIGHT] < roi.shape[0] * 0.3)
        if ok.sum() >= 4:
            ys, xs = cents[1:, 1][ok], cents[1:, 0][ok]
            hist, edges = np.histogram(ys, bins=24, range=(0, roi.shape[0]))
            c = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
            half = roi.shape[0] * 0.17
            sel = np.abs(ys - c) < half
            if sel.sum() >= 4:
                mx_ = int(w * 0.03)
                bx0, bx1 = max(0, int(xs[sel].min()) - mx_), min(roi.shape[1], int(xs[sel].max()) + mx_)
                by0, by1 = max(0, int(c - half)), min(roi.shape[0], int(c + half))
                # 2) looser threshold inside the text block (second line is fainter)
                loose = ((mn - local) > 0.07) & ((mx - mn) < 0.13) & (mn > 0.45)
                if coin is not None:
                    loose &= inner[y0:y1, x0:x1]
                keep[by0:by1, bx0:bx1] = loose[by0:by1, bx0:bx1]
    if keep.sum() < 30:
        keep[:] = 0
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    out = np.zeros((h, w), bool)
    out[y0:y1, x0:x1] = keep.astype(bool)
    return out


def remove_watermark(img: np.ndarray, coin: np.ndarray | None = None,
                     window=DEFAULT_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Return (cleaned image, mask of inpainted pixels)."""
    m = watermark_mask(img, coin, window)
    if not m.any():
        return img, m
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    radius = max(3, int(img.shape[0] * 0.004))
    clean = cv2.inpaint(u8, m.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return clean.astype(np.float32) / 255.0, m
