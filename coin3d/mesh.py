"""Height fields (obverse + reverse) -> closed, textured, real-scale coin mesh exported as GLB.

Conventions (see docs/catalog-format.md): metres, coin in the XY plane centred on the origin,
obverse facing +Z, reverse facing -Z, obverse "up" = +Y.

Geometry is a radial mesh following the actual outline of the coin (ancient coins are not round),
fine relief detail is carried by a tangent-space normal map baked from the full-resolution height.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import map_coordinates
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

DENSITY_G_CM3 = {"or": 19.3, "argent": 10.5, "électrum": 15.0, "electrum": 15.0, "bronze": 8.8,
                 "cuivre": 8.9, "billon": 9.5, "potin": 8.6, "orichalque": 8.5, "plomb": 11.3,
                 # English names
                 "gold": 19.3, "silver": 10.5, "copper": 8.9, "orichalcum": 8.5, "lead": 11.3}

# viewer-friendly PBR presets (a web viewer usually has no environment map, so keep metalness moderate)
MATERIAL_PRESET = {"or": (0.45, 0.35), "argent": (0.4, 0.4), "électrum": (0.45, 0.38),
                   "gold": (0.45, 0.35), "silver": (0.4, 0.4), "electrum": (0.45, 0.38)}
DEFAULT_MATERIAL = (0.25, 0.55)


@dataclass
class Face:
    rgb: np.ndarray       # (T, T, 3) texture crop, float [0,1]
    mask: np.ndarray      # (R, R) bool, at height-map resolution
    height: np.ndarray    # (R, R) relief in millimetres (0 = field)


@dataclass
class CoinPhysical:
    diameter_mm: float = 20.0
    thickness_mm: float = 1.6

    @staticmethod
    def estimate(diameter_mm: float | None, weight_g: float | None, metal: str | None) -> "CoinPhysical":
        d = diameter_mm if diameter_mm and 5 <= diameter_mm <= 60 else 20.0
        t = 1.6
        rho = DENSITY_G_CM3.get((metal or "").strip().lower())
        if weight_g and rho:
            area_cm2 = np.pi * (d / 20) ** 2  # radius in cm
            t = float(np.clip(weight_g / rho / area_cm2 * 10, 0.6, 6.0))  # mm
        return CoinPhysical(d, t)

    @property
    def relief_mm(self) -> float:
        return float(np.clip(0.28 * self.thickness_mm, 0.12, 0.9))


def outline_polar(mask: np.ndarray, n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    """Radius of the coin outline for n_theta angles, in crop units ([-1, 1] square). Returns (centre xy, r)."""
    R = mask.shape[0]
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    steps = np.linspace(0, R * 0.75, R * 2)
    r = np.zeros(n_theta)
    for i, t in enumerate(th):
        px = cx + steps * np.cos(t)
        py = cy - steps * np.sin(t)  # image rows go down
        inside = map_coordinates(mask.astype(np.float32), [py, px], order=1, mode="constant") > 0.5
        idx = np.nonzero(inside)[0]
        r[i] = steps[idx.max()] if len(idx) else 0.0
    # light circular smoothing to remove pixel jaggies
    k = np.array([1, 2, 3, 2, 1], np.float64)
    k /= k.sum()
    r = np.convolve(np.concatenate([r[-2:], r, r[:2]]), k, mode="valid")
    to_unit = 2.0 / R
    centre = np.array([(cx + 0.5) * to_unit - 1, 1 - (cy + 0.5) * to_unit])
    return centre, r * to_unit


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a (R, R) map at crop-unit coordinates (x right, y up, in [-1, 1])."""
    R = img.shape[0]
    col = (x + 1) / 2 * R - 0.5
    row = (1 - y) / 2 * R - 0.5
    return map_coordinates(img, [row, col], order=1, mode="nearest")


def normal_map(height_mm: np.ndarray, mask: np.ndarray, px_mm: float, out_size: int) -> Image.Image:
    """glTF tangent-space normal map (+X right, +Y up in image space)."""
    h = cv2.resize(height_mm, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(mask.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    px = px_mm * height_mm.shape[0] / out_size
    gy, gx = np.gradient(h * m, px)
    n = np.stack([-gx, gy, np.ones_like(h)], -1)  # gy: rows go down -> +Y up flips sign twice
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).round().astype(np.uint8))


def _bleed(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace background pixels by the nearest coin pixel (no white halo when sampling near the rim)."""
    from scipy.ndimage import distance_transform_edt
    m = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool)  # drop the anti-aliased edge too
    if m.all() or not m.any():
        return rgb
    _, (iy, ix) = distance_transform_edt(~m, return_indices=True)
    return rgb[iy, ix]


def _as_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf)


def build_coin_mesh(obverse: Face, reverse: Face | None, phys: CoinPhysical, metal: str | None = None,
                    rings: int = 48, n_theta: int = 192) -> trimesh.Trimesh:
    R = obverse.mask.shape[0]
    centre, r_out = outline_polar(obverse.mask, n_theta)
    scale_mm = phys.diameter_mm / (2 * r_out.max())       # crop units -> mm
    half_t = phys.thickness_mm / 2

    # --- radial grid in obverse crop units
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    s = np.linspace(0, 1, rings + 1)[1:]                  # ring fractions (centre handled separately)
    rr = s[:, None] * r_out[None, :]
    X = centre[0] + rr * np.cos(th)[None, :]
    Y = centre[1] + rr * np.sin(th)[None, :]
    px, py = np.concatenate([[centre[0]], X.ravel()]), np.concatenate([[centre[1]], Y.ravel()])

    def ring_idx(k, j):  # k in 1..rings, j in 0..n_theta-1  -> vertex index (0 = centre)
        return 1 + (k - 1) * n_theta + (j % n_theta)

    tris = []
    for j in range(n_theta):
        tris.append((0, ring_idx(1, j), ring_idx(1, j + 1)))
    for k in range(1, rings):
        for j in range(n_theta):
            a, b = ring_idx(k, j), ring_idx(k, j + 1)
            c, d = ring_idx(k + 1, j), ring_idx(k + 1, j + 1)
            tris += [(a, c, d), (a, d, b)]
    tris = np.array(tris, np.int64)          # counter-clockwise seen from +Z
    nv = len(px)

    # --- back sampling: the reverse photo has its own outline. Seen from behind, world angle th maps to
    # photo angle pi - th (mirror). Map ring fraction s of the obverse outline onto fraction s of the
    # reverse outline so the reverse texture fills the mesh exactly (no background showing at the rim).
    s_all = np.concatenate([[0.0], np.repeat(s, n_theta)])
    j_all = np.concatenate([[0], np.tile(np.arange(n_theta), rings)])
    if reverse is not None:
        cb, rb = outline_polar(reverse.mask, n_theta)
        jm = (n_theta // 2 - j_all) % n_theta            # mirrored angle index (n_theta even)
        bx = cb[0] + s_all * rb[jm] * np.cos(th[jm])
        by = cb[1] + s_all * rb[jm] * np.sin(th[jm])
    else:
        bx, by = px, py

    hf = _sample(obverse.height, px, py)
    hb = _sample(reverse.height, bx, by) if reverse is not None else np.zeros_like(hf)
    to_m = 1e-3
    front_v = np.stack([(px - centre[0]) * scale_mm, (py - centre[1]) * scale_mm, half_t + hf], -1) * to_m
    back_v = np.stack([(px - centre[0]) * scale_mm, (py - centre[1]) * scale_mm, -half_t - hb], -1) * to_m

    # texture atlas: [obverse | reverse], uv origin bottom-left (trimesh converts to glTF)
    uf = np.stack([(px + 1) / 2 * 0.5, (py + 1) / 2], -1)
    ub = np.stack([0.5 + (bx + 1) / 2 * 0.5, (by + 1) / 2], -1) if reverse is not None else uf.copy()

    # --- wall: 4 rings (front rim, mid-front, mid-back, back rim). Each half shows a thin band of its
    # face's border (rim -> 93 % radius) instead of one stretched pixel.
    rim = [ring_idx(rings, j) for j in range(n_theta)]
    rim_xy = np.stack([px[rim], py[rim]], -1)
    band = 0.93
    uf_band = np.stack([(centre[0] + band * (rim_xy[:, 0] - centre[0]) + 1) / 2 * 0.5,
                        (centre[1] + band * (rim_xy[:, 1] - centre[1]) + 1) / 2], -1)
    if reverse is not None:
        jr = (n_theta // 2 - np.arange(n_theta)) % n_theta
        bbx, bby = cb[0] + band * rb[jr] * np.cos(th[jr]), cb[1] + band * rb[jr] * np.sin(th[jr])
        ub_band = np.stack([0.5 + (bbx + 1) / 2 * 0.5, (bby + 1) / 2], -1)
    else:
        ub_band = uf_band
    def wall_ring(z_mm, uv):
        v = np.stack([(rim_xy[:, 0] - centre[0]) * scale_mm, (rim_xy[:, 1] - centre[1]) * scale_mm, z_mm], -1)
        return v * to_m, uv
    w0 = wall_ring(half_t + hf[rim], uf[rim])
    w1 = wall_ring(np.zeros(n_theta), uf_band)
    w2 = wall_ring(np.zeros(n_theta), ub_band)
    w3 = wall_ring(-half_t - hb[rim], ub[rim])

    verts = np.concatenate([front_v, back_v, w0[0], w1[0], w2[0], w3[0]])
    uvs = np.concatenate([uf, ub, w0[1], w1[1], w2[1], w3[1]])
    faces = [tris, tris[:, ::-1] + nv]
    base = 2 * nv
    def strip(a0, b0, flip=False):
        out = []
        for j in range(n_theta):
            a, b = a0 + j, a0 + (j + 1) % n_theta
            c, d = b0 + j, b0 + (j + 1) % n_theta
            out += [(a, c, d), (a, d, b)] if not flip else [(a, d, c), (a, b, d)]
        return np.array(out, np.int64)
    faces += [strip(base, base + n_theta), strip(base + 2 * n_theta, base + 3 * n_theta)]  # outward
    faces = np.concatenate(faces)

    # --- textures
    T = obverse.rgb.shape[0]
    front_tex = _bleed(obverse.rgb, obverse.mask)
    back_tex = _bleed(reverse.rgb, reverse.mask) if reverse is not None else front_tex
    if back_tex.shape[0] != T:
        back_tex = cv2.resize(back_tex, (T, T), interpolation=cv2.INTER_AREA)
    atlas = Image.fromarray((np.concatenate([front_tex, back_tex], 1) * 255).astype(np.uint8))
    px_mm = scale_mm * 2 / R
    nf = normal_map(obverse.height, obverse.mask, px_mm, T)
    nb = normal_map(reverse.height, reverse.mask, px_mm, T) if reverse is not None else nf
    natlas = Image.new("RGB", (2 * T, T))
    natlas.paste(nf, (0, 0))
    natlas.paste(nb, (T, 0))

    atlas, natlas = _as_jpeg(atlas, 88), _as_jpeg(natlas, 92)  # trimesh keeps the source format in the GLB

    metal_key = (metal or "").strip().lower()
    metallic, rough = MATERIAL_PRESET.get(metal_key, DEFAULT_MATERIAL)
    mat = PBRMaterial(name="coin", baseColorTexture=atlas, normalTexture=natlas,
                      metallicFactor=metallic, roughnessFactor=rough)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False,
                           visual=TextureVisuals(uv=uvs, material=mat))
    return mesh


def export_glb(mesh: trimesh.Trimesh, path) -> None:
    _ = mesh.vertex_normals  # cache smooth normals so the exporter writes them (else viewers shade flat)
    data = mesh.export(file_type="glb", include_normals=True)
    with open(path, "wb") as f:
        f.write(data)
