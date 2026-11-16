"""Coin photos (obverse, optional reverse) -> relief height maps -> real-scale textured coin (.glb)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .debug import debug_sheet
from .mesh import CoinPhysical, Face, build_coin_mesh, export_glb
from .relief import ReliefParams, normalise_height, reconstruct_relief
from .segment import coin_mask, crop_coin, load_rgb
from .watermark import remove_watermark

TEXTURE_SIZE = 1024


def prepare(path: Path, resolution: int, watermark: bool = False):
    """Load a photo, cut the coin out of its background, return (texture crop, small crop, inpainted px)."""
    img = load_rgb(path)
    wm_px = 0
    if watermark:
        cm, _ = coin_mask(img)
        img, wm = remove_watermark(img, cm)
        wm_px = int(wm.sum())
    return crop_coin(img, TEXTURE_SIZE), crop_coin(img, resolution), wm_px


def reconstruct(obverse: Path, reverse: Path | None, out: Path, *, diameter_mm: float | None = None,
                weight_g: float | None = None, metal: str | None = None, resolution: int = 384,
                iters: int = 600, debug: bool = True, watermark: bool = False, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    phys = CoinPhysical.estimate(diameter_mm, weight_g, metal)
    params = ReliefParams(iters=iters)
    faces, report = {}, {"physical": phys.__dict__, "relief_mm": phys.relief_mm, "sides": {}}
    t0 = time.time()
    for side, path in (("obverse", obverse), ("reverse", reverse)):
        if path is None:
            continue
        tex, small, wm_px = prepare(Path(path), resolution, watermark)
        res = reconstruct_relief(small.rgb, small.mask, params)
        h_mm = normalise_height(res.height, small.mask, phys.relief_mm)
        faces[side] = Face(rgb=tex.rgb, mask=small.mask, height=h_mm)
        np.save(out / f"{side}_height_mm.npy", h_mm)
        if debug:
            debug_sheet(small.rgb, small.mask, res, h_mm / max(phys.relief_mm, 1e-6) * 0.04,
                        out / f"debug_{side}.png", title=side)
        report["sides"][side] = {"rmse": round(res.photometric_rmse, 4), "seconds": round(res.seconds, 1),
                                 "coverage": round(small.coverage, 3), "watermark_px": wm_px}
        log(f"{side}: rmse={res.photometric_rmse:.4f} in {res.seconds:.1f}s")
    mesh = build_coin_mesh(faces["obverse"], faces.get("reverse"), phys, metal=metal)
    export_glb(mesh, out / "model.glb")
    Image.fromarray((faces["obverse"].rgb * 255).astype(np.uint8)).resize((320, 320), Image.LANCZOS) \
        .save(out / "thumb.jpg", quality=85)
    report.update({"seconds": round(time.time() - t0, 1), "vertices": int(len(mesh.vertices)),
                   "glb_bytes": (out / "model.glb").stat().st_size})
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"wrote {out / 'model.glb'} ({report['glb_bytes'] // 1024} KB, {report['vertices']} vertices)")
    return report
