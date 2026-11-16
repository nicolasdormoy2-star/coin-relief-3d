"""python -m coin3d OBVERSE [REVERSE] -o OUT_DIR [--diameter MM] [--weight G] [--metal NAME]"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="coin3d", description="Coin photo(s) -> 3D coin (.glb)")
    p.add_argument("obverse", type=Path, help="photo of the obverse (heads), coin on a plain background")
    p.add_argument("reverse", type=Path, nargs="?", help="photo of the reverse (optional: flat back otherwise)")
    p.add_argument("-o", "--out", type=Path, default=Path("out"), help="output folder (default: out/)")
    p.add_argument("--diameter", type=float, help="diameter in mm (default 20)")
    p.add_argument("--weight", type=float, help="weight in g (with --metal, gives the thickness)")
    p.add_argument("--metal", help="gold, silver, bronze, copper, billon, electrum, lead (or French names)")
    p.add_argument("--iters", type=int, default=600, help="gradient steps per face")
    p.add_argument("--resolution", type=int, default=384, help="height map resolution in pixels")
    p.add_argument("--watermark", action="store_true", help="inpaint a light text watermark at the bottom")
    p.add_argument("--no-debug", action="store_true", help="skip the debug_*.png sheets")
    a = p.parse_args(argv)

    from .pipeline import reconstruct

    reconstruct(a.obverse, a.reverse, a.out, diameter_mm=a.diameter, weight_g=a.weight, metal=a.metal,
                resolution=a.resolution, iters=a.iters, debug=not a.no_debug, watermark=a.watermark)


if __name__ == "__main__":
    main()
