"""Write NonLinLoc model grids (``SLOW_LEN``) directly, for 3D models Vel2Grid cannot build.

Values are ``dx / v`` at cell centres in C order ``(nx, ny, nz)``, little-endian float32,
with the same header Vel2Grid writes, so Grid2Time reads them unchanged. For a layered model
without heterogeneity the result matches Vel2Grid (tests/unit/test_heterogeneity.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from netdesign.frame import GridGeometry, LocalFrame

__all__ = ["write_model_grid"]


def write_model_grid(
    root: Path, velocity: npt.NDArray[np.float64], geom: GridGeometry, frame: LocalFrame
) -> Path:
    """Write ``<root>.hdr`` / ``<root>.buf`` from velocities (km/s) of shape (nx, ny, nz)."""
    if velocity.shape != (geom.nx, geom.ny, geom.nz):
        raise ValueError(f"velocity shape {velocity.shape} does not match the grid")
    root.parent.mkdir(parents=True, exist_ok=True)
    d = geom.dx
    header = (
        f"{geom.nx} {geom.ny} {geom.nz}  {geom.x0:.6f} {geom.y0:.6f} {geom.z0:.6f}  "
        f"{d:.6f} {d:.6f} {d:.6f} SLOW_LEN FLOAT\n" + frame.nll_transform_header() + "\n"
    )
    root.with_name(root.name + ".hdr").write_text(header)
    (d / velocity).astype("<f4").tofile(root.with_name(root.name + ".buf"))
    return root
