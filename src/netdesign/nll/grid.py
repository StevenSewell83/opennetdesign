"""Read NonLinLoc ``.hdr`` / ``.buf`` grid pairs.

The buffer is little-endian float32 (or float64 if the header says ``DOUBLE``) with ``z``
varying fastest, i.e. C order ``(nx, ny, nz)``. Vel2Grid samples the model at cell centres:
the value at node ``(i, j, k)`` is the velocity at ``z0 + (k + 0.5) * dz``. Time grids
(``TIME``, ``TIME2D``) are node values at ``z0 + k * dz`` and carry a source (station) line
``label x y z`` after the first header line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from netdesign.errors import DataError

__all__ = ["NLLGrid", "read_grid"]


@dataclass(frozen=True)
class NLLGrid:
    """One grid: header values and data of shape ``(nx, ny, nz)``."""

    nx: int
    ny: int
    nz: int
    x0: float
    y0: float
    z0: float
    dx: float
    dy: float
    dz: float
    grid_type: str
    transform: str
    data: npt.NDArray[np.floating]
    source: tuple[str, float, float, float] | None = None

    def velocity(self) -> npt.NDArray[np.float64]:
        """Velocity (km/s) from a ``SLOW_LEN``, ``SLOWNESS`` or ``VELOCITY`` grid."""
        values = self.data.astype(np.float64)
        if self.grid_type == "SLOW_LEN":
            result: npt.NDArray[np.float64] = self.dx / values
        elif self.grid_type == "SLOWNESS":
            result = 1.0 / values
        elif self.grid_type == "VELOCITY":
            result = values
        else:
            raise DataError(f"grid type {self.grid_type} is not a velocity-type grid")
        return result


def read_grid(root: str | Path) -> NLLGrid:
    """Read ``<root>.hdr`` and ``<root>.buf``.

    Raises
    ------
    DataError
        If either file is missing or their sizes disagree.
    """
    root = Path(root)
    hdr, buf = root.with_name(root.name + ".hdr"), root.with_name(root.name + ".buf")
    try:
        lines = hdr.read_text().splitlines()
        tokens = lines[0].split()
        nx, ny, nz = (int(t) for t in tokens[:3])
        x0, y0, z0, dx, dy, dz = (float(t) for t in tokens[3:9])
        grid_type = tokens[9]
        precision = tokens[10] if len(tokens) > 10 else "FLOAT"
    except (OSError, IndexError, ValueError) as exc:
        raise DataError(f"{hdr}: cannot read NonLinLoc header ({exc})") from exc
    transform = next((ln.strip() for ln in lines[1:] if ln.startswith("TRANSFORM")), "")
    source = None
    if len(lines) > 1 and lines[1].strip() and not lines[1].startswith("TRANSFORM"):
        try:
            label, *xyz = lines[1].split()[:4]
            sx, sy, sz = (float(t) for t in xyz)
        except ValueError as exc:
            raise DataError(f"{hdr}: bad source line {lines[1]!r}") from exc
        source = (label, sx, sy, sz)
    dtype = "<f8" if precision == "DOUBLE" else "<f4"
    n = nx * ny * nz
    try:
        # Like NonLinLoc, read only the first n values: Grid2Time writes its whole working
        # buffer (two x-sheets) for 2D time grids whose header says nx = 1.
        data = np.fromfile(buf, dtype=dtype, count=n)
    except OSError as exc:
        raise DataError(f"{buf}: cannot read grid buffer ({exc})") from exc
    if data.size != n:
        raise DataError(f"{buf}: {data.size} values, header expects {n}")
    return NLLGrid(
        nx,
        ny,
        nz,
        x0,
        y0,
        z0,
        dx,
        dy,
        dz,
        grid_type,
        transform,
        data.reshape(nx, ny, nz),
        source,
    )
