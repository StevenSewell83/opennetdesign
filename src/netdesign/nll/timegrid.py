"""Evaluate NonLinLoc travel-time grids at arbitrary source positions.

Time grids hold node values at ``(x0 + i*dx, y0 + j*dy, z0 + k*dz)``. A ``TIME2D`` grid
(``nx = 1``) is a function of horizontal distance from the station (along ``y``) and depth.
Coordinates are local km (see :mod:`netdesign.frame`); depth is km below sea level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from netdesign.errors import DataError
from netdesign.nll.grid import NLLGrid, read_grid

__all__ = ["TravelTimeGrid"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TravelTimeGrid:
    """A station's travel-time grid with trilinear (3D) or bilinear (2D) interpolation."""

    grid: NLLGrid
    _interp: RegularGridInterpolator

    @classmethod
    def read(cls, root: str | Path) -> TravelTimeGrid:
        """Read ``<root>.hdr/.buf``.

        Raises
        ------
        DataError
            If the grid is not a ``TIME``/``TIME2D`` grid with a station line.
        """
        g = read_grid(root)
        if g.grid_type not in ("TIME", "TIME2D") or g.source is None:
            raise DataError(f"{root}: not a travel-time grid with a station line")
        ys = g.y0 + np.arange(g.ny) * g.dy
        zs = g.z0 + np.arange(g.nz) * g.dz
        data = g.data.astype(np.float64)
        if g.grid_type == "TIME2D":
            interp = RegularGridInterpolator(
                (ys, zs), data[0], bounds_error=False, fill_value=np.nan
            )
        else:
            xs = g.x0 + np.arange(g.nx) * g.dx
            interp = RegularGridInterpolator(
                (xs, ys, zs), data, bounds_error=False, fill_value=np.nan
            )
        return cls(g, interp)

    @property
    def station(self) -> str:
        """Station label."""
        assert self.grid.source is not None
        return self.grid.source[0]

    @property
    def station_xyz(self) -> tuple[float, float, float]:
        """Station position, local km."""
        assert self.grid.source is not None
        return self.grid.source[1:]

    def at(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike) -> FloatArray:
        """Travel time (s) from sources at ``(x, y, z)`` to the station; NaN outside the grid."""
        xa, ya, za = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
        )
        if self.grid.grid_type == "TIME2D":
            sx, sy, _ = self.station_xyz
            pts = np.stack([np.hypot(xa - sx, ya - sy), za], axis=-1)
        else:
            pts = np.stack([xa, ya, za], axis=-1)
        result: FloatArray = self._interp(pts)
        return result
