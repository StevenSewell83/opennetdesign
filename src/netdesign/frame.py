"""Local Cartesian frame and NonLinLoc grid geometry.

The local frame is a transverse Mercator projection (WGS-84, scale 1, no false easting) about
an origin, matching NonLinLoc ``TRANS TRANS_MERC WGS-84 <lat> <long> 0.0``. ``x`` is km east,
``y`` km north, ``z`` km depth below sea level (positive down).

Grid geometry
-------------
- Horizontal region: bounding box of the candidates plus ``padding_km`` on every side.
- Top: the velocity model top, which must be at or above the highest candidate (NonLinLoc
  needs stations inside the grid). Bottom: at least ``depth_max_km``.
- ``2d`` (1D model, ``GTMODE GRID2D``): ``nx = 2``; ``y`` is horizontal distance from the
  station, long enough for the padded region's diagonal.
- ``3d``: covers the padded region, with the origin snapped to a multiple of ``dx_km``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import Transformer

from netdesign.crs import InputCRS
from netdesign.errors import ConfigError
from netdesign.stations import Candidates

__all__ = ["GridGeometry", "GridSpec", "LocalFrame", "Origin", "grid_geometry"]

FloatArray = npt.NDArray[np.float64]


class Origin(BaseModel):
    """Local-frame origin in the input ``crs``: ``lat``/``long`` (geographic) or
    ``easting``/``northing`` (projected)."""

    model_config = ConfigDict(extra="forbid")

    lat: float | None = Field(None, ge=-90, le=90)
    long: float | None = Field(None, ge=-180, le=180)
    easting: float | None = None
    northing: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        geo = self.lat is not None and self.long is not None
        proj = self.easting is not None and self.northing is not None
        if geo == proj:
            raise ConfigError("grid.origin: give either lat and long, or easting and northing")
        return self

    def to_wgs84(self, crs: InputCRS) -> tuple[float, float]:
        """``(lat, long)`` WGS-84.

        Raises
        ------
        ConfigError
            If the origin uses the other kind of coordinates than ``crs``.
        """
        if crs.geographic:
            if self.long is None or self.lat is None:
                raise ConfigError(f"grid.origin: {crs.name} needs lat and long")
            lon, lat = crs.to_wgs84(self.long, self.lat)
        else:
            if self.easting is None or self.northing is None:
                raise ConfigError(f"grid.origin: {crs.name} needs easting and northing")
            lon, lat = crs.to_wgs84(self.easting, self.northing)
        return float(lat), float(lon)


class GridSpec(BaseModel):
    """``grid`` config section."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["2d", "3d"] = "2d"
    dx_km: float = Field(0.05, gt=0)
    depth_max_km: float = Field(gt=0, description="grid bottom, km below sea level")
    padding_km: float = Field(2.0, ge=0)
    origin: Origin | None = None


@dataclass(frozen=True)
class LocalFrame:
    """Transverse Mercator frame about ``(lat0, long0)``."""

    lat0: float
    long0: float

    def _transformer(self) -> Transformer:
        proj = f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.long0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84"
        return Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    def to_local(self, long: npt.ArrayLike, lat: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Geographic degrees to local ``(x_km, y_km)``."""
        x, y = self._transformer().transform(np.asarray(long), np.asarray(lat))
        return np.asarray(x, dtype=np.float64) / 1000.0, np.asarray(y, dtype=np.float64) / 1000.0

    def to_geographic(
        self, x_km: npt.ArrayLike, y_km: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Local ``(x_km, y_km)`` to geographic ``(long, lat)`` degrees."""
        x = np.asarray(x_km, dtype=np.float64) * 1000.0
        y = np.asarray(y_km, dtype=np.float64) * 1000.0
        lon, lat = self._transformer().transform(x, y, direction="INVERSE")
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    def nll_trans(self) -> str:
        """NonLinLoc ``TRANS`` line."""
        return f"TRANS TRANS_MERC WGS-84 {self.lat0:.6f} {self.long0:.6f} 0.0\n"

    def nll_transform_header(self) -> str:
        """``TRANSFORM`` line as NonLinLoc writes it in grid headers."""
        return (
            f"TRANSFORM  TRANS_MERC RefEllipsoid WGS-84  LatOrig {self.lat0:.6f}  "
            f"LongOrig {self.long0:.6f}  RotCW 0.000000  UseFalseEasting 0  "
            "FalseEasting 500000  ScaleFactor 1.000000\n"
        )


@dataclass(frozen=True)
class GridGeometry:
    """NonLinLoc grid: node counts, origin (km) and spacing (km)."""

    mode: Literal["2d", "3d"]
    nx: int
    ny: int
    nz: int
    x0: float
    y0: float
    z0: float
    dx: float

    def vggrid(self, grid_type: str = "SLOW_LEN") -> str:
        """``VGGRID`` line (cubic cells)."""
        d = self.dx
        return (
            f"VGGRID {self.nx} {self.ny} {self.nz} {self.x0:.4f} {self.y0:.4f} {self.z0:.4f} "
            f"{d:.4f} {d:.4f} {d:.4f} {grid_type}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (for JSON)."""
        return asdict(self)


def grid_geometry(
    candidates: Candidates, frame: LocalFrame, spec: GridSpec, model_top_km: float
) -> GridGeometry:
    """Compute the grid that holds every candidate and the padded region.

    Raises
    ------
    ConfigError
        If a candidate lies above the model top, or ``depth_max_km`` is above the model top.
    """
    highest = float(candidates.depth_km.min())
    if highest < model_top_km:
        raise ConfigError(
            f"highest candidate is at {-highest * 1000:.0f} m elevation, above the velocity "
            f"model top ({-model_top_km * 1000:.0f} m); raise the first LAYER top"
        )
    if spec.depth_max_km <= model_top_km:
        raise ConfigError("grid.depth_max_km must be below the velocity model top")

    d = spec.dx_km
    nz = math.ceil((spec.depth_max_km - model_top_km) / d - 1e-9) + 1
    x, y = frame.to_local(candidates.long, candidates.lat)
    pad = spec.padding_km
    xmin, xmax = float(x.min()) - pad, float(x.max()) + pad
    ymin, ymax = float(y.min()) - pad, float(y.max()) + pad

    if spec.mode == "2d":
        ny = math.ceil(math.hypot(xmax - xmin, ymax - ymin) / d - 1e-9) + 1
        return GridGeometry("2d", 2, ny, nz, 0.0, 0.0, model_top_km, d)

    x0, y0 = math.floor(xmin / d) * d, math.floor(ymin / d) * d
    nx = math.ceil((xmax - x0) / d - 1e-9) + 1
    ny = math.ceil((ymax - y0) / d - 1e-9) + 1
    return GridGeometry("3d", nx, ny, nz, x0, y0, model_top_km, d)
