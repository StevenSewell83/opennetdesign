"""Coordinate reference system of the geographic inputs (``crs`` config section).

All horizontal input coordinates (candidates, target box, grid origin, 3D model nodes) are in
one CRS given by EPSG code: geographic (degrees: columns/keys ``long``, ``lat``) or projected
(metres: ``easting``, ``northing``). They are converted to WGS-84 longitude/latitude on input;
everything downstream (the local frame, NonLinLoc, grids) works from WGS-84.

Elevations (``z``, m) are orthometric: metres above mean sea level, e.g. from a DEM, the same
reference as the velocity model's depths. No geoid conversion is applied.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

from netdesign.errors import ConfigError

__all__ = ["WGS84", "CRSSpec", "InputCRS"]

FloatArray = npt.NDArray[np.float64]


class CRSSpec(BaseModel):
    """``crs`` config section."""

    model_config = ConfigDict(extra="forbid")

    epsg: int = Field(4326, description="EPSG code of all horizontal input coordinates")


class InputCRS:
    """Conversions between the input CRS and WGS-84 (always easting/longitude first)."""

    def __init__(self, epsg: int) -> None:
        try:
            self.crs = CRS.from_epsg(epsg)
        except CRSError as exc:
            raise ConfigError(f"crs.epsg: {epsg} is not a known EPSG code ({exc})") from exc
        if not (self.crs.is_geographic or self.crs.is_projected):
            raise ConfigError(f"crs.epsg: EPSG:{epsg} is neither geographic nor projected")
        self.epsg = epsg

    @property
    def geographic(self) -> bool:
        """True for degrees (long/lat), False for projected metres (easting/northing)."""
        return bool(self.crs.is_geographic)

    @property
    def axes(self) -> tuple[str, str]:
        """Column/key names of the two horizontal coordinates."""
        return ("long", "lat") if self.geographic else ("easting", "northing")

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``EPSG:2193 (NZGD2000 / NZTM 2000)``."""
        return f"EPSG:{self.epsg} ({self.crs.name})"

    @cached_property
    def _to(self) -> Transformer:
        return Transformer.from_crs(self.crs, CRS.from_epsg(4326), always_xy=True)

    @cached_property
    def _from(self) -> Transformer:
        return Transformer.from_crs(CRS.from_epsg(4326), self.crs, always_xy=True)

    def to_wgs84(self, a: npt.ArrayLike, b: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Input coordinates to WGS-84 ``(long, lat)``."""
        if self.epsg == 4326:
            return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        lon, lat = self._to.transform(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        )
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    def from_wgs84(self, lon: npt.ArrayLike, lat: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """WGS-84 ``(long, lat)`` to input coordinates."""
        if self.epsg == 4326:
            return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
        a, b = self._from.transform(
            np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
        )
        return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


WGS84 = InputCRS(4326)
"""The default: WGS-84 longitude/latitude."""
