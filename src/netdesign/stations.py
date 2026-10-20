"""Candidate station sites.

Input CSV, one row per candidate, header exactly::

    station,long,lat,z            (geographic crs, e.g. EPSG:4326, EPSG:4167)
    station,easting,northing,z    (projected crs, e.g. EPSG:2193 NZTM2000)

- ``station``: unique label (no whitespace; used as the NonLinLoc station name).
- horizontal coordinates in the config's ``crs`` (see :mod:`netdesign.crs`); converted to
  WGS-84 on input, with the original values kept for the outputs.
- ``z``: sensor elevation in metres above mean sea level (orthometric, e.g. from a DEM;
  positive up, below sea level negative). For a borehole sensor give the sensor's elevation,
  not the wellhead's.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from netdesign.crs import WGS84, InputCRS
from netdesign.errors import DataError

__all__ = ["Candidates"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Candidates:
    """Candidate sites; arrays are aligned with ``names``. ``long``/``lat`` are WGS-84;
    ``input_xy`` (n, 2) holds the coordinates as given, in ``crs``."""

    names: tuple[str, ...]
    long: FloatArray
    lat: FloatArray
    elev_m: FloatArray
    input_xy: FloatArray | None = None
    crs: InputCRS = WGS84

    def __len__(self) -> int:
        return len(self.names)

    @property
    def elev_km(self) -> FloatArray:
        """Elevation, km above sea level."""
        result: FloatArray = self.elev_m / 1000.0
        return result

    @property
    def depth_km(self) -> FloatArray:
        """Depth, km below sea level (the package's z axis)."""
        return -self.elev_km

    @classmethod
    def read(cls, path: str | Path, crs: InputCRS = WGS84) -> Candidates:
        """Read the candidate CSV described in the module docstring, in ``crs``.

        Raises
        ------
        DataError
            On a missing file, wrong header, unparseable or out-of-range values, or
            duplicate station names. Messages give the file and row.
        """
        path = Path(path)
        try:
            with path.open(newline="") as fh:
                reader = csv.DictReader(line for line in fh if not line.startswith("#"))
                header = tuple(h.strip() for h in reader.fieldnames or ())
                rows = list(reader)
        except OSError as exc:
            raise DataError(f"{path}: cannot read candidates ({exc})") from exc
        a_name, b_name = crs.axes
        expected = ("station", a_name, b_name, "z")
        if header != expected:
            raise DataError(
                f"{path}: header must be {','.join(expected)} for {crs.name}, got {header}"
            )
        if not rows:
            raise DataError(f"{path}: no candidate stations")

        names: list[str] = []
        values: list[tuple[float, float, float]] = []
        for lineno, row in enumerate(rows, start=2):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}
            name = row["station"]
            if not name or any(c.isspace() for c in name):
                raise DataError(f"{path}: row {lineno}: bad station name {name!r}")
            try:
                a, b, z = float(row[a_name]), float(row[b_name]), float(row["z"])
            except ValueError as exc:
                raise DataError(f"{path}: row {lineno} ({name}): {exc}") from exc
            if crs.geographic and not (-180 <= a <= 180 and -90 <= b <= 90):
                raise DataError(f"{path}: row {lineno} ({name}): long/lat out of range")
            names.append(name)
            values.append((a, b, z))

        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise DataError(f"{path}: duplicate station names {dupes}")
        arr = np.array(values, dtype=np.float64)
        lon, lat = crs.to_wgs84(arr[:, 0], arr[:, 1])
        if not (np.all(np.isfinite(lon)) and np.all(np.abs(lat) <= 90)):
            raise DataError(f"{path}: coordinates are not valid in {crs.name}")
        return cls(tuple(names), lon, lat, arr[:, 2], arr[:, :2].copy(), crs)

    def subset(self, indices: list[int] | range) -> Candidates:
        """Candidates at the given positions, in that order."""
        idx = list(indices)
        return Candidates(
            tuple(self.names[i] for i in idx),
            self.long[idx],
            self.lat[idx],
            self.elev_m[idx],
            None if self.input_xy is None else self.input_xy[idx],
            self.crs,
        )

    def to_gtsrce(self) -> str:
        """NonLinLoc ``GTSRCE`` lines (``LATLON``, depth 0, elevation in km)."""
        return "".join(
            f"GTSRCE {n} LATLON {la:.6f} {lo:.6f} 0.0 {e:.4f}\n"
            for n, lo, la, e in zip(self.names, self.long, self.lat, self.elev_km, strict=True)
        )
