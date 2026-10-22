"""Candidate sites on a regular grid, with elevations from a DEM (``sites`` config section).

The grid is laid out in the local transverse Mercator frame about ``centre`` (see
:mod:`netdesign.frame`): nodes every ``spacing_m`` along two perpendicular axes, rotated
``rotation_deg`` clockwise from north, out to half of ``extent_m`` either side of the centre
(the centre is always a node). ``shape: ellipse`` keeps only the nodes inside the ellipse
inscribed in that rectangle.

Elevations come from a DEM, sampled bilinearly at each site:

- ``copernicus30`` / ``copernicus90``: Copernicus GLO-30 / GLO-90 DSM tiles, read from the
  public AWS bucket (no account needed). Only the area around the sites is fetched; the clip
  is cached as a GeoTIFF in ``dem.cache_dir`` and reused. Heights are relative to the EGM2008
  geoid, i.e. metres above mean sea level as :mod:`netdesign.stations` expects. GLO-30 is a
  surface model: over forest and buildings it gives the canopy/roof height.
- ``file``: a local GeoTIFF in any CRS, heights in metres above mean sea level.

Sites where the DEM has no data (e.g. open sea, where Copernicus has no tiles), or below
``min_elevation_m``, are dropped. ``z`` written is the DEM height minus ``sensor_depth_m``.
Needs the ``dem`` extra (rasterio).
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates

from netdesign.crs import InputCRS
from netdesign.errors import ConfigError, DataError
from netdesign.frame import LocalFrame, Origin

__all__ = [
    "DEM",
    "DEMSpec",
    "SiteGrid",
    "SitesResult",
    "SitesSpec",
    "build_sites",
    "copernicus_tile_url",
    "load_dem",
    "site_grid",
]

FloatArray = npt.NDArray[np.float64]

_COPERNICUS = {
    # source: (bucket, resolution code in the tile name)
    "copernicus30": ("copernicus-dem-30m", "10"),
    "copernicus90": ("copernicus-dem-90m", "30"),
}


class DEMSpec(BaseModel):
    """``sites.dem`` config section."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["copernicus30", "copernicus90", "file"] = "copernicus30"
    path: Path | None = Field(None, description="file: local GeoTIFF")
    cache_dir: Path = Field(Path("dem"), description="downloaded DEM clips")
    margin_m: float = Field(500.0, ge=0, description="DEM area beyond the outermost site")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.source == "file") != (self.path is not None):
            raise ConfigError("sites.dem: 'path' is needed for source 'file', and only then")
        return self


class SitesSpec(BaseModel):
    """``sites`` config section."""

    model_config = ConfigDict(extra="forbid")

    centre: Origin
    spacing_m: float = Field(gt=0)
    extent_m: tuple[float, float] = Field(description="full width across, along the grid axes")
    shape: Literal["rectangle", "ellipse"] = "rectangle"
    rotation_deg: float = Field(0.0, description="grid 'north' axis, clockwise from north")
    prefix: str = Field("G", pattern=r"^\S+$")
    sensor_depth_m: float = Field(0.0, description="sensor below the DEM surface")
    min_elevation_m: float | None = Field(None, description="drop sites lower than this")
    dem: DEMSpec = DEMSpec()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if min(self.extent_m) < 0:
            raise ConfigError("sites.extent_m must not be negative")
        return self


@dataclass(frozen=True)
class SiteGrid:
    """Grid nodes: WGS-84 ``long``/``lat`` and local ``x``/``y`` (km) about the centre."""

    long: FloatArray
    lat: FloatArray
    x_km: FloatArray
    y_km: FloatArray
    frame: LocalFrame


def site_grid(spec: SitesSpec, crs: InputCRS) -> SiteGrid:
    """Grid nodes, ordered by rows from the grid's top (far end of its north axis), each row
    left to right."""
    lat0, lon0 = spec.centre.to_wgs84(crs, where="sites.centre")
    frame = LocalFrame(lat0, lon0)
    half_u, half_v = spec.extent_m[0] / 2, spec.extent_m[1] / 2
    nu = math.floor(half_u / spec.spacing_m + 1e-9)
    nv = math.floor(half_v / spec.spacing_m + 1e-9)
    v, u = np.meshgrid(
        np.arange(nv, -nv - 1, -1) * spec.spacing_m,
        np.arange(-nu, nu + 1) * spec.spacing_m,
        indexing="ij",
    )
    u, v = u.ravel(), v.ravel()
    if spec.shape == "ellipse":
        # a zero half-width axis has only its centre node, so it adds nothing to r2
        r2 = sum(((c / h) ** 2 for c, h in ((u, half_u), (v, half_v)) if h > 0), np.zeros(u.size))
        keep = r2 <= 1 + 1e-9
        u, v = u[keep], v[keep]
    t = math.radians(spec.rotation_deg)
    x = (u * math.cos(t) + v * math.sin(t)) / 1000.0
    y = (-u * math.sin(t) + v * math.cos(t)) / 1000.0
    lon, lat = frame.to_geographic(x, y)
    return SiteGrid(lon, lat, x, y, frame)


# ---------------------------------------------------------------- DEM


@dataclass(frozen=True)
class DEM:
    """Heights (m, NaN = no data) on a north-up raster: ``transform`` is the GDAL geotransform
    ``(x0, dx, 0, y0, 0, dy)`` of the pixel corners; ``crs`` is WKT."""

    z: FloatArray
    transform: tuple[float, float, float, float, float, float]
    crs: str
    path: Path

    def sample(self, long: npt.ArrayLike, lat: npt.ArrayLike) -> FloatArray:
        """Bilinear heights at WGS-84 points; NaN outside the DEM or next to no-data."""
        to_dem = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_wkt(self.crs), always_xy=True)
        x, y = to_dem.transform(np.asarray(long, dtype=float), np.asarray(lat, dtype=float))
        x0, dx, _, y0, _, dy = self.transform
        col = (np.asarray(x) - x0) / dx - 0.5
        row = (np.asarray(y) - y0) / dy - 0.5
        out = map_coordinates(self.z, [row, col], order=1, mode="constant", cval=np.nan)
        return np.asarray(out, dtype=np.float64)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` in the DEM's CRS."""
        x0, dx, _, y0, _, dy = self.transform
        ny, nx = self.z.shape
        return x0, y0 + ny * dy, x0 + nx * dx, y0


def _rasterio() -> Any:
    try:
        import rasterio
    except ImportError as exc:
        raise ConfigError(
            "sites.dem needs rasterio: install with `uv sync --extra dem` (or ./install.sh)"
        ) from exc
    # Use the wheel's own PROJ database: rasterio otherwise follows PROJ_DATA, which may point
    # at another (incompatible) PROJ install, e.g. conda's. It reads the variable when its
    # first GDAL environment starts, so set it before any dataset is opened. (pyproj uses its
    # own bundled database regardless.)
    bundled = Path(rasterio.__file__).parent / "proj_data"
    if bundled.is_dir():
        os.environ["PROJ_DATA"] = str(bundled)
    return rasterio


def _read(path: Path) -> DEM:
    rasterio = _rasterio()
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                raise DataError(f"{path}: DEM has no coordinate reference system")
            if ds.transform.b != 0 or ds.transform.d != 0 or ds.transform.e >= 0:
                raise DataError(f"{path}: DEM must be north-up (no rotation)")
            z = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
            return DEM(z, tuple(ds.transform.to_gdal()), ds.crs.to_wkt(), path)
    except rasterio.errors.RasterioIOError as exc:
        raise DataError(f"{path}: cannot read DEM ({exc})") from exc


def copernicus_tile_url(source: str, lat: int, lon: int) -> str:
    """URL of the Copernicus tile whose south-west corner is ``(lat, lon)``."""
    bucket, res = _COPERNICUS[source]
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    name = f"Copernicus_DSM_COG_{res}_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"https://{bucket}.s3.eu-central-1.amazonaws.com/{name}/{name}.tif"


def _download(source: str, bounds: tuple[float, float, float, float], out: Path) -> None:
    """Mosaic and clip the Copernicus tiles covering WGS-84 ``bounds`` into ``out``."""
    rasterio = _rasterio()
    from rasterio.merge import merge

    west, south, east, north = bounds
    urls = [
        copernicus_tile_url(source, la, lo)
        for la in range(math.floor(south), math.floor(north) + 1)
        for lo in range(math.floor(west), math.floor(east) + 1)
    ]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3"):
        datasets = []
        try:
            for url in urls:
                try:
                    datasets.append(rasterio.open(url))
                except rasterio.errors.RasterioIOError as exc:
                    if "404" not in str(exc):
                        raise DataError(f"cannot download DEM tile {url} ({exc})") from exc
                    # no tile: open sea
            if not datasets:
                raise DataError(
                    f"no {source} DEM tiles for long {west:.3f}..{east:.3f}, "
                    f"lat {south:.3f}..{north:.3f} (all sea)"
                )
            # snap outward to the tiles' pixel grid, or merge resamples (shifts) the pixels
            t = datasets[0].transform
            snapped = (
                t.c + math.floor((west - t.c) / t.a) * t.a,
                t.f + math.ceil((south - t.f) / t.e) * t.e,
                t.c + math.ceil((east - t.c) / t.a) * t.a,
                t.f + math.floor((north - t.f) / t.e) * t.e,
            )
            z, transform = merge(datasets, bounds=snapped, nodata=np.nan, dtype="float32")
            profile = {
                "driver": "GTiff",
                "height": z.shape[1],
                "width": z.shape[2],
                "count": 1,
                "dtype": "float32",
                "crs": datasets[0].crs,
                "transform": transform,
                "nodata": np.nan,
                "compress": "deflate",
            }
        finally:
            for ds in datasets:
                ds.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(z)
    tmp.replace(out)


def load_dem(spec: DEMSpec, long: FloatArray, lat: FloatArray, frame: LocalFrame) -> DEM:
    """The DEM covering the points plus ``spec.margin_m``: the local file, or a (cached)
    Copernicus clip."""
    if spec.source == "file":
        assert spec.path is not None
        return _read(spec.path)
    x, y = frame.to_local(long, lat)
    m = spec.margin_m / 1000.0
    corners = (
        np.array([x.min() - m, x.min() - m, x.max() + m, x.max() + m]),
        np.array([y.min() - m, y.max() + m, y.min() - m, y.max() + m]),
    )
    clon, clat = frame.to_geographic(*corners)
    bounds = (
        math.floor(clon.min() * 1e4) / 1e4,
        math.floor(clat.min() * 1e4) / 1e4,
        math.ceil(clon.max() * 1e4) / 1e4,
        math.ceil(clat.max() * 1e4) / 1e4,
    )
    path = spec.cache_dir / (spec.source + "_" + "_".join(f"{b:.4f}" for b in bounds) + ".tif")
    if not path.is_file():
        _download(spec.source, bounds, path)
    return _read(path)


# ---------------------------------------------------------------- candidates


@dataclass(frozen=True)
class SitesResult:
    """What :func:`build_sites` wrote."""

    path: Path
    n_grid: int
    n_written: int
    n_no_data: int
    n_low: int
    dem: DEM
    grid: SiteGrid
    keep: npt.NDArray[np.bool_]
    elev_m: FloatArray


def build_sites(spec: SitesSpec, crs: InputCRS, out: Path, overwrite: bool = False) -> SitesResult:
    """Lay out the grid, sample the DEM and write the candidate CSV ``out``
    (``station,long,lat,z`` or ``station,easting,northing,z`` in ``crs``).

    Raises
    ------
    DataError
        If ``out`` exists and not ``overwrite``, the DEM cannot be read, or no site is left.
    """
    if out.exists() and not overwrite:
        raise DataError(f"{out} exists; use --force to overwrite it")
    grid = site_grid(spec, crs)
    dem = load_dem(spec.dem, grid.long, grid.lat, grid.frame)
    surface = dem.sample(grid.long, grid.lat)
    elev = surface - spec.sensor_depth_m
    no_data = ~np.isfinite(surface)
    low = np.zeros_like(no_data)
    if spec.min_elevation_m is not None:
        low = ~no_data & (elev < spec.min_elevation_m)
    keep = ~(no_data | low)
    if not keep.any():
        raise DataError(f"no site has DEM data{'' if not low.any() else ' above min_elevation_m'}")

    a, b = crs.from_wgs84(grid.long[keep], grid.lat[keep])
    width = len(str(int(keep.sum())))
    fmt = ".6f" if crs.geographic else ".1f"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("station", *crs.axes, "z"))
        for i, (ai, bi, zi) in enumerate(zip(a, b, elev[keep], strict=True), start=1):
            w.writerow((f"{spec.prefix}{i:0{width}d}", f"{ai:{fmt}}", f"{bi:{fmt}}", f"{zi:.1f}"))
    return SitesResult(
        out,
        len(keep),
        int(keep.sum()),
        int(no_data.sum()),
        int(low.sum()),
        dem,
        grid,
        keep,
        elev,
    )
