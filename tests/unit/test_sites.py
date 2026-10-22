"""Candidate grid generation and DEM sampling (synthetic GeoTIFFs; no network)."""

import re
import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from netdesign.cli import app
from netdesign.crs import WGS84, InputCRS
from netdesign.errors import ConfigError, DataError
from netdesign.sites import (
    DEMSpec,
    SitesSpec,
    _rasterio,
    build_sites,
    copernicus_tile_url,
    site_grid,
)
from netdesign.stations import Candidates

rasterio = pytest.importorskip("rasterio")
_rasterio()  # bundled PROJ data, as the package uses
from affine import Affine  # noqa: E402

LAT0, LON0 = -38.6, 176.1


def _plane(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 300.0 + 2000.0 * (x - LON0) - 1000.0 * (y - LAT0)


def _write_dem(path: Path, west: float, north: float, nx: int, ny: int, d: float, epsg: int,
               fn=None, nodata_cols: slice | None = None) -> Path:  # fmt: skip
    """GeoTIFF with heights ``fn(x, y)`` at pixel centres (default: the plane)."""
    xc = west + (np.arange(nx) + 0.5) * d
    yc = north - (np.arange(ny) + 0.5) * d
    gx, gy = np.meshgrid(xc, yc)
    z = (fn or _plane)(gx, gy).astype(np.float32)
    if nodata_cols is not None:
        z[:, nodata_cols] = np.nan
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32",
        crs=f"EPSG:{epsg}", transform=Affine(d, 0, west, 0, -d, north), nodata=np.nan,
    ) as dst:  # fmt: skip
        dst.write(z, 1)
    return path


def _spec(dem: DEMSpec, **kw: object) -> SitesSpec:
    base: dict[str, object] = {
        "centre": {"lat": LAT0, "long": LON0},
        "spacing_m": 500.0,
        "extent_m": (2000.0, 1000.0),
        "dem": dem,
    }
    return SitesSpec.model_validate(base | kw)


def test_grid_layout() -> None:
    g = site_grid(_spec(DEMSpec()), WGS84)
    assert len(g.x_km) == 5 * 3
    assert np.isclose(g.x_km, 0).sum() == 3 and np.any((g.x_km == 0) & (g.y_km == 0))
    assert np.allclose(np.unique(np.round(g.x_km, 9)), [-1.0, -0.5, 0, 0.5, 1.0])
    assert g.y_km[0] == pytest.approx(0.5) and g.x_km[0] == pytest.approx(-1.0)  # top-left first
    lon, lat = g.frame.to_geographic(0.0, 0.0)
    assert float(lon) == pytest.approx(LON0) and float(lat) == pytest.approx(LAT0)


def test_grid_rotation_and_ellipse() -> None:
    g = site_grid(_spec(DEMSpec(), rotation_deg=90.0), WGS84)
    # the grid's 'north' axis (1000 m wide) now points east
    assert np.ptp(g.x_km) == pytest.approx(1.0) and np.ptp(g.y_km) == pytest.approx(2.0)
    e = site_grid(_spec(DEMSpec(), extent_m=(2000.0, 2000.0), shape="ellipse"), WGS84)
    r = np.hypot(e.x_km, e.y_km)
    assert len(r) == 13 and r.max() <= 1.0 + 1e-9


def test_centre_must_match_crs() -> None:
    with pytest.raises(ConfigError, match=r"sites\.centre"):
        site_grid(_spec(DEMSpec()), InputCRS(2193))


def test_dem_spec_needs_path_for_file() -> None:
    with pytest.raises(ConfigError, match="path"):
        DEMSpec(source="file")


def test_file_dem_geographic(tmp_path: Path) -> None:
    dem = _write_dem(tmp_path / "dem.tif", LON0 - 0.05, LAT0 + 0.05, 400, 400, 0.00025, 4326)
    out = tmp_path / "cand.csv"
    spec = _spec(DEMSpec(source="file", path=dem), sensor_depth_m=2.0)
    result = build_sites(spec, WGS84, out)
    assert result.n_written == 15
    c = Candidates.read(out)
    assert c.names[0] == "G01" and c.names[-1] == "G15"
    # bilinear interpolation reproduces a plane exactly (to the 0.1 m written)
    assert np.allclose(c.elev_m, _plane(c.long, c.lat) - 2.0, atol=0.06)


def test_file_dem_projected_and_projected_output(tmp_path: Path) -> None:
    nztm = InputCRS(2193)
    e0, n0 = nztm.from_wgs84(LON0, LAT0)
    dem = _write_dem(
        tmp_path / "dem.tif", float(e0) - 3000, float(n0) + 3000, 300, 300, 20.0, 2193,
        fn=lambda x, y: 100.0 + 0.01 * (x - float(e0)),
    )  # fmt: skip
    spec = _spec(
        DEMSpec(source="file", path=dem), centre={"easting": float(e0), "northing": float(n0)}
    )
    out = tmp_path / "cand.csv"
    build_sites(spec, nztm, out)
    c = Candidates.read(out, nztm)
    assert c.input_xy is not None
    assert np.allclose(c.elev_m, 100.0 + 0.01 * (c.input_xy[:, 0] - float(e0)), atol=0.06)


def test_no_data_and_low_sites_dropped(tmp_path: Path) -> None:
    # west half no data, centre 300 m; min_elevation drops sites below 300 m (south-east)
    dem = _write_dem(
        tmp_path / "dem.tif", LON0 - 0.05, LAT0 + 0.05, 400, 400, 0.00025, 4326,
        nodata_cols=slice(0, 196),
    )  # fmt: skip
    spec = _spec(DEMSpec(source="file", path=dem), min_elevation_m=299.95)
    result = build_sites(spec, WGS84, tmp_path / "cand.csv")
    assert result.n_no_data == 6  # x = -1, -0.5 km
    assert result.n_written + result.n_low == 9
    assert result.n_low > 0
    assert (Candidates.read(result.path).elev_m >= 299.95).all()


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "cand.csv"
    out.write_text("keep me")
    with pytest.raises(DataError, match="--force"):
        build_sites(_spec(DEMSpec()), WGS84, out)
    assert out.read_text() == "keep me"


def test_tile_url() -> None:
    assert copernicus_tile_url("copernicus30", -39, 176).endswith(
        "/Copernicus_DSM_COG_10_S39_00_E176_00_DEM/Copernicus_DSM_COG_10_S39_00_E176_00_DEM.tif"
    )
    assert "copernicus-dem-90m" in copernicus_tile_url("copernicus90", 45, -3)
    assert "_30_N45_00_W003_00_" in copernicus_tile_url("copernicus90", 45, -3)


class _RangeHandler(SimpleHTTPRequestHandler):
    """Static files with single byte-range GETs, as GDAL's /vsicurl/ needs."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        path = Path(self.translate_path(self.path))
        if m is None or not path.is_file():
            super().do_GET()
            return
        data = path.read_bytes()
        start = int(m[1])
        end = min(int(m[2]) if m[2] else len(data) - 1, len(data) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start : end + 1])


@pytest.fixture
def tile_server(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """HTTP server for a tile directory (missing tiles: 404, like the S3 bucket)."""
    root = tmp_path / "tiles"
    root.mkdir()
    handler = partial(_RangeHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_copernicus_download_and_cache(
    tmp_path: Path, tile_server: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tile east of 176 E is land; the one to the west is sea (no tile: 404)."""
    root, base = tile_server
    _write_dem(root / "-39_176.tif", 176.0, -38.0, 1200, 1200, 1 / 1200, 4326)
    requested: list[str] = []

    def url(source: str, lat: int, lon: int) -> str:
        requested.append(f"{lat}_{lon}")
        return f"{base}/{lat}_{lon}.tif"

    monkeypatch.setattr("netdesign.sites.copernicus_tile_url", url)
    dem = DEMSpec(cache_dir=tmp_path / "cache", margin_m=200.0)
    spec = _spec(dem, centre={"lat": LAT0, "long": 176.01})
    result = build_sites(spec, WGS84, tmp_path / "a.csv")
    assert sorted(requested) == ["-39_175", "-39_176"]
    assert result.n_no_data == 3  # x = -1 km is west of 176 E
    c = Candidates.read(result.path)
    assert len(c) == 12
    assert np.allclose(c.elev_m, _plane(c.long, c.lat), atol=0.06)
    assert len(list((tmp_path / "cache").glob("*.tif"))) == 1
    requested.clear()
    build_sites(spec, WGS84, tmp_path / "b.csv")
    assert requested == []  # second run reads the cache


def test_copernicus_all_sea(
    tmp_path: Path, tile_server: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, base = tile_server
    monkeypatch.setattr("netdesign.sites.copernicus_tile_url", lambda s, la, lo: f"{base}/x.tif")
    with pytest.raises(DataError, match="all sea"):
        build_sites(_spec(DEMSpec(cache_dir=tmp_path)), WGS84, tmp_path / "a.csv")


def test_sites_command(tmp_path: Path) -> None:
    _write_dem(tmp_path / "dem.tif", LON0 - 0.05, LAT0 + 0.05, 400, 400, 0.00025, 4326)
    model = Path(__file__).resolve().parents[2] / "examples" / "start_1d.layer"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"velocity: {{base: {model}}}\nstations: {{candidates: cand.csv}}\n"
        f"sites:\n  centre: {{lat: {LAT0}, long: {LON0}}}\n  spacing_m: 500\n"
        "  extent_m: [1000, 1000]\n  dem: {source: file, path: dem.tif}\n"
    )
    result = CliRunner().invoke(app, ["sites", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "wrote 9 of 9" in result.output
    assert (tmp_path / "out" / "qc" / "sites_map.png").is_file()
    assert len(Candidates.read(tmp_path / "cand.csv")) == 9
    again = CliRunner().invoke(app, ["sites", str(cfg)])
    assert again.exit_code == 1 and "--force" in again.output
    assert CliRunner().invoke(app, ["sites", str(cfg), "--force"]).exit_code == 0
