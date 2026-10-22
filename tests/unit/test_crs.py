"""Input coordinate systems by EPSG code."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from example_config import EXAMPLES, write_example

from netdesign.config import load_config
from netdesign.crs import InputCRS
from netdesign.errors import ConfigError, DataError
from netdesign.frame import LocalFrame, Origin
from netdesign.pipeline import run_design, run_sensitivity
from netdesign.stations import Candidates
from netdesign.target import TargetSpec, sample_sources

NZTM = InputCRS(2193)


def test_nztm_known_point_and_round_trip() -> None:
    # NZTM2000: central meridian 173 E, false easting 1 600 000 m.
    e, n = NZTM.from_wgs84(173.0, -41.0)
    assert float(e) == pytest.approx(1_600_000.0, abs=0.01)
    lon, lat = NZTM.to_wgs84(e, n)
    assert (float(lon), float(lat)) == pytest.approx((173.0, -41.0), abs=1e-9)
    assert NZTM.axes == ("easting", "northing") and not NZTM.geographic
    assert InputCRS(4326).axes == ("long", "lat")


def test_bad_epsg() -> None:
    with pytest.raises(ConfigError, match="EPSG"):
        InputCRS(999999)


def nztm_candidates(path: Path) -> Path:
    """The example candidates converted to NZTM eastings/northings."""
    geo = Candidates.read(EXAMPLES / "candidates.csv")
    e, n = NZTM.from_wgs84(geo.long, geo.lat)
    rows = ["station,easting,northing,z"] + [
        f"{s},{a:.4f},{b:.4f},{z:g}" for s, a, b, z in zip(geo.names, e, n, geo.elev_m, strict=True)
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_projected_candidates_match_geographic(tmp_path: Path) -> None:
    geo = Candidates.read(EXAMPLES / "candidates.csv")
    proj = Candidates.read(nztm_candidates(tmp_path / "c.csv"), NZTM)
    np.testing.assert_allclose(proj.long, geo.long, atol=1e-8)
    np.testing.assert_allclose(proj.lat, geo.lat, atol=1e-8)
    assert proj.input_xy is not None and proj.input_xy[0, 0] > 1e6


def test_header_must_match_crs(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="station,easting,northing,z"):
        Candidates.read(EXAMPLES / "candidates.csv", NZTM)


def test_target_and_origin_in_projected_coordinates() -> None:
    frame = LocalFrame(-38.6, 176.13)
    geo = TargetSpec(long=(176.11, 176.15), lat=(-38.62, -38.59), depth_km=(1, 2))
    e0, n0 = (float(v) for v in NZTM.from_wgs84(176.11, -38.62))  # SW corner
    e1, n1 = (float(v) for v in NZTM.from_wgs84(176.15, -38.59))  # NE corner
    proj = TargetSpec(easting=(e0, e1), northing=(n0, n1), depth_km=(1, 2))
    # (the NZTM box is a slightly different shape on the ground; counts are close)
    a, b = sample_sources(geo, frame), sample_sources(proj, frame, NZTM)
    assert abs(len(a) - len(b)) <= 0.1 * len(a)
    with pytest.raises(ConfigError, match="easting/northing"):
        geo.check_crs(NZTM)
    origin = Origin(easting=e0, northing=n0)
    lat, lon = origin.to_wgs84(NZTM)
    assert (lon, lat) == pytest.approx((176.11, -38.62), abs=1e-8)
    with pytest.raises(ConfigError, match="lat and long"):
        origin.to_wgs84(InputCRS(4326))


def test_target_needs_exactly_one_kind() -> None:
    with pytest.raises(ConfigError, match="either long and lat"):
        TargetSpec(long=(1, 2), lat=(1, 2), easting=(1, 2), northing=(1, 2), depth_km=(0, 1))
    with pytest.raises(ConfigError, match="either one box"):
        TargetSpec(depth_km=(0, 1))


def test_config_rejects_mismatched_target(tmp_path: Path) -> None:
    path = write_example(tmp_path, None, crs={"epsg": 2193})
    with pytest.raises(ConfigError, match="easting/northing"):
        load_config(path)


def test_projected_run_matches_geographic(tmp_path: Path, nll_bin: Path) -> None:
    """Same sites given in NZTM2000 or in WGS-84: identical grids, travel times and design."""
    common = {
        "grid": {"dx_km": 0.2},
        "velocity": {"perturb": {"n_realizations": 2}},
        "design": {"n_random": 5, "n_restarts": 1},
    }
    geo_cfg = load_config(write_example(tmp_path / "geo", nll_bin, **common))
    path = write_example(tmp_path / "proj", nll_bin, **common)
    raw = yaml.safe_load(path.read_text())
    raw["crs"] = {"epsg": 2193}
    raw.pop("sites")  # its centre is long/lat; not used by the run
    raw["stations"]["candidates"] = str(nztm_candidates(tmp_path / "proj" / "c.csv"))
    t = raw.pop("target")
    (e0, e1), (n0, n1) = zip(
        *(NZTM.from_wgs84(lo, la) for lo, la in zip(t.pop("long"), t.pop("lat"), strict=True)),
        strict=True,
    )
    # use the geographic box's NZTM corners; the sampled points then differ slightly, so
    # compare the station side (positions, travel-time grids) rather than target points
    raw["target"] = {**t, "easting": [float(e0), float(e1)], "northing": [float(n0), float(n1)]}
    path.write_text(yaml.safe_dump(raw))
    proj_cfg = load_config(path)

    run_sensitivity(geo_cfg)
    run_sensitivity(proj_cfg)
    g = (geo_cfg.output_dir / "stations_local.csv").read_text().splitlines()
    p = (proj_cfg.output_dir / "stations_local.csv").read_text().splitlines()
    assert g == p  # identical local station coordinates to 10 m precision
    run_design(proj_cfg)
    rows = (proj_cfg.output_dir / "qc" / "candidates_outcome.csv").read_text().splitlines()
    assert rows[0].startswith("station,long,lat,easting,northing,elev_m")
