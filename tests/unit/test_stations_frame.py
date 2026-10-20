"""Candidate CSV, local frame and grid geometry."""

import math
from pathlib import Path

import numpy as np
import pytest

from netdesign.errors import ConfigError, DataError
from netdesign.frame import GridSpec, LocalFrame, grid_geometry
from netdesign.stations import Candidates

CSV = "station,long,lat,z\nA,176.10,-38.60,350\nB,176.15,-38.62,-150\n"


@pytest.fixture
def cands(tmp_path: Path) -> Candidates:
    path = tmp_path / "c.csv"
    path.write_text(CSV)
    return Candidates.read(path)


def test_read(cands: Candidates) -> None:
    assert cands.names == ("A", "B")
    np.testing.assert_allclose(cands.depth_km, [-0.35, 0.15])


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("station,lon,lat,z\nA,1,2,3\n", "header"),
        ("station,long,lat,z\nA,1,2,x\n", "row 2"),
        ("station,long,lat,z\nA,200,2,3\n", "out of range"),
        ("station,long,lat,z\nA,1,2,3\nA,1,2,3\n", "duplicate"),
        ("station,long,lat,z\nA B,1,2,3\n", "bad station"),
    ],
)
def test_read_errors(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "c.csv"
    path.write_text(text)
    with pytest.raises(DataError, match=match):
        Candidates.read(path)


def test_gtsrce(cands: Candidates) -> None:
    assert cands.to_gtsrce().splitlines()[1] == "GTSRCE B LATLON -38.620000 176.150000 0.0 -0.1500"


def test_frame_distances() -> None:
    frame = LocalFrame(-38.6, 176.1)
    x, y = frame.to_local([176.1, 176.1], [-38.6, -38.5])
    assert x[0] == pytest.approx(0, abs=1e-9)
    assert y[1] == pytest.approx(11.09, rel=1e-3)  # 0.1 deg of latitude


def test_geometry_2d(cands: Candidates) -> None:
    frame = LocalFrame(-38.61, 176.125)
    spec = GridSpec(mode="2d", dx_km=0.1, depth_max_km=5.0, padding_km=1.0)
    g = grid_geometry(cands, frame, spec, model_top_km=-0.5)
    x, y = frame.to_local(cands.long, cands.lat)
    diag = math.hypot(np.ptp(x) + 2, np.ptp(y) + 2)
    assert (g.nx, g.z0) == (2, -0.5)
    assert (g.ny - 1) * 0.1 >= diag > (g.ny - 2) * 0.1
    assert g.z0 + (g.nz - 1) * 0.1 >= 5.0


def test_geometry_3d_contains_padded_region(cands: Candidates) -> None:
    frame = LocalFrame(-38.61, 176.125)
    spec = GridSpec(mode="3d", dx_km=0.1, depth_max_km=5.0, padding_km=1.0)
    g = grid_geometry(cands, frame, spec, model_top_km=-0.5)
    x, y = frame.to_local(cands.long, cands.lat)
    assert g.x0 <= x.min() - 1 and g.x0 + (g.nx - 1) * 0.1 >= x.max() + 1
    assert g.y0 <= y.min() - 1 and g.y0 + (g.ny - 1) * 0.1 >= y.max() + 1


def test_station_above_model_top(cands: Candidates) -> None:
    spec = GridSpec(depth_max_km=5.0)
    with pytest.raises(ConfigError, match="above the velocity model top"):
        grid_geometry(cands, LocalFrame(-38.6, 176.1), spec, model_top_km=-0.3)
