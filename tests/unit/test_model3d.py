"""3D starting model from a CSV of nodes."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from analytic import GRAD, TOP, V0, VPVS, write_case

from netdesign.config import load_config
from netdesign.errors import ConfigError, DataError
from netdesign.frame import GridGeometry, LocalFrame
from netdesign.nll import read_grid
from netdesign.pipeline import run_sensitivity, run_vel_grids
from netdesign.velocity.heterogeneity import cell_centres
from netdesign.velocity.model3d import NodeModel, sample_model

FRAME = LocalFrame(-38.61, 176.115)
GEOM = GridGeometry("3d", 30, 25, 20, -1.5, -1.25, -0.5, 0.1)


def write_nodes(path: Path, lon, lat, dep, vp_fn, vs_col="vs", shuffle=True) -> Path:
    rows = []
    for lo in lon:
        for la in lat:
            for d in dep:
                vp = vp_fn(lo, la, d)
                second = vp / VPVS if vs_col == "vs" else VPVS
                rows.append(f"{lo:.6f},{la:.6f},{d:.4f},{vp:.6f},{second:.6f}")
    if shuffle:
        np.random.default_rng(0).shuffle(rows)
    path.write_text(f"long,lat,depth_km,vp,{vs_col}\n" + "\n".join(rows) + "\n")
    return path


def gradient_vp(lo: float, la: float, d: float) -> float:
    return V0 + GRAD * (d - TOP)


LON = np.linspace(176.05, 176.18, 8)
LAT = np.linspace(-38.65, -38.57, 7)
DEP = np.linspace(-0.6, 3.2, 9)


def test_read_any_row_order_and_vpvs_column(tmp_path: Path) -> None:
    m = NodeModel.read(write_nodes(tmp_path / "a.csv", LON, LAT, DEP, gradient_vp, "vpvs"))
    assert m.vp.shape == (8, 7, 9)
    np.testing.assert_allclose(m.vp / m.vs, VPVS)
    np.testing.assert_allclose(m.vp[2, 3], V0 + GRAD * (DEP - TOP), rtol=1e-6)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda lines: lines[:-1], "full regular grid"),
        (lambda lines: [*lines[:-1], lines[1]], "duplicate"),
        (lambda lines: ["long,lat,depth,vp,vs", *lines[1:]], "need columns"),
        (
            lambda lines: [lines[0], lines[1].replace(lines[1].split(",")[3], "x", 1), *lines[2:]],
            "bad value",
        ),
    ],
)
def test_read_errors(tmp_path: Path, mutate, match: str) -> None:
    path = write_nodes(tmp_path / "a.csv", LON[:3], LAT[:3], DEP[:3], gradient_vp)
    path.write_text("\n".join(mutate(path.read_text().splitlines())) + "\n")
    with pytest.raises(DataError, match=match):
        NodeModel.read(path)


def test_vpvs_bound(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    write_nodes(path, LON[:2], LAT[:2], DEP[:2], gradient_vp, "vpvs")
    path.write_text(path.read_text().replace(f",{VPVS:.6f}", ",1.1"))
    with pytest.raises(DataError, match="Vp/Vs"):
        NodeModel.read(path)


def test_sampling_exact_for_linear_field_and_full_coverage(tmp_path: Path) -> None:
    def linear(lo: float, la: float, d: float) -> float:
        return 3.0 + 0.4 * d + 2.0 * (lo - 176.1) - 3.0 * (la + 38.6)

    m = NodeModel.read(write_nodes(tmp_path / "a.csv", LON, LAT, DEP, linear))
    s = sample_model(m, FRAME, GEOM)
    xc, yc, zc = cell_centres(GEOM)
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    lon, lat = FRAME.to_geographic(xx, yy)
    expected = 3.0 + 0.4 * zc[None, None, :] + (2.0 * (lon - 176.1) - 3.0 * (lat + 38.6))[..., None]
    np.testing.assert_allclose(s.vp, expected, rtol=1e-5)
    assert s.coverage == 1.0


def test_partial_coverage_extends_edges(tmp_path: Path) -> None:
    m = NodeModel.read(
        write_nodes(tmp_path / "a.csv", LON, LAT, np.linspace(0.0, 1.0, 3), gradient_vp)
    )
    s = sample_model(m, FRAME, GEOM)
    _, _, zc = cell_centres(GEOM)
    assert 0 < s.coverage < 1
    # Above the model (z < 0) the top value (depth 0) is used; below 1 km the bottom value.
    np.testing.assert_allclose(s.vp[:, :, zc < 0], gradient_vp(0, 0, 0.0), rtol=1e-6)
    np.testing.assert_allclose(s.vp[:, :, zc > 1], gradient_vp(0, 0, 1.0), rtol=1e-6)


# ---------------------------------------------------------------- config


def config_dict(tmp_path: Path, nll_bin: Path, **velocity) -> Path:
    path = write_case(tmp_path, nll_bin, "3d")
    cfg = yaml.safe_load(path.read_text())
    nodes = write_nodes(tmp_path / "nodes.csv", LON, LAT, DEP, gradient_vp)
    cfg["velocity"] = {"model_3d": str(nodes), **velocity}
    cfg["target"] = {
        "long": [176.105, 176.125],
        "lat": [-38.615, -38.6],
        "depth_km": [0.5, 2.5],
        "sampling": "random",
        "n_points": 200,
        "seed": 1,
    }
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize(
    ("velocity", "match"),
    [
        ({"base": "m.layer"}, "exactly one"),
        ({"perturb": {"enabled": True, "interface_sd_km": 0.1}}, "no interfaces"),
        ({"heterogeneity": {"enabled": True, "vp_pct": [1.0, 2.0]}}, "single value"),
        ({"attenuation": {"model": "layered"}}, "vs_scaled"),
    ],
)
def test_config_rejects_layer_settings(tmp_path: Path, velocity: dict, match: str) -> None:
    path = config_dict(tmp_path, Path("/nonexistent"), **velocity)
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_needs_3d_grid(tmp_path: Path, nll_bin: Path) -> None:
    path = config_dict(tmp_path, nll_bin)
    cfg = yaml.safe_load(path.read_text())
    cfg["grid"]["mode"] = "2d"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match=r"grid\.mode: 3d"):
        run_vel_grids(load_config(path))


# ---------------------------------------------------------------- with NonLinLoc


def test_matches_equivalent_layered_model(tmp_path: Path, nll_bin: Path) -> None:
    """A node model of the linear-gradient medium gives the layered model's travel times."""
    nodes_path = config_dict(tmp_path / "n", nll_bin)
    target = yaml.safe_load(nodes_path.read_text())["target"]
    layered = load_config(write_case(tmp_path / "l", nll_bin, "3d", target=target))
    a, b = run_sensitivity(layered), run_sensitivity(load_config(nodes_path))
    # Same grid apart from the top: layered starts at the model top (-0.6 km), the 3D model
    # at the highest station (-0.3 km); travel times agree to Grid2Time's accuracy.
    np.testing.assert_allclose(b.T, a.T, rtol=2e-3)
    np.testing.assert_allclose(b.G, a.G, rtol=0.02, atol=2e-3)


def test_global_perturbation_and_heterogeneity(tmp_path: Path, nll_bin: Path) -> None:
    path = config_dict(
        tmp_path,
        nll_bin,
        perturb={
            "enabled": True,
            "n_realizations": 2,
            "vp_pct": {"sd": 5.0, "mode": "global"},
            "vpvs": {"sd": 0.03},
        },
        heterogeneity={
            "enabled": True,
            "vp_pct": 4.0,
            "structure": {"corr_len_km": [1.0, 1.0, 0.3]},
        },
    )
    cfg = load_config(path)
    grids = run_vel_grids(cfg)
    meta = json.loads((cfg.output_dir / "models" / "ensemble.json").read_text())
    assert meta["starting_model"] == "3d"
    assert not list((cfg.output_dir / "models").glob("model_*.csv"))
    assert json.loads((cfg.output_dir / "models" / "model3d.json").read_text())["coverage"] > 0
    base = read_grid(grids.roots[0]["P"]).velocity()
    for r in meta["realizations"][1:]:
        v = read_grid(grids.roots[r["index"]]["P"]).velocity()
        ratio = v / base
        # Mean ratio = the global factor; spread around it = heterogeneity (~4 %).
        assert np.mean(ratio) == pytest.approx(r["params"]["vp_factor"][0], abs=0.01)
        assert 0.02 < np.std(ratio) < 0.06
