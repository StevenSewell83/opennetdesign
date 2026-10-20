"""3D correlated heterogeneity: random fields, model grids, writer and streaming."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from example_config import write_example

from netdesign.config import load_config
from netdesign.errors import ConfigError
from netdesign.frame import GridGeometry, LocalFrame
from netdesign.nll import read_grid
from netdesign.nll.modelgrid import write_model_grid
from netdesign.pipeline import run_sensitivity, run_vel_grids
from netdesign.velocity import Layer, LayeredModel
from netdesign.velocity.heterogeneity import HeterogeneitySpec, build_model_grid, cell_centres
from netdesign.velocity.layered import MIN_VPVS
from netdesign.velocity.random_field import CorrelationSpec, correlated_field

MODEL = LayeredModel(
    (
        Layer(-0.5, 3.0, 1.7),
        Layer(0.53, 4.0, 2.3),
        Layer(1.27, 5.0, 2.9, vp_grad=0.2, vs_grad=0.1),
    )
)
GEOM = GridGeometry("3d", 60, 50, 50, -3.0, -2.5, -0.5, 0.05)


def acf_along(field: np.ndarray, axis: int, lag: int) -> float:
    a = np.take(field, range(field.shape[axis] - lag), axis=axis)
    b = np.take(field, range(lag, field.shape[axis]), axis=axis)
    return float(np.mean(a * b))


# ---------------------------------------------------------------- random fields


def test_field_is_standardised_and_reproducible() -> None:
    spec = CorrelationSpec(corr_len_km=(1.0, 1.0, 0.3))
    a = correlated_field((40, 30, 20), 0.05, spec, np.random.default_rng(1))
    b = correlated_field((40, 30, 20), 0.05, spec, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert abs(a.mean()) < 1e-12 and a.std() == pytest.approx(1.0)


def test_exponential_correlation_length() -> None:
    """Exponential ACF is exp(-r/a): about 1/e at lag a (averaged over many fields)."""
    spec = CorrelationSpec(correlation="exponential", corr_len_km=(0.5, 0.5, 0.5))
    rng = np.random.default_rng(2)
    lag = 10  # cells; 0.5 km at 0.05 km
    acf = np.mean(
        [acf_along(correlated_field((120, 120), 0.05, spec, rng), 0, lag) for _ in range(20)]
    )
    assert acf == pytest.approx(math.exp(-1), abs=0.08)


def test_anisotropy_follows_strike() -> None:
    """Long correlation along strike: N-S for strike 0, E-W for strike 90."""
    rng = np.random.default_rng(3)
    for strike, long_axis in ((0.0, 1), (90.0, 0)):
        spec = CorrelationSpec(corr_len_km=(2.0, 0.3, 0.3), strike_deg=strike)
        f = correlated_field((150, 150), 0.05, spec, rng)
        lag = 8
        assert acf_along(f, long_axis, lag) > 2 * acf_along(f, 1 - long_axis, lag)


def test_rougher_with_smaller_hurst() -> None:
    rng = np.random.default_rng(4)
    rough = correlated_field((200, 200), 0.05, CorrelationSpec(hurst=0.1), rng)
    smooth = correlated_field((200, 200), 0.05, CorrelationSpec(correlation="gaussian"), rng)
    assert acf_along(rough, 0, 1) < acf_along(smooth, 0, 1)


# ---------------------------------------------------------------- model grids


def het(**kw: object) -> HeterogeneitySpec:
    base = {"enabled": True, "seed": 5, "structure": {"corr_len_km": [1.0, 0.6, 0.3]}}
    return HeterogeneitySpec.model_validate({**base, **kw})


def test_no_heterogeneity_is_layered_model() -> None:
    grid = build_model_grid(MODEL, GEOM, None, 3)
    _, _, zc = cell_centres(GEOM)
    np.testing.assert_allclose(grid.vp[7, 9], MODEL.velocity_at(zc, "P"))
    np.testing.assert_allclose(grid.vs[0, 0], MODEL.velocity_at(zc, "S"))


def test_realizations_reproducible_and_distinct() -> None:
    spec = het(vp_pct=5.0)
    a = build_model_grid(MODEL, GEOM, spec, 1)
    np.testing.assert_array_equal(a.vp, build_model_grid(MODEL, GEOM, spec, 1).vp)
    assert not np.array_equal(a.vp, build_model_grid(MODEL, GEOM, spec, 2).vp)


def test_vp_amplitude_per_layer() -> None:
    spec = het(vp_pct=[8.0, 4.0, 0.0])
    grid = build_model_grid(MODEL, GEOM, spec, 1)
    plain = build_model_grid(MODEL, GEOM, None, 1)
    _, _, zc = cell_centres(GEOM)
    rel = grid.vp / plain.vp - 1
    for k, sd in enumerate([0.08, 0.04, 0.0]):
        top = MODEL.tops_km[k]
        bot = MODEL.tops_km[k + 1] if k + 1 < MODEL.n_layers else np.inf
        inside = (zc >= top + 0.05) & (zc < bot - 0.05)
        assert rel[:, :, inside].std() == pytest.approx(sd, abs=0.25 * sd + 1e-12)


def test_vpvs_perturbed_independently_and_bounded() -> None:
    spec = het(vp_pct=5.0, vpvs_sd=0.05, structure={"corr_len_km": [0.3, 0.3, 0.15]})
    plain = build_model_grid(MODEL, GEOM, None, 1)
    corrs = []
    for index in range(1, 6):
        grid = build_model_grid(MODEL, GEOM, spec, index)
        vpvs = grid.vp / grid.vs
        assert vpvs.min() >= MIN_VPVS + 0.01 - 1e-12
        dvp = (grid.vp / plain.vp - 1).ravel()
        dvpvs = (vpvs - plain.vp / plain.vs).ravel()
        corrs.append(np.corrcoef(dvp, dvpvs)[0, 1])
    assert abs(np.mean(corrs)) < 0.1


def test_per_layer_fields_change_across_interface() -> None:
    """Independent fields per layer: no correlation across an interface (shared field: high)."""
    _, _, zc = cell_centres(GEOM)
    above = np.searchsorted(zc, MODEL.tops_km[1]) - 1  # last cell in layer 0
    plain = build_model_grid(MODEL, GEOM, None, 1)
    structure = {"corr_len_km": [0.3, 0.3, 0.3]}
    corr: dict[bool, float] = {}
    for per_layer in (True, False):
        spec = het(vp_pct=5.0, per_layer=per_layer, structure=structure)
        values = []
        for index in range(1, 6):
            rel = build_model_grid(MODEL, GEOM, spec, index).vp / plain.vp - 1
            values.append(np.corrcoef(rel[:, :, above].ravel(), rel[:, :, above + 1].ravel())[0, 1])
        corr[per_layer] = float(np.mean(values))
    assert corr[False] > 0.5 and abs(corr[True]) < 0.1


def test_interface_relief_keeps_layer_order() -> None:
    spec = het(interface_sd_km=[0.15, 0.15])
    grid = build_model_grid(MODEL, GEOM, spec, 1)
    plain = build_model_grid(MODEL, GEOM, None, 1)
    # Layers are identified by their (unperturbed) velocity; per column they stay in order.
    layer = np.searchsorted([3.0, 4.0, 5.0], grid.vp, side="right") - 1
    assert np.all(np.diff(layer, axis=2) >= 0)
    # The interfaces actually move: the layer-0/1 boundary depth varies laterally.
    first_l1 = np.argmax(layer >= 1, axis=2)
    assert first_l1.std() > 1.0
    assert not np.array_equal(grid.vp, plain.vp)


def test_spec_validation() -> None:
    with pytest.raises(ConfigError, match="every sd is zero"):
        HeterogeneitySpec(enabled=True)
    with pytest.raises(ConfigError, match="has 2 values"):
        het(vp_pct=[1.0, 2.0]).per_layer_values("vp_pct", 3)
    with pytest.raises(ConfigError, match="mode: 3d"):
        het(vp_pct=1.0).check_grid(GridGeometry("2d", 2, 10, 10, 0, 0, 0, 0.05))
    with pytest.raises(ConfigError, match="two grid"):
        het(vp_pct=1.0, structure={"corr_len_km": [1.0, 1.0, 0.05]}).check_grid(GEOM)


def test_writer_round_trip(tmp_path: Path) -> None:
    grid = build_model_grid(MODEL, GEOM, het(vp_pct=5.0), 1)
    root = write_model_grid(tmp_path / "m.P.mod", grid.vp, GEOM, LocalFrame(-38.6, 176.1))
    back = read_grid(root)
    np.testing.assert_allclose(back.velocity(), grid.vp, rtol=1e-6)
    assert "TRANS_MERC" in back.transform


# ---------------------------------------------------------------- with NonLinLoc


def test_writer_matches_vel2grid(tmp_path: Path, nll_bin: Path) -> None:
    """Direct writer == Vel2Grid for the layered model (header and values)."""
    path = write_example(
        tmp_path,
        nll_bin,
        velocity={"perturb": {"enabled": False}, "heterogeneity": {"enabled": False}},
    )
    cfg = load_config(path)
    v2g = run_vel_grids(cfg)
    frame = json.loads((cfg.output_dir / "grids" / "grids.json").read_text())["frame"]
    grid = build_model_grid(LayeredModel.read(cfg.velocity.base), v2g.geometry, None, 0)
    for phase, values in (("P", grid.vp), ("S", grid.vs)):
        ref = v2g.roots[0][phase]
        own = write_model_grid(
            tmp_path / f"own.{phase}.mod",
            values,
            v2g.geometry,
            LocalFrame(frame["lat0"], frame["long0"]),
        )
        assert (
            own.with_name(own.name + ".hdr").read_text()
            == ref.with_name(ref.name + ".hdr").read_text()
        )
        np.testing.assert_allclose(read_grid(own).data, read_grid(ref).data, rtol=1e-6)


def test_heterogeneity_increases_travel_time_spread(tmp_path: Path, nll_bin: Path) -> None:
    """Streamed run with 3D heterogeneity only (no 1D perturbation): travel times now vary
    across the ensemble, and keep_grids: none leaves no grids behind."""
    path = write_example(
        tmp_path,
        nll_bin,
        nll={"keep_grids": "none"},
        velocity={
            "perturb": {"enabled": False, "n_realizations": 3},
            "heterogeneity": {"enabled": True},
        },
    )
    cfg = load_config(path)
    sens = run_sensitivity(cfg)
    assert len(sens.realizations) == 4  # base + 3 heterogeneous copies
    assert not list((cfg.output_dir / "grids").glob("model_*"))
    assert not list((cfg.output_dir / "traveltimes").glob("model_*"))
    spread = sens.T[1:].std(axis=0)  # among the heterogeneous models
    assert np.median(spread) > 1e-3  # > 1 ms
    assert not np.allclose(sens.T[1], sens.T[2])
