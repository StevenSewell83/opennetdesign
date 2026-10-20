"""Sensitivities against the analytic linear-gradient solution (needs NonLinLoc)."""

from pathlib import Path

import numpy as np
import pytest
from analytic import PHASE_PARAMS, gradient_time, write_case
from example_config import write_example
from typer.testing import CliRunner

from netdesign.cli import app
from netdesign.config import load_config
from netdesign.errors import ConfigError
from netdesign.pipeline import run_sensitivity
from netdesign.sensitivity import Sensitivity

TARGET = {
    "long": [176.105, 176.125],
    "lat": [-38.615, -38.6],
    "depth_km": [0.5, 2.5],
    "sampling": "random",
    "n_points": 300,
    "seed": 1,
}


def exact_gradient(phase: str, station: tuple[float, float, float], xyz: np.ndarray) -> np.ndarray:
    v_top, grad = PHASE_PARAMS[phase]
    sx, sy, sz = station
    h = 1e-5

    def t(p: np.ndarray) -> np.ndarray:
        return gradient_time(v_top, grad, np.array([[sx, sy]]), sz, p[:, :2], p[:, 2])

    cols = []
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        cols.append((t(xyz + d) - t(xyz - d)) / (2 * h))
    return np.stack(cols, axis=-1)


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_gradients_match_exact(tmp_path: Path, nll_bin: Path, mode: str) -> None:
    cfg = load_config(write_case(tmp_path, nll_bin, mode, target=TARGET))
    sens = run_sensitivity(cfg)
    assert sens.T.shape == (1, 2, 2, 300) and sens.G.shape == (1, 2, 2, 300, 3)
    assert (cfg.output_dir / "sensitivity" / "sensitivity.npz").is_file()

    stations = (cfg.output_dir / "stations_local.csv").read_text().splitlines()[1:]
    for is_, row in enumerate(stations):
        name, *xyz = row.split(",")
        assert name == sens.stations[is_]
        sta = tuple(float(v) for v in xyz)
        dist = np.linalg.norm(sens.sources.xyz - np.array(sta), axis=1)
        far = dist > 0.5
        for ip, phase in enumerate(sens.phases):
            exact = exact_gradient(phase, sta, sens.sources.xyz)
            got = sens.G[0, is_, ip]
            # Vector error relative to slowness magnitude.
            rel = np.linalg.norm(got - exact, axis=1) / np.linalg.norm(exact, axis=1)
            # Measured (dx = 50 m, h = dx): median 1.4-1.9 %, p95 <= 2.8 %, max <= 3.7 %.
            assert np.median(rel[far]) < 0.025
            assert np.percentile(rel[far], 95) < 0.04


def test_target_outside_grid(tmp_path: Path, nll_bin: Path) -> None:
    deep = {**TARGET, "depth_km": [0.5, 2.99]}
    cfg = load_config(write_case(tmp_path, nll_bin, "2d", target=deep))
    with pytest.raises(ConfigError, match="outside the travel-time grid"):
        run_sensitivity(cfg)


def test_missing_target(tmp_path: Path, nll_bin: Path) -> None:
    cfg = load_config(write_case(tmp_path, nll_bin, "2d"))
    with pytest.raises(ConfigError, match="'target'"):
        run_sensitivity(cfg)


def test_cli_on_example(tmp_path: Path, nll_bin: Path) -> None:
    """The example as shipped (3D with heterogeneity), just smaller."""
    path = write_example(tmp_path, nll_bin)
    result = CliRunner().invoke(app, ["sensitivity", str(path)])
    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    sens = Sensitivity.load(out / "sensitivity" / "sensitivity.npz")
    assert sens.T.shape[:3] == (3, 10, 2)
    assert (out / "sensitivity" / "sources.csv").is_file()
    # keep_grids: base -> only the base model's grids remain.
    assert sorted(d.name for d in (out / "grids").glob("model_*")) == ["model_000"]
    assert sorted(d.name for d in (out / "traveltimes").glob("model_*")) == ["model_000"]
