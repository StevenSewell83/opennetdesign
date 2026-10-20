"""Grid2Time travel times against the analytic linear-gradient solution (needs NonLinLoc)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from analytic import PHASE_PARAMS, gradient_time, write_case

from netdesign.config import load_config
from netdesign.nll import TravelTimeGrid
from netdesign.pipeline import run_traveltimes

# mode -> (max rel error 0.3-1 km, max rel error > 1 km, max mean bias)
TOLERANCES = {"2d": (0.02, 0.0125, 0.005), "3d": (0.035, 0.02, 0.008)}


@pytest.fixture
def config(tmp_path: Path, nll_bin: Path):
    return lambda mode: write_case(tmp_path, nll_bin, mode)


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_times_match_gradient_solution(config, mode: str) -> None:
    cfg = load_config(config(mode))
    times = run_traveltimes(cfg)
    manifest = json.loads((cfg.output_dir / "traveltimes" / "traveltimes.json").read_text())
    assert set(manifest["realizations"]["000"]) == {"P", "S"}
    assert not list((cfg.output_dir / "traveltimes" / "model_000").glob("*.mod.*"))

    rng = np.random.default_rng(3)
    for phase, (v_top, grad) in PHASE_PARAMS.items():
        for sta, root in times.roots[0][phase].items():
            tt = TravelTimeGrid.read(root)
            assert root.with_name(root.name + ".buf").stat().st_size == tt.grid.data.nbytes
            assert tt.station == sta
            sx, sy, sz = tt.station_xyz
            x = sx + rng.uniform(-2.5, 2.5, 400)
            y = sy + rng.uniform(-2.5, 2.5, 400)
            z = rng.uniform(0.0, 2.9, 400)
            t = tt.at(x, y, z)
            dist = np.hypot(np.hypot(x - sx, y - sy), z - sz)
            ok = np.isfinite(t) & (dist > 0.3)
            assert ok.sum() > 100
            exact = gradient_time(
                v_top, grad, np.array([[sx, sy]]), sz, np.stack([x, y], -1)[ok], z[ok]
            )
            err = t[ok] - exact
            rel = np.abs(err) / exact
            far = dist[ok] > 1.0
            # Accuracy measured with plfd_eps = 0.01 is documented in netdesign.nll.grid2time.
            near_tol, far_tol, bias_tol = TOLERANCES[mode]
            assert rel[~far].max() < near_tol, f"{phase} {sta}: near rel {rel[~far].max():.4f}"
            assert rel[far].max() < far_tol, f"{phase} {sta}: far rel {rel[far].max():.4f}"
            assert abs(np.mean(err / exact)) < bias_tol, f"{phase} {sta}: bias"


def test_angles_option(config) -> None:
    path = config("2d")
    raw = yaml.safe_load(path.read_text())
    raw["nll"]["angles"] = True
    raw["phases"] = ["P"]
    path.write_text(yaml.safe_dump(raw))
    times = run_traveltimes(load_config(path))
    for root in times.roots[0]["P"].values():
        angle = root.with_name(root.name.removesuffix(".time") + ".angle.buf")
        assert angle.is_file()
