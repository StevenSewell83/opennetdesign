"""Vel2Grid runs (needs NonLinLoc) and grid reading."""

import json
from pathlib import Path

import numpy as np
import pytest
from example_config import write_example
from typer.testing import CliRunner

from netdesign.cli import app
from netdesign.errors import ConfigError, DataError
from netdesign.nll import NLLSpec, find_program, read_grid
from netdesign.velocity import LayeredModel


def test_find_program_bad_dir(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="bin_dir"):
        find_program("Vel2Grid", NLLSpec(bin_dir=tmp_path))


def test_read_grid_size_mismatch(tmp_path: Path) -> None:
    (tmp_path / "g.hdr").write_text("2 2 2 0 0 0 1 1 1 SLOW_LEN FLOAT\nTRANSFORM NONE\n")
    np.zeros(7, dtype="<f4").tofile(tmp_path / "g.buf")
    with pytest.raises(DataError, match="expects 8"):
        read_grid(tmp_path / "g")


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_grids_match_models(tmp_path: Path, nll_bin: Path, mode: str) -> None:
    path = write_example(
        tmp_path,
        nll_bin,
        grid={"mode": mode},
        velocity={"perturb": {"n_realizations": 3}, "heterogeneity": {"enabled": False}},
    )

    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["grids", str(path), "-o", str(out)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "grids" / "grids.json").read_text())
    assert len(manifest["realizations"]) == 4
    assert (out / "stations.gtsrce").read_text().count("GTSRCE") == 10

    for idx, roots in manifest["realizations"].items():
        model = LayeredModel.read(out / "models" / f"model_{idx}.csv")
        for phase, rel in roots.items():
            grid = read_grid(out / "grids" / rel)
            assert grid.nx == (2 if mode == "2d" else grid.nx)
            assert "TRANS_MERC" in grid.transform
            z_mid = grid.z0 + (np.arange(grid.nz) + 0.5) * grid.dz
            v = grid.velocity()
            assert np.ptp(v, axis=(0, 1)).max() < 1e-6  # laterally uniform
            expected = model.velocity_at(z_mid, phase)
            # Exact away from interfaces; a cell straddling one may take either layer.
            near = np.min(np.abs(z_mid[:, None] - model.tops_km[None, 1:]), axis=1) < grid.dz
            np.testing.assert_allclose(v[0, 0][~near], expected[~near], rtol=1e-5)
