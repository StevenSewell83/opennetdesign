"""QC checks, figures and report."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from example_config import write_example

from netdesign.config import load_config
from netdesign.frame import GridGeometry
from netdesign.pipeline import run_sensitivity, run_validate
from netdesign.qc.checks import Check, write_checks
from netdesign.qc.report import build_report
from netdesign.qc.stages import _measured_lengths, _theoretical_efold, write_candidates_table
from netdesign.stations import Candidates
from netdesign.velocity.random_field import CorrelationSpec


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (CorrelationSpec(correlation="exponential"), 1.0),
        (CorrelationSpec(correlation="gaussian"), 1.0),
        (CorrelationSpec(correlation="von_karman", hurst=0.5), 1.0),
    ],
)
def test_theoretical_efold(spec: CorrelationSpec, expected: float) -> None:
    assert _theoretical_efold(spec, 1.0) == pytest.approx(expected, rel=1e-6)


def test_von_karman_efold_shorter_for_rougher_fields() -> None:
    rough = _theoretical_efold(CorrelationSpec(hurst=0.1), 1.0)
    smooth = _theoretical_efold(CorrelationSpec(hurst=0.9), 1.0)
    assert 0 < rough < smooth


def test_measured_lengths_recover_spec() -> None:
    spec = CorrelationSpec(correlation="exponential", corr_len_km=(0.6, 0.25, 0.15), strike_deg=30)
    geom = GridGeometry("3d", 160, 160, 60, 0, 0, 0, 0.05)
    got = _measured_lengths(spec, geom, seed=4)
    for name, a in zip(("along", "across", "vertical"), spec.corr_len_km, strict=True):
        assert got[name] == pytest.approx(a, rel=0.3), name


def test_candidates_table(tmp_path: Path) -> None:
    c = Candidates(
        ("A", "B"), np.array([176.1, 176.2]), np.array([-38.6, -38.7]), np.array([300.0, 250.0])
    )
    write_candidates_table(c, np.array([5.0, 1.0]), np.array([2.0, -0.5]), tmp_path, [1, None])
    rows = (tmp_path / "candidates_outcome.csv").read_text().splitlines()
    assert rows[0].split(",")[4:8] == [
        "status",
        "selection_order",
        "greedy_order",
        "gain_if_added_pct",
    ]
    assert rows[1] == "A,176.100000,-38.600000,300.0,selected,1,,,5.00,2.00"
    assert rows[2].split(",")[4:6] == ["candidate", ""]
    gj = json.loads((tmp_path / "candidates_outcome.geojson").read_text())
    assert gj["features"][1]["properties"]["status"] == "candidate"
    assert gj["features"][0]["geometry"]["coordinates"] == [176.1, -38.6]


def test_report_from_minimal_files(tmp_path: Path) -> None:
    write_checks(
        tmp_path / "qc" / "inputs.json",
        [Check("inputs", "a check", "pass", 1, "1"), Check("inputs", "b", "warn", 2, "1", "note")],
        {"figures": []},
    )
    html = build_report(tmp_path).read_text()
    assert "<h2>Inputs</h2>" in html and "✓ PASS" in html and "! WARN" in html


def test_qc_end_to_end(tmp_path: Path, nll_bin: Path) -> None:
    cfg = load_config(
        write_example(
            tmp_path,
            nll_bin,
            grid={"dx_km": 0.1},
            velocity={"perturb": {"n_realizations": 4}},
            validation={"n_events": 20},
        )
    )
    run_sensitivity(cfg)
    run_validate(cfg)
    q = cfg.output_dir / "qc"
    for stage in (
        "inputs",
        "ensemble",
        "sensitivity",
        "validation",
        "run_sensitivity",
        "run_validate",
    ):
        data = json.loads((q / f"{stage}.json").read_text())
        assert not [c for c in data["checks"] if c["status"] == "fail"], stage
        for fig in data.get("figures", []):
            assert (q / fig).stat().st_size > 10_000, fig
    run = json.loads((q / "run_sensitivity.json").read_text())
    assert run["versions"]["nonlinloc"] and run["peak_disk_mb"] > 0
    assert set(run["timings_s"]) >= {"setup", "grids_traveltimes_sensitivity", "qc"}
    html = (q / "report.html").read_text()
    assert html.count("<img") >= 9
    assert (q / "candidates_outcome.csv").is_file()
    assert math.isfinite(
        json.loads((q / "sensitivity.json").read_text())["median_uncertainty"][
            "achievable (NonLinLoc-like)"
        ]["horizontal 1-sigma (m)"]
    )


def test_qc_disabled(tmp_path: Path, nll_bin: Path) -> None:
    cfg = load_config(write_example(tmp_path, nll_bin, grid={"dx_km": 0.2}, qc={"enabled": False}))
    run_sensitivity(cfg)
    assert not (cfg.output_dir / "qc").exists()
