"""NonLinLoc validation of the location-error predictions."""

from pathlib import Path

import numpy as np
import pytest
from example_config import write_example

from netdesign.config import load_config
from netdesign.errors import ConfigError
from netdesign.pipeline import run_sensitivity, run_validate
from netdesign.validate import ValidateSpec, nll_data_covariance, parse_summary

HYP = """NLLOC "./loc/x.20260101.000300.grid0" "LOCATED" "Location completed."
HYPOCENTER  x 1.5 y -0.25 z 2.0  OT 1.2  ix -1 iy -1 iz -1 MAXIMUM_LIKELIHOOD
GEOGRAPHIC  OT 2026 01 01  00 03 0.012500  Lat -38.6 Long 176.1 Depth 2.0
QUALITY  Pmax 1e10 MFmin 1 MFmax 2 RMS 0.0042 Nphs 20 Gap 40 Dist 1.2 Mamp -9.9 0 Mdur -9.9 0
STATISTICS  ExpectX 1.5 Y -0.25 Z 2.0  CovXX 0.01 XY 0 XZ 0 YY 0.01 YZ 0 ZZ 0.04 EllAz1 0
END_NLLOC

NLLOC "./loc/x.20260101.000400.grid0" "REJECTED" "too few phases"
GEOGRAPHIC  OT 2026 01 01  00 04 0.0  Lat -38.6 Long 176.1 Depth 2.0
END_NLLOC
"""


def test_parse_summary(tmp_path: Path) -> None:
    path = tmp_path / "s.hyp"
    path.write_text(HYP)
    events = parse_summary(path)
    assert set(events) == {3, 4}
    assert events[3]["status"] == "LOCATED" and events[3]["xyz"] == (1.5, -0.25, 2.0)
    assert events[3]["rms_s"] == 0.0042 and events[3]["nphs"] == 20
    assert events[3]["cov"] == {"xx": 0.01, "xy": 0, "xz": 0, "yy": 0.01, "yz": 0, "zz": 0.04}
    assert events[4]["status"] == "REJECTED"


def test_nll_covariance_matches_nlloc_rules() -> None:
    sigma = np.array([0.005, 0.01, 0.005, 0.01])
    phases = ["P", "S", "P", "S"]
    xyz = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    plain = nll_data_covariance(sigma, phases, xyz, ValidateSpec())
    np.testing.assert_allclose(plain, np.diag(sigma**2))
    spec = ValidateSpec(locgau_sigma_s=0.02, locgau_corr_len_km=1.0)
    cov = nll_data_covariance(sigma, phases, xyz, spec)
    np.testing.assert_allclose(np.diag(cov), sigma**2 + 0.02**2)
    assert cov[0, 2] == pytest.approx(0.02**2 * np.exp(-0.5))  # same phase, 1 km apart
    assert cov[0, 1] == 0.0 and cov[0, 3] == 0.0  # P-S pairs are uncorrelated


def small(tmp_path: Path, nll_bin: Path, **over) -> Path:
    return write_example(tmp_path, nll_bin, grid={"dx_km": 0.1}, **over)


def test_noise_free_base_model_recovers_truth(tmp_path: Path, nll_bin: Path) -> None:
    """True model = location model, no noise: NonLinLoc finds the true hypocentres."""
    path = small(
        tmp_path,
        nll_bin,
        velocity={"perturb": {"enabled": False}, "heterogeneity": {"enabled": False}},
        validation={"n_events": 20, "pick_noise": False},
    )
    cfg = load_config(path)
    run_sensitivity(cfg)
    result = run_validate(cfg)
    assert result.summary["n_located"] == 20
    for ax in ("x", "y", "z"):
        # Only interpolation and oct-tree resolution remain (a few metres).
        assert result.summary[ax]["rms_nll_m"] < 10.0
        assert result.summary[ax]["rms_linear_m"] < 1e-6


def test_nll_matches_linear_prediction(tmp_path: Path, nll_bin: Path) -> None:
    """Heterogeneous truth models: NonLinLoc's mislocations follow the linear prediction."""
    path = small(
        tmp_path,
        nll_bin,
        velocity={"perturb": {"n_realizations": 6}},
        validation={"n_events": 40},
    )
    cfg = load_config(path)
    run_sensitivity(cfg)
    s = run_validate(cfg).summary
    assert s["n_located"] == 40
    for ax in ("x", "y", "z"):
        a = s[ax]
        assert a["corr_nll_vs_linear"] > 0.9
        assert 0.6 < a["rms_nll_m"] / a["rms_predicted_achievable_m"] < 1.6
        # NonLinLoc does not know the correlated model error: it does worse than GLS.
        assert a["rms_predicted_design_m"] < a["rms_predicted_achievable_m"]
    assert 0.4 < s["fraction_within_achievable_68pct"] < 0.9
    out = cfg.output_dir / "validation"
    assert (out / "events.csv").is_file() and (out / "summary.json").is_file()


def test_requires_sensitivity_and_base_grids(tmp_path: Path, nll_bin: Path) -> None:
    cfg = load_config(small(tmp_path, nll_bin, nll={"keep_grids": "none"}))
    with pytest.raises(ConfigError, match="run `netdesign sensitivity`"):
        run_validate(cfg)
    run_sensitivity(cfg)
    with pytest.raises(ConfigError, match="keep_grids"):
        run_validate(cfg)


def test_unknown_station(tmp_path: Path, nll_bin: Path) -> None:
    cfg = load_config(small(tmp_path, nll_bin, validation={"stations": ["C01", "NOPE"]}))
    run_sensitivity(cfg)
    with pytest.raises(ConfigError, match="NOPE"):
        run_validate(cfg)
