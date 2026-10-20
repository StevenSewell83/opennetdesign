"""Relative detectability."""

import json
from pathlib import Path

import numpy as np
import pytest
from example_config import write_example
from test_design import VP, VPVS, ring_with_decoys, straight_ray, target_points

from netdesign.config import load_config
from netdesign.design import DesignSpec, Evaluator
from netdesign.detection import DetectionSpec, SiteType, relative_detection
from netdesign.errors import ConfigError
from netdesign.information import PickSpec
from netdesign.pipeline import run_design, run_sensitivity


def constant_profile(q: float):
    def profile(z):
        one = np.ones_like(z)
        return VP * one, VP / VPVS * one, q * one, q / 2 * one

    return profile


def det(sens, stations, q=1e12, **kw):
    defaults = {
        "site_types": {"surface": SiteType(kappa_s=0.0, amplification=1.0)},
        "reference_rank": 1,  # simple geometry checks use the single best arrival
    }
    spec = DetectionSpec(**{**defaults, **kw})
    n_r = sens.T.shape[0]
    return relative_detection(sens, stations, [constant_profile(q)] * n_r, [0.0] * n_r, 1.0, spec)


def test_geometric_spreading_only() -> None:
    stations = ring_with_decoys()
    sens = straight_ray(stations, target_points(), n_r=1)
    p, q = det(sens, stations)
    dist = np.linalg.norm(sens.sources.xyz[None] - stations[:, None], axis=-1)  # (S, J)
    # Per-phase reference: each phase's relative level = -20 log10(R / R_min).
    expected = -20 * np.log10(dist / dist.min(axis=0))
    np.testing.assert_allclose(q[0, :, 0], expected, atol=1e-9)
    np.testing.assert_allclose(q[0, :, 1], expected, atol=1e-9)
    assert p.shape == sens.T.shape and np.all((p >= 0) & (p <= 1))
    # One reference over both phases: P sits 20 log10(S/P) below S everywhere.
    _, q_all = det(sens, stations, reference="all_phases")
    np.testing.assert_allclose(q_all[0, :, 1] - q_all[0, :, 0], 20 * np.log10(5.0), atol=1e-9)
    np.testing.assert_allclose(q_all[0, :, 1], expected, atol=1e-9)


def test_far_decoys_rarely_detected_and_margin_sets_half_point() -> None:
    stations = ring_with_decoys()
    sens = straight_ray(stations, target_points(), n_r=1)
    p, q = det(sens, stations, q=100.0, margin_db=10.0)
    assert p[0, 8:].mean() < p[0, :8].mean()  # decoys 12 km away vs the ring at 3 km
    i = np.unravel_index(np.argmin(np.abs(q + 10.0)), q.shape)
    assert p[i] == pytest.approx(0.5, abs=0.02)


def test_attenuation_lowers_distant_arrivals() -> None:
    stations = ring_with_decoys()
    sens = straight_ray(stations, target_points(), n_r=1)
    _, q_none = det(sens, stations)
    _, q_att = det(sens, stations, q=50.0)
    assert np.mean(q_att[0, 8:]) < np.mean(q_none[0, 8:]) - 3.0


def test_quieter_site_gains_its_noise_difference() -> None:
    stations = ring_with_decoys()
    sens = straight_ray(stations, target_points(), n_r=1)
    surface = SiteType(kappa_s=0, amplification=1)
    types = {"surface": surface, "quiet": SiteType(kappa_s=0, amplification=1, noise_db=-20)}
    flat = [constant_profile(1e12)]
    _, q0 = relative_detection(
        sens, stations, flat, [0.0], 1.0, DetectionSpec(site_types=types, reference_rank=1)
    )
    _, q1 = relative_detection(
        sens,
        stations,
        flat,
        [0.0],
        1.0,
        DetectionSpec(site_types=types, sites={"S03": "quiet"}, reference_rank=1),
    )
    # Relative to any other station, S03 is exactly 20 dB better than before.
    np.testing.assert_allclose((q1[0, 3] - q1[0, 0]) - (q0[0, 3] - q0[0, 0]), 20.0, atol=1e-9)


def test_zero_detection_equals_excluding_station() -> None:
    stations = ring_with_decoys()
    sens = straight_ray(stations, target_points())
    picks = PickSpec(sigma_s={"P": 0.005, "S": 0.01})
    detection = np.ones(sens.T.shape)
    detection[:, 5] = 0.0
    ev_det = Evaluator(sens, picks, DesignSpec(), detection)
    ev = Evaluator(sens, picks, DesignSpec())
    assert ev_det.score([0, 2, 5, 6]) == pytest.approx(ev.score([0, 2, 6]), rel=1e-9)


def test_unknown_site_type() -> None:
    with pytest.raises(ConfigError, match="undefined site type"):
        DetectionSpec(sites={"C01": "cave"})


def test_design_with_detection_end_to_end(tmp_path: Path, nll_bin: Path) -> None:
    cfg = load_config(
        write_example(
            tmp_path,
            nll_bin,
            grid={"dx_km": 0.1},
            velocity={"perturb": {"n_realizations": 4}},
            design={"n_random": 10, "n_restarts": 1},
        )
    )
    assert cfg.detection.enabled
    run_sensitivity(cfg)
    run_design(cfg)
    d = json.loads((cfg.output_dir / "design" / "design.json").read_text())
    rows = d["detection"]["margin_sensitivity"]
    assert [r["margin_db"] for r in rows] == [5.0, 10.0, 15.0, 20.0]
    # Smaller margins (smaller events): fewer picks, larger error.
    crit = [r["criterion_m"] for r in rows]
    assert crit[0] >= crit[-1]
    q = cfg.output_dir / "qc"
    for fig in ("detection_by_station.png", "detection_picks_map.png"):
        assert (q / fig).stat().st_size > 10_000
    checks = json.loads((q / "design.json").read_text())["checks"]
    assert any(c["name"] == "target points with enough expected picks" for c in checks)


def test_unknown_station_in_sites(tmp_path: Path, nll_bin: Path) -> None:
    cfg = load_config(
        write_example(
            tmp_path, nll_bin, grid={"dx_km": 0.2}, detection={"sites": {"NOPE": "borehole"}}
        )
    )
    run_sensitivity(cfg)
    with pytest.raises(ConfigError, match="NOPE"):
        run_design(cfg)


def test_reference_rank_limits_one_exceptional_site() -> None:
    """A site made 30 dB quieter drags every other station's level down by up to 30 dB with
    rank 1, but only by the small gaps between the top arrivals with rank 4."""
    stations = ring_with_decoys()
    sens = straight_ray(stations, target_points(), n_r=1)
    base = {
        "surface": SiteType(kappa_s=0, amplification=1),
        "quiet": SiteType(kappa_s=0, amplification=1, noise_db=-30),
    }
    flat = [constant_profile(1e12)]
    others = list(range(1, len(stations)))
    shift = {}
    for rank in (1, 4):
        _, q0 = relative_detection(
            sens, stations, flat, [0.0], 1.0, DetectionSpec(site_types=base, reference_rank=rank)
        )
        _, q1 = relative_detection(
            sens,
            stations,
            flat,
            [0.0],
            1.0,
            DetectionSpec(site_types=base, reference_rank=rank, sites={"S00": "quiet"}),
        )
        shift[rank] = float(np.median(np.abs(q1[0, others] - q0[0, others])))
        # The rank-th strongest arrival for each event and phase sits at 0 dB.
        np.testing.assert_allclose(np.sort(q0[0], axis=0)[-rank], 0.0, atol=1e-9)
    assert shift[1] > 10.0
    assert shift[4] < shift[1] / 5
