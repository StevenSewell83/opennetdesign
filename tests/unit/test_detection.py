"""Detectability: relative and magnitude modes."""

import json
from pathlib import Path

import numpy as np
import pytest
from example_config import write_example
from scipy.special import ndtr
from test_design import VP, VPVS, ring_with_decoys, straight_ray, target_points

from netdesign.config import load_config
from netdesign.design import DesignSpec, Evaluator, optimise
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


def _magnitude(sens, stations, magnitude=0.0, **kw):
    site = {"surface": SiteType(kappa_s=0.0, amplification=1.0, noise_psd_db=-140.0)}
    return det(sens, stations, mode="magnitude", magnitude=magnitude, site_types=site, **kw)


def test_magnitude_mode_matches_hand_calculation() -> None:
    stations = ring_with_decoys()
    sens = straight_ray(stations, target_points(), n_r=1)
    f, mw, rho, drop, window, thr = 10.0, 0.5, 2700.0, 1e6, 0.5, 10.0
    _, q = _magnitude(
        sens, stations, mw, frequency_band_hz=(f, f), n_frequencies=1, snr_threshold_db=thr
    )
    m0 = 10 ** (1.5 * mw + 9.1)
    vs = VP / VPVS * 1000
    fc = 0.4906 * vs * (drop / m0) ** (1 / 3)
    dist = np.linalg.norm(sens.sources.xyz[None] - stations[:, None], axis=-1) * 1000  # (S, J)
    for ip, (v, rad) in enumerate(((VP * 1000, 0.52), (vs, 0.63))):
        u = rad * m0 / (4 * np.pi * rho * v**3 * dist) / (1 + (f / fc) ** 2)
        signal = 2 * (2 * np.pi * f * u) ** 2 / window
        noise = 10 ** (-140.0 / 10) / (2 * np.pi * f) ** 2
        np.testing.assert_allclose(q[0, :, ip], 10 * np.log10(signal / noise) - thr, atol=1e-9)


def test_magnitude_mode_bigger_and_nearer_events_detected_more() -> None:
    stations = ring_with_decoys()
    sens = straight_ray(stations, target_points(), n_r=1)
    p0, q0 = _magnitude(sens, stations, 0.0, q=200.0)
    p1, q1 = _magnitude(sens, stations, 1.0, q=200.0)
    assert np.all(q1 > q0) and np.all(p1 >= p0)
    assert p0[0, 8:].mean() < p0[0, :8].mean()  # far decoys
    np.testing.assert_allclose(p0, ndtr(q0 / 5.0))  # 50 % at the SNR threshold, width 5 dB


def test_magnitude_mode_needs_noise_psd() -> None:
    with pytest.raises(ConfigError, match="noise_psd_db"):
        DetectionSpec(mode="magnitude", site_types={"surface": SiteType()})
    assert DetectionSpec(mode="magnitude").sensitivity_values[0] == "magnitude"


def _uniform_detection(sens, p_p: float, p_s: float) -> np.ndarray:
    det = np.empty(sens.T.shape)
    det[:, :, 0], det[:, :, 1] = p_p, p_s
    return det


@pytest.mark.parametrize(("k", "a"), [(4, 1), (3, 0), (5, 2)])
def test_located_probability_matches_enumeration(k: int, a: int) -> None:
    import itertools

    sens = straight_ray(ring_with_decoys(), target_points(), n_r=2)
    det = np.random.default_rng(4).uniform(0.05, 0.95, sens.T.shape)
    spec = DesignSpec(locate_min_picks=k, locate_min_p_picks=a)
    ev = Evaluator(sens, PickSpec(), spec, det)
    stations = [0, 3, 5, 9]
    p = det.mean(axis=0)[stations]  # (n, P, J)
    expected = np.zeros(p.shape[-1])
    for outcome in itertools.product([0, 1], repeat=2 * len(stations)):
        o = np.array(outcome).reshape(len(stations), 2)
        prob = np.prod(np.where(o[:, :, None] == 1, p, 1 - p), axis=(0, 1))
        if o.sum() >= k and o[:, 0].sum() >= a:
            expected += prob
    np.testing.assert_allclose(ev.located_probability(stations), expected, atol=1e-12)


def test_certain_picks_locate_everything_and_change_nothing() -> None:
    sens = straight_ray(ring_with_decoys(), target_points())
    ones = np.ones(sens.T.shape)
    with_loc = Evaluator(sens, PickSpec(), DesignSpec(), ones)
    without = Evaluator(sens, PickSpec(), DesignSpec(locatability=False), ones)
    for net in ([0, 2, 4], [1, 3, 5, 7, 9]):
        assert with_loc.evaluate(net)[1] == pytest.approx(1.0)
        assert with_loc.score(net) == pytest.approx(without.score(net), rel=1e-12)
    # two stations x 2 phases = 4 picks: located; one station: never
    assert with_loc.evaluate([0, 2])[1] == pytest.approx(1.0)
    assert with_loc.evaluate([0])[1] == pytest.approx(0.0)


def test_too_small_design_event_is_flagged_not_recommended() -> None:
    sens = straight_ray(ring_with_decoys(), target_points())
    det = _uniform_detection(sens, 0.05, 0.2)  # picks unlikely everywhere
    spec = DesignSpec(n_min=3, n_max=6, n_random=10, n_restarts=1)
    result = optimise(
        Evaluator(sens, PickSpec(), spec, det), sens.stations, spec, np.random.default_rng(0)
    )
    assert not any(net.meets_located for net in result.networks.values())
    assert result.recommended_n == 6
    assert any("too small" in note for note in result.notes)
    assert all(net.criterion_m < 1e6 for net in result.networks.values())  # reported unpenalised
    relaxed = spec.model_copy(update={"min_located_fraction": 0.0})
    ok = optimise(
        Evaluator(sens, PickSpec(), relaxed, det), sens.stations, relaxed, np.random.default_rng(0)
    )
    assert all(net.meets_located for net in ok.networks.values())


def test_located_fraction_raised_before_error() -> None:
    """Below the required located fraction, a network that locates more events wins even if
    its (located-event) error is larger."""
    sens = straight_ray(ring_with_decoys(), target_points())
    det = _uniform_detection(sens, 0.6, 0.6)
    spec = DesignSpec(min_located_fraction=0.99)
    ev = Evaluator(sens, PickSpec(), spec, det)
    small, big = [0, 2, 4], [0, 2, 4, 6, 8, 9]
    (f_small, f_big) = (ev.evaluate(small)[1], ev.evaluate(big)[1])
    assert f_small < f_big < 0.99
    assert ev.score(big) < ev.score(small)
