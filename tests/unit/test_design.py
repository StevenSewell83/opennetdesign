"""Network optimisation."""

import itertools
from pathlib import Path

import numpy as np
import pytest
from example_config import write_example

from netdesign.config import load_config
from netdesign.design import DesignSpec, Evaluator, NetworkResult, _recommend, optimise
from netdesign.errors import ConfigError
from netdesign.information import PickSpec, achievable_stats
from netdesign.pipeline import run_design, run_sensitivity
from netdesign.sensitivity import Sensitivity
from netdesign.target import Sources

VP, VPVS = 5.0, 1.75


def straight_ray(stations: np.ndarray, sources: np.ndarray, n_r: int = 6, seed: int = 0):
    """Homogeneous medium: T = r / v; perturbed models scale velocity per station slightly."""
    rng = np.random.default_rng(seed)
    d = sources[None, :, :] - stations[:, None, :]  # (S, J, 3)
    r = np.linalg.norm(d, axis=-1)
    vs = np.array([VP, VP / VPVS])
    T0 = r[:, None, :] / vs[None, :, None]  # (S, P, J)
    G0 = d[:, None, :, :] / (r[:, None, :, None] * vs[None, :, None, None])
    f = 1 + rng.normal(0, 0.03, (n_r, stations.shape[0], 1, 1))
    f[0] = 1.0
    T = (T0[None] * f).astype(np.float32)
    G = np.repeat(G0[None], n_r, axis=0).astype(np.float32)
    src = Sources(sources, sources[:, 0], sources[:, 1], np.full(len(sources), 1 / len(sources)))
    names = tuple(f"S{i:02d}" for i in range(len(stations)))
    return Sensitivity(tuple(range(n_r)), names, ("P", "S"), src, T, G, 0.05)


def target_points() -> np.ndarray:
    g = np.linspace(-1, 1, 5)
    x, y, z = np.meshgrid(g, g, 2 + g, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def ring_with_decoys() -> np.ndarray:
    ang = np.radians(np.arange(0, 360, 45))
    ring = np.stack([3 * np.cos(ang), 3 * np.sin(ang), np.zeros(8)], axis=1)
    decoys = np.array([[12.0, 0, 0], [12.5, 0.5, 0], [12.0, 1.0, 0], [12.5, -0.5, 0]])
    return np.vstack([ring, decoys])


PICKS = PickSpec(sigma_s={"P": 0.005, "S": 0.01})


def test_known_answer_ring_beats_decoys() -> None:
    sens = straight_ray(ring_with_decoys(), target_points())
    spec = DesignSpec(n_min=3, n_max=6, n_random=50)
    ev = Evaluator(sens, PICKS, spec)
    result = optimise(ev, sens.stations, spec, np.random.default_rng(0))
    for n, net in result.networks.items():
        assert all(s < 8 for s in net.stations), f"n={n} picked a decoy"
    # Four stations: one per quadrant (spread round the target).
    angles = np.degrees(np.arctan2(*ring_with_decoys()[list(result.networks[4].stations), 1::-1].T))
    assert len({int((a % 360) // 90) for a in angles}) == 4


def test_matches_exhaustive_search() -> None:
    rng = np.random.default_rng(3)
    stations = np.column_stack([rng.uniform(-6, 6, (9, 2)), np.zeros(9)])
    sens = straight_ray(stations, target_points(), seed=1)
    spec = DesignSpec(n_min=3, n_max=6, n_random=0)
    ev = Evaluator(sens, PICKS, spec)
    result = optimise(ev, sens.stations, spec, np.random.default_rng(0))
    for n in range(3, 7):
        best = min(ev.score(c) for c in itertools.combinations(range(9), n))
        assert result.networks[n].criterion_m <= best * 1.001, n


def test_fixed_and_excluded() -> None:
    sens = straight_ray(ring_with_decoys(), target_points())
    spec = DesignSpec(fixed=["S09"], exclude=["S00", "S01"], n_min=3, n_max=5, n_random=10)
    ev = Evaluator(sens, PICKS, spec)
    result = optimise(ev, sens.stations, spec, np.random.default_rng(0))
    for net in result.networks.values():
        assert 9 in net.stations and 0 not in net.stations and 1 not in net.stations


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"fixed": ["NOPE"]}, "unknown"),
        ({"n_min": 20}, "n_min"),
    ],
)
def test_bad_settings(kw: dict, match: str) -> None:
    sens = straight_ray(ring_with_decoys(), target_points())
    spec = DesignSpec(**kw)
    with pytest.raises(ConfigError, match=match):
        optimise(Evaluator(sens, PICKS, spec), sens.stations, spec, np.random.default_rng(0))


def test_fixed_and_excluded_overlap_rejected() -> None:
    with pytest.raises(ConfigError, match="both fixed and excluded"):
        DesignSpec(fixed=["A"], exclude=["A"])


def test_evaluator_matches_achievable_without_prior() -> None:
    sens = straight_ray(ring_with_decoys(), target_points())
    spec = DesignSpec(prior_sd_km=1e6, prior_sd_t0_s=1e6)
    ev = Evaluator(sens, PICKS, spec)
    stations = [0, 2, 4, 6, 9]
    mse = achievable_stats(sens, PICKS).mse(stations)
    np.testing.assert_allclose(ev.covariance(ev.sums(stations)), mse, rtol=1e-5, atol=1e-12)
    var = np.diagonal(mse, axis1=-2, axis2=-1)
    expected = np.sqrt(np.mean(var[:, 0] + var[:, 1] + var[:, 2])) * 1000
    assert ev.score(stations) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("criterion", ["rms", "percentile", "logdet"])
def test_gls_criteria_never_worse_with_more_stations(criterion: str) -> None:
    sens = straight_ray(ring_with_decoys(), target_points())
    ev = Evaluator(sens, PICKS, DesignSpec(measure="gls", criterion=criterion))
    scores = [ev.score(list(range(n))) for n in range(3, 12)]
    assert all(b <= a * (1 + 1e-9) for a, b in itertools.pairwise(scores))


def _net(n: int, c: float) -> NetworkResult:
    return NetworkResult(n, tuple(range(n)), c, c, 1, None, None, {})


def test_recommend() -> None:
    nets = {n: _net(n, c) for n, c in zip(range(3, 8), [200, 120, 100, 97, 96], strict=True)}
    assert _recommend(nets, DesignSpec(min_gain_pct=5), 95) == 5  # 100 -> 97 is < 5 %
    assert _recommend(nets, DesignSpec(target_m=110), 95) == 5
    assert _recommend(nets, DesignSpec(target_m=10), 95) == 7  # unreachable: largest


def test_design_end_to_end(tmp_path: Path, nll_bin: Path) -> None:
    cfg = load_config(
        write_example(
            tmp_path,
            nll_bin,
            grid={"dx_km": 0.1},
            velocity={"perturb": {"n_realizations": 4}},
            design={"n_random": 20, "n_restarts": 2},
        )
    )
    run_sensitivity(cfg)
    result = run_design(cfg)
    assert 3 <= result.recommended_n <= 10
    d = cfg.output_dir / "design"
    assert (d / "design.json").is_file() and (d / "networks.csv").is_file()
    q = cfg.output_dir / "qc"
    for fig in (
        "criterion_vs_n.png",
        "robustness.png",
        "design_map.png",
        "design_uncertainty_maps.png",
    ):
        assert (q / fig).stat().st_size > 10_000, fig
    rows = (q / "candidates_outcome.csv").read_text().splitlines()
    statuses = [r.split(",")[4] for r in rows[1:]]
    assert statuses.count("selected") == result.recommended_n
    assert "<h2>Network design</h2>" in (q / "report.html").read_text()
