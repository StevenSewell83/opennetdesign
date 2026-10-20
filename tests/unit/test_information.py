"""Information matrices from synthetic sensitivities (no NonLinLoc needed)."""

from pathlib import Path

import numpy as np
import pytest

from netdesign.errors import ConfigError
from netdesign.information import (
    PickSpec,
    model_error_factor,
    pick_variance,
    station_stats,
)
from netdesign.sensitivity import Sensitivity
from netdesign.target import Sources

SIGMA = {"P": 0.01, "S": 0.02}


def make_sens(n_r: int = 6, n_s: int = 4, n_j: int = 5, seed: int = 0, spread: float = 0.02):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1, 1, (n_j, 3))
    src = Sources(xyz, xyz[:, 0], xyz[:, 1], np.full(n_j, 1 / n_j))
    T0 = rng.uniform(0.5, 1.5, (1, n_s, 2, n_j))
    T = (T0 + rng.normal(0, spread, (n_r, n_s, 2, n_j))).astype(np.float32)
    G = rng.normal(0, 0.3, (n_r, n_s, 2, n_j, 3)).astype(np.float32)
    names = tuple(f"S{i}" for i in range(n_s))
    return Sensitivity(tuple(range(n_r)), names, ("P", "S"), src, T, G, 0.05)


def brute_force(sens: Sensitivity, stations: list[int], detection=None) -> np.ndarray:
    """G^T C_d^-1 G with the full sample covariance, picks of the given stations only."""
    n_r, _, n_p, n_j = sens.T.shape
    sig = np.array([SIGMA[p] for p in sens.phases])
    det = np.ones(sens.T.shape) if detection is None else detection
    out = np.empty((n_r, n_j, 4, 4))
    t = sens.T.astype(np.float64)
    for r in range(n_r):
        for j in range(n_j):
            rows, w, dev = [], [], []
            for s in stations:
                for p in range(n_p):
                    if det[r, s, p, j] == 0:
                        continue
                    rows.append(np.append(sens.G[r, s, p, j], 1.0))
                    w.append(det[r, s, p, j] / sig[p] ** 2)
                    dev.append(t[:, s, p, j] - t[:, s, p, j].mean())
            g, dev = np.array(rows), np.array(dev)
            c_d = np.diag(1 / np.array(w)) + dev @ dev.T / (n_r - 1)
            out[r, j] = g.T @ np.linalg.solve(c_d, g)
    return out


def test_none_matches_explicit_sum() -> None:
    sens = make_sens()
    stats = station_stats(sens, PickSpec(sigma_s=SIGMA, model_error="none"))
    assert stats.B is None and stats.M is None
    r, s, j = 1, 2, 3
    expected = np.zeros((4, 4))
    for ip, sigma in enumerate(SIGMA.values()):
        g = np.append(sens.G[r, s, ip, j], 1.0)
        expected += np.outer(g, g) / sigma**2
    np.testing.assert_allclose(stats.A[r, s, j], expected, rtol=1e-5)
    np.testing.assert_allclose(stats.network_information([s])[r, j], expected, rtol=1e-5)


@pytest.mark.parametrize("subset", [[0], [1, 3], [0, 1, 2, 3]])
def test_correlated_matches_full_covariance(subset: list[int]) -> None:
    sens = make_sens()
    stats = station_stats(sens, PickSpec(sigma_s=SIGMA, model_error_energy=1.0))
    got = stats.network_information(subset)
    np.testing.assert_allclose(got, brute_force(sens, subset), rtol=1e-6, atol=1e-6)


def test_correlated_with_detection_matches_full_covariance() -> None:
    sens = make_sens()
    det = np.ones(sens.T.shape)
    det[:, 1, 1] = 0.0  # station 1 never picks S
    det[:, 2] = 0.5
    stats = station_stats(sens, PickSpec(sigma_s=SIGMA, model_error_energy=1.0), det)
    got = stats.network_information([0, 1, 2])
    np.testing.assert_allclose(got, brute_force(sens, [0, 1, 2], det), rtol=1e-6, atol=1e-6)


def test_compression_keeps_requested_energy() -> None:
    sens = make_sens(n_r=21)
    full = model_error_factor(sens, 1.0)
    part = model_error_factor(sens, 0.9)
    assert part.shape[-1] < full.shape[-1] <= 20
    cov = np.einsum("spjk,spjk->spj", full, full)
    np.testing.assert_allclose(cov, sens.T.astype(np.float64).var(axis=0, ddof=1), rtol=1e-6)
    kept = np.einsum("spjk,spjk->j", part, part) / np.einsum("spjk,spjk->j", full, full)
    assert np.all(kept >= 0.9 - 1e-9)


def test_common_shift_absorbed_by_origin_time() -> None:
    """Model error identical at every pick only inflates origin-time uncertainty."""
    rng = np.random.default_rng(5)
    base = make_sens(n_r=1)
    shift = rng.normal(0, 0.05, (8, 1, 1, 1))
    T = (base.T.astype(np.float64) + shift).astype(np.float32)
    G = np.repeat(base.G, 8, axis=0)
    sens = Sensitivity(tuple(range(8)), base.stations, base.phases, base.sources, T, G, 0.05)
    all_sta = list(range(4))
    cov = {
        mode: np.linalg.inv(
            station_stats(sens, PickSpec(sigma_s=SIGMA, model_error=mode)).network_information(
                all_sta
            )
        )
        for mode in ("none", "independent", "correlated")
    }
    sd = {m: np.sqrt(np.diagonal(c, axis1=-2, axis2=-1)) for m, c in cov.items()}
    np.testing.assert_allclose(sd["correlated"][..., :3], sd["none"][..., :3], rtol=1e-3)
    assert np.all(sd["correlated"][..., 3] > sd["none"][..., 3])
    assert np.all(sd["independent"][..., :3] > 2 * sd["none"][..., :3])


def test_independent_variance() -> None:
    sens = make_sens()
    var = pick_variance(sens, PickSpec(sigma_s=SIGMA, model_error="independent"))
    expected = 0.01**2 + sens.T[:, 0, 0, 0].astype(np.float64).var(ddof=1)
    assert var[0, 0, 0] == pytest.approx(expected)
    np.testing.assert_allclose(pick_variance(sens, PickSpec(model_error="none"))[:, 0], 0.005**2)


def test_single_realization_has_no_model_error() -> None:
    stats = station_stats(make_sens(n_r=1), PickSpec())
    assert stats.B is None


def test_symmetric_psd_and_full_rank_network() -> None:
    stats = station_stats(make_sens(), PickSpec(sigma_s=SIGMA))
    one = stats.network_information([0])
    eig = np.linalg.eigvalsh(one)
    assert eig.min() > -1e-6 * eig.max()
    assert np.all(np.sum(eig > 1e-9 * eig.max(), axis=-1) == 2)  # one station, P+S: rank 2
    net = stats.network_information(range(4))
    np.testing.assert_allclose(net, np.swapaxes(net, -1, -2), atol=1e-8)
    assert np.all(np.linalg.eigvalsh(net).min(axis=-1) > 0)


def test_detection_shape_checked() -> None:
    sens = make_sens()
    with pytest.raises(ConfigError, match="shape"):
        station_stats(sens, PickSpec(), np.ones((1, 2)))


def test_missing_phase_sigma() -> None:
    with pytest.raises(ConfigError, match="no entry"):
        pick_variance(make_sens(), PickSpec(sigma_s={"P": 0.01}))


def test_bad_sigma() -> None:
    with pytest.raises(ConfigError, match="> 0"):
        PickSpec(sigma_s={"P": 0.0})


def test_save_load_round_trip(tmp_path: Path) -> None:
    sens = make_sens()
    back = Sensitivity.load(sens.save(tmp_path / "s.npz"))
    assert back.stations == sens.stations and back.phases == sens.phases
    np.testing.assert_array_equal(back.G, sens.G)
    np.testing.assert_array_equal(back.sources.weights, sens.sources.weights)


def test_achievable_matches_sandwich() -> None:
    """AchievableStats.mse == A^-1 G^T W (S2 + D) W G A^-1 computed directly."""
    from netdesign.information import achievable_stats

    sens = make_sens(n_r=7)
    picks = PickSpec(sigma_s=SIGMA, model_error_energy=1.0)
    subset = [0, 2, 3]
    got = achievable_stats(sens, picks).mse(subset)
    t = sens.T.astype(np.float64)
    sig = np.array([SIGMA[p] for p in sens.phases])
    for j in range(sens.T.shape[-1]):
        g = np.concatenate(
            [sens.G[0][subset][:, :, j].reshape(-1, 3), np.ones((2 * len(subset), 1))], axis=1
        )
        w = np.diag(np.tile(1 / sig**2, len(subset)))
        tj = t[..., j][:, subset]  # (R, S', P)
        dt = (tj[1:] - tj[0]).reshape(6, -1)
        s2 = dt.T @ dt / 6
        a_inv = np.linalg.inv(g.T @ w @ g)
        k = a_inv @ g.T @ w
        expected = k @ (s2 + np.diag(np.tile(sig**2, len(subset)))) @ k.T
        np.testing.assert_allclose(got[j], expected, rtol=1e-6, atol=1e-12)


def test_achievable_worse_than_gls() -> None:
    from netdesign.information import achievable_stats

    sens = make_sens(n_r=7)
    picks = PickSpec(sigma_s=SIGMA, model_error_energy=1.0)
    ach = achievable_stats(sens, picks).mse(range(4))
    gls = np.linalg.inv(station_stats(sens, picks).network_information(range(4))[0])
    # The GLS covariance is a lower bound for the spatial variances (it uses the true C_m).
    assert np.all(np.diagonal(ach, axis1=-2, axis2=-1)[:, :3] >= 0.0)
    assert np.mean(np.diagonal(ach, axis1=-2, axis2=-1)[:, :3]) > np.mean(
        np.diagonal(gls, axis1=-2, axis2=-1)[:, :3]
    )


def test_detection_correction_matches_random_detection_patterns() -> None:
    """With picks present (full weight) or absent at random, the corrected expected MSE is
    much closer to the Monte-Carlo average than the uncorrected first-order version
    (measured: 3-32 % vs 35-58 % here)."""
    from netdesign.information import AchievableStats, achievable_stats

    rng = np.random.default_rng(8)
    sens = make_sens(n_r=9, n_s=12, n_j=3, seed=4, spread=0.03)
    picks = PickSpec(sigma_s=SIGMA, model_error_energy=1.0)
    prob = rng.uniform(0.3, 1.0, sens.T.shape[1:])  # (S, P, J), same in every model
    det = np.broadcast_to(prob, sens.T.shape)
    corrected = achievable_stats(sens, picks, det)
    uncorrected = AchievableStats(corrected.A, corrected.B, None)
    stations = list(range(12))
    m_corr, m_unc = corrected.mse(stations), uncorrected.mse(stations)

    t = sens.T.astype(np.float64)
    sig = np.array([SIGMA[p] for p in sens.phases])
    for j in range(sens.T.shape[-1]):
        g = np.concatenate([sens.G[0][:, :, j].reshape(-1, 3), np.ones((24, 1))], axis=1)
        dt = (t[1:, :, :, j] - t[0, :, :, j]).reshape(8, -1)
        s2 = dt.T @ dt / 8
        w = np.tile(1 / sig**2, 12)
        acc = np.zeros((4, 4))
        n_mc = 4000
        for _ in range(n_mc):
            d = (rng.random(24) < prob[:, :, j].reshape(-1)).astype(float)
            wd = np.diag(w * d)
            a = g.T @ wd @ g
            k = np.linalg.solve(a, g.T @ wd)
            acc += k @ (s2 + np.diag(np.tile(sig**2, 12))) @ k.T
        mc = acc / n_mc
        err_corr = np.abs(np.trace(m_corr[j][:3, :3]) / np.trace(mc[:3, :3]) - 1)
        err_unc = np.abs(np.trace(m_unc[j][:3, :3]) / np.trace(mc[:3, :3]) - 1)
        assert err_corr < err_unc
        # What remains is the ignored randomness of A: optimistic, never pessimistic.
        assert np.trace(mc[:3, :3]) >= 0.98 * np.trace(m_corr[j][:3, :3])
        assert err_corr < 0.35  # harsh case: p down to 0.3, random geometry
