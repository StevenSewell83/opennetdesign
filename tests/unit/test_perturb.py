"""Attenuation and perturbation ensembles."""

import json
from pathlib import Path

import numpy as np
import pytest

from netdesign.errors import ConfigError
from netdesign.velocity import AttenuationSpec, Layer, LayeredModel, PerturbSpec, make_ensemble
from netdesign.velocity.perturb import Perturbation

BASE = LayeredModel(
    (
        Layer(-0.5, 2.0, 1.1, vp_grad=0.8, vs_grad=0.45),
        Layer(0.5, 3.2, 1.8, vp_grad=0.4, vs_grad=0.22),
        Layer(1.5, 4.2, 2.4),
        Layer(3.0, 5.5, 3.18),
    )
)


def spec(**kwargs: object) -> PerturbSpec:
    return PerturbSpec.model_validate({"enabled": True, "n_realizations": 50, **kwargs})


def test_vs_scaled_q_follows_vs_and_gradient() -> None:
    model = make_ensemble(BASE, PerturbSpec(), AttenuationSpec()).realizations[0].model
    np.testing.assert_allclose(
        model.q_at([0.0, 1.0, 2.0], "S"), 50 * BASE.velocity_at([0, 1, 2], "S")
    )
    np.testing.assert_allclose(model.q_at(0.0, "P"), 2 * model.q_at(0.0, "S"))
    assert model.q_at(2.9, "S") > model.q_at(-0.4, "S")  # Q increases with depth


def test_disabled_gives_base_only() -> None:
    ens = make_ensemble(BASE, PerturbSpec())
    assert len(ens) == 1
    assert ens.realizations[0].is_base
    assert ens.realizations[0].model == BASE


def test_include_base_and_count() -> None:
    ens = make_ensemble(BASE, spec(vp_pct={"sd": 5, "mode": "per_layer"}))
    assert len(ens) == 51
    assert ens.realizations[0].model == BASE
    assert not ens.realizations[1].is_base


def test_reproducible_with_seed() -> None:
    s = spec(vp_pct={"sd": 5, "mode": "per_layer"}, seed=7)
    a, b = make_ensemble(BASE, s), make_ensemble(BASE, s)
    assert [r.model for r in a.realizations] == [r.model for r in b.realizations]
    c = make_ensemble(BASE, spec(vp_pct={"sd": 5, "mode": "per_layer"}, seed=8))
    assert a.realizations[1].model != c.realizations[1].model


def test_global_vp_scales_all_layers_equally() -> None:
    ens = make_ensemble(BASE, spec(vp_pct={"sd": 5, "mode": "global"}))
    for r in ens.realizations[1:]:
        ratios = [p.vp / b.vp for p, b in zip(r.model.layers, BASE.layers, strict=True)]
        assert np.ptp(ratios) < 1e-12
        # Vp/Vs unchanged when only Vp is perturbed
        np.testing.assert_allclose(r.model.vpvs, BASE.vpvs)


def test_vpvs_perturbation_keeps_vp() -> None:
    ens = make_ensemble(BASE, spec(vpvs={"sd": 0.03, "mode": "global"}))
    deltas = []
    for r in ens.realizations[1:]:
        np.testing.assert_allclose([la.vp for la in r.model.layers], [la.vp for la in BASE.layers])
        d = r.model.vpvs - BASE.vpvs
        assert np.ptp(d) < 1e-12
        deltas.append(d[0])
    assert np.std(deltas) == pytest.approx(0.03, rel=0.3)


def test_lhs_stratifies_global_parameter() -> None:
    n = 40
    ens = make_ensemble(BASE, spec(n_realizations=n, vp_pct={"sd": 5, "mode": "global"}))
    factors = np.array([r.params["vp_factor"][0] for r in ens.realizations[1:]])
    # One draw per probability stratum: sorted quantiles land in their own bins.
    from scipy.special import ndtr

    u = np.sort(ndtr((factors - 1) / 0.05))
    assert np.all((u >= np.arange(n) / n) & (u <= np.arange(1, n + 1) / n))


def test_layer_correlation() -> None:
    s = spec(
        n_realizations=2000,
        sampling="random",
        layer_corr=0.8,
        vp_pct={"sd": 5, "mode": "per_layer"},
    )
    ens = make_ensemble(BASE, s)
    f = np.array([r.params["vp_factor"] for r in ens.realizations[1:]])
    c = np.corrcoef(f.T)
    assert c[0, 1] == pytest.approx(0.8, abs=0.05)
    assert c[0, 2] == pytest.approx(0.64, abs=0.06)


def test_interfaces_stay_ordered_and_clipped() -> None:
    ens = make_ensemble(BASE, spec(interface_sd_km=5.0, n_realizations=200, sampling="random"))
    thick = np.diff(BASE.tops_km)
    for r in ens.realizations[1:]:
        shift = np.array(r.params["top_shift_km"])
        assert shift[0] == 0.0
        assert np.all(np.diff(r.model.tops_km) > 0)
        assert abs(shift[1]) <= 0.45 * min(thick[0], thick[1]) + 1e-12


def test_q_follows_perturbed_vs() -> None:
    att = AttenuationSpec()
    ens = make_ensemble(BASE, spec(vp_pct={"sd": 5, "mode": "per_layer"}), att)
    for r in ens.realizations:
        qs = [la.qs for la in r.model.layers]
        np.testing.assert_allclose(qs, [50 * la.vs for la in r.model.layers])


def test_q_and_ratio_perturbation() -> None:
    s = spec(q_log10={"sd": 0.3}, qpqs_sd_log10=0.1, sampling="random", n_realizations=500)
    ens = make_ensemble(BASE, s, AttenuationSpec())
    log_f = np.log10([r.params["q_factor"][0] for r in ens.realizations[1:]])
    assert np.std(log_f) == pytest.approx(0.3, rel=0.1)
    for r in ens.realizations[1:]:
        la = r.model.layers[2]
        assert la.qp / la.qs == pytest.approx(2.0 * r.params["qpqs_factor"][0])


def test_eta_perturbation_non_negative() -> None:
    s = spec(eta_sd=0.5, sampling="random", n_realizations=200)
    ens = make_ensemble(BASE, s, AttenuationSpec(eta=0.1))
    etas = [r.eta for r in ens.realizations]
    assert min(e for e in etas if e is not None) >= 0.0
    assert ens.realizations[0].eta == 0.1


def test_q_perturbation_needs_q() -> None:
    with pytest.raises(ConfigError, match="no Q"):
        make_ensemble(BASE, spec(q_log10={"sd": 0.3}))


def test_layered_mode_requires_q_in_model() -> None:
    with pytest.raises(ConfigError, match="layered"):
        make_ensemble(BASE, PerturbSpec(), AttenuationSpec(model="layered"))


def test_enabled_with_all_zero_rejected() -> None:
    with pytest.raises(ConfigError, match="every sd is zero"):
        PerturbSpec(enabled=True)


def test_vp_sd_too_large_rejected() -> None:
    with pytest.raises(ConfigError, match="vp_pct"):
        PerturbSpec(vp_pct=Perturbation(sd=40), truncate_sd=3)


def test_write(tmp_path: Path) -> None:
    ens = make_ensemble(BASE, spec(n_realizations=3, vp_pct={"sd": 5}), AttenuationSpec())
    meta_path = ens.write(tmp_path)
    meta = json.loads(meta_path.read_text())
    assert len(meta["realizations"]) == 4
    for entry, r in zip(meta["realizations"], ens.realizations, strict=True):
        csv_model = LayeredModel.read(tmp_path / entry["files"]["csv"])
        nll_model = LayeredModel.read(tmp_path / entry["files"]["nll"])
        assert csv_model.has_q and not nll_model.has_q
        np.testing.assert_allclose(nll_model.tops_km, r.model.tops_km, atol=1e-6)


def test_copies_when_disabled() -> None:
    ens = make_ensemble(BASE, PerturbSpec(n_realizations=3), copies=True)
    assert [r.is_base for r in ens.realizations] == [True, False, False, False]
    assert all(r.model == BASE for r in ens.realizations)
    no_base = make_ensemble(BASE, PerturbSpec(n_realizations=2, include_base=False), copies=True)
    assert [r.index for r in no_base.realizations] == [0, 1]
    assert not any(r.is_base for r in no_base.realizations)
