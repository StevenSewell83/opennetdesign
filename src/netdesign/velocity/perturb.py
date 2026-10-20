"""Ensembles of perturbed layered models about a starting model.

Perturbed quantities (each optional; ``sd = 0`` disables it):

========================  ==========================================================
``vp_pct``                Vp multiplier ``1 + sd/100 * z`` (top and gradient)
``vpvs``                  Vp/Vs additive ``sd * z``; Vs follows from the perturbed Vp
``interface_sd_km``       interface depth shift ``sd * z`` (model top is fixed)
``q_log10``               Q multiplier ``10 ** (sd * z)`` on Qs and Qp
``qpqs_sd_log10``         Qp/Qs multiplier ``10 ** (sd * z)`` (global)
``eta_sd``                frequency exponent additive ``sd * z``, clipped at 0 (global)
========================  ==========================================================

``z`` are standard normals truncated at ``truncate_sd``. ``global`` mode uses one draw for all
layers; ``per_layer`` draws per layer with adjacent-layer correlation ``layer_corr`` (AR(1),
correlation ``layer_corr ** |i - j|``), which also applies to interface shifts.

Interface shifts are clipped to ``interface_clip`` times the thinner adjacent layer, so layer
order is always preserved (hence ``interface_clip < 0.5``). Latin hypercube sampling
stratifies each active dimension; the correlation transform then mixes per-layer dimensions,
so stratification is exact only for global parameters and uncorrelated per-layer ones.

Q is set after the velocity perturbation (see :func:`~netdesign.velocity.attenuation.
apply_attenuation`), so in ``vs_scaled`` mode Q tracks the perturbed Vs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import ndtri

from netdesign.errors import ConfigError, PhysicsError
from netdesign.velocity.attenuation import AttenuationSpec, apply_attenuation
from netdesign.velocity.layered import Layer, LayeredModel

__all__ = [
    "Ensemble",
    "PerturbSpec",
    "Perturbation",
    "Realization",
    "make_ensemble",
]

FloatArray = npt.NDArray[np.float64]


class Perturbation(BaseModel):
    """Standard deviation plus whether it is drawn once (global) or per layer."""

    model_config = ConfigDict(extra="forbid")

    sd: float = Field(0.0, ge=0)
    mode: Literal["global", "per_layer"] = "global"


class PerturbSpec(BaseModel):
    """``velocity.perturb`` config section."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    n_realizations: int = Field(20, ge=1)
    include_base: bool = True
    sampling: Literal["random", "latin_hypercube"] = "latin_hypercube"
    seed: int = 0
    truncate_sd: float = Field(3.0, gt=0)
    layer_corr: float = Field(0.0, ge=0, lt=1)
    vp_pct: Perturbation = Perturbation(mode="per_layer")
    vpvs: Perturbation = Perturbation()
    interface_sd_km: float = Field(0.0, ge=0)
    interface_clip: float = Field(
        0.45, gt=0, lt=0.5, description="max shift / thinner adjacent layer thickness"
    )
    q_log10: Perturbation = Perturbation()
    qpqs_sd_log10: float = Field(0.0, ge=0)
    eta_sd: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.vp_pct.sd * self.truncate_sd >= 100:
            raise ConfigError(
                "velocity.perturb.vp_pct.sd * truncate_sd must be < 100 (Vp would reach 0)"
            )
        if self.enabled and not any(
            (
                self.vp_pct.sd,
                self.vpvs.sd,
                self.interface_sd_km,
                self.q_log10.sd,
                self.qpqs_sd_log10,
                self.eta_sd,
            )
        ):
            raise ConfigError("velocity.perturb.enabled is true but every sd is zero")
        return self

    @property
    def perturbs_q(self) -> bool:
        """True if any attenuation parameter is perturbed."""
        return bool(self.q_log10.sd or self.qpqs_sd_log10 or self.eta_sd)


@dataclass(frozen=True)
class Realization:
    """One model of the ensemble and the draws that produced it."""

    index: int
    model: LayeredModel
    is_base: bool
    eta: float | None
    fref_hz: float | None
    params: dict[str, list[float]]


@dataclass(frozen=True)
class Ensemble:
    """Ordered realizations; index 0 is the starting model when ``include_base`` is set."""

    realizations: tuple[Realization, ...]
    spec: PerturbSpec
    attenuation: AttenuationSpec | None

    def __len__(self) -> int:
        return len(self.realizations)

    def write(
        self,
        outdir: str | Path,
        base_path: str | Path | None = None,
        extra: dict[str, Any] | None = None,
        write_models: bool = True,
    ) -> Path:
        """Write ``model_NNN.csv`` (with Q), ``model_NNN.layer`` (NonLinLoc) and
        ``ensemble.json`` (spec and per-realization draws) to ``outdir``. With
        ``write_models`` false only ``ensemble.json`` is written (3D starting model, where
        the layered models are just carriers of the global draws).

        Returns
        -------
        Path
            Path to ``ensemble.json``.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        entries = []
        for r in self.realizations:
            stem = f"model_{r.index:03d}"
            if write_models:
                r.model.write(outdir / f"{stem}.csv")
                r.model.write(outdir / f"{stem}.layer")
            entries.append(
                {
                    "index": r.index,
                    "is_base": r.is_base,
                    "eta": r.eta,
                    "fref_hz": r.fref_hz,
                    "params": r.params,
                    "files": (
                        {"csv": f"{stem}.csv", "nll": f"{stem}.layer"} if write_models else {}
                    ),
                }
            )
        meta: dict[str, Any] = {
            "base": None if base_path is None else str(base_path),
            "perturb": self.spec.model_dump(),
            "attenuation": None if self.attenuation is None else self.attenuation.model_dump(),
            "realizations": entries,
            **(extra or {}),
        }
        path = outdir / "ensemble.json"
        path.write_text(json.dumps(meta, indent=2))
        return path


def _latin_hypercube(rng: np.random.Generator, n: int, d: int) -> FloatArray:
    """``n`` x ``d`` uniforms in (0, 1), one sample per stratum in every column."""
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def _ar1_cholesky(width: int, rho: float) -> FloatArray:
    idx = np.arange(width)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])
    chol: FloatArray = np.linalg.cholesky(corr)
    return chol


def _draw(
    spec: PerturbSpec, n_layers: int, rng: np.random.Generator
) -> list[dict[str, FloatArray]]:
    """Truncated standard normals per active parameter, one dict per realization."""

    def width(p: Perturbation) -> int:
        return n_layers if p.mode == "per_layer" else 1

    blocks: list[tuple[str, int, bool]] = []  # name, width, correlated across layers
    if spec.vp_pct.sd:
        blocks.append(("vp", width(spec.vp_pct), spec.vp_pct.mode == "per_layer"))
    if spec.vpvs.sd:
        blocks.append(("vpvs", width(spec.vpvs), spec.vpvs.mode == "per_layer"))
    if spec.interface_sd_km and n_layers > 1:
        blocks.append(("interface", n_layers - 1, True))
    if spec.q_log10.sd:
        blocks.append(("q", width(spec.q_log10), spec.q_log10.mode == "per_layer"))
    if spec.qpqs_sd_log10:
        blocks.append(("qpqs", 1, False))
    if spec.eta_sd:
        blocks.append(("eta", 1, False))

    n, d = spec.n_realizations, sum(w for _, w, _ in blocks)
    if spec.sampling == "latin_hypercube":
        z = ndtri(_latin_hypercube(rng, n, d))
    else:
        z = rng.standard_normal((n, d))

    out: list[dict[str, FloatArray]] = [{} for _ in range(n)]
    col = 0
    for name, w, correlated in blocks:
        zb = z[:, col : col + w]
        col += w
        if correlated and w > 1 and spec.layer_corr > 0:
            zb = zb @ _ar1_cholesky(w, spec.layer_corr).T
        zb = np.clip(zb, -spec.truncate_sd, spec.truncate_sd)
        for i in range(n):
            out[i][name] = zb[i]
    return out


def _perturb_velocity(
    base: LayeredModel, spec: PerturbSpec, z: dict[str, FloatArray]
) -> tuple[list[Layer], dict[str, list[float]]]:
    n = base.n_layers
    ones = np.ones(n)
    vp_factor = 1 + spec.vp_pct.sd / 100 * z["vp"] * ones if "vp" in z else ones
    vpvs_delta = spec.vpvs.sd * z["vpvs"] * ones if "vpvs" in z else np.zeros(n)

    tops = base.tops_km
    shift = np.zeros(n)
    if "interface" in z:
        thick = np.diff(tops)
        for k in range(1, n):  # interface k is the top of layer k
            adjacent = thick[k - 1] if k == n - 1 else min(thick[k - 1], thick[k])
            limit = spec.interface_clip * adjacent
            shift[k] = np.clip(spec.interface_sd_km * z["interface"][k - 1], -limit, limit)

    layers = []
    for k, la in enumerate(base.layers):
        r0 = la.vpvs
        r1 = r0 + vpvs_delta[k]
        vs_scale = vp_factor[k] * r0 / r1
        layers.append(
            replace(
                la,
                top_km=la.top_km + shift[k],
                vp=la.vp * vp_factor[k],
                vp_grad=la.vp_grad * vp_factor[k],
                vs=la.vs * vs_scale,
                vs_grad=la.vs_grad * vs_scale,
            )
        )
    params = {
        "vp_factor": vp_factor.tolist(),
        "vpvs_delta": vpvs_delta.tolist(),
        "top_shift_km": shift.tolist(),
    }
    return layers, params


def make_ensemble(
    base: LayeredModel,
    spec: PerturbSpec,
    attenuation: AttenuationSpec | None = None,
    copies: bool = False,
) -> Ensemble:
    """Build the ensemble of models about ``base``.

    Parameters
    ----------
    base : LayeredModel
        Starting model.
    spec : PerturbSpec
        Perturbation settings; if ``enabled`` is false the ensemble is the base model only.
    attenuation : AttenuationSpec, optional
        How to set Q. ``None`` keeps whatever Q the base model has (possibly none).
    copies : bool
        If ``spec`` is disabled, still return ``n_realizations`` unperturbed copies (after the
        base if ``include_base``) — used when 3D heterogeneity supplies the variation.

    Raises
    ------
    ConfigError
        If attenuation is perturbed but there is no Q to perturb.
    PhysicsError
        If a realization is unphysical (message names the realization; reduce the sds).
    """
    n = base.n_layers
    if spec.enabled and spec.perturbs_q and attenuation is None and not base.has_q:
        raise ConfigError(
            "velocity.perturb perturbs Q but the model has no Q and velocity.attenuation is unset"
        )
    eta0 = attenuation.eta if attenuation is not None else None
    fref = attenuation.fref_hz if attenuation is not None else None
    if spec.enabled and spec.eta_sd and eta0 is None:
        raise ConfigError("velocity.perturb.eta_sd needs velocity.attenuation (for eta)")

    def finish(layers: list[Layer], q_factor: FloatArray, qpqs: float) -> LayeredModel:
        model = LayeredModel(tuple(layers))
        if attenuation is not None:
            return apply_attenuation(
                model, attenuation, q_factor=tuple(q_factor.tolist()), qpqs_factor=qpqs
            )
        if base.has_q:  # keep the base model's Q, perturbed
            layers = [
                replace(
                    la,
                    qs=la.qs * f if la.qs is not None else None,
                    qs_grad=la.qs_grad * f,
                    qp=la.qp * f * qpqs if la.qp is not None else None,
                    qp_grad=la.qp_grad * f * qpqs,
                )
                for la, f in zip(model.layers, q_factor, strict=True)
            ]
            return LayeredModel(tuple(layers))
        return model

    realizations: list[Realization] = []
    unperturbed = finish(list(base.layers), np.ones(n), 1.0)
    if spec.include_base or not (spec.enabled or copies):
        realizations.append(Realization(0, unperturbed, True, eta0, fref, {}))
    if not spec.enabled:
        if copies:
            start = len(realizations)
            realizations += [
                Realization(start + i, unperturbed, False, eta0, fref, {})
                for i in range(spec.n_realizations)
            ]
        return Ensemble(tuple(realizations), spec, attenuation)

    rng = np.random.default_rng(spec.seed)
    for draw in _draw(spec, n, rng):
        index = len(realizations)
        layers, params = _perturb_velocity(base, spec, draw)
        ones = np.ones(n)
        q_factor = 10 ** (spec.q_log10.sd * draw["q"]) * ones if "q" in draw else ones
        qpqs = float(10 ** (spec.qpqs_sd_log10 * draw["qpqs"][0])) if "qpqs" in draw else 1.0
        eta = eta0
        if eta0 is not None and "eta" in draw:
            eta = max(0.0, eta0 + spec.eta_sd * float(draw["eta"][0]))
        try:
            model = finish(layers, q_factor, qpqs)
        except PhysicsError as exc:
            raise PhysicsError(
                f"realization {index}: {exc} (reduce the velocity.perturb sds)"
            ) from exc
        params |= {
            "q_factor": q_factor.tolist(),
            "qpqs_factor": [qpqs],
            "eta": [] if eta is None else [eta],
        }
        realizations.append(Realization(index, model, False, eta, fref, params))
    return Ensemble(tuple(realizations), spec, attenuation)
