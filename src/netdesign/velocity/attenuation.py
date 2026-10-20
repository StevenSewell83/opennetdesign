"""Attenuation (Q) for layered models.

Two modes:

- ``vs_scaled`` (default): ``Qs = qs_per_vs * Vs`` (Vs in km/s) and ``Qp = qp_qs_ratio * Qs``,
  applied to layer tops and gradients, so Q follows the velocity layering and any velocity
  perturbation. Useful when no local Q estimates exist.
- ``layered``: Q is read per layer from the model file (CSV ``qp``, ``qs``, ``qp_grad``,
  ``qs_grad``) and left unchanged here.

Frequency dependence is ``Q(f) = Q0 * (f / fref_hz) ** eta`` and is carried alongside the model
(it is not a per-layer property).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from netdesign.errors import ConfigError
from netdesign.velocity.layered import LayeredModel

__all__ = ["AttenuationSpec", "apply_attenuation"]


class AttenuationSpec(BaseModel):
    """``velocity.attenuation`` config section.

    The default coefficients are placeholders, not calibrated values for any site.
    """

    model_config = ConfigDict(extra="forbid")

    model: Literal["vs_scaled", "layered"] = "vs_scaled"
    qs_per_vs: float = Field(50.0, gt=0, description="Qs per km/s of Vs (vs_scaled mode)")
    qp_qs_ratio: float = Field(2.0, gt=0, description="Qp / Qs (vs_scaled mode)")
    eta: float = Field(0.0, ge=0, description="frequency exponent in Q0 * (f/fref)**eta")
    fref_hz: float = Field(1.0, gt=0)


def apply_attenuation(
    model: LayeredModel,
    spec: AttenuationSpec,
    *,
    q_factor: tuple[float, ...] | None = None,
    qpqs_factor: float = 1.0,
) -> LayeredModel:
    """Return ``model`` with Q set according to ``spec``, optionally perturbed.

    Parameters
    ----------
    model : LayeredModel
        Velocity model (for ``layered`` mode it must already carry Q).
    spec : AttenuationSpec
        Attenuation settings.
    q_factor : tuple of float, optional
        Per-layer multiplier on Qs (and Qp); defaults to 1 everywhere.
    qpqs_factor : float
        Multiplier on the Qp/Qs ratio.

    Raises
    ------
    ConfigError
        If ``spec.model`` is ``layered`` but the model has no Q, or ``q_factor`` has the
        wrong length.
    """
    factors = q_factor if q_factor is not None else (1.0,) * model.n_layers
    if len(factors) != model.n_layers:
        raise ConfigError(f"q_factor has {len(factors)} entries for {model.n_layers} layers")
    layers = []
    for la, f in zip(model.layers, factors, strict=True):
        if spec.model == "vs_scaled":
            qs, qs_grad = spec.qs_per_vs * la.vs, spec.qs_per_vs * la.vs_grad
            ratio = spec.qp_qs_ratio
            qp, qp_grad = ratio * qs, ratio * qs_grad
        else:
            if la.qp is None or la.qs is None:
                raise ConfigError(
                    "velocity.attenuation.model is 'layered' but the model file has no qp/qs"
                )
            qp, qp_grad, qs, qs_grad = la.qp, la.qp_grad, la.qs, la.qs_grad
        layers.append(
            replace(
                la,
                qs=qs * f,
                qs_grad=qs_grad * f,
                qp=qp * f * qpqs_factor,
                qp_grad=qp_grad * f * qpqs_factor,
            )
        )
    return LayeredModel(tuple(layers))
