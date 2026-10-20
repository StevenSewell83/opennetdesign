"""Fisher information for the linearised location problem, with velocity-model error.

For source ``j`` the unknowns are ``(x, y, z, t0)``. A pick ``i`` (station ``s``, phase ``p``)
has the row ``g_i = (dT/dx, dT/dy, dT/dz, 1)`` and weight ``w_i = detection_i / sigma_p**2``.
The data covariance is ``C_d = W^-1 + C_m`` with ``C_m`` the velocity-model error, and a
network's information is ``F = G^T C_d^-1 G``.

``picks.model_error`` selects ``C_m``:

- ``none``: ``C_m = 0``; ``F = sum_i w_i g_i g_i^T`` (additive over stations).
- ``independent``: the diagonal of the ensemble covariance is added to each pick's variance;
  still additive, but it ignores that model error is largely shared between stations.
- ``correlated`` (default): the full ensemble covariance of travel times between all picks at
  each source, ``C_m = U U^T`` with ``U[i, k] = (T_k[i] - mean_k T[i]) / sqrt(R - 1)``. ``U`` is
  compressed per source by SVD to the ``K`` components holding ``model_error_energy`` of the
  variance (rank <= R - 1). By the Woodbury identity::

      F = A - B (I_K + M)^-1 B^T,
      A = sum_i w_i g_i g_i^T,   B = sum_i w_i g_i u_i^T,   M = sum_i w_i u_i u_i^T,

  and ``A``, ``B``, ``M`` are sums over picks, so they are precomputed per station and a
  network's information needs only their sums plus one ``K x K`` solve per source. A model error
  common to every pick (a pure time shift) is then absorbed by the origin time, as it should be.

The information is evaluated with each realization's gradients ``G_r`` (so a design can be
scored in every model of the ensemble) and the same ``C_m`` from the whole ensemble.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from netdesign.errors import ConfigError
from netdesign.nll import Phase
from netdesign.sensitivity import Sensitivity

__all__ = [
    "AchievableStats",
    "PickSpec",
    "StationStats",
    "achievable_stats",
    "model_error_factor",
    "pick_variance",
    "station_stats",
]

FloatArray = npt.NDArray[np.float64]


def _default_sigma() -> dict[Phase, float]:
    return {"P": 0.005, "S": 0.010}


class PickSpec(BaseModel):
    """``picks`` config section."""

    model_config = ConfigDict(extra="forbid")

    sigma_s: dict[Phase, float] = Field(
        default_factory=_default_sigma,
        description="pick standard deviation per phase, seconds",
    )
    model_error: Literal["none", "independent", "correlated"] = "correlated"
    model_error_energy: float = Field(
        0.99, gt=0, le=1, description="fraction of ensemble variance kept (correlated)"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        bad = {p: v for p, v in self.sigma_s.items() if not v > 0}
        if bad:
            raise ConfigError(f"picks.sigma_s must be > 0, got {bad}")
        return self


@dataclass(frozen=True)
class StationStats:
    """Per-station additive statistics; shapes (R, S, J, 4, 4), (R, S, J, 4, K), (R, S, J, K, K).

    ``B`` and ``M`` are ``None`` unless ``model_error`` is ``correlated``.
    """

    A: FloatArray
    B: FloatArray | None
    M: FloatArray | None

    @property
    def n_stations(self) -> int:
        """Number of candidate stations."""
        return int(self.A.shape[1])

    def network_information(self, stations: Sequence[int] | npt.ArrayLike) -> FloatArray:
        """Information (R, J, 4, 4) of the network made of the given station indices."""
        idx = np.asarray(stations, dtype=np.intp)
        a = self.A[:, idx].sum(axis=1)
        if self.B is None or self.M is None:
            return a
        b = self.B[:, idx].sum(axis=1)
        m = self.M[:, idx].sum(axis=1)
        k = m.shape[-1]
        m[..., np.arange(k), np.arange(k)] += 1.0
        info: FloatArray = a - b @ np.linalg.solve(m, np.swapaxes(b, -1, -2))
        return info


def pick_variance(sens: Sensitivity, picks: PickSpec) -> FloatArray:
    """Pick variance (s²) per station, phase and source, shape (S, P, J).

    Includes the ensemble variance only for ``model_error: independent``.

    Raises
    ------
    ConfigError
        If a phase in ``sens`` has no ``picks.sigma_s`` entry.
    """
    sigmas: dict[str, float] = {str(k): v for k, v in picks.sigma_s.items()}
    missing = [p for p in sens.phases if p not in sigmas]
    if missing:
        raise ConfigError(f"picks.sigma_s has no entry for phase(s) {missing}")
    sigma = np.array([sigmas[p] for p in sens.phases])
    var = np.broadcast_to((sigma**2)[None, :, None], sens.T.shape[1:]).astype(np.float64)
    if picks.model_error == "independent" and len(sens.realizations) > 1:
        var = var + sens.T.astype(np.float64).var(axis=0, ddof=1)
    return var


def model_error_factor(sens: Sensitivity, energy: float) -> FloatArray:
    """Compressed model-error factor ``U``, shape (S, P, J, K), with ``C_m ~= U U^T`` per source.

    ``K`` is the largest number of components any source needs to keep ``energy`` of its
    variance; sources needing fewer get zero columns (which contribute nothing).
    """
    t = sens.T.astype(np.float64)
    n_r = t.shape[0]
    return _compress((t - t.mean(axis=0)) / np.sqrt(n_r - 1), energy)


def station_stats(
    sens: Sensitivity,
    picks: PickSpec,
    detection: npt.ArrayLike | None = None,
) -> StationStats:
    """Precompute the per-station statistics for :meth:`StationStats.network_information`.

    Parameters
    ----------
    sens : Sensitivity
        Travel times and gradients.
    picks : PickSpec
        Pick uncertainties and model-error treatment.
    detection : array_like, optional
        Detection probability per pick, shape (R, S, P, J); default 1.

    Memory is ``R*S*J*(16 + 4K + K²)*8`` bytes (``K`` model-error components).
    """
    var = pick_variance(sens, picks)
    weight = np.broadcast_to(1.0 / var, sens.T.shape)
    if detection is not None:
        det = np.asarray(detection, dtype=np.float64)
        if det.shape != sens.T.shape:
            raise ConfigError(f"detection has shape {det.shape}, expected {sens.T.shape}")
        weight = weight * det
    n_r, n_s, n_p, n_j = sens.T.shape
    g = np.empty((n_r, n_s, n_p, n_j, 4))
    g[..., :3] = sens.G
    g[..., 3] = 1.0
    a = np.einsum("rspj,rspja,rspjb->rsjab", weight, g, g, optimize=True)
    if picks.model_error != "correlated" or n_r < 2:
        return StationStats(a, None, None)
    u = model_error_factor(sens, picks.model_error_energy)  # (S, P, J, K)
    b = np.einsum("rspj,rspja,spjk->rsjak", weight, g, u, optimize=True)
    m = np.einsum("rspj,spjk,spjl->rsjkl", weight, u, u, optimize=True)
    return StationStats(a, b, m)


@dataclass(frozen=True)
class AchievableStats:
    """Per-station sums for the error of a NonLinLoc-like location in the base model.

    NonLinLoc (L2, pick errors only) solves with ``W = diag(detection / sigma**2)`` and does
    not know the model error, so its mean-square error over the ensemble is the "sandwich"
    ``M = A^-1 (A + B B^T) A^-1`` with ``A = sum w g g^T`` (base gradients) and
    ``B B^T = G^T W S2 W G``, ``S2`` the second moment of ``T_r - T_base`` over the perturbed
    realizations (so it includes the base model's bias), compressed by SVD as in
    :func:`model_error_factor`. Validated against NonLinLoc in tests/unit/test_validate.py.

    With detection probabilities ``p``, a pick is either present at full weight or absent.
    To first order in the detection pattern, ``A`` and ``B`` use ``w = p / sigma^2`` and the
    model-error term gains the per-pick correction
    ``C = sum p (1 - p) / sigma^4 * s2 * g g^T`` (``s2`` the pick's mean-square model error):
    a rarely detected pick's bias enters with weight ``p``, not ``p^2``. The randomness of
    ``A`` itself (which picks enter the solve) is ignored; that is optimistic, by a few % when
    detection is high and up to ~30-45 % in a harsh test with ``p`` down to 0.3
    (tests/unit/test_information.py), against 35-58 % without the correction.
    Shapes: ``A`` and ``C`` (S, J, 4, 4), ``B`` (S, J, 4, K); ``B``/``C`` may be ``None``.
    """

    A: FloatArray
    B: FloatArray | None
    C: FloatArray | None = None

    def mse(self, stations: Sequence[int] | npt.ArrayLike) -> FloatArray:
        """Mean-square-error matrix (J, 4, 4) of the network of the given station indices."""
        idx = np.asarray(stations, dtype=np.intp)
        a = self.A[idx].sum(axis=0)
        a_inv = np.linalg.inv(a)
        if self.B is None:
            return a_inv
        b = self.B[idx].sum(axis=0)
        inner = a + b @ np.swapaxes(b, -1, -2)
        if self.C is not None:
            inner = inner + self.C[idx].sum(axis=0)
        out: FloatArray = a_inv @ inner @ a_inv
        return out


def achievable_stats(
    sens: Sensitivity,
    picks: PickSpec,
    detection: npt.ArrayLike | None = None,
) -> AchievableStats:
    """Precompute :class:`AchievableStats`; realization index 0 must be the base model.

    ``detection`` (R, S, P, J) is optional; its mean over realizations (the expected detection,
    so Q uncertainty counts) weights the picks.
    """
    sigmas = np.array([dict(picks.sigma_s)[p] for p in sens.phases])  # type: ignore[index]
    n_r, n_s, n_p, n_j = sens.T.shape
    inv_var = np.broadcast_to((1.0 / sigmas**2)[None, :, None], (n_s, n_p, n_j))
    prob = None if detection is None else np.asarray(detection, dtype=np.float64).mean(axis=0)
    weight = inv_var.copy() if prob is None else inv_var * prob
    g = np.empty((n_s, n_p, n_j, 4))
    g[..., :3] = sens.G[0]
    g[..., 3] = 1.0
    a = np.einsum("spj,spja,spjb->sjab", weight, g, g, optimize=True)
    if n_r < 2:
        return AchievableStats(a, None)
    t = sens.T.astype(np.float64)
    dev = (t[1:] - t[0]) / np.sqrt(n_r - 1)  # second moment about the base, (R-1, S, P, J)
    u = _compress(dev, picks.model_error_energy)  # (S, P, J, K)
    b = np.einsum("spj,spja,spjk->sjak", weight, g, u, optimize=True)
    c = None
    if prob is not None:
        s2 = np.einsum("spjk,spjk->spj", u, u)  # each pick's mean-square model error
        c_w = prob * (1.0 - prob) * inv_var**2 * s2
        c = np.einsum("spj,spja,spjb->sjab", c_w, g, g, optimize=True)
    return AchievableStats(a, b, c)


def _compress(dev: FloatArray, energy: float) -> FloatArray:
    """Factor ``U`` (S, P, J, K) with ``U U^T ~= dev^T dev`` per source (SVD, ``energy``)."""
    n_r, n_s, n_p, n_j = dev.shape
    mat = dev.reshape(n_r, n_s * n_p, n_j).transpose(2, 1, 0)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    power = s**2
    total = power.sum(axis=1, keepdims=True)
    frac = np.divide(np.cumsum(power, axis=1), total, out=np.ones_like(power), where=total > 0)
    keep = np.argmax(frac >= energy * (1 - 1e-12), axis=1) + 1
    k = max(1, int(keep.max()))
    factor = u[:, :, :k] * s[:, None, :k]
    factor *= (np.arange(k)[None, :] < keep[:, None])[:, None, :]
    out: FloatArray = factor.reshape(n_j, n_s, n_p, k).transpose(1, 2, 0, 3)
    return out
