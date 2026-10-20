"""Choose station subsets that minimise the predicted absolute location error.

Error model (``measure``)
-------------------------
- ``achievable`` (default): mean-square error of a NonLinLoc-like location in the starting
  model, over the velocity ensemble (:class:`netdesign.information.AchievableStats`), the
  measure validated against NonLinLoc.
- ``gls``: covariance of the generalised least-squares location that knows the correlated
  model error (:class:`netdesign.information.StationStats`); a lower bound.

Both add a weak prior ``P = diag(prior_sd_km^-2 x3, prior_sd_t0_s^-2)`` so that networks too
small to determine a location still score finitely; for the achievable measure the error is
``(A + P)^-1 (A + B B^T + P) (A + P)^-1``. With realistic networks the prior is negligible.

Criterion (lower is better), over target points ``j`` with weights ``w_j``
--------------------------------------------------------------------------
``e_j = s_x^2 + s_y^2 + vertical_weight * s_z^2`` (spatial error variance).

- ``rms``: ``sqrt(sum_j w_j e_j)``, the RMS 3D location error (m);
- ``percentile``: the weighted ``percentile`` of ``sqrt(e_j)`` (m), e.g. 90 = worst 10 %;
- ``logdet``: ``exp(sum_j w_j log det C_xyz,j / 6)`` (m), the geometric-mean error-ellipsoid
  semi-axis (D-optimality).

Search
------
For every network size ``n`` in ``[n_min, n_max]``: greedy forward selection (from the
``fixed`` stations) gives a nested sequence; each greedy network is then improved by
exchange (swap one selected for one unselected station while the criterion improves), also
from ``n_restarts`` random starts, keeping the best. ``n_random`` random networks per size
give a baseline, and every best network is scored with each single station removed
(robustness to one station failing).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from netdesign.errors import ConfigError
from netdesign.information import PickSpec, achievable_stats, station_stats
from netdesign.sensitivity import Sensitivity

__all__ = ["DesignResult", "DesignSpec", "Evaluator", "NetworkResult", "optimise"]

FloatArray = npt.NDArray[np.float64]


class DesignSpec(BaseModel):
    """``design`` config section."""

    model_config = ConfigDict(extra="forbid")

    measure: Literal["achievable", "gls"] = "achievable"
    criterion: Literal["rms", "percentile", "logdet"] = "rms"
    percentile: float = Field(90.0, gt=0, lt=100)
    vertical_weight: float = Field(1.0, ge=0)
    prior_sd_km: float = Field(5.0, gt=0)
    prior_sd_t0_s: float = Field(1.0, gt=0)
    fixed: list[str] = Field(default_factory=list, description="always included (existing)")
    exclude: list[str] = Field(default_factory=list, description="never used")
    n_min: int | None = Field(None, ge=1, description="default: max(len(fixed), 3)")
    n_max: int | None = Field(None, ge=1, description="default: all usable candidates")
    exchange: bool = True
    n_restarts: int = Field(5, ge=0)
    n_random: int = Field(200, ge=0)
    seed: int = 0
    target_m: float | None = Field(None, gt=0, description="recommend smallest n meeting it")
    min_gain_pct: float = Field(5.0, ge=0, description="knee: stop when a station adds less")

    @model_validator(mode="after")
    def _check(self) -> Self:
        both = set(self.fixed) & set(self.exclude)
        if both:
            raise ConfigError(f"design: stations both fixed and excluded: {sorted(both)}")
        if self.n_min and self.n_max and self.n_min > self.n_max:
            raise ConfigError("design.n_min must be <= design.n_max")
        return self


class Evaluator:
    """Scores networks from per-station sums; ``score(stations)`` is the criterion (m)."""

    def __init__(
        self,
        sens: Sensitivity,
        picks: PickSpec,
        spec: DesignSpec,
        detection: npt.ArrayLike | None = None,
    ) -> None:
        self.spec = spec
        self.weights = sens.sources.weights / sens.sources.weights.sum()
        prior = np.diag([spec.prior_sd_km**-2] * 3 + [spec.prior_sd_t0_s**-2]).astype(np.float64)
        self.prior = prior
        if spec.measure == "achievable":
            st = achievable_stats(sens, picks, detection)
            self.A, self.B, self.M, self.C = st.A, st.B, None, st.C
        else:
            gs = station_stats(sens, picks, detection)
            # base-model gradients (realization 0), as for the achievable measure
            self.A = gs.A[0]
            self.B = None if gs.B is None else gs.B[0]
            self.M = None if gs.M is None else gs.M[0]
            self.C = None
        self.n_stations = self.A.shape[0]

    def sums(self, stations: Sequence[int]) -> tuple[FloatArray, ...]:
        """Summed statistics of a network (A, then B, M, C as available)."""
        idx = np.asarray(list(stations), dtype=np.intp)
        out = [self.A[idx].sum(axis=0)]
        for extra in (self.B, self.M, self.C):
            if extra is not None:
                out.append(extra[idx].sum(axis=0))
        return tuple(out)

    def station_terms(self, s: int) -> tuple[FloatArray, ...]:
        """One station's statistics, in the same order as :meth:`sums`."""
        out = [self.A[s]]
        for extra in (self.B, self.M, self.C):
            if extra is not None:
                out.append(extra[s])
        return tuple(out)

    def covariance(self, sums: tuple[FloatArray, ...]) -> FloatArray:
        """Error covariance / MSE matrices (J, 4, 4) from summed statistics."""
        a = sums[0] + self.prior
        if self.spec.measure == "achievable":
            a_inv = np.linalg.inv(a)
            inner = a.copy()
            if self.B is not None:
                b = sums[1]
                inner = inner + b @ np.swapaxes(b, -1, -2)
            if self.C is not None:
                inner = inner + sums[-1]  # detection-pattern correction
            out: FloatArray = a_inv @ inner @ a_inv
            return out
        if self.B is not None and self.M is not None:
            b, m = sums[1], sums[2].copy()
            k = m.shape[-1]
            m[..., np.arange(k), np.arange(k)] += 1.0
            a = np.asarray(a - b @ np.linalg.solve(m, np.swapaxes(b, -1, -2)), dtype=np.float64)
        result: FloatArray = np.asarray(np.linalg.inv(a), dtype=np.float64)
        return result

    def criterion_from_cov(self, cov: FloatArray) -> float:
        """Criterion (m) from covariance matrices (J, 4, 4)."""
        spec = self.spec
        var = np.diagonal(cov, axis1=-2, axis2=-1)
        e = var[:, 0] + var[:, 1] + spec.vertical_weight * var[:, 2]
        if spec.criterion == "rms":
            return float(np.sqrt(np.sum(self.weights * e)) * 1000)
        if spec.criterion == "percentile":
            return weighted_percentile(np.sqrt(np.maximum(e, 0)), self.weights, spec.percentile)
        sign, logdet = np.linalg.slogdet(cov[:, :3, :3])
        logdet = np.where(sign > 0, logdet, np.inf)
        return float(np.exp(np.sum(self.weights * logdet) / 6) * 1000)

    def score_sums(self, sums: tuple[FloatArray, ...]) -> float:
        """Criterion of the network with these summed statistics."""
        return self.criterion_from_cov(self.covariance(sums))

    def score(self, stations: Sequence[int]) -> float:
        """Criterion of a network given by station indices."""
        return self.score_sums(self.sums(stations))


def weighted_percentile(values: FloatArray, weights: FloatArray, q: float) -> float:
    """Weighted percentile ``q`` (0-100) of ``values``."""
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    cum /= cum[-1]
    return float(np.interp(q / 100, cum, values[order]))


@dataclass(frozen=True)
class NetworkResult:
    """Best network found for one size."""

    n: int
    stations: tuple[int, ...]
    criterion_m: float
    greedy_criterion_m: float
    restarts_reaching_best: int
    random_percentiles_m: tuple[float, float, float] | None  # 5th, 50th, 95th
    fraction_random_better: float | None
    loss_one_station_m: dict[int, float]  # station removed -> criterion


@dataclass
class DesignResult:
    """Everything the design stage produces."""

    greedy_order: list[int]
    networks: dict[int, NetworkResult]
    recommended_n: int
    usable: list[int]
    fixed: list[int]
    all_candidates_m: float
    notes: list[str] = field(default_factory=list)


def _add(a: tuple[FloatArray, ...], b: tuple[FloatArray, ...]) -> tuple[FloatArray, ...]:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _sub(a: tuple[FloatArray, ...], b: tuple[FloatArray, ...]) -> tuple[FloatArray, ...]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _exchange(ev: Evaluator, start: list[int], usable: list[int], fixed: set[int]) -> list[int]:
    """Best-improvement swaps until no swap improves the criterion."""
    current = list(start)
    sums = ev.sums(current)
    best = ev.score_sums(sums)
    terms = {s: ev.station_terms(s) for s in usable}
    while True:
        move, move_score = None, best
        for out_s in [s for s in current if s not in fixed]:
            without = _sub(sums, terms[out_s])
            for in_s in [s for s in usable if s not in current]:
                score = ev.score_sums(_add(without, terms[in_s]))
                if score < move_score * (1 - 1e-9):
                    move, move_score = (out_s, in_s), score
        if move is None:
            return current
        current = [move[1] if s == move[0] else s for s in current]
        sums = _add(_sub(sums, terms[move[0]]), terms[move[1]])
        best = move_score


def optimise(
    ev: Evaluator, names: Sequence[str], spec: DesignSpec, rng: np.random.Generator
) -> DesignResult:
    """Run the search described in the module docstring.

    Raises
    ------
    ConfigError
        If fixed/excluded stations are unknown or the size range is impossible.
    """
    index = {n: i for i, n in enumerate(names)}
    unknown = sorted((set(spec.fixed) | set(spec.exclude)) - set(index))
    if unknown:
        raise ConfigError(f"design: unknown stations {unknown}")
    fixed = [index[n] for n in spec.fixed]
    usable = [i for i in range(len(names)) if names[i] not in spec.exclude]
    n_max = min(spec.n_max or len(usable), len(usable))
    n_min = spec.n_min or max(len(fixed), 3)
    n_min = max(n_min, len(fixed), 1)
    if n_min > n_max:
        raise ConfigError(f"design: n_min {n_min} > usable stations {n_max}")
    notes: list[str] = []

    # Greedy forward selection: a nested order over all usable candidates.
    order = list(fixed)
    sums = ev.sums(order) if order else None
    terms = {s: ev.station_terms(s) for s in usable}
    greedy_score: dict[int, float] = {}
    while len(order) < n_max:
        best_s, best_score = None, np.inf
        for s in usable:
            if s in order:
                continue
            trial = terms[s] if sums is None else _add(sums, terms[s])
            score = ev.score_sums(trial)
            if score < best_score:
                best_s, best_score = s, score
        assert best_s is not None
        order.append(best_s)
        sums = terms[best_s] if sums is None else _add(sums, terms[best_s])
        greedy_score[len(order)] = best_score
    for n in range(1, len(fixed) + 1):
        greedy_score.setdefault(n, ev.score(order[:n]))

    networks: dict[int, NetworkResult] = {}
    fixed_set = set(fixed)
    free = [s for s in usable if s not in fixed_set]
    for n in range(n_min, n_max + 1):
        start = order[:n]
        best = _exchange(ev, start, usable, fixed_set) if spec.exchange else list(start)
        best_score = ev.score(best)
        candidates = [best]
        if spec.exchange and n - len(fixed) > 0 and n < len(usable):
            for _ in range(spec.n_restarts):
                pick = list(rng.choice(free, size=n - len(fixed), replace=False))
                candidates.append(_exchange(ev, fixed + [int(s) for s in pick], usable, fixed_set))
        scores = [ev.score(c) for c in candidates]
        i_best = int(np.argmin(scores))
        best, best_score = candidates[i_best], scores[i_best]
        reached = int(np.sum(np.isclose(scores, best_score, rtol=1e-6)))
        if scores[0] > best_score * (1 + 1e-6):
            notes.append(f"n={n}: a random restart beat the greedy+exchange network")

        rand_pct, frac_better = None, None
        if spec.n_random and n - len(fixed) > 0 and n < len(usable):
            rs = [
                ev.score(
                    fixed + [int(s) for s in rng.choice(free, size=n - len(fixed), replace=False)]
                )
                for _ in range(spec.n_random)
            ]
            p5, p50, p95 = np.percentile(rs, [5, 50, 95])
            rand_pct = (float(p5), float(p50), float(p95))
            frac_better = float(np.mean(np.array(rs) < best_score * (1 - 1e-9)))

        loss = {s: ev.score([t for t in best if t != s]) for s in best if n > 1}
        networks[n] = NetworkResult(
            n,
            tuple(sorted(best)),
            best_score,
            greedy_score[n],
            reached,
            rand_pct,
            frac_better,
            loss,
        )

    all_score = ev.score(usable)
    recommended = _recommend(networks, spec, all_score)
    return DesignResult(order, networks, recommended, usable, fixed, all_score, notes)


def _recommend(networks: dict[int, NetworkResult], spec: DesignSpec, all_m: float) -> int:
    ns = sorted(networks)
    if spec.target_m is not None:
        meeting = [n for n in ns if networks[n].criterion_m <= spec.target_m]
        return meeting[0] if meeting else ns[-1]
    for n in ns[:-1]:
        nxt = networks[n + 1].criterion_m if n + 1 in networks else all_m
        gain = 100 * (1 - nxt / networks[n].criterion_m)
        if gain < spec.min_gain_pct:
            return n
    return ns[-1]
