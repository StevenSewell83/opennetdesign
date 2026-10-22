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

Locatability (with detection)
-----------------------------
With detection probabilities, an event is only located if it gets at least
``locate_min_picks`` picks including ``locate_min_p_picks`` P picks. ``L_j``, the probability
of that at target point ``j``, follows exactly from the stations' independent pick
probabilities (a capped count distribution). The criterion is then taken over located events
(point weights ``w_j L_j``), and the located fraction ``F = sum_j w_j L_j`` must reach
``min_located_fraction``: networks below it score ``PENALTY_M * (1 + gap)``, so the search
raises coverage first, and only sizes that reach it can be recommended. This stops the
averaged-information approximation (valid when picks are likely) from being optimised where
most picks are unlikely, e.g. by stacking neighbouring stations with small pick chances.
``locatability: false`` restores the plain averaged-information criterion.

With ``min_separation_km`` > 0, no chosen station may be within that horizontal distance of
another (``fixed`` stations are exempt among themselves): the greedy step, exchanges, restart
starts and random baselines all respect it, and sizes no feasible network reaches are skipped.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Self

import numba
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from netdesign._score import point_scores
from netdesign.errors import ConfigError
from netdesign.information import PickSpec, achievable_stats, station_stats
from netdesign.sensitivity import Sensitivity

__all__ = [
    "DesignResult",
    "DesignSpec",
    "Evaluator",
    "NetworkResult",
    "meets_targets",
    "optimise",
    "separation_conflicts",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

PENALTY_M = 1e9
"""Score of networks that locate too few design events (plus the shortfall)."""


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
    target_horizontal_m: float | None = Field(
        None, gt=0, description="horizontal RMS goal (located events); with target_depth_m"
    )
    target_depth_m: float | None = Field(None, gt=0, description="depth RMS goal")
    min_gain_pct: float = Field(5.0, ge=0, description="knee: stop when a station adds less")
    min_separation_km: float = Field(
        0.0, ge=0, description="chosen stations must be farther apart than this (horizontal)"
    )
    locatability: bool = Field(True, description="with detection: score located events only")
    locate_min_picks: int = Field(4, ge=1, description="picks an event needs to be located")
    locate_min_p_picks: int = Field(1, ge=0, description="of which at least this many P")
    min_located_fraction: float = Field(
        0.9, ge=0, le=1, description="share of design events a network must locate"
    )
    jobs: int = Field(
        default_factory=lambda: os.cpu_count() or 1, ge=1, description="threads for the search"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        both = set(self.fixed) & set(self.exclude)
        if both:
            raise ConfigError(f"design: stations both fixed and excluded: {sorted(both)}")
        if self.n_min and self.n_max and self.n_min > self.n_max:
            raise ConfigError("design.n_min must be <= design.n_max")
        if self.locate_min_p_picks > self.locate_min_picks:
            raise ConfigError("design.locate_min_p_picks must be <= locate_min_picks")
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
        self.A = np.ascontiguousarray(self.A, dtype=np.float64)
        self.B, self.M, self.C = (
            None if x is None else np.ascontiguousarray(x, dtype=np.float64)
            for x in (self.B, self.M, self.C)
        )
        self.n_stations = self.A.shape[0]
        # pick probabilities (S, J, P) for locatability; None: every event is located
        self.pick_p: FloatArray | None = None
        self.is_p = np.array([p == "P" for p in sens.phases], dtype=np.bool_)
        if detection is not None and spec.locatability:
            mean = np.asarray(detection, dtype=np.float64).mean(axis=0)  # (S, P, J)
            self.pick_p = np.ascontiguousarray(mean.transpose(0, 2, 1))

    def _empty_counts(self) -> FloatArray:
        """P / S pick-count distributions (2, J, K+1) of an empty network."""
        d = np.zeros((2, len(self.weights), self.spec.locate_min_picks + 1))
        d[:, :, 0] = 1.0
        return d

    def _fold(self, dist: FloatArray, s: int) -> FloatArray:
        """``dist`` with station ``s``'s picks added (counts capped at K = "K or more")."""
        assert self.pick_p is not None
        k = self.spec.locate_min_picks
        out = dist.copy()
        for ph, is_p in enumerate(self.is_p):
            i = 0 if is_p else 1
            d, p = out[i], self.pick_p[s, :, ph][:, None]
            new = d * (1 - p)
            new[:, 1:] += d[:, :-1] * p
            new[:, k] += d[:, k] * p[:, 0]
            out[i] = new
        return out

    def _counts(self, stations: Sequence[int]) -> FloatArray:
        dist = self._empty_counts()
        for s in stations:
            dist = self._fold(dist, s)
        return dist

    @staticmethod
    def _combine(a: FloatArray, b: FloatArray) -> FloatArray:
        """Pick-count distributions of two disjoint station sets together (capped)."""
        k = a.shape[-1] - 1
        out = np.zeros_like(a)
        for i in range(k + 1):
            for j in range(k + 1 - i):
                out[..., i + j] += a[..., i] * b[..., j]
        out[..., k] = 1.0 - out[..., :k].sum(axis=-1)  # the rest is "K or more"
        return out

    def _located_from(self, dist: FloatArray) -> FloatArray:
        k, a = self.spec.locate_min_picks, self.spec.locate_min_p_picks
        tail_s = np.cumsum(dist[1][:, ::-1], axis=1)[:, ::-1]  # P(S picks >= m)
        return np.sum(
            [dist[0][:, i] * (tail_s[:, k - i] if i < k else 1.0) for i in range(a, k + 1)],
            axis=0,
        )

    def located_probability(self, stations: Sequence[int]) -> FloatArray:
        """``L_j`` (J,): probability that an event at each point is located by the network
        (numpy reference for the compiled kernel); ones without locatability."""
        if self.pick_p is None:
            return np.ones(len(self.weights))
        return self._located_from(self._counts(stations))

    def _variants(self, base: Sequence[int], sub: IntArray) -> tuple[FloatArray, IntArray]:
        """Pick-count distributions of the base network minus each row's ``sub`` stations
        (V, 2, J, K+1), and each row's variant. Leave-one-out variants come from prefix and
        suffix products (O(n) for all of them); other removals are counted directly."""
        base = list(base)
        if sub.shape[1] <= 1:  # the search's case: at most one removal per row (vectorised)
            col = sub[:, 0] if sub.shape[1] else np.full(len(sub), -1, np.intp)
            col = np.where(col >= 0, col, -1)
            values, inverse = np.unique(col, return_inverse=True)
            unique: list[tuple[int, ...]] = [() if v < 0 else (int(v),) for v in values]
            row_variant = inverse.astype(np.intp)
        else:
            keys = [tuple(sorted(int(x) for x in row if x >= 0)) for row in sub]
            unique = sorted(set(keys))
            where = {key: i for i, key in enumerate(unique)}
            row_variant = np.array([where[key] for key in keys], dtype=np.intp)
        singles = {key[0] for key in unique if len(key) == 1 and key[0] in base}
        loo: dict[int, FloatArray] = {}
        if singles:
            prefix = [self._empty_counts()]
            for s in base:
                prefix.append(self._fold(prefix[-1], s))
            suffix = [self._empty_counts()]
            for s in reversed(base):
                suffix.append(self._fold(suffix[-1], s))
            suffix.reverse()  # suffix[i] = stations base[i:]
            for i, s in enumerate(base):
                if s in singles:
                    loo[s] = self._combine(prefix[i], suffix[i + 1])
        out = np.empty((len(unique), 2, len(self.weights), self.spec.locate_min_picks + 1))
        for i, key in enumerate(unique):
            if len(key) == 1 and key[0] in loo:
                out[i] = loo[key[0]]
            else:
                out[i] = self._counts([s for s in base if s not in key])
        return out, row_variant

    def penalised(self, metric: FloatArray, located: FloatArray) -> FloatArray:
        """Search score: ``metric`` where the located fraction reaches
        ``min_located_fraction``, else ``PENALTY_M * (1 + shortfall)``."""
        if self.pick_p is None:
            return metric
        gap = self.spec.min_located_fraction - located
        return np.where(gap <= 1e-12, metric, PENALTY_M * (1 + gap))

    def _zero_sums(self) -> tuple[FloatArray, ...]:
        return tuple(
            np.zeros(x.shape[1:]) for x in (self.A, self.B, self.M, self.C) if x is not None
        )

    def evaluate_batch(
        self, base: Sequence[int], add: npt.ArrayLike, sub: npt.ArrayLike | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Criteria (m, over located events) and located fractions of many networks at once:
        ``base`` plus the stations in each row of ``add`` minus those in the same row of
        ``sub`` (rows padded with -1). Compiled and multithreaded (``design.jobs``); equal
        to :meth:`evaluate` of each network."""
        add_a = np.asarray(add, dtype=np.intp)
        if add_a.ndim == 1:
            add_a = add_a[:, None]
        n = add_a.shape[0]
        sub_a = np.full((n, 0), -1, np.intp) if sub is None else np.asarray(sub, dtype=np.intp)
        if sub_a.ndim == 1:
            sub_a = sub_a[:, None]
        sums = self.sums(base) if len(base) else self._zero_sums()
        gls = self.spec.measure == "gls"
        extra = self.M if gls else self.C
        base_t = (sums[0], None if self.B is None else sums[1], None if extra is None else sums[-1])
        numba.set_num_threads(min(self.spec.jobs, numba.config.NUMBA_NUM_THREADS))  # type: ignore[attr-defined]
        spec = self.spec
        locate = None
        if self.pick_p is not None:
            variants, row_variant = self._variants(base, sub_a)
            locate = (
                self.pick_p,
                self.is_p,
                variants,
                row_variant,
                spec.locate_min_picks,
                spec.locate_min_p_picks,
            )
        raw, frac, located = point_scores(
            base_t,
            (self.A, self.B, extra),
            add_a,
            sub_a,
            self.prior,
            self.weights,
            spec.vertical_weight,
            gls,
            spec.criterion,
            locate,
        )
        norm = frac if self.pick_p is not None else np.ones_like(frac)
        with np.errstate(divide="ignore", invalid="ignore"):
            if spec.criterion == "rms":
                metric = np.sqrt(raw / norm) * 1000
            elif spec.criterion == "logdet":
                metric = np.exp(raw / norm / 6) * 1000
            else:
                assert located is not None
                metric = np.array(
                    [
                        weighted_percentile(
                            np.sqrt(np.maximum(e, 0)), self.weights * l_, spec.percentile
                        )
                        if np.sum(self.weights * l_) > 0
                        else np.inf
                        for e, l_ in zip(raw, located, strict=True)
                    ]
                )
        metric = np.where(np.isfinite(metric), metric, np.inf)
        return metric, frac

    def score_batch(
        self, base: Sequence[int], add: npt.ArrayLike, sub: npt.ArrayLike | None = None
    ) -> FloatArray:
        """Search scores of many networks (see :meth:`evaluate_batch`, :meth:`penalised`)."""
        return self.penalised(*self.evaluate_batch(base, add, sub))

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

    def criterion_from_cov(self, cov: FloatArray, located: FloatArray | None = None) -> float:
        """Criterion (m) from covariance matrices (J, 4, 4), over located events when
        ``located`` (``L_j``) is given."""
        spec = self.spec
        w = self.weights if located is None else self.weights * located
        if located is not None and not np.sum(w) > 0:
            return float("inf")
        norm = 1.0 if located is None else float(np.sum(w))
        var = np.diagonal(cov, axis1=-2, axis2=-1)
        e = var[:, 0] + var[:, 1] + spec.vertical_weight * var[:, 2]
        if spec.criterion == "rms":
            return float(np.sqrt(np.sum(w * e) / norm) * 1000)
        if spec.criterion == "percentile":
            return weighted_percentile(np.sqrt(np.maximum(e, 0)), w, spec.percentile)
        sign, logdet = np.linalg.slogdet(cov[:, :3, :3])
        logdet = np.where(sign > 0, logdet, np.inf)
        use = w > 0
        return float(np.exp(np.sum(w[use] * logdet[use]) / norm / 6) * 1000)

    def score_sums(self, sums: tuple[FloatArray, ...]) -> float:
        """Criterion of the network with these summed statistics (every event located)."""
        return self.criterion_from_cov(self.covariance(sums))

    def evaluate(self, stations: Sequence[int]) -> tuple[float, float]:
        """Criterion (m, over located events) and located fraction of a network."""
        located = self.located_probability(stations)
        cov = self.covariance(self.sums(stations))
        if self.pick_p is None:
            return self.criterion_from_cov(cov), 1.0
        return self.criterion_from_cov(cov, located), float(np.sum(self.weights * located))

    def axis_errors(self, stations: Sequence[int]) -> tuple[float, float]:
        """Horizontal (``sqrt(s_x^2 + s_y^2)``) and depth RMS errors (m) over located
        events, whatever the criterion."""
        located = self.located_probability(stations)
        w = self.weights * located
        if not np.sum(w) > 0:
            return float("inf"), float("inf")
        var = np.diagonal(self.covariance(self.sums(stations)), axis1=-2, axis2=-1)
        h = np.sum(w * (var[:, 0] + var[:, 1])) / np.sum(w)
        z = np.sum(w * var[:, 2]) / np.sum(w)
        return float(np.sqrt(h) * 1000), float(np.sqrt(z) * 1000)

    def score(self, stations: Sequence[int]) -> float:
        """Search score of a network given by station indices (numpy reference)."""
        metric, frac = self.evaluate(stations)
        return float(self.penalised(np.array([metric]), np.array([frac]))[0])


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
    located_fraction: float = 1.0  # share of design events located (1 without detection)
    meets_located: bool = True  # located_fraction >= min_located_fraction
    loss_located_fraction: dict[int, float] = field(default_factory=dict)
    horizontal_m: float | None = None  # RMS horizontal error over located events
    depth_m: float | None = None  # RMS depth error over located events


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


BoolArray = npt.NDArray[np.bool_]


def _exchange(
    ev: Evaluator,
    start: list[int],
    usable: list[int],
    fixed: set[int],
    conflict: BoolArray | None = None,
) -> list[int]:
    """Best-improvement swaps until no swap improves the criterion. Each pass scores every
    allowed (selected, unselected) swap in one batch; the first best swap in (out, in) order
    wins. With ``conflict`` (S, S), an incoming station may not conflict with any station
    that stays."""
    current = list(start)
    best = float(ev.score_batch(current, np.full((1, 0), -1))[0])
    while True:
        outs = [s for s in current if s not in fixed]
        ins = np.array([s for s in usable if s not in current], dtype=np.intp)
        if not outs or not len(ins):
            return current
        if conflict is None:
            add = np.tile(ins, len(outs))
            sub = np.repeat(outs, len(ins))
        else:
            hits = conflict[np.ix_(ins, np.asarray(current, dtype=np.intp))]  # (m, n)
            count = hits.sum(axis=1)
            pos = {s: k for k, s in enumerate(current)}
            allowed = [ins[count - hits[:, pos[o]] == 0] for o in outs]
            add = np.concatenate(allowed)
            sub = np.repeat(outs, [len(a) for a in allowed])
            if not len(add):
                return current
        scores = ev.score_batch(current, add, sub)
        i = int(np.argmin(scores))
        if not scores[i] < best * (1 - 1e-9):
            return current
        current = [int(add[i]) if s == sub[i] else s for s in current]
        best = float(scores[i])


def _random_network(
    rng: np.random.Generator,
    free: list[int],
    k: int,
    chosen: list[int],
    conflict: BoolArray | None,
) -> list[int] | None:
    """``k`` random stations from ``free`` to add to ``chosen``; with ``conflict``, drawn one
    at a time among those still allowed (None if fewer than ``k`` fit)."""
    if conflict is None:
        return [int(s) for s in rng.choice(free, size=k, replace=False)]
    blocked: BoolArray = np.zeros(len(conflict), dtype=bool)
    if chosen:
        blocked = np.asarray(conflict[chosen].any(axis=0))
    picked: list[int] = []
    for s in rng.permutation(free):
        if len(picked) == k:
            break
        if not blocked[s]:
            picked.append(int(s))
            blocked = blocked | conflict[s]
    return picked if len(picked) == k else None


def separation_conflicts(xy: npt.ArrayLike, min_separation_km: float) -> BoolArray | None:
    """(S, S) True where two stations are within ``min_separation_km`` horizontally (1 m
    tolerance, so grid neighbours exactly that far apart conflict); None if it is 0."""
    if min_separation_km <= 0:
        return None
    p = np.asarray(xy, dtype=np.float64)[:, :2]
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    out: BoolArray = d <= min_separation_km + 1e-3
    np.fill_diagonal(out, False)
    return out


def optimise(
    ev: Evaluator,
    names: Sequence[str],
    spec: DesignSpec,
    rng: np.random.Generator,
    xy: npt.ArrayLike | None = None,
    progress: Callable[[str], None] | None = None,
) -> DesignResult:
    """Run the search described in the module docstring. ``xy`` (S, 2+) are the stations'
    local coordinates in km, needed for ``min_separation_km``. ``progress`` receives a line
    per finished network size (with a rough time remaining).

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
    conflict = None
    if spec.min_separation_km > 0:
        if xy is None:
            raise ConfigError("design.min_separation_km needs the station coordinates")
        conflict = separation_conflicts(xy, spec.min_separation_km)

    # Greedy forward selection: a nested order over all usable candidates.
    order = list(fixed)
    while len(order) < n_max:
        trial = [s for s in usable if s not in order]
        if conflict is not None:
            blocked: BoolArray = np.zeros(len(names), dtype=bool)
            if order:
                blocked = np.asarray(conflict[order].any(axis=0))
            trial = [s for s in trial if not blocked[s]]
            if not trial:
                if len(order) < n_min:
                    raise ConfigError(
                        f"design: only {len(order)} stations fit at min_separation_km "
                        f"{spec.min_separation_km:g}, fewer than n_min {n_min}"
                    )
                notes.append(
                    f"only {len(order)} stations fit at min_separation_km "
                    f"{spec.min_separation_km:g}; larger sizes skipped"
                )
                break
        scores = ev.score_batch(order, trial)
        i = int(np.argmin(scores))
        order.append(trial[i])

    networks: dict[int, NetworkResult] = {}
    fixed_set = set(fixed)
    free = [s for s in usable if s not in fixed_set]
    start_time = time.perf_counter()
    sizes = list(range(n_min, len(order) + 1))
    # rough relative cost of a size: exchange passes x swaps per pass ~ (n (S - n))^2
    work = {k: float(k * max(len(usable) - k, 1)) ** 2 for k in sizes}
    total_work, done_work = sum(work.values()), 0.0
    for n in sizes:
        start = order[:n]
        best = _exchange(ev, start, usable, fixed_set, conflict) if spec.exchange else list(start)
        candidates = [best]
        if spec.exchange and n - len(fixed) > 0 and n < len(usable):
            for _ in range(spec.n_restarts):
                pick = _random_network(rng, free, n - len(fixed), fixed, conflict)
                if pick is not None:
                    candidates.append(_exchange(ev, fixed + pick, usable, fixed_set, conflict))
        scores = ev.score_batch([], candidates)
        i_best = int(np.argmin(scores))
        best, best_score = candidates[i_best], float(scores[i_best])
        reached = int(np.sum(np.isclose(scores, best_score, rtol=1e-6)))
        if scores[0] > best_score * (1 + 1e-6):
            notes.append(f"n={n}: a random restart beat the greedy+exchange network")

        rand_pct, frac_better = None, None
        if spec.n_random and n - len(fixed) > 0 and n < len(usable):
            draws = [
                _random_network(rng, free, n - len(fixed), fixed, conflict)
                for _ in range(spec.n_random)
            ]
            nets = [fixed + d for d in draws if d is not None]
            if nets:
                r_metric, r_frac = ev.evaluate_batch([], nets)
                p5, p50, p95 = np.percentile(r_metric, [5, 50, 95])
                rand_pct = (float(p5), float(p50), float(p95))
                rs = ev.penalised(r_metric, r_frac)
                frac_better = float(np.mean(rs < best_score * (1 - 1e-9)))

        # Report the criterion over located events and the located fraction (the search
        # score adds the shortfall penalty).
        (metric,), (located,) = ev.evaluate_batch([], [best])
        (g_metric,), _ = ev.evaluate_batch([], [order[:n]])
        loss: dict[int, float] = {}
        loss_frac: dict[int, float] = {}
        if n > 1:
            l_metric, l_frac = ev.evaluate_batch(best, np.full((n, 0), -1), best)
            loss = {s: float(v) for s, v in zip(best, l_metric, strict=True)}
            loss_frac = {s: float(v) for s, v in zip(best, l_frac, strict=True)}
        networks[n] = NetworkResult(
            n,
            tuple(sorted(best)),
            float(metric),
            float(g_metric),
            reached,
            rand_pct,
            frac_better,
            loss,
            float(located),
            bool(best_score < PENALTY_M),
            loss_frac,
            *ev.axis_errors(best),
        )

        if progress is not None:
            done_work += work[n]
            elapsed = time.perf_counter() - start_time
            left = elapsed * (total_work - done_work) / max(done_work, 1e-9)
            net = networks[n]
            progress(
                f"n={n} of {sizes[0]}-{sizes[-1]}: {net.criterion_m:.0f} m, "
                f"located {100 * net.located_fraction:.0f}%  "
                f"[{elapsed / 60:.1f} min, ~{left / 60:.0f} min left]"
            )

    all_score = float(ev.evaluate_batch([], [usable])[0][0])
    recommended = _recommend(networks, spec, all_score)
    if not networks[recommended].meets_located:
        best_n = max(networks, key=lambda k: networks[k].located_fraction)
        notes.append(
            f"no network up to {max(networks)} stations locates "
            f"{100 * spec.min_located_fraction:.0f}% of the design events (best "
            f"{100 * networks[best_n].located_fraction:.0f}% at n={best_n}); the design event "
            "is too small for these candidates"
        )
    return DesignResult(order, networks, recommended, usable, fixed, all_score, notes)


def meets_targets(net: NetworkResult, spec: DesignSpec) -> bool:
    """Whether a network meets every accuracy goal that is set."""
    goals = [
        (net.criterion_m, spec.target_m),
        (net.horizontal_m, spec.target_horizontal_m),
        (net.depth_m, spec.target_depth_m),
    ]
    return all(v is not None and v <= t for v, t in goals if t is not None)


def _recommend(networks: dict[int, NetworkResult], spec: DesignSpec, all_m: float) -> int:
    """Smallest size meeting the accuracy goals (``target_m``, ``target_horizontal_m``,
    ``target_depth_m``), else the knee; only sizes that locate enough design events count
    (the largest size if none does, or if no size meets the goals)."""
    ns = [n for n in sorted(networks) if networks[n].meets_located]
    if not ns:
        return max(networks)
    if any(t is not None for t in (spec.target_m, spec.target_horizontal_m, spec.target_depth_m)):
        meeting = [n for n in ns if meets_targets(networks[n], spec)]
        return meeting[0] if meeting else ns[-1]
    for n in ns[:-1]:
        nxt = networks[n + 1].criterion_m if n + 1 in ns else all_m
        gain = 100 * (1 - nxt / networks[n].criterion_m)
        if gain < spec.min_gain_pct:
            return n
    return ns[-1]
