"""Compiled batch scoring of networks for :class:`netdesign.design.Evaluator`.

One call scores ``N`` networks, each given as a base network's summed statistics plus the
stations in a row of ``add`` minus those in a row of ``sub`` (``-1`` pads rows). Networks are
spread over threads; each target point's 4x4 algebra is written out, which is ~100x faster
than numpy's batched ``inv`` on tiny matrices. Formulas are those of
:meth:`netdesign.design.Evaluator.covariance` and ``criterion_from_cov``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange

__all__ = ["CRITERIA", "point_scores"]

CRITERIA = {"rms": 0, "percentile": 1, "logdet": 2}

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


@njit(cache=True, inline="always")
def _inv4(a: FloatArray, out: FloatArray, work: FloatArray) -> bool:
    """Gauss-Jordan inverse with partial pivoting of the 4x4 ``a`` into ``out``."""
    for i in range(4):
        for j in range(4):
            work[i, j] = a[i, j]
            out[i, j] = 1.0 if i == j else 0.0
    for c in range(4):
        p = c
        for r in range(c + 1, 4):
            if abs(work[r, c]) > abs(work[p, c]):
                p = r
        if work[p, c] == 0.0:
            return False
        if p != c:
            for k in range(4):
                work[c, k], work[p, k] = work[p, k], work[c, k]
                out[c, k], out[p, k] = out[p, k], out[c, k]
        d = 1.0 / work[c, c]
        for k in range(4):
            work[c, k] *= d
            out[c, k] *= d
        for r in range(4):
            if r != c:
                f = work[r, c]
                if f != 0.0:
                    for k in range(4):
                        work[r, k] -= f * work[c, k]
                        out[r, k] -= f * out[c, k]
    return True


@njit(cache=True, inline="always")
def _chol_solve(m: FloatArray, rhs: FloatArray, k: int) -> None:
    """Solve ``m x = rhs`` in place (``m`` SPD, k x k, overwritten by its Cholesky factor;
    ``rhs`` k x 4)."""
    for j in range(k):
        s = m[j, j]
        for p in range(j):
            s -= m[j, p] * m[j, p]
        m[j, j] = np.sqrt(s)
        for i in range(j + 1, k):
            s = m[i, j]
            for p in range(j):
                s -= m[i, p] * m[j, p]
            m[i, j] = s / m[j, j]
    for c in range(rhs.shape[1]):
        for i in range(k):
            s = rhs[i, c]
            for p in range(i):
                s -= m[i, p] * rhs[p, c]
            rhs[i, c] = s / m[i, i]
        for i in range(k - 1, -1, -1):
            s = rhs[i, c]
            for p in range(i + 1, k):
                s -= m[p, i] * rhs[p, c]
            rhs[i, c] = s / m[i, i]


@njit(cache=True, inline="always")
def _add_pick(dist: FloatArray, p: float, kk: int) -> None:
    """Fold one pick with probability ``p`` into a count distribution capped at ``kk``
    (``dist[kk]`` holds "kk or more")."""
    dist[kk] += dist[kk - 1] * p
    for c in range(kk - 1, 0, -1):
        dist[c] = dist[c] * (1.0 - p) + dist[c - 1] * p
    dist[0] *= 1.0 - p


@njit(cache=True, inline="always")
def _located(
    pick_p: FloatArray,
    is_p: npt.NDArray[np.bool_],
    start: FloatArray,
    add: IntArray,
    row: int,
    j: int,
    n_add: int,
    n_ph: int,
    kk: int,
    min_p_picks: int,
    dist_p: FloatArray,
    dist_s: FloatArray,
) -> float:
    """Probability that an event at point ``j`` gets >= ``kk`` picks including >=
    ``min_p_picks`` P picks: the precomputed P / S pick-count distributions of the row's
    base network (``start``, (2, J, kk + 1)) plus this row's added stations (independent
    picks)."""
    for c in range(kk + 1):
        dist_p[c] = start[0, j, c]
        dist_s[c] = start[1, j, c]
    for t in range(n_add):
        s = add[row, t]
        if s < 0:
            continue
        for ph in range(n_ph):
            if is_p[ph]:
                _add_pick(dist_p, pick_p[s, j, ph], kk)
            else:
                _add_pick(dist_s, pick_p[s, j, ph], kk)
    total = 0.0
    for i in range(min_p_picks, kk + 1):
        tail = 0.0  # P(S picks >= kk - i)
        for c in range(max(kk - i, 0), kk + 1):
            tail += dist_s[c]
        total += dist_p[i] * tail
    return total


@njit(parallel=True, cache=True)
def _kernel(
    base_a: FloatArray,  # (J, 4, 4)
    base_b: FloatArray,  # (J, 4, K) or (0, 4, 0)
    base_x: FloatArray,  # (J, X, X): C (achievable) or M (gls), or (0, 0, 0)
    sta_a: FloatArray,  # (S, J, 4, 4)
    sta_b: FloatArray,  # (S, J, 4, K) or (0, 0, 4, 0)
    sta_x: FloatArray,  # (S, J, X, X) or (0, 0, 0, 0)
    add: IntArray,  # (N, na)
    sub: IntArray,  # (N, ns)
    prior: FloatArray,  # (4, 4)
    weights: FloatArray,  # (J,)
    vertical_weight: float,
    gls: bool,
    criterion: int,
    out: FloatArray,  # (N,) sum_j w_j L_j e_j (rms) or ... log det (logdet), or
    out_e: FloatArray,  # (N, J) per-point error variance (percentile)
    pick_p: FloatArray,  # (S, J, P) pick probabilities, or (0, 0, 0): every event located
    is_p: npt.NDArray[np.bool_],  # (P,) phase is a P phase
    variants: FloatArray,  # (V, 2, J, K+1) P / S pick-count distributions of base variants
    row_variant: IntArray,  # (N,) which variant each row starts from (base minus its sub)
    min_picks: int,
    min_p_picks: int,
    out_frac: FloatArray,  # (N,) located fraction sum_j w_j L_j
    out_l: FloatArray,  # (N, J) L_j (percentile)
) -> None:
    n_rows, n_j = add.shape[0], base_a.shape[0]
    n_add, n_sub = add.shape[1], sub.shape[1]
    has_loc = pick_p.shape[0] > 0
    n_ph, kk = pick_p.shape[2], max(min_picks, 1)
    has_b = base_b.shape[0] > 0
    has_x = base_x.shape[0] > 0
    k = base_b.shape[2]
    nx = base_x.shape[1]
    for row in prange(n_rows):
        a = np.empty((4, 4))
        ainv = np.empty((4, 4))
        work = np.empty((4, 4))
        cov = np.empty((4, 4))
        tmp = np.empty((4, 4))
        b = np.empty((4, max(k, 1)))
        x = np.empty((max(nx, 1), max(nx, 1)))
        rhs = np.empty((max(k, 1), 4))
        dist_p = np.empty(kk + 1)
        dist_s = np.empty(kk + 1)
        acc = 0.0
        frac = 0.0
        for j in range(n_j):
            located = 1.0
            if has_loc:
                located = _located(
                    pick_p, is_p, variants[row_variant[row]], add, row, j, n_add, n_ph, kk,
                    min_p_picks, dist_p, dist_s,
                )  # fmt: skip
            # summed statistics of this network at point j
            for r in range(4):
                for c in range(4):
                    a[r, c] = base_a[j, r, c] + prior[r, c]
            if has_b:
                for r in range(4):
                    for c in range(k):
                        b[r, c] = base_b[j, r, c]
            if has_x:
                for r in range(nx):
                    for c in range(nx):
                        x[r, c] = base_x[j, r, c]
            for t in range(n_add + n_sub):
                s = add[row, t] if t < n_add else sub[row, t - n_add]
                if s < 0:
                    continue
                sign = 1.0 if t < n_add else -1.0
                for r in range(4):
                    for c in range(4):
                        a[r, c] += sign * sta_a[s, j, r, c]
                if has_b:
                    for r in range(4):
                        for c in range(k):
                            b[r, c] += sign * sta_b[s, j, r, c]
                if has_x:
                    for r in range(nx):
                        for c in range(nx):
                            x[r, c] += sign * sta_x[s, j, r, c]

            ok = True
            if not gls:
                # (A + P)^-1 (A + P + B B^T + C) (A + P)^-1
                ok = _inv4(a, ainv, work)
                for r in range(4):
                    for c in range(4):
                        v = a[r, c]
                        if has_b:
                            for p in range(k):
                                v += b[r, p] * b[c, p]
                        if has_x:
                            v += x[r, c]
                        work[r, c] = v
                for r in range(4):
                    for c in range(4):
                        v = 0.0
                        for p in range(4):
                            v += work[r, p] * ainv[p, c]
                        tmp[r, c] = v
                for r in range(4):
                    for c in range(4):
                        v = 0.0
                        for p in range(4):
                            v += ainv[r, p] * tmp[p, c]
                        cov[r, c] = v
            else:
                if has_b and has_x:
                    # A + P - B (I + M)^-1 B^T
                    for r in range(nx):
                        x[r, r] += 1.0
                    for r in range(k):
                        for c in range(4):
                            rhs[r, c] = b[c, r]
                    _chol_solve(x, rhs, k)
                    for r in range(4):
                        for c in range(4):
                            v = 0.0
                            for p in range(k):
                                v += b[r, p] * rhs[p, c]
                            a[r, c] -= v
                ok = _inv4(a, cov, work)

            if not ok:
                acc = np.inf
                if criterion == 1:
                    for jj in range(n_j):
                        out_e[row, jj] = np.inf
                break
            e = cov[0, 0] + cov[1, 1] + vertical_weight * cov[2, 2]
            wl = weights[j] * located
            frac += wl
            if criterion == 0:
                acc += wl * e
            elif criterion == 1:
                out_e[row, j] = e
                if has_loc:
                    out_l[row, j] = located
            elif wl > 0:
                det = (
                    cov[0, 0] * (cov[1, 1] * cov[2, 2] - cov[1, 2] * cov[2, 1])
                    - cov[0, 1] * (cov[1, 0] * cov[2, 2] - cov[1, 2] * cov[2, 0])
                    + cov[0, 2] * (cov[1, 0] * cov[2, 1] - cov[1, 1] * cov[2, 0])
                )
                acc += wl * (np.log(det) if det > 0 else np.inf)
        out[row] = acc
        out_frac[row] = frac


def _empty(ndim: int) -> FloatArray:
    return np.zeros((0,) * ndim, dtype=np.float64)


def point_scores(
    base: tuple[FloatArray, FloatArray | None, FloatArray | None],
    stations: tuple[FloatArray, FloatArray | None, FloatArray | None],
    add: IntArray,
    sub: IntArray,
    prior: FloatArray,
    weights: FloatArray,
    vertical_weight: float,
    gls: bool,
    criterion: str,
    locate: tuple[FloatArray, npt.NDArray[np.bool_], FloatArray, IntArray, int, int] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """Per network: the located-weighted sum ``sum_j w_j L_j e_j`` (rms) or ``sum_j w_j L_j
    log det C_xyz,j`` (logdet), and the located fraction ``sum_j w_j L_j``; for percentile the
    per-point error variances ``e`` (N, J) instead of the sum, and ``L`` (N, J).

    ``base``/``stations`` hold the ``A``, ``B`` and ``C`` (achievable) or ``M`` (gls) arrays,
    ``None`` when absent. ``locate`` = (pick probabilities (S, J, P), P-phase flags, pick-count
    distributions of the base-network variants (V, 2, J, K+1), the variant of each row,
    min picks, min P picks); without it every event counts as located (``L = 1``)."""
    (base_a, base_b, base_x), (sta_a, sta_b, sta_x) = base, stations
    n, n_j = add.shape[0], base_a.shape[0]
    code = CRITERIA[criterion]
    out = np.empty(n)
    out_frac = np.empty(n)
    out_e = np.empty((n, n_j)) if code == 1 else _empty(2)
    if locate is None:
        pick_p, is_p = _empty(3), np.zeros(0, dtype=np.bool_)
        variants, row_variant, min_picks, min_p = _empty(4), np.zeros(n, np.intp), 1, 0
    else:
        pick_p, is_p, variants, row_variant, min_picks, min_p = locate
    out_l = np.ones((n, n_j)) if code == 1 else _empty(2)
    _kernel(
        np.ascontiguousarray(base_a, dtype=np.float64),
        _empty(3) if base_b is None else np.ascontiguousarray(base_b, dtype=np.float64),
        _empty(3) if base_x is None else np.ascontiguousarray(base_x, dtype=np.float64),
        sta_a,
        _empty(4) if sta_b is None else sta_b,
        _empty(4) if sta_x is None else sta_x,
        np.ascontiguousarray(add, dtype=np.intp),
        np.ascontiguousarray(sub, dtype=np.intp),
        prior,
        weights,
        float(vertical_weight),
        gls,
        code,
        out,
        out_e,
        pick_p,
        is_p,
        np.ascontiguousarray(variants, dtype=np.float64),
        np.ascontiguousarray(row_variant, dtype=np.intp),
        int(min_picks),
        int(min_p),
        out_frac,
        out_l,
    )
    if code == 1:
        return out_e, out_frac, out_l
    return out, out_frac, None
