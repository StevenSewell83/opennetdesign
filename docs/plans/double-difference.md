# Plan: double-difference (relative location) accuracy in the design

Status: planned, not started (written 2026-10-21).

## Goal

Let the network design take relative (double-difference, DD) location accuracy into account,
alongside the absolute accuracy it optimises today. Absolute accuracy stays the default
(`design.measure: achievable`); DD is opt-in.

- `design.measure: dd` — optimise relative accuracy only (for comparison).
- `design.measure: combined` — `sqrt(abs_ms + dd.weight * dd_ms)`: absolute first, relative
  accuracy as a secondary term.

## Why it changes the answer

Most of the predicted absolute error is velocity-model bias that is shared by nearby events.
DD differencing cancels that shared part, so relative errors are far smaller and are driven by
pick precision, azimuthal / take-off-angle coverage, S picks, and whether *both* events of a
pair are picked at a station. The optimiser will therefore favour different stations
(wider coverage, more stations for small events) than the absolute measure does.

## Method

### Event pairs

- Pair each target point with its neighbours within `dd.pair_separation_km` (default 0.25 km,
  typical inter-event distance in a cluster), using `scipy.spatial.cKDTree.query_pairs`.
  Works for grid and random sampling and across `target.boxes`.
- Cap at `dd.max_pairs_per_point` (default 6: the ±x, ±y, ±z neighbours on a grid) to bound
  cost. Pair weight = mean of the two points' weights (renormalised).

### Linearised relative problem, per pair (j, k)

- Unknowns: relative position `r = x_j - x_k` (3) and origin-time difference (1).
- Datum per station s and phase p: differential time `dT = T_sj - T_sk`, row
  `g = ((G_sj + G_sk) / 2, 1)` (mean gradient; exact as the separation goes to 0).
- Pick variance per datum: `sigma_j^2 + sigma_k^2` (catalogue picks, = `2 sigma^2`), or
  `dd.sigma_cc_s[p]^2` when cross-correlation differential times are modelled.
- Detection: the datum exists only if both picks exist, probability `q = p_sj * p_sk`.
  Weight `w = q / var`. The first-order presence correction is the same form as the absolute
  one with p -> q: `C = sum q (1 - q) / var^2 * s2 * g g^T`.
- Model error: deviations `dev[r, s, p, j] - dev[r, s, p, k]` over the ensemble (base model
  index 0, as in `achievable_stats`), compressed by the existing `_compress` SVD to `U`.
  Shared shifts cancel exactly here; nothing is assumed about how much cancels.
- Error of the pair's relative position: the same "achievable" sandwich as absolute,
  `(A + P)^-1 (A + P + B B^T + C) (A + P)^-1`, so the compiled kernel (`_score.py`) and the
  search are reused unchanged; only the per-station sums differ.

### Code changes

| File | Change |
|---|---|
| `src/netdesign/information.py` | `dd_pairs(sources, separation, max_per_point)`; `dd_stats(sens, picks, pairs, dd_spec, detection) -> AchievableStats` over pairs (J_pairs instead of J) |
| `src/netdesign/design.py` | `DesignSpec.measure` gains `dd`, `combined`; `Evaluator` holds a second stats set + pair weights; `score_batch` calls the kernel once per measure and combines raw sums (`rms` only for `combined` at first; error for `percentile`/`logdet`) |
| `src/netdesign/config.py` | new `dd` section: `pair_separation_km`, `max_pairs_per_point`, `sigma_cc_s` (null or {P, S}), `weight` |
| `src/netdesign/pipeline.py` | build pairs and DD stats in `run_design`; write per-network `dd_criterion_m` (and `abs_criterion_m` for `combined`) to `design.json` / `networks.csv` |
| `src/netdesign/qc/stages.py`, `figures.py` | relative-error curve on `criterion_vs_n.png`; relative-error map at mid depth; report rows |
| `examples/config.yaml`, `templates/config.yaml` | document `dd` and the new measures |

### Tests

- `dd_stats` vs a brute-force full covariance of differential times for small networks
  (as `test_correlated_matches_full_covariance` does for absolute), with and without
  detection and `sigma_cc_s`.
- Identical events (separation -> 0, same gradients): model-error term vanishes.
- Presence correction vs random detection patterns (as
  `test_detection_correction_matches_random_detection_patterns`).
- `combined` with `dd.weight = 0` equals the absolute design; batch scores equal the numpy
  reference (extend `test_score_batch_matches_score`).
- Example config documents every new option (existing test).

### Validation (separate, later step)

The absolute measure is validated against NonLinLoc; DD would only have the brute-force
algebra check. A later step: generate synthetic event pairs/clusters with the ensemble
travel times, relocate with hypoDD (`/home/steve/seismic/HYPODD`), and compare the
relative-location scatter with the prediction (like `netdesign validate`).

## Cost and risks

- Run time: scoring cost scales with points; up to ~3x the absolute measure (plus 2x for
  `combined`, which scores both). The last Mak-Ban run took 51 min; mitigate with fewer
  restarts for the magnitude re-runs (e.g. a `detection.sensitivity_restarts` setting) or a
  smaller `max_pairs_per_point`.
- Memory: B for pairs is (S, J_pairs, 4, K); with S = 625, J_pairs ~ 3 x 2000, K ~ 10 that is
  ~1.2 GB in float64 — check, and use float32 storage or fewer pairs if needed.
- Mean-gradient approximation degrades for widely separated pairs; keep
  `pair_separation_km` small relative to source depth and station distances.
- Pairwise proxy, not a full cluster inversion (hypoDD solves all events jointly); expected
  to rank networks correctly, absolute values to be checked by the hypoDD validation.

## Effort

One focused session for the measure, config, outputs and tests; the hypoDD validation is
a separate piece of similar size.
