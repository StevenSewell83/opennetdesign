"""QC checks and figures per pipeline stage, written to ``<output_dir>/qc/``.

Each stage writes ``qc/<stage>.json`` (checks, tables, figure names) and its PNGs;
:mod:`netdesign.qc.report` assembles them into ``qc/report.html``.
"""

from __future__ import annotations

import json
import math
import os
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy
from scipy.optimize import brentq
from scipy.special import gamma, kv

import netdesign
from netdesign.crs import InputCRS
from netdesign.design import DesignResult, DesignSpec, Evaluator, meets_targets
from netdesign.frame import GridGeometry, LocalFrame
from netdesign.information import PickSpec, achievable_stats, station_stats
from netdesign.qc import figures
from netdesign.qc.checks import Check, QCSpec, Status, within, write_checks
from netdesign.sensitivity import Sensitivity
from netdesign.stations import Candidates
from netdesign.target import TargetSpec
from netdesign.velocity import Ensemble
from netdesign.velocity.heterogeneity import (
    HeterogeneitySpec,
    build_model_grid,
    cell_centres,
    perturb_grid,
)
from netdesign.velocity.layered import MIN_VPVS
from netdesign.velocity.model3d import SampledModel
from netdesign.velocity.random_field import CorrelationSpec, correlated_field

__all__ = [
    "LogScan",
    "qc_design",
    "qc_ensemble",
    "qc_inputs",
    "qc_sensitivity",
    "qc_validation",
    "run_record",
    "scan_logs",
]

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------- run record and logs


@dataclass
class LogScan:
    """``ERROR``/``WARNING`` lines found in NonLinLoc logs (accumulated while streaming)."""

    issues: list[str]
    nll_version: str | None = None

    def scan(self, *dirs: Path) -> None:
        """Scan every ``*.log`` under ``dirs`` (before they are deleted)."""
        for d in dirs:
            for log in sorted(d.rglob("*.log")) if d.is_dir() else []:
                for line in log.read_text(errors="replace").splitlines():
                    if self.nll_version is None and "(NonLinLoc v" in line:
                        self.nll_version = line.split("(NonLinLoc ", 1)[1].split(")")[0]
                    if ("ERROR" in line or "WARNING" in line) and len(self.issues) < 200:
                        self.issues.append(f"{log.name}: {line.strip()}")


def scan_logs(*dirs: Path) -> LogScan:
    """Convenience: a :class:`LogScan` of ``dirs``."""
    scan = LogScan([])
    scan.scan(*dirs)
    return scan


def run_record(
    out: Path,
    stage: str,
    config: dict[str, Any],
    timings_s: dict[str, float],
    peak_disk_mb: float | None,
    warnings_seen: Sequence[str],
    logs: LogScan,
) -> None:
    """``qc/run_<stage>.json``: config, versions, seeds, timings, disk, warnings, log issues."""
    seeds = {
        "perturb": config["velocity"]["perturb"]["seed"],
        "heterogeneity": config["velocity"]["heterogeneity"]["seed"],
        "target": (config.get("target") or {}).get("seed"),
        "validation": config["validation"]["seed"],
    }
    checks = [
        Check(
            stage,
            "warnings",
            "pass" if not warnings_seen else "warn",
            len(warnings_seen),
            "0",
            "; ".join(warnings_seen)[:500],
        ),
        Check(
            stage,
            "NonLinLoc log errors/warnings",
            "pass" if not logs.issues else "warn",
            len(logs.issues),
            "0",
            "; ".join(logs.issues[:5]),
        ),
    ]
    write_checks(
        out / "qc" / f"run_{stage}.json",
        checks,
        {
            "stage": stage,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "versions": {
                "netdesign": netdesign.__version__,
                "nonlinloc": logs.nll_version,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "seeds": seeds,
            "timings_s": timings_s,
            "peak_disk_mb": peak_disk_mb,
            "warnings": list(warnings_seen),
            "log_issues": logs.issues,
            "config": config,
        },
    )


def dir_size_mb(*dirs: Path) -> float:
    """Total size of the files under ``dirs`` (MB)."""
    total = 0
    for d in dirs:
        if d.is_dir():
            for root, _, files in os.walk(d):
                total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 1e6


# ---------------------------------------------------------------- helpers


def _goal_text(d: DesignSpec) -> str:
    goals = [
        f"{name} {t:g} m"
        for name, t in (
            ("target", d.target_m),
            ("horizontal", d.target_horizontal_m),
            ("depth", d.target_depth_m),
        )
        if t is not None
    ]
    return "goals: " + ", ".join(goals) if goals else f"knee: gain < {d.min_gain_pct}%"


def _target_polygon(
    target: TargetSpec | None, frame: LocalFrame, crs: InputCRS
) -> list[FloatArray]:
    return [] if target is None else target.polygons(frame, crs)


def _grid_rect(geom: GridGeometry, xy: FloatArray, padding: float) -> FloatArray:
    if geom.mode == "3d":
        x1, y1 = geom.x0 + (geom.nx - 1) * geom.dx, geom.y0 + (geom.ny - 1) * geom.dx
        x0, y0 = geom.x0, geom.y0
    else:  # 2D grids are distance-only; draw the padded region they were sized for
        x0, y0 = xy.min(axis=0) - padding
        x1, y1 = xy.max(axis=0) + padding
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _station_xy(candidates: Candidates, frame: LocalFrame) -> FloatArray:
    x, y = frame.to_local(candidates.long, candidates.lat)
    return np.stack([x, y], axis=1)


# ---------------------------------------------------------------- inputs


def qc_inputs(
    candidates: Candidates,
    frame: LocalFrame,
    geom: GridGeometry,
    padding_km: float,
    target: TargetSpec | None,
    base3d: SampledModel | None,
    spec: QCSpec,
    out: Path,
) -> list[Check]:
    """Candidate count and spacing, containment in the grid, 3D-model coverage; input map."""
    xy = _station_xy(candidates, frame)
    checks = [Check("inputs", "candidate stations", "info", len(candidates), "-")]
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    close = [
        f"{candidates.names[i]}-{candidates.names[j]} {1000 * d[i, j]:.0f} m"
        for i in range(len(xy))
        for j in range(i + 1, len(xy))
        if d[i, j] < spec.station_min_spacing_km
    ]
    checks.append(
        Check(
            "inputs",
            "closely spaced candidates",
            "pass" if not close else "warn",
            len(close),
            f">= {1000 * spec.station_min_spacing_km:.0f} m apart",
            ", ".join(close[:10]),
        )
    )
    inside = candidates.depth_km >= geom.z0 - 1e-9
    if geom.mode == "3d":
        x1, y1 = geom.x0 + (geom.nx - 1) * geom.dx, geom.y0 + (geom.ny - 1) * geom.dx
        inside &= (
            (xy[:, 0] >= geom.x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= geom.y0) & (xy[:, 1] <= y1)
        )
    outside = [n for n, ok in zip(candidates.names, inside, strict=True) if not ok]
    checks.append(
        Check(
            "inputs",
            "stations inside grid",
            "pass" if not outside else "fail",
            len(candidates) - len(outside),
            f"{len(candidates)}",
            ", ".join(outside),
        )
    )
    checks.append(
        Check(
            "inputs",
            "target inside travel-time grids",
            "pass",
            True,
            "true",
            "checked when sensitivities were computed",
        )
    )
    if base3d is not None:
        status: Status = "pass" if base3d.coverage >= spec.min_model_coverage else "warn"
        checks.append(
            Check(
                "inputs",
                "3D model coverage of grid",
                status,
                round(base3d.coverage, 3),
                f">= {spec.min_model_coverage}",
                "outside: nearest-edge values",
            )
        )
    fig = figures.inputs_map(
        xy,
        candidates.names,
        candidates.elev_m,
        _target_polygon(target, frame, candidates.crs),
        _grid_rect(geom, xy, padding_km),
        out / "qc" / "inputs_map.png",
        spec.dpi,
    )
    write_checks(out / "qc" / "inputs.json", checks, {"figures": [fig.name]})
    return checks


# ---------------------------------------------------------------- ensemble


def _profiles_1d(
    ensemble: Ensemble, zc: FloatArray, base3d: SampledModel | None
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    if base3d is None:
        vp = np.array([r.model.velocity_at(zc, "P") for r in ensemble.realizations])
        vs = np.array([r.model.velocity_at(zc, "S") for r in ensemble.realizations])
    else:  # horizontal mean of the 3D model, scaled by each realization's global draws
        vp0 = base3d.vp.mean(axis=(0, 1))
        r0 = (base3d.vp / base3d.vs).mean(axis=(0, 1))
        f = np.array([r.params.get("vp_factor", [1.0])[0] for r in ensemble.realizations])
        dv = np.array([r.params.get("vpvs_delta", [0.0])[0] for r in ensemble.realizations])
        vp = f[:, None] * vp0[None, :]
        vs = vp / (r0[None, :] + dv[:, None])
    ens = {"vp": vp, "vs": vs, "vpvs": vp / vs}
    return {k: v[0] for k, v in ens.items()}, ens


def _theoretical_efold(spec: CorrelationSpec, a: float) -> float:
    """Lag at which the model correlation function falls to 1/e."""
    h = spec.exponent()
    if h is None:
        return a  # exp(-r^2 / a^2)

    def acf(r: float) -> float:
        x = r / a
        return float(2 ** (1 - h) / gamma(h) * x**h * kv(h, x)) - math.exp(-1)

    return float(brentq(acf, 1e-6 * a, 20 * a))


def _efold(acf: FloatArray, lag_km: FloatArray) -> float:
    below = np.flatnonzero(acf < math.exp(-1))
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return 0.0
    t = (acf[i - 1] - math.exp(-1)) / (acf[i - 1] - acf[i])
    return float(lag_km[i - 1] + t * (lag_km[i] - lag_km[i - 1]))


def _acf_2d(f: FloatArray) -> FloatArray:
    """Unbiased 2D autocorrelation (zero padding, overlap-normalised), centred."""
    shape = tuple(2 * n for n in f.shape)
    spec = np.fft.rfft2(f, s=shape)
    ones = np.fft.rfft2(np.ones_like(f), s=shape)
    num = np.fft.irfft2(np.abs(spec) ** 2, s=shape)
    den = np.fft.irfft2(np.abs(ones) ** 2, s=shape)
    acf = np.fft.fftshift(num / np.maximum(den, 1.0)) / f.var()
    return acf


def _measured_lengths(spec: CorrelationSpec, geom: GridGeometry, seed: int) -> dict[str, float]:
    """Measured e-folding lengths of generated fields along strike, across, vertical (km)."""
    from scipy.ndimage import map_coordinates

    rng = np.random.default_rng([seed, 99])
    d = geom.dx
    theta = math.radians(spec.strike_deg)
    along = np.array([math.sin(theta), math.cos(theta)])  # (east, north)
    across = np.array([math.cos(theta), -math.sin(theta)])
    out: dict[str, list[float]] = {"along": [], "across": [], "vertical": []}
    for _ in range(3):
        f2 = correlated_field((geom.nx, geom.ny), d, spec, rng)
        acf = _acf_2d(f2)
        cx, cy = np.array(acf.shape) // 2
        max_lag = min(geom.nx, geom.ny) // 2
        lags = np.arange(max_lag, dtype=np.float64)
        for name, u in (("along", along), ("across", across)):
            vals = map_coordinates(acf, [cx + lags * u[0], cy + lags * u[1]], order=1)
            out[name].append(_efold(vals, lags * d))
        f3 = correlated_field((min(geom.nx, 48), min(geom.ny, 48), geom.nz), d, spec, rng)
        fz = np.fft.rfft(f3, n=2 * geom.nz, axis=2)
        num = np.fft.irfft(np.abs(fz) ** 2, axis=2)[..., : geom.nz].mean(axis=(0, 1))
        cnt = geom.nz - np.arange(geom.nz)
        vz = num / cnt / f3.var()
        out["vertical"].append(
            _efold(vz.astype(np.float64), np.arange(geom.nz, dtype=np.float64) * d)
        )
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def qc_ensemble(
    ensemble: Ensemble,
    geom: GridGeometry,
    heterogeneity: HeterogeneitySpec,
    base3d: SampledModel | None,
    stations_xy: FloatArray,
    target_mid_depth_km: float,
    spec: QCSpec,
    out: Path,
) -> list[Check]:
    """1D profiles, clip fractions, achieved heterogeneity sd and correlation lengths."""
    _, _, zc = cell_centres(geom)
    base, ens = _profiles_1d(ensemble, zc, base3d)
    figs = [
        figures.profiles(zc, base, ens, spec.profile_samples, out / "qc" / "profiles.png", spec.dpi)
    ]
    checks = [Check("ensemble", "models", "info", len(ensemble), "-")]
    checks.append(
        Check(
            "ensemble",
            "Vp/Vs range",
            "pass" if ens["vpvs"].min() > MIN_VPVS else "fail",
            [round(float(ens["vpvs"].min()), 3), round(float(ens["vpvs"].max()), 3)],
            f"> {MIN_VPVS:.3f}",
        )
    )

    perturb = ensemble.spec
    perturbed = [r for r in ensemble.realizations if not r.is_base]
    if base3d is None and perturb.enabled and perturb.interface_sd_km > 0 and perturbed:
        thick = np.diff(ensemble.realizations[0].model.tops_km)
        n_l = len(thick) + 1
        limits = np.array(
            [0.0]
            + [
                perturb.interface_clip
                * (thick[k - 1] if k == n_l - 1 else min(thick[k - 1], thick[k]))
                for k in range(1, n_l)
            ]
        )
        shifts = np.abs(np.array([r.params["top_shift_km"] for r in perturbed]))[:, 1:]
        frac = float(np.mean(shifts >= limits[1:] * (1 - 1e-9)))
        checks.append(
            Check(
                "ensemble",
                "interface shifts at clip limit",
                "pass" if frac <= 0.05 else "warn",
                round(frac, 3),
                "<= 0.05",
                "high values: requested sd is not realised",
            )
        )

    tables: dict[str, Any] = {}
    if heterogeneity.enabled:
        models = [m for m in spec.slice_models if 0 < m < len(ensemble)]
        perts: dict[int, FloatArray] = {}
        sd_vp: dict[int, list[float]] = {}
        sd_vpvs: dict[int, list[float]] = {}
        n_clip, n_cells = 0, 0
        n_l = 1 if base3d is not None else ensemble.realizations[0].model.n_layers
        for m in models:
            r = ensemble.realizations[m]
            if base3d is None:
                plain = build_model_grid(r.model, geom, None, r.index)
                het = build_model_grid(r.model, geom, heterogeneity, r.index)
            else:
                fac = r.params.get("vp_factor", [1.0])[0]
                dv = r.params.get("vpvs_delta", [0.0])[0]
                plain = perturb_grid(base3d.vp, base3d.vs, geom, None, r.index, fac, dv)
                het = perturb_grid(base3d.vp, base3d.vs, geom, heterogeneity, r.index, fac, dv)
            # Perturbation relative to the same model with relief but without the fields.
            assert het.vp_background is not None
            rel = (het.vp / het.vp_background - 1.0) * 100.0
            perts[m] = rel
            layer = np.zeros(het.vp.shape, dtype=np.intp) if het.layer is None else het.layer
            # Vp/Vs change, where the layer is the same with and without relief.
            same = (
                np.ones(het.vp.shape, dtype=bool)
                if het.layer is None or plain.layer is None
                else het.layer == plain.layer
            )
            dvpvs = het.vp / het.vs - plain.vp / plain.vs
            for k in range(n_l):
                sel = layer == k
                if sel.sum() > 1000:
                    sd_vp.setdefault(k, []).append(float(np.std(rel[sel])))
                if (sel & same).sum() > 1000:
                    sd_vpvs.setdefault(k, []).append(float(np.std(dvpvs[sel & same])))
            n_clip += int(np.sum(het.vp / het.vs <= MIN_VPVS + 0.01 + 1e-9))
            n_cells += het.vp.size
        if perts:
            figs.append(
                figures.heterogeneity_slices(
                    *cell_centres(geom),
                    perts,
                    target_mid_depth_km,
                    stations_xy,
                    out / "qc" / "heterogeneity_slices.png",
                    spec.dpi,
                )
            )
        want_vp = heterogeneity.per_layer_values("vp_pct", n_l)
        want_vpvs = heterogeneity.per_layer_values("vpvs_sd", n_l)
        rows = []
        for k in sorted(set(sd_vp) | set(sd_vpvs)):
            for name, got_list, want in (
                ("Vp sd (%)", sd_vp.get(k), want_vp[k]),
                ("Vp/Vs sd", sd_vpvs.get(k), want_vpvs[k]),
            ):
                if want <= 0 or not got_list:  # not requested, or too few cells to measure
                    continue
                got = float(np.mean(got_list))
                ok = abs(got / want - 1) <= spec.het_sd_tolerance
                rows.append({"layer": k, "quantity": name, "requested": want, "achieved": got})
                checks.append(
                    Check(
                        "ensemble",
                        f"layer {k} {name}",
                        "pass" if ok else "warn",
                        round(got, 4),
                        f"{want:g} ± {100 * spec.het_sd_tolerance:.0f}%",
                    )
                )
        tables["heterogeneity_sd"] = rows
        if n_cells:
            frac = n_clip / n_cells
            checks.append(
                Check(
                    "ensemble",
                    "Vp/Vs cells at lower clip",
                    "pass" if frac <= 1e-3 else "warn",
                    round(frac, 5),
                    "<= 0.001",
                )
            )
        lengths = heterogeneity.structure.corr_len_km
        measured = _measured_lengths(heterogeneity.structure, geom, heterogeneity.seed)
        clen_rows = []
        for name, a in zip(("along", "across", "vertical"), lengths, strict=True):
            expect = _theoretical_efold(heterogeneity.structure, a)
            got = measured[name]
            ok = bool(np.isfinite(got)) and abs(got / expect - 1) <= spec.het_corr_len_tolerance
            clen_rows.append(
                {
                    "direction": name,
                    "corr_len_km": a,
                    "expected_efold_km": expect,
                    "measured_efold_km": got,
                }
            )
            checks.append(
                Check(
                    "ensemble",
                    f"correlation length {name}",
                    "pass" if ok else "warn",
                    round(got, 3),
                    f"{expect:.3f} km ± {100 * spec.het_corr_len_tolerance:.0f}%",
                    "e-folding length of generated fields on this grid",
                )
            )
        tables["correlation_lengths"] = clen_rows
    write_checks(
        out / "qc" / "ensemble.json", checks, {"figures": [f.name for f in figs], **tables}
    )
    return checks


# ---------------------------------------------------------------- sensitivity


def _depth_bands(z: FloatArray, wanted: list[float] | None) -> list[tuple[float, float]]:
    levels = np.unique(np.round(z, 6))
    if wanted:
        if len(levels) <= 100:
            picks = [float(levels[np.argmin(np.abs(levels - w))]) for w in wanted]
            return [(p - 1e-6, p + 1e-6) for p in picks]
        half = (z.max() - z.min()) / 12
        return [(w - half, w + half) for w in wanted]
    if len(levels) <= 100:
        picks = [levels[0], levels[len(levels) // 2], levels[-1]]
        return [(float(p) - 1e-6, float(p) + 1e-6) for p in picks]
    lo, hi = float(z.min()), float(z.max())
    w = (hi - lo) / 6
    return [(lo, lo + w), ((lo + hi) / 2 - w / 2, (lo + hi) / 2 + w / 2), (hi - w, hi)]


def _sd(cov: FloatArray) -> dict[str, FloatArray]:
    """1-sigma per quantity; negative variances (ill-determined points) become NaN."""
    var = np.diagonal(cov, axis1=-2, axis2=-1).copy()
    var[~(var >= 0)] = np.nan
    return {
        "horizontal 1-sigma (m)": np.sqrt(var[:, 0] + var[:, 1]) * 1000,
        "depth 1-sigma (m)": np.sqrt(var[:, 2]) * 1000,
        "origin-time 1-sigma (ms)": np.sqrt(var[:, 3]) * 1000,
    }


def qc_sensitivity(
    sens: Sensitivity,
    ensemble: Ensemble,
    candidates: Candidates,
    frame: LocalFrame,
    geom: GridGeometry,
    base3d: SampledModel | None,
    picks: PickSpec,
    target: TargetSpec | None,
    spec: QCSpec,
    out: Path,
) -> list[Check]:
    """Eikonal check, predicted uncertainty maps, station contribution, candidate map."""
    qdir = out / "qc"
    xyz = sens.sources.xyz
    h = sens.step_km
    checks: list[Check] = []
    figs: list[Path] = []

    # Eikonal check in the base model: |grad T| * v = 1.
    keep = np.ones(len(xyz), dtype=bool)
    if base3d is None:
        model = ensemble.realizations[0].model
        for top in model.tops_km[1:]:
            keep &= np.abs(xyz[:, 2] - top) > h + geom.dx
    ratios: dict[str, FloatArray] = {}
    for ip, phase in enumerate(sens.phases):
        if base3d is None:
            v = ensemble.realizations[0].model.velocity_at(xyz[:, 2], phase)  # type: ignore[arg-type]
        else:
            xc, yc, zc = cell_centres(geom)
            ix = np.clip(np.searchsorted(xc, xyz[:, 0]) - 1, 0, geom.nx - 1)
            iy = np.clip(np.searchsorted(yc, xyz[:, 1]) - 1, 0, geom.ny - 1)
            iz = np.clip(np.searchsorted(zc, xyz[:, 2]) - 1, 0, geom.nz - 1)
            v = (base3d.vp if phase == "P" else base3d.vs)[ix, iy, iz]
        g = np.linalg.norm(sens.G[0, :, ip].astype(np.float64), axis=-1)  # (S, J)
        ratios[phase] = (g * v[None, :])[:, keep].ravel()
    for phase, r in ratios.items():
        dev = np.abs(r - 1.0)
        med, p99 = float(np.median(dev)), float(np.percentile(dev, 99))
        status: Status = "pass"
        if med > spec.eikonal_median_tol or p99 > spec.eikonal_p99_tol:
            status = "warn"
        if med > 3 * spec.eikonal_median_tol:
            status = "fail"
        checks.append(
            Check(
                "sensitivity",
                f"eikonal |grad T|·v ({phase})",
                status,
                {"median_dev": round(med, 4), "p99_dev": round(p99, 4)},
                f"median <= {spec.eikonal_median_tol}, p99 <= {spec.eikonal_p99_tol}",
                "points within one fd step of an interface excluded",
            )
        )
    figs.append(figures.eikonal_histogram(ratios, qdir / "eikonal.png", spec.dpi))

    # Predicted uncertainty, all candidates, base-model gradients.
    all_s = list(range(len(sens.stations)))
    ach_stats = achievable_stats(sens, picks)
    ach = ach_stats.mse(all_s)
    gls = np.linalg.inv(station_stats(sens, picks).network_information(all_s)[0])
    none = np.linalg.inv(
        station_stats(sens, picks.model_copy(update={"model_error": "none"})).network_information(
            all_s
        )[0]
    )
    sd_ach = _sd(ach)
    determined = float(np.mean(np.isfinite(np.column_stack(list(sd_ach.values()))).all(axis=1)))
    det_status: Status = "pass" if determined == 1.0 else ("warn" if determined >= 0.9 else "fail")
    checks.append(
        Check(
            "sensitivity",
            "points well determined by all candidates",
            det_status,
            round(determined, 4),
            "1.0",
            "ill-determined: too few or poorly placed stations",
        )
    )
    medians = {
        name: {k: float(np.nanmedian(v)) for k, v in _sd(c).items()}
        for name, c in (
            ("achievable (NonLinLoc-like)", ach),
            ("design (GLS)", gls),
            ("pick errors only", none),
        )
    }
    bands = _depth_bands(xyz[:, 2], spec.map_depths_km)
    n_band = max(1, int(np.max([np.sum((xyz[:, 2] >= a) & (xyz[:, 2] <= b)) for a, b in bands])))
    figs.append(
        figures.uncertainty_maps(
            xyz,
            sd_ach,
            bands,
            _station_xy(candidates, frame),
            max(4.0, 9000.0 / n_band),
            qdir / "uncertainty_maps.png",
            spec.dpi,
        )
    )

    # Station contribution: median 1-sigma increase when each station is removed.
    base_h = float(np.nanmedian(sd_ach["horizontal 1-sigma (m)"]))
    base_z = float(np.nanmedian(sd_ach["depth 1-sigma (m)"]))
    inc_h, inc_z = np.zeros(len(all_s)), np.zeros(len(all_s))
    for s in all_s:
        rest = [i for i in all_s if i != s]
        if len(rest) < 2:
            continue
        sd = _sd(ach_stats.mse(rest))
        inc_h[s] = 100 * (float(np.nanmedian(sd["horizontal 1-sigma (m)"])) / base_h - 1)
        inc_z[s] = 100 * (float(np.nanmedian(sd["depth 1-sigma (m)"])) / base_z - 1)
    figs.append(
        figures.station_contribution(
            sens.stations, inc_h, inc_z, qdir / "station_contribution.png", spec.dpi
        )
    )

    # Candidate map (selection is added by the design stage).
    xy = _station_xy(candidates, frame)
    mid = bands[len(bands) // 2]
    sel = (xyz[:, 2] >= mid[0]) & (xyz[:, 2] <= mid[1])
    sel &= np.isfinite(sd_ach["horizontal 1-sigma (m)"])
    figs.append(
        figures.candidate_outcome_map(
            xy,
            candidates.names,
            inc_h,
            xyz[sel, :2],
            sd_ach["horizontal 1-sigma (m)"][sel],
            _target_polygon(target, frame, candidates.crs),
            qdir / "candidates_map.png",
            spec.dpi,
            "Candidates: contribution (contours: horizontal 1-sigma, mid depth)",
        )
    )
    write_candidates_table(candidates, inc_h, inc_z, qdir)
    write_checks(
        qdir / "sensitivity.json",
        checks,
        {
            "figures": [f.name for f in figs],
            "median_uncertainty": medians,
            "station_contribution": [
                {
                    "station": n,
                    "horizontal_pct": round(float(a), 2),
                    "depth_pct": round(float(b), 2),
                }
                for n, a, b in zip(sens.stations, inc_h, inc_z, strict=True)
            ],
        },
    )
    return checks


def write_candidates_table(
    candidates: Candidates,
    inc_h: FloatArray,
    inc_z: FloatArray,
    qdir: Path,
    selection_order: Sequence[int | None] | None = None,
    status: Sequence[str] | None = None,
    greedy_order: Sequence[int | None] | None = None,
    gain_pct: Sequence[float | None] | None = None,
) -> None:
    """``candidates_outcome.csv`` and ``.geojson`` per candidate: status, rank in the chosen
    network (``selection_order``, 1 = most important), position in the nested greedy order,
    improvement if added (unselected) and contribution to the full candidate network."""
    n_c = len(candidates)
    order = list(selection_order) if selection_order is not None else [None] * n_c
    status = (
        list(status)
        if status is not None
        else ["selected" if o is not None else "candidate" for o in order]
    )
    greedy = list(greedy_order) if greedy_order is not None else [None] * n_c
    gains = list(gain_pct) if gain_pct is not None else [None] * n_c
    # Projected input: also give the coordinates as supplied (e.g. NZTM eastings/northings).
    extra = candidates.input_xy is not None and not candidates.crs.geographic
    ax, ay = candidates.crs.axes
    rows = [
        "station,long,lat,"
        + (f"{ax},{ay}," if extra else "")
        + "elev_m,status,selection_order,greedy_order,gain_if_added_pct,"
        "contribution_horizontal_pct,contribution_depth_pct"
    ]
    features = []
    for i, n in enumerate(candidates.names):
        o = "" if order[i] is None else str(order[i])
        g = "" if greedy[i] is None else str(greedy[i])
        gain = gains[i]
        gain_s = "" if gain is None else f"{gain:.2f}"
        xy_in = ""
        if extra:
            assert candidates.input_xy is not None
            xy_in = f"{candidates.input_xy[i, 0]:.2f},{candidates.input_xy[i, 1]:.2f},"
        rows.append(
            f"{n},{candidates.long[i]:.6f},{candidates.lat[i]:.6f},{xy_in}"
            f"{candidates.elev_m[i]:.1f},"
            f"{status[i]},{o},{g},{gain_s},{inc_h[i]:.2f},{inc_z[i]:.2f}"
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(candidates.long[i]), float(candidates.lat[i])],
                },
                "properties": {
                    "station": n,
                    **(
                        {
                            ax: float(candidates.input_xy[i, 0]),
                            ay: float(candidates.input_xy[i, 1]),
                            "input_crs": f"EPSG:{candidates.crs.epsg}",
                        }
                        if extra and candidates.input_xy is not None
                        else {}
                    ),
                    "elev_m": float(candidates.elev_m[i]),
                    "status": status[i],
                    "selection_order": order[i],
                    "greedy_order": greedy[i],
                    "gain_if_added_pct": None if gain is None else round(float(gain), 2),
                    "contribution_horizontal_pct": round(float(inc_h[i]), 2),
                    "contribution_depth_pct": round(float(inc_z[i]), 2),
                },
            }
        )
    (qdir / "candidates_outcome.csv").write_text("\n".join(rows) + "\n")
    (qdir / "candidates_outcome.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=1)
    )


# ---------------------------------------------------------------- validation


def qc_validation(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    stations_xy: FloatArray,
    spec: QCSpec,
    out: Path,
) -> list[Check]:
    """Located fraction, NonLinLoc vs prediction per axis, ellipsoid coverage; figures."""
    qdir = out / "qc"
    n, n_ok = summary["n_events"], summary["n_located"]
    frac = n_ok / n if n else 0.0
    status: Status = "pass" if frac >= spec.validation_min_located else "warn"
    if frac < 0.5:
        status = "fail"
    checks = [
        Check(
            "validation",
            "events located",
            status,
            f"{n_ok}/{n}",
            f">= {100 * spec.validation_min_located:.0f}%",
        )
    ]
    figs: list[Path] = []
    if n_ok:
        for ax in ("x", "y", "z"):
            a = summary[ax]
            predicted = a["rms_predicted_achievable_m"]
            if predicted > 0:
                ratio = a["rms_nll_m"] / predicted
                checks.append(
                    Check(
                        "validation",
                        f"{ax}: NonLinLoc rms / predicted",
                        within(ratio, *spec.validation_rms_ratio),
                        round(ratio, 3),
                        f"{spec.validation_rms_ratio[0]}-{spec.validation_rms_ratio[1]}",
                    )
                )
            else:  # no noise and no model error: nothing to compare against
                checks.append(
                    Check(
                        "validation",
                        f"{ax}: NonLinLoc rms (m)",
                        "info",
                        round(a["rms_nll_m"], 2),
                        "predicted 0",
                    )
                )
            corr = a["corr_nll_vs_linear"]
            checks.append(
                Check(
                    "validation",
                    f"{ax}: corr(NonLinLoc, linear)",
                    "info"
                    if corr is None
                    else ("pass" if corr >= spec.validation_min_corr else "warn"),
                    None if corr is None else round(corr, 3),
                    f">= {spec.validation_min_corr}",
                )
            )
        for key, rng, label in (
            ("fraction_within_achievable_68pct", spec.validation_68_range, "68% ellipsoid"),
            ("fraction_within_achievable_95pct", spec.validation_95_range, "95% ellipsoid"),
        ):
            if key in summary:
                checks.append(
                    Check(
                        "validation",
                        f"fraction within predicted {label}",
                        within(summary[key], *rng),
                        round(summary[key], 3),
                        f"{rng[0]}-{rng[1]}",
                    )
                )
        ok = [r for r in rows if "dx_nll" in r]
        nll = np.array([[r["dx_nll"], r["dy_nll"], r["dz_nll"]] for r in ok]) * 1000
        lin = np.array([[r["dx_lin"], r["dy_lin"], r["dz_lin"]] for r in ok]) * 1000
        true = np.array([[r["x_true"], r["y_true"], r["z_true"]] for r in ok])
        figs.append(figures.validation_scatter(nll, lin, qdir / "validation_scatter.png", spec.dpi))
        figs.append(
            figures.mislocation_vectors(
                true, true + nll / 1000, stations_xy, qdir / "mislocation_vectors.png", spec.dpi
            )
        )
    write_checks(
        qdir / "validation.json", checks, {"figures": [f.name for f in figs], "summary": summary}
    )
    return checks


# ---------------------------------------------------------------- design


def qc_design(
    result: DesignResult,
    ev: Evaluator,
    sens: Sensitivity,
    candidates: Candidates,
    frame: LocalFrame,
    target: TargetSpec | None,
    spec: QCSpec,
    out: Path,
    detection: FloatArray | None = None,
    margin_rows: list[dict[str, Any]] | None = None,
    design_event: str = "",
) -> list[Check]:
    """Design checks, criterion and robustness curves, final candidate map and table; with
    ``detection`` also detectability checks and figures. ``design_event`` describes the
    detection setting (e.g. ``Mw 0.5``)."""
    qdir = out / "qc"
    d = ev.spec
    label = {
        "rms": "RMS 3D location error (m)",
        "percentile": f"{d.percentile:g}th-percentile 3D error (m)",
        "logdet": "geometric-mean error semi-axis (m)",
    }[d.criterion]
    ns = sorted(result.networks)
    nets = [result.networks[n] for n in ns]
    rec = result.networks[result.recommended_n]
    checks: list[Check] = [
        Check(
            "design",
            "recommended network size",
            "info",
            result.recommended_n,
            _goal_text(d),
        ),
        Check(
            "design",
            f"criterion at n={result.recommended_n} ({d.criterion}, {d.measure})",
            "info",
            round(rec.criterion_m, 1),
            "-",
            f"all candidates: {result.all_candidates_m:.1f} m",
        ),
    ]
    if any(t is not None for t in (d.target_m, d.target_horizontal_m, d.target_depth_m)):
        ok = meets_targets(rec, d)
        checks.append(
            Check(
                "design",
                "target met",
                "pass" if ok else "warn",
                ok,
                "true",
                "" if ok else "even the largest network misses the target",
            )
        )
    beaten = [
        n
        for n, r in zip(ns, nets, strict=True)
        if r.fraction_random_better is not None and r.fraction_random_better > 0
    ]
    checks.append(
        Check(
            "design",
            "optimum beats every random network",
            "pass" if not beaten else "warn",
            beaten if beaten else "all sizes",
            "no random better",
        )
    )
    if d.exchange and d.n_restarts:
        partial = [
            n
            for n, r in zip(ns, nets, strict=True)
            if r.restarts_reaching_best < 1 + d.n_restarts and r.n < len(result.usable)
        ]
        checks.append(
            Check(
                "design",
                "restarts agree on the best network",
                "pass" if not partial else "info",
                partial if partial else "all sizes",
                "-",
                "sizes listed: some restarts found worse local optima",
            )
        )
    improved = [
        n
        for n, r in zip(ns, nets, strict=True)
        if r.criterion_m < r.greedy_criterion_m * (1 - 1e-6)
    ]
    checks.append(
        Check(
            "design",
            "exchange improved on greedy",
            "info",
            improved or "none",
            "-",
            "sizes where swapping beat greedy selection",
        )
    )
    for note in result.notes:
        checks.append(Check("design", "note", "info", note, "-"))

    worst = [max(r.loss_one_station_m.values()) if r.loss_one_station_m else np.nan for r in nets]
    median = [
        float(np.median(list(r.loss_one_station_m.values()))) if r.loss_one_station_m else np.nan
        for r in nets
    ]
    figs = [
        figures.criterion_vs_n(
            ns,
            [r.criterion_m for r in nets],
            [r.greedy_criterion_m for r in nets],
            [r.random_percentiles_m for r in nets],
            result.recommended_n,
            d.target_m,
            label,
            qdir / "criterion_vs_n.png",
            spec.dpi,
        ),
        figures.robustness(
            ns,
            [r.criterion_m for r in nets],
            worst,
            median,
            label,
            qdir / "robustness.png",
            spec.dpi,
        ),
    ]

    # Final map: status, rank within the recommended network, gain of unselected stations.
    names = list(sens.stations)
    chosen = set(rec.stations)
    fixed = set(result.fixed)
    loss = rec.loss_one_station_m
    by_importance = sorted(chosen, key=lambda s: -loss.get(s, 0.0))
    rank = {s: i + 1 for i, s in enumerate(by_importance)}
    base_sums = ev.sums(sorted(chosen))
    status: list[str] = []
    ranks: list[int | None] = []
    gains: list[float | None] = []
    for s in range(len(names)):
        if s in fixed:
            status.append("fixed")
        elif s in chosen:
            status.append("selected")
        elif s in result.usable:
            status.append("candidate")
        else:
            status.append("excluded")
        ranks.append(rank.get(s))
        if status[-1] == "candidate":
            with_s = ev.score_sums(
                tuple(x + y for x, y in zip(base_sums, ev.station_terms(s), strict=True))
            )
            gains.append(100 * (1 - with_s / rec.criterion_m))
        else:
            gains.append(None)
    cov = ev.covariance(base_sums)
    sd = _sd(cov)["horizontal 1-sigma (m)"]
    xyz = sens.sources.xyz
    bands = _depth_bands(xyz[:, 2], spec.map_depths_km)
    mid = bands[len(bands) // 2]
    sel = (xyz[:, 2] >= mid[0]) & (xyz[:, 2] <= mid[1]) & np.isfinite(sd)
    xy = _station_xy(candidates, frame)
    figs.append(
        figures.design_map(
            xy,
            names,
            status,
            ranks,
            gains,
            xyz[sel, :2],
            sd[sel],
            _target_polygon(target, frame, candidates.crs),
            qdir / "design_map.png",
            spec.dpi,
            f"Recommended network: {result.recommended_n} stations "
            f"(contours: horizontal 1-sigma, mid depth)",
        )
    )
    sds = _sd(cov)
    figs.append(
        figures.uncertainty_maps(
            xyz,
            sds,
            bands,
            xy[[s for s in sorted(chosen)]],
            max(4.0, 9000.0 / max(1, sel.sum())),
            qdir / "design_uncertainty_maps.png",
            spec.dpi,
        )
    )

    if detection is not None:
        p_mean = np.asarray(detection, dtype=np.float64).mean(axis=0)  # (S, P, J)
        frac = (p_mean > 0.5).mean(axis=-1)  # (S, P)
        figs.append(
            figures.detection_by_station(
                names, frac, sens.phases, qdir / "detection_by_station.png", spec.dpi
            )
        )
        expected = p_mean[sorted(chosen)].sum(axis=0)  # (P, J)
        vals = {f"expected {ph} picks": expected[k] for k, ph in enumerate(sens.phases)}
        figs.append(
            figures.uncertainty_maps(
                xyz,
                vals,
                bands,
                xy[sorted(chosen)],
                max(4.0, 9000.0 / max(1, sel.sum())),
                qdir / "detection_picks_map.png",
                spec.dpi,
                title=f"Expected picks, recommended network ({result.recommended_n} stations)",
                limits={k: (0.0, float(len(chosen))) for k in vals},  # counts: fixed 0..n scale
            )
        )
        need = {"P": spec.min_expected_p_picks, "S": spec.min_expected_s_picks}
        enough = np.ones(expected.shape[1], dtype=bool)
        for k, ph in enumerate(sens.phases):
            enough &= expected[k] >= need.get(ph, 0.0)
        cover = float(enough.mean())
        checks.append(
            Check(
                "design",
                "target points with enough expected picks",
                "pass" if cover >= spec.min_detection_coverage else "warn",
                round(cover, 3),
                f">= {spec.min_detection_coverage} with >= {need['P']:g} P, >= {need['S']:g} S",
                f"detection at {design_event}",
            )
        )
        if margin_rows:
            ns_m = {row["recommended_n"] for row in margin_rows}
            checks.append(
                Check(
                    "design",
                    "recommended size stable across design event sizes",
                    "pass" if len(ns_m) == 1 else "info",
                    sorted(ns_m),
                    "one size",
                    "see the event-size sensitivity table",
                )
            )

    # Candidates table with the outcome (overwrites the pre-design version).
    contrib = json.loads((qdir / "sensitivity.json").read_text()).get("station_contribution", [])
    inc = {r["station"]: (r["horizontal_pct"], r["depth_pct"]) for r in contrib}
    inc_h = np.array([inc.get(n, (np.nan, np.nan))[0] for n in names])
    inc_z = np.array([inc.get(n, (np.nan, np.nan))[1] for n in names])
    greedy_pos = {s: i + 1 for i, s in enumerate(result.greedy_order)}
    write_candidates_table(
        candidates,
        inc_h,
        inc_z,
        qdir,
        [rank.get(s) for s in range(len(names))],
        status=status,
        greedy_order=[greedy_pos.get(s) for s in range(len(names))],
        gain_pct=gains,
    )
    networks = [
        {
            "n": r.n,
            "criterion_m": round(r.criterion_m, 2),
            "greedy_m": round(r.greedy_criterion_m, 2),
            "random_median_m": None
            if r.random_percentiles_m is None
            else round(r.random_percentiles_m[1], 2),
            "worst_station_lost_m": round(w, 2) if np.isfinite(w) else None,
            "stations": " ".join(names[s] for s in r.stations),
        }
        for r, w in zip(nets, worst, strict=True)
    ]
    write_checks(
        qdir / "design.json",
        checks,
        {
            "figures": [f.name for f in figs],
            "networks": networks,
            "margin_sensitivity": margin_rows or [],
            "recommended": {
                "n": result.recommended_n,
                "stations": [names[s] for s in by_importance],
            },
        },
    )
    return checks
