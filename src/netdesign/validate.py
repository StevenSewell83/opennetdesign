"""Validate the linearised location-error predictions with NonLinLoc.

For ``n_events`` target points, one perturbed realization is drawn as the "true" earth. The
synthetic arrival times are that realization's travel times (already in ``sensitivity.npz``)
plus Gaussian pick noise (``picks.sigma_s``) and an origin time. NonLinLoc then locates every
event in the *base* (starting) model, as it would real data, and the mislocation (location
minus truth) is compared with:

- ``linear``: the linearised mislocation of that event from the same data,
  ``dm = A^-1 G^T W d`` with ``d = T_true - T_base + noise``, ``A = G^T W G``, ``G`` the base
  model's rows ``(dT/dx, dT/dy, dT/dz, 1)`` and ``W`` the inverse of NonLinLoc's own data
  covariance (pick variances plus ``LOCGAU``, correlated between same-phase picks by station
  distance, exactly as NLLocLib.c builds it);
- ``achievable``: the mean-square-error matrix of that estimator over the ensemble,
  ``M = A^-1 G^T W (S2 + D) W G A^-1`` with ``S2`` the second moment of ``T_r - T_base`` over
  the truth realizations (so it includes the base model's bias) and ``D`` the pick noise;
- ``design``: the GLS covariance with correlated model error used for design
  (:meth:`netdesign.information.StationStats.network_information` with the base gradients).
  It is the best any estimator that knew the ensemble covariance could do, so NonLinLoc
  (which does not) should scatter more.

``M`` is what NonLinLoc should match if linearisation holds; the ratio of ``design`` to
``achievable`` shows how optimistic the design measure is for NonLinLoc locations.
"""

from __future__ import annotations

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import chi2

from netdesign.errors import ConfigError, DataError
from netdesign.frame import GridGeometry, LocalFrame
from netdesign.information import PickSpec, station_stats
from netdesign.nll.vel2grid import NLLSpec, find_program
from netdesign.sensitivity import Sensitivity

__all__ = ["ValidateSpec", "ValidationResult", "run_validation"]

FloatArray = npt.NDArray[np.float64]

REFERENCE_TIME = datetime(2026, 1, 1)
EVENT_SPACING_S = 60.0
"""Synthetic events are one minute apart, so a location's origin time identifies its event."""


class ValidateSpec(BaseModel):
    """``validation`` config section."""

    model_config = ConfigDict(extra="forbid")

    n_events: int = Field(200, ge=1)
    seed: int = 0
    stations: list[str] | None = Field(None, description="network to test; default all")
    pick_noise: bool = True
    method: Literal["GAU_ANALYTIC", "EDT_OT_WT"] = "GAU_ANALYTIC"
    locgau_sigma_s: float = Field(0.0, ge=0, description="NonLinLoc LOCGAU model error, s")
    locgau_corr_len_km: float = Field(0.0, ge=0, description="NonLinLoc LOCGAU CorrLen, km")
    locgrid_dx_km: float = Field(0.1, gt=0, description="initial search grid spacing")
    locgrid_margin_km: float = Field(1.0, ge=0, description="search grid beyond the target")
    oct_init_cells: tuple[int, int, int] = (10, 10, 5)
    oct_min_node_km: float = Field(0.002, gt=0)
    oct_max_nodes: int = Field(20000, ge=100)


@dataclass(frozen=True)
class ValidationResult:
    """Per-event table rows and the summary written to ``validation/``."""

    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def nll_data_covariance(
    sigma: FloatArray, phases: list[str], xyz: FloatArray, spec: ValidateSpec
) -> FloatArray:
    """NonLinLoc's L2 data covariance for picks with errors ``sigma`` at stations ``xyz``."""
    cov = np.diag(sigma**2)
    if spec.locgau_sigma_s > 0:
        s2 = spec.locgau_sigma_s**2
        same = np.array(phases)[:, None] == np.array(phases)[None, :]
        if spec.locgau_corr_len_km > 0:
            d2 = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=-1)
            corr = np.exp(-0.5 * d2 / spec.locgau_corr_len_km**2)
        else:
            corr = np.eye(len(sigma))
        cov = cov + s2 * np.where(same, corr, 0.0)
    return cov


def _obs_line(station: str, phase: str, t: datetime, err: float) -> str:
    sec = t.second + t.microsecond / 1e6
    return (
        f"{station:<8s} ?    ?    ? {phase:<6s} ? {t:%Y%m%d} {t:%H%M} {sec:9.5f} GAU "
        f"{err:9.2e} -1.00e+00 -1.00e+00 -1.00e+00\n"
    )


def _control(
    spec: ValidateSpec,
    frame: LocalFrame,
    obs: Path,
    ttroot: Path,
    outroot: Path,
    locgrid: tuple[int, int, int, float, float, float, float],
    n_grids: int,
) -> str:
    nx, ny, nz, x0, y0, z0, d = locgrid
    ix, iy, iz = spec.oct_init_cells
    return (
        "CONTROL 1 54321\n"
        f"{frame.nll_trans()}"
        "LOCSIG netdesign validation\n"
        "LOCCOM synthetic events\n"
        f"LOCFILES {obs} NLLOC_OBS {ttroot} {outroot}\n"
        "LOCHYPOUT SAVE_NLLOC_SUM\n"
        f"LOCSEARCH OCT {ix} {iy} {iz} {spec.oct_min_node_km:g} {spec.oct_max_nodes} 0 0 1\n"
        f"LOCGRID {nx} {ny} {nz} {x0:.4f} {y0:.4f} {z0:.4f} {d:g} {d:g} {d:g} "
        # SAVE is required: NLLoc only writes a location if its LOCGRID is saved (for an
        # oct-tree search that is just a small header).
        "PROB_DENSITY SAVE\n"
        f"LOCMETH {spec.method} 9999.0 4 -1 -1 -1 {n_grids} -1.0 0\n"
        f"LOCGAU {spec.locgau_sigma_s:g} {spec.locgau_corr_len_km:g}\n"
        "LOCPHASEID P P\n"
        "LOCPHASEID S S\n"
        "LOCQUAL2ERR 0.01 0.02 0.05 0.1 99999.9\n"
        "LOCANGLES ANGLES_NO 5\n"
    )


def _locgrid(
    sens: Sensitivity, geom: GridGeometry, spec: ValidateSpec
) -> tuple[int, int, int, float, float, float, float]:
    d, m = spec.locgrid_dx_km, spec.locgrid_margin_km
    lo = sens.sources.xyz.min(axis=0) - m
    hi = sens.sources.xyz.max(axis=0) + m
    zmax = geom.z0 + (geom.nz - 1) * geom.dx
    lo[2], hi[2] = max(lo[2], geom.z0 + geom.dx), min(hi[2], zmax - geom.dx)
    if geom.mode == "3d":
        gx1, gy1 = geom.x0 + (geom.nx - 1) * geom.dx, geom.y0 + (geom.ny - 1) * geom.dx
        lo[0], hi[0] = max(lo[0], geom.x0 + geom.dx), min(hi[0], gx1 - geom.dx)
        lo[1], hi[1] = max(lo[1], geom.y0 + geom.dx), min(hi[1], gy1 - geom.dx)
    n = [max(2, math.floor((hi[k] - lo[k]) / d) + 1) for k in range(3)]
    return n[0], n[1], n[2], float(lo[0]), float(lo[1]), float(lo[2]), d


def parse_summary(path: Path) -> dict[int, dict[str, Any]]:
    """Locations from a ``.sum.grid0.loc.hyp`` file, keyed by event index (from origin time)."""
    events: dict[int, dict[str, Any]] = {}
    block: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "NLLOC":
            block = {"status": line.split('"')[3] if line.count('"') >= 4 else "UNKNOWN"}
        elif tok[0] == "HYPOCENTER":
            block["xyz"] = (float(tok[2]), float(tok[4]), float(tok[6]))
        elif tok[0] == "GEOGRAPHIC":
            sec = float(tok[7])
            y, mo, dd, hh, mi = (int(v) for v in tok[2:7])
            ot = datetime(y, mo, dd, hh, mi) + timedelta(seconds=sec)
            block["ot_s"] = (ot - REFERENCE_TIME).total_seconds()
        elif tok[0] == "QUALITY":
            block["rms_s"] = float(tok[tok.index("RMS") + 1])
            block["nphs"] = int(tok[tok.index("Nphs") + 1])
        elif tok[0] == "STATISTICS" and "CovXX" in tok:
            # "... CovXX v XY v XZ v YY v YZ v ZZ v ..." (km^2): only the first has "Cov".
            i = tok.index("CovXX")
            names = ("xx", "xy", "xz", "yy", "yz", "zz")
            block["cov"] = {n: float(tok[i + 1 + 2 * k]) for k, n in enumerate(names)}
        elif tok[0] == "END_NLLOC" and "ot_s" in block:
            events[round(block["ot_s"] / EVENT_SPACING_S)] = block
    return events


def run_validation(
    sens: Sensitivity,
    base_roots: dict[str, dict[str, Path]],
    station_xyz: dict[str, tuple[float, float, float]],
    frame: LocalFrame,
    geom: GridGeometry,
    picks: PickSpec,
    nll: NLLSpec,
    spec: ValidateSpec,
    outdir: Path,
) -> ValidationResult:
    """Run the validation; see the module docstring.

    Parameters
    ----------
    sens : Sensitivity
        Travel times and gradients; realization index 0 must be the unperturbed base model.
    base_roots : dict
        Base-model time-grid roots, phase -> station -> root.
    station_xyz : dict
        Local coordinates of every candidate (for LOCGAU correlation).

    Raises
    ------
    ConfigError
        If ``validate.stations`` names unknown stations or NonLinLoc cannot be found.
    DataError
        If NonLinLoc fails.
    """
    exe = find_program("NLLoc", nll)
    names = list(sens.stations)
    use = spec.stations or names
    unknown = sorted(set(use) - set(names))
    if unknown:
        raise ConfigError(f"validation.stations: unknown station(s) {unknown}")
    s_idx = [names.index(n) for n in use]
    phases = list(sens.phases)
    n_r, _, n_p, n_j = sens.T.shape
    rng = np.random.default_rng(spec.seed)
    events = rng.choice(n_j, size=min(spec.n_events, n_j), replace=False)
    pool = list(range(1, n_r)) or [0]
    truth = rng.choice(pool, size=len(events))
    sig_p = np.array([picks.sigma_s[p] for p in phases])  # type: ignore[index]
    noise = (
        rng.normal(0.0, 1.0, (len(events), len(s_idx), n_p)) * sig_p
        if spec.pick_noise
        else np.zeros((len(events), len(s_idx), n_p))
    )

    outdir.mkdir(parents=True, exist_ok=True)
    tdir = outdir / "ttimes"
    tdir.mkdir(exist_ok=True)
    for p in phases:
        for s in use:
            src = base_roots[p][s]
            for ext in (".hdr", ".buf"):
                link = tdir / f"time.{p}.{s}.time{ext}"
                link.unlink(missing_ok=True)
                link.symlink_to(src.with_name(src.name + ext))

    # Synthetic observations, split into chunks run in parallel.
    n_chunks = max(1, min(nll.jobs, len(events)))
    chunks = np.array_split(np.arange(len(events)), n_chunks)
    locgrid = _locgrid(sens, geom, spec)
    jobs = []
    for c, members in enumerate(chunks):
        if members.size == 0:
            continue
        cdir = outdir / f"run_{c:02d}"
        cdir.mkdir(exist_ok=True)
        lines = []
        for e in members:
            j, r = int(events[e]), int(truth[e])
            t0 = REFERENCE_TIME + timedelta(seconds=EVENT_SPACING_S * e)
            for k, si in enumerate(s_idx):
                for ip, p in enumerate(phases):
                    t = float(sens.T[r, si, ip, j]) + float(noise[e, k, ip])
                    lines.append(_obs_line(names[si], p, t0 + timedelta(seconds=t), sig_p[ip]))
            lines.append("\n")
        obs = cdir / "events.obs"
        obs.write_text("".join(lines))
        ctrl = cdir / "nlloc.in"
        ctrl.write_text(
            _control(spec, frame, obs, tdir / "time", cdir / "loc", locgrid, len(use) * n_p)
        )
        jobs.append(ctrl)

    def run(ctrl: Path) -> None:
        proc = subprocess.run(
            [str(exe), ctrl.name], cwd=ctrl.parent, capture_output=True, text=True, check=False
        )
        (ctrl.parent / "nlloc.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0 or not (ctrl.parent / "loc.sum.grid0.loc.hyp").is_file():
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
            raise DataError(f"NLLoc failed for {ctrl} (exit {proc.returncode}):\n{tail}")

    with ThreadPoolExecutor(max_workers=nll.jobs) as ex:
        for f in [ex.submit(run, c) for c in jobs]:
            f.result()
    located: dict[int, dict[str, Any]] = {}
    for ctrl in jobs:
        located.update(parse_summary(ctrl.parent / "loc.sum.grid0.loc.hyp"))

    # Predictions.
    design_info = station_stats(sens, picks).network_information(s_idx)[0]  # (J, 4, 4)
    sta_xyz = np.array([station_xyz[names[si]] for si in s_idx for _ in phases])
    pick_phase = [p for _ in s_idx for p in phases]
    sig = np.tile(sig_p, len(s_idx))
    w = np.linalg.inv(nll_data_covariance(sig, pick_phase, sta_xyz, spec))
    d_noise = np.diag(sig**2) if spec.pick_noise else np.zeros((len(sig), len(sig)))
    tt = sens.T.astype(np.float64)[:, s_idx]  # (R, S', P, J)
    g = sens.G.astype(np.float64)[0, s_idx]  # (S', P, J, 3)

    rows = []
    for e, j in enumerate(events):
        j, r = int(j), int(truth[e])
        g4 = np.concatenate([g[:, :, j].reshape(-1, 3), np.ones((len(sig), 1))], axis=1)
        a_inv = np.linalg.inv(g4.T @ w @ g4)
        k = a_inv @ g4.T @ w  # (4, n_obs)
        d = (tt[r, :, :, j] - tt[0, :, :, j]).reshape(-1) + noise[e].reshape(-1)
        lin = k @ d
        dt_pool = (tt[pool, :, :, j] - tt[0, :, :, j]).reshape(len(pool), -1)
        s2 = dt_pool.T @ dt_pool / len(pool)
        mse = k @ (s2 + d_noise) @ k.T
        design_cov = np.linalg.inv(design_info[j])
        true_xyz = sens.sources.xyz[j]
        row: dict[str, Any] = {
            "event": e,
            "source": j,
            "truth_model": r,
            "x_true": true_xyz[0],
            "y_true": true_xyz[1],
            "z_true": true_xyz[2],
            "dx_lin": lin[0],
            "dy_lin": lin[1],
            "dz_lin": lin[2],
        }
        for ax, i in (("x", 0), ("y", 1), ("z", 2)):
            row[f"sd_{ax}_achievable"] = math.sqrt(mse[i, i])
            row[f"sd_{ax}_design"] = math.sqrt(design_cov[i, i])
        loc = located.get(e)
        ok = loc is not None and loc.get("status") == "LOCATED" and "xyz" in loc
        row["status"] = loc.get("status", "MISSING") if loc else "MISSING"
        if ok:
            assert loc is not None
            err = np.array(loc["xyz"]) - true_xyz
            row |= {"dx_nll": err[0], "dy_nll": err[1], "dz_nll": err[2]}
            row["rms_s"] = loc.get("rms_s")
            try:
                row["mahal_achievable"] = float(err @ np.linalg.solve(mse[:3, :3], err))
            except np.linalg.LinAlgError:  # no noise and no model error: zero covariance
                row["mahal_achievable"] = None
        rows.append(row)

    summary = _summarise(rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    header = list(rows[0])
    for extra in ("dx_nll", "dy_nll", "dz_nll", "rms_s", "mahal_achievable"):
        if extra not in header:
            header.append(extra)
    lines = [",".join(header)] + [",".join(_fmt(row.get(h)) for h in header) for row in rows]
    (outdir / "events.csv").write_text("\n".join(lines) + "\n")
    return ValidationResult(rows, summary)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "dx_nll" in r]
    out: dict[str, Any] = {"n_events": len(rows), "n_located": len(ok)}
    if not ok:
        return out
    for ax in ("x", "y", "z"):
        nll = np.array([r[f"d{ax}_nll"] for r in ok]) * 1000
        lin = np.array([r[f"d{ax}_lin"] for r in ok]) * 1000
        ach = np.array([r[f"sd_{ax}_achievable"] for r in ok]) * 1000
        des = np.array([r[f"sd_{ax}_design"] for r in ok]) * 1000
        out[ax] = {
            "rms_nll_m": float(np.sqrt(np.mean(nll**2))),
            "rms_linear_m": float(np.sqrt(np.mean(lin**2))),
            "rms_predicted_achievable_m": float(np.sqrt(np.mean(ach**2))),
            "rms_predicted_design_m": float(np.sqrt(np.mean(des**2))),
            "corr_nll_vs_linear": (
                float(np.corrcoef(nll, lin)[0, 1])
                if len(ok) > 2 and nll.std() > 0 and lin.std() > 0
                else None
            ),
            "rms_nll_minus_linear_m": float(np.sqrt(np.mean((nll - lin) ** 2))),
        }
    mahal = np.array([r["mahal_achievable"] for r in ok if r["mahal_achievable"] is not None])
    if mahal.size:
        out["fraction_within_achievable_68pct"] = float(np.mean(mahal <= chi2.ppf(0.683, 3)))
        out["fraction_within_achievable_95pct"] = float(np.mean(mahal <= chi2.ppf(0.95, 3)))
    return out
