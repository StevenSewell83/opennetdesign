"""End-to-end steps driven by a :class:`~netdesign.config.Config`.

``run_vel_grids`` and ``run_traveltimes`` keep every grid they build (for inspection).
``run_sensitivity`` streams: for each realization it builds the model and travel-time grids,
extracts travel times and gradients at the target points, then deletes the grids unless
``nll.keep_grids`` says to keep them. The next realization's model grids are built while
Grid2Time runs, so peak disk use is about two realizations' grids.
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from netdesign.config import Config
from netdesign.design import DesignResult, Evaluator, optimise
from netdesign.detection import profiles_from_outputs, relative_detection
from netdesign.errors import ConfigError
from netdesign.frame import GridGeometry, LocalFrame, grid_geometry
from netdesign.gis import export_gis
from netdesign.nll import Phase, TimeGridSet, VelGridSet, build_time_grids, build_vel_grids
from netdesign.nll.grid2time import time_grids
from netdesign.nll.vel2grid import model_grids
from netdesign.qc.report import build_report
from netdesign.qc.stages import (
    LogScan,
    dir_size_mb,
    qc_design,
    qc_ensemble,
    qc_inputs,
    qc_sensitivity,
    qc_validation,
    run_record,
    scan_logs,
)
from netdesign.sensitivity import Sensitivity, realization_sensitivity
from netdesign.stations import Candidates
from netdesign.target import sample_sources
from netdesign.validate import ValidationResult, run_validation
from netdesign.velocity import Ensemble, Layer, LayeredModel, make_ensemble
from netdesign.velocity.model3d import NodeModel, SampledModel, sample_model

__all__ = [
    "check_run",
    "run_design",
    "run_ensemble",
    "run_sensitivity",
    "run_traveltimes",
    "run_validate",
    "run_vel_grids",
]


@dataclass(frozen=True)
class _Context:
    ensemble: Ensemble
    candidates: Candidates
    frame: LocalFrame
    geometry: GridGeometry
    base3d: SampledModel | None


# Carrier for the global (1D-section) draws when the starting model is 3D: a single layer with
# Vp = 1 and a typical Vp/Vs, so the realization's vp_factor and vpvs_delta can be read back.
_REFERENCE_3D = LayeredModel((Layer(0.0, 1.0, 1.0 / 1.75),))


def run_ensemble(cfg: Config, out: Path | None = None) -> Ensemble:
    """Build the velocity/attenuation ensemble and write it to ``out/models``.

    ``out`` defaults to ``cfg.output_dir``.
    """
    out = cfg.output_dir if out is None else out
    vel = cfg.velocity
    layered = vel.base is not None
    ensemble = make_ensemble(
        LayeredModel.read(vel.base) if vel.base is not None else _REFERENCE_3D,
        vel.perturb,
        vel.attenuation,
        copies=vel.heterogeneity.enabled,
    )
    het = vel.heterogeneity.model_dump() if vel.heterogeneity.enabled else None
    ensemble.write(
        out / "models",
        base_path=vel.base if layered else vel.model_3d,
        extra={"heterogeneity": het, "starting_model": "layered" if layered else "3d"},
        write_models=layered,
    )
    return ensemble


def _frame(cfg: Config, candidates: Candidates) -> LocalFrame:
    """Local frame about ``grid.origin`` (in the input crs) or the candidates' centroid."""
    origin = cfg.grid.origin if cfg.grid is not None else None
    if origin is not None:
        return LocalFrame(*origin.to_wgs84(cfg.input_crs))
    return LocalFrame(float(candidates.lat.mean()), float(candidates.long.mean()))


def _context(cfg: Config, out: Path) -> _Context:
    """Ensemble, candidates, frame and grid geometry; writes the station files."""
    if cfg.stations is None or cfg.grid is None:
        raise ConfigError("building grids needs the 'stations' and 'grid' config sections")
    candidates = Candidates.read(cfg.stations.candidates, cfg.input_crs)
    frame = _frame(cfg, candidates)
    out.mkdir(parents=True, exist_ok=True)
    ensemble = run_ensemble(cfg, out)
    base3d = None
    if cfg.velocity.model_3d is None:
        top = float(ensemble.realizations[0].model.tops_km[0])
    else:
        if cfg.grid.mode != "3d":
            raise ConfigError("velocity.model_3d needs grid.mode: 3d")
        # Grid top: the highest station, snapped up to a multiple of dx.
        dx = cfg.grid.dx_km
        top = float(np.floor(candidates.depth_km.min() / dx + 1e-9) * dx)
    geom = grid_geometry(candidates, frame, cfg.grid, top)
    if cfg.velocity.heterogeneity.enabled:
        cfg.velocity.heterogeneity.check_grid(geom)
    if cfg.velocity.model_3d is not None:
        base3d = sample_model(NodeModel.read(cfg.velocity.model_3d, cfg.input_crs), frame, geom)
        (out / "models" / "model3d.json").write_text(
            json.dumps({"model_3d": str(cfg.velocity.model_3d), "coverage": base3d.coverage})
        )
        if base3d.coverage < 1.0:
            warnings.warn(
                f"the 3D model covers {100 * base3d.coverage:.0f}% of the design grid; values "
                "outside it are extended from the nearest edge",
                stacklevel=2,
            )

    (out / "stations.gtsrce").write_text(candidates.to_gtsrce())
    x, y = frame.to_local(candidates.long, candidates.lat)
    rows = ["station,x_km,y_km,z_km"] + [
        f"{n},{xi:.5f},{yi:.5f},{zi:.5f}"
        for n, xi, yi, zi in zip(candidates.names, x, y, candidates.depth_km, strict=True)
    ]
    (out / "stations_local.csv").write_text("\n".join(rows) + "\n")
    return _Context(ensemble, candidates, frame, geom, base3d)


def _write_grids_manifest(ctx: _Context, out: Path, grids: VelGridSet) -> None:
    manifest = {
        "frame": {
            "projection": "TRANS_MERC WGS-84",
            "lat0": ctx.frame.lat0,
            "long0": ctx.frame.long0,
        },
        "geometry": ctx.geometry.to_dict(),
        "realizations": {
            f"{i:03d}": {p: str(root.relative_to(out / "grids")) for p, root in roots.items()}
            for i, roots in grids.roots.items()
        },
    }
    (out / "grids" / "grids.json").write_text(json.dumps(manifest, indent=2))


def _write_times_manifest(cfg: Config, tdir: Path, roots: dict) -> None:
    manifest = {
        "angles": cfg.nll.angles,
        "realizations": {
            f"{i:03d}": {
                phase: {sta: str(root.relative_to(tdir)) for sta, root in st.items()}
                for phase, st in by_phase.items()
            }
            for i, by_phase in roots.items()
        },
    }
    (tdir / "traveltimes.json").write_text(json.dumps(manifest, indent=2))


def run_vel_grids(cfg: Config, out: Path | None = None) -> VelGridSet:
    """Ensemble plus NonLinLoc model grids (``cfg.phases``) for every realization.

    Writes ``out/models/``, ``out/grids/model_NNN/``, ``out/grids/grids.json``,
    ``out/stations.gtsrce`` and ``out/stations_local.csv``.

    Raises
    ------
    ConfigError
        If the ``stations`` or ``grid`` section is missing, or the geometry is invalid.
    """
    out = cfg.output_dir if out is None else out
    return _vel_grids(cfg, out)[1]


def _vel_grids(cfg: Config, out: Path) -> tuple[_Context, VelGridSet]:
    ctx = _context(cfg, out)
    grids = build_vel_grids(
        ctx.ensemble,
        ctx.frame,
        ctx.geometry,
        out / "grids",
        cfg.nll,
        cfg.phases,
        cfg.velocity.heterogeneity,
        ctx.base3d,
    )
    _write_grids_manifest(ctx, out, grids)
    return ctx, grids


def run_traveltimes(cfg: Config, out: Path | None = None) -> TimeGridSet:
    """Model grids plus Grid2Time travel-time grids for every candidate, phase and realization.

    Writes everything :func:`run_vel_grids` does, plus ``out/traveltimes/model_NNN/`` and
    ``out/traveltimes/traveltimes.json`` (realization -> phase -> station -> grid root).
    """
    out = cfg.output_dir if out is None else out
    ctx, vel = _vel_grids(cfg, out)
    tdir = (out / "traveltimes").resolve()
    times = build_time_grids(vel, ctx.candidates, ctx.frame, tdir, cfg.nll)
    _write_times_manifest(cfg, tdir, times.roots)
    return times


def run_sensitivity(cfg: Config, out: Path | None = None) -> Sensitivity:
    """Travel times and gradients at the target points for every realization (streamed).

    Writes ``out/models/``, the station files, ``out/sensitivity/sensitivity.npz`` and
    ``out/sensitivity/sources.csv``; grids are kept according to ``nll.keep_grids``. With
    ``qc.enabled`` it also writes the inputs, ensemble and sensitivity QC (``out/qc/``) and
    rebuilds ``out/qc/report.html``.

    Raises
    ------
    ConfigError
        If the ``target`` section is missing or the target lies outside the grids.
    """
    if cfg.target is None:
        raise ConfigError("sensitivities need the 'target' config section")
    out = cfg.output_dir if out is None else out
    logs, timings = LogScan([]), dict[str, float]()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sens, ctx, peak = _sensitivity(cfg, out, logs, timings)
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if cfg.qc.enabled:
        start = time.perf_counter()
        _qc_sensitivity(cfg, ctx, sens, out)
        timings["qc"] = time.perf_counter() - start
        run_record(
            out,
            "sensitivity",
            cfg.model_dump(mode="json"),
            timings,
            peak,
            [str(w.message) for w in caught],
            logs,
        )
        build_report(out)
    return sens


def _qc_sensitivity(cfg: Config, ctx: _Context, sens: Sensitivity, out: Path) -> None:
    assert cfg.grid is not None
    xy = np.stack(ctx.frame.to_local(ctx.candidates.long, ctx.candidates.lat), axis=1)
    qc_inputs(
        ctx.candidates,
        ctx.frame,
        ctx.geometry,
        cfg.grid.padding_km,
        cfg.target,
        ctx.base3d,
        cfg.qc,
        out,
    )
    qc_ensemble(
        ctx.ensemble,
        ctx.geometry,
        cfg.velocity.heterogeneity,
        ctx.base3d,
        xy,
        float(np.median(sens.sources.xyz[:, 2])),
        cfg.qc,
        out,
    )
    qc_sensitivity(
        sens,
        ctx.ensemble,
        ctx.candidates,
        ctx.frame,
        ctx.geometry,
        ctx.base3d,
        cfg.picks,
        cfg.target,
        cfg.qc,
        out,
    )


def _station_batch(cfg: Config, ctx: _Context, tdir: Path) -> int:
    """Stations per Grid2Time batch so that one batch's travel-time grids fit in
    ``nll.max_disk_gb`` (default half the free disk space).

    Raises
    ------
    ConfigError
        If the grids that ``nll.keep_grids`` keeps would not fit.
    """
    g = ctx.geometry
    grid_bytes = 4 * g.nx * g.ny * g.nz * len(cfg.phases)  # float32 buf, all phases
    tdir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(tdir).free
    budget = cfg.nll.max_disk_gb * 1e9 if cfg.nll.max_disk_gb else 0.5 * free
    n_s = len(ctx.candidates)
    kept = {"none": 0, "base": 1, "all": len(ctx.ensemble.realizations)}[cfg.nll.keep_grids]
    if kept and kept * n_s * grid_bytes > 0.9 * free:
        raise ConfigError(
            f"nll.keep_grids: {cfg.nll.keep_grids} keeps {kept * n_s * grid_bytes / 1e9:.1f} GB "
            f"of travel-time grids but only {free / 1e9:.1f} GB is free; use keep_grids: none "
            "(and validate a smaller network separately), fewer candidates or a coarser grid"
        )
    return max(1, min(n_s, int(budget // grid_bytes)))


def _sensitivity(
    cfg: Config, out: Path, logs: LogScan, timings: dict[str, float]
) -> tuple[Sensitivity, _Context, float]:
    """The streamed computation; returns the sensitivities, context and peak disk (MB)."""
    assert cfg.target is not None
    start = time.perf_counter()
    ctx = _context(cfg, out)
    sources = sample_sources(cfg.target, ctx.frame, cfg.input_crs)
    step = cfg.target.fd_step_km or ctx.geometry.dx
    gdir, tdir = (out / "grids").resolve(), (out / "traveltimes").resolve()
    keep = cfg.nll.keep_grids
    peak = 0.0
    batch = _station_batch(cfg, ctx, tdir)
    timings["setup"] = time.perf_counter() - start
    start = time.perf_counter()

    T, G, kept_models, kept_times = [], [], {}, {}
    realizations = ctx.ensemble.realizations

    def build(i: int) -> dict[Phase, Path]:
        r = realizations[i]
        return model_grids(
            r,
            ctx.frame,
            ctx.geometry,
            gdir / f"model_{r.index:03d}",
            cfg.nll,
            cfg.phases,
            cfg.velocity.heterogeneity,
            ctx.base3d,
        )

    # Build the next realization's model grids while Grid2Time runs on the current one.
    with (
        ThreadPoolExecutor(max_workers=cfg.nll.jobs) as pool,
        ThreadPoolExecutor(max_workers=1) as builder,
    ):
        upcoming = builder.submit(build, 0)
        for i, r in enumerate(realizations):
            name = f"model_{r.index:03d}"
            mroots = upcoming.result()
            if i + 1 < len(realizations):
                upcoming = builder.submit(build, i + 1)
            keeping = keep == "all" or (keep == "base" and r.is_base)
            if keeping:
                troots = time_grids(
                    mroots, ctx.candidates, ctx.frame, ctx.geometry.mode, tdir / name, cfg.nll, pool
                )
                t, g = realization_sensitivity(troots, sources, step)
                peak = max(peak, dir_size_mb(gdir, tdir))
                logs.scan(gdir / name, tdir / name)
                kept_models[r.index], kept_times[r.index] = mroots, troots
            else:
                # Station batches that fit the disk budget, each deleted once read.
                parts_t, parts_g = [], []
                n_s = len(ctx.candidates)
                for k, lo in enumerate(range(0, n_s, batch)):
                    part = tdir / name / f"part{k:03d}"
                    troots = time_grids(
                        mroots,
                        ctx.candidates.subset(range(lo, min(lo + batch, n_s))),
                        ctx.frame,
                        ctx.geometry.mode,
                        part,
                        cfg.nll,
                        pool,
                    )
                    t_k, g_k = realization_sensitivity(troots, sources, step)
                    parts_t.append(t_k)
                    parts_g.append(g_k)
                    peak = max(peak, dir_size_mb(gdir, tdir))
                    logs.scan(gdir / name, part)
                    shutil.rmtree(part)
                t, g = np.concatenate(parts_t), np.concatenate(parts_g)
                shutil.rmtree(gdir / name)
                shutil.rmtree(tdir / name)
            T.append(t)
            G.append(g)
    if kept_models:
        _write_grids_manifest(ctx, out, VelGridSet(ctx.geometry, kept_models))
        _write_times_manifest(cfg, tdir, kept_times)
    timings["grids_traveltimes_sensitivity"] = time.perf_counter() - start

    sens = Sensitivity(
        tuple(r.index for r in ctx.ensemble.realizations),
        ctx.candidates.names,
        tuple(cfg.phases),
        sources,
        np.stack(T),
        np.stack(G),
        step,
    )
    sdir = out / "sensitivity"
    sens.save(sdir / "sensitivity.npz")
    rows = ["long,lat,x_km,y_km,z_km,weight"] + [
        f"{lo:.6f},{la:.6f},{x:.5f},{y:.5f},{z:.5f},{w:.6g}"
        for lo, la, (x, y, z), w in zip(
            sources.long, sources.lat, sources.xyz, sources.weights, strict=True
        )
    ]
    (sdir / "sources.csv").write_text("\n".join(rows) + "\n")
    return sens, ctx, peak


def run_validate(cfg: Config, out: Path | None = None) -> ValidationResult:
    """Locate synthetic events with NonLinLoc and compare with the predictions.

    Uses the outputs of :func:`run_sensitivity` in ``out`` (run that first, with
    ``nll.keep_grids`` ``base`` or ``all``); writes ``out/validation/``.

    Raises
    ------
    ConfigError
        If the sensitivity outputs or the base model's grids are missing.
    """
    out = cfg.output_dir if out is None else out
    sens_path = out / "sensitivity" / "sensitivity.npz"
    times_manifest = out / "traveltimes" / "traveltimes.json"
    grids_manifest = out / "grids" / "grids.json"
    if not sens_path.is_file():
        raise ConfigError(f"{sens_path} not found; run `netdesign sensitivity` first")
    meta = json.loads((out / "models" / "ensemble.json").read_text())
    if not meta["realizations"][0]["is_base"]:
        raise ConfigError("validation needs the unperturbed base model: set perturb.include_base")
    if (
        not times_manifest.is_file()
        or "000" not in json.loads(times_manifest.read_text())["realizations"]
    ):
        raise ConfigError(
            "the base model's grids were not kept; rerun `sensitivity` with nll.keep_grids: base"
        )
    roots = {
        p: {s: (out / "traveltimes" / rel).resolve() for s, rel in st.items()}
        for p, st in json.loads(times_manifest.read_text())["realizations"]["000"].items()
    }
    grids = json.loads(grids_manifest.read_text())
    frame = LocalFrame(grids["frame"]["lat0"], grids["frame"]["long0"])
    geom = GridGeometry(**grids["geometry"])
    rows = (out / "stations_local.csv").read_text().splitlines()[1:]
    station_xyz = {n: (float(x), float(y), float(z)) for n, x, y, z in (r.split(",") for r in rows)}
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_validation(
            Sensitivity.load(sens_path),
            roots,
            station_xyz,
            frame,
            geom,
            cfg.picks,
            cfg.nll,
            cfg.validation,
            out / "validation",
        )
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if cfg.qc.enabled:
        timings = {"validation": time.perf_counter() - start}
        xy = np.array([v[:2] for v in station_xyz.values()])
        qc_validation(result.rows, result.summary, xy, cfg.qc, out)
        run_record(
            out,
            "validate",
            cfg.model_dump(mode="json"),
            timings,
            None,
            [str(w.message) for w in caught],
            scan_logs(out / "validation"),
        )
        build_report(out)
    return result


def _prefixed(progress: Callable[[str], None] | None, prefix: str) -> Callable[[str], None] | None:
    """``progress`` with every message prefixed."""
    if progress is None:
        return None
    report = progress
    return lambda message: report(prefix + message)


def run_design(
    cfg: Config, out: Path | None = None, progress: Callable[[str], None] | None = None
) -> DesignResult:
    """Choose the network: best station subset per size and a recommended size.

    Uses ``out/sensitivity/sensitivity.npz`` (run ``sensitivity`` first); writes
    ``out/design/design.json`` and ``out/design/networks.csv``, plus design QC and the report.

    Raises
    ------
    ConfigError
        If the sensitivity outputs are missing or the design settings are inconsistent.
    """
    out = cfg.output_dir if out is None else out
    sens_path = out / "sensitivity" / "sensitivity.npz"
    if not sens_path.is_file():
        raise ConfigError(f"{sens_path} not found; run `netdesign sensitivity` first")
    # The design QC reads it only after the (possibly hours-long) search, so check now.
    sens_qc = out / "qc" / "sensitivity.json"
    if cfg.qc.enabled and not sens_qc.is_file():
        raise ConfigError(
            f"{sens_qc} not found (the `sensitivity` run's QC did not finish); rerun "
            "`netdesign sensitivity`, or set qc.enabled: false"
        )
    if cfg.stations is None:
        raise ConfigError("design needs the 'stations' config section")
    start = time.perf_counter()
    sens = Sensitivity.load(sens_path)
    det = None
    margin_rows: list[dict[str, Any]] = []
    if cfg.detection.enabled:
        xyz, profiles, eta, fref, usable = _detection_inputs(cfg, out, sens)
        det, _ = relative_detection(sens, xyz, profiles, eta, fref, cfg.detection, usable)
    ev = Evaluator(sens, cfg.picks, cfg.design, det)
    sta_xyz = _station_xyz(out, sens)
    result = optimise(
        ev, sens.stations, cfg.design, np.random.default_rng(cfg.design.seed), sta_xyz, progress
    )
    if cfg.detection.enabled:
        # How much does the "how small an event" knob (margin or magnitude) change the answer?
        quick = cfg.design.model_copy(update={"n_random": 0})
        key, values, design_value = cfg.detection.sensitivity_values
        for value in values:
            if value == design_value:
                res = result
            else:
                det_m, _ = relative_detection(
                    sens, xyz, profiles, eta, fref, cfg.detection, usable, **{key: value}
                )
                res = optimise(
                    Evaluator(sens, cfg.picks, quick, det_m),
                    sens.stations,
                    quick,
                    np.random.default_rng(cfg.design.seed),
                    sta_xyz,
                    _prefixed(progress, f"[{key} {value:g}] "),
                )
            rec_m = res.networks[res.recommended_n]
            margin_rows.append(
                {
                    key: value,
                    "recommended_n": res.recommended_n,
                    "criterion_m": round(rec_m.criterion_m, 1),
                    "located_fraction": round(rec_m.located_fraction, 3),
                    "meets_located_fraction": rec_m.meets_located,
                    "stations": " ".join(sens.stations[i] for i in rec_m.stations),
                }
            )
    elapsed = time.perf_counter() - start

    ddir = out / "design"
    ddir.mkdir(parents=True, exist_ok=True)
    names = list(sens.stations)
    nets: list[dict[str, Any]] = []
    for n in sorted(result.networks):
        r = result.networks[n]
        nets.append(
            {
                "n": n,
                "stations": [names[s] for s in r.stations],
                "criterion_m": r.criterion_m,
                "greedy_criterion_m": r.greedy_criterion_m,
                "random_5_50_95_m": r.random_percentiles_m,
                "fraction_random_better": r.fraction_random_better,
                "restarts_reaching_best": r.restarts_reaching_best,
                "criterion_if_station_lost_m": {
                    names[k]: v for k, v in r.loss_one_station_m.items()
                },
                "located_fraction": r.located_fraction,
                "horizontal_m": r.horizontal_m,
                "depth_m": r.depth_m,
                "meets_located_fraction": r.meets_located,
                "located_fraction_if_station_lost": {
                    names[k]: v for k, v in r.loss_located_fraction.items()
                },
            }
        )
    rec = result.networks[result.recommended_n]
    locating = cfg.detection.enabled and cfg.design.locatability
    warnings = []
    if not rec.meets_located:
        warnings.append(
            f"the recommended network locates only {100 * rec.located_fraction:.0f}% of the "
            f"design events (needs {100 * cfg.design.min_located_fraction:.0f}%): the design "
            "event is too small for these candidates; use a larger detection.magnitude "
            "(or margin), quieter sites, or more candidates near the target"
        )
    (ddir / "design.json").write_text(
        json.dumps(
            {
                "measure": cfg.design.measure,
                "criterion": cfg.design.criterion,
                "recommended_n": result.recommended_n,
                "recommended_stations": [names[s] for s in rec.stations],
                "all_candidates_m": result.all_candidates_m,
                "greedy_order": [names[s] for s in result.greedy_order],
                "networks": nets,
                "notes": result.notes,
                "warnings": warnings,
                "recommendation_reliable": not warnings,
                "min_located_fraction": cfg.design.min_located_fraction if locating else None,
                "detection": None
                if not cfg.detection.enabled
                else {
                    **cfg.detection.model_dump(mode="json"),
                    "margin_sensitivity": margin_rows,
                },
            },
            indent=2,
        )
    )
    lines = ["n,criterion_m,horizontal_m,depth_m,located_fraction,stations"] + [
        f"{x['n']},{x['criterion_m']:.2f},{x['horizontal_m']:.2f},{x['depth_m']:.2f},"
        f"{x['located_fraction']:.3f},{' '.join(x['stations'])}"
        for x in nets
    ]
    (ddir / "networks.csv").write_text("\n".join(lines) + "\n")

    if cfg.qc.enabled:
        candidates = Candidates.read(cfg.stations.candidates, cfg.input_crs)
        frame = _frame(cfg, candidates)
        start = time.perf_counter()
        dspec = cfg.detection
        event = (
            f"Mw {dspec.magnitude:g} (SNR {dspec.snr_threshold_db:g} dB)"
            if dspec.mode == "magnitude"
            else f"relative margin {dspec.margin_db:g} dB"
        )
        qc_design(
            result, ev, sens, candidates, frame, cfg.target, cfg.qc, out, det, margin_rows, event
        )
        run_record(
            out,
            "design",
            cfg.model_dump(mode="json"),
            {"design": elapsed, "qc": time.perf_counter() - start},
            None,
            [],
            LogScan([]),
        )
        build_report(out)
    return result


def _station_xyz(out: Path, sens: Sensitivity) -> np.ndarray:
    """Local station coordinates (S, 3) km, in ``sens.stations`` order."""
    rows = (out / "stations_local.csv").read_text().splitlines()[1:]
    xyz = {n: (float(x), float(y), float(z)) for n, x, y, z in (r.split(",") for r in rows)}
    return np.array([xyz[n] for n in sens.stations])


def _detection_inputs(
    cfg: Config, out: Path, sens: Sensitivity
) -> tuple[np.ndarray, list[Any], list[float], float, list[int]]:
    """Station coordinates, per-realization profiles, eta, fref and usable candidates."""
    station_xyz = _station_xyz(out, sens)
    profiles, eta, fref = profiles_from_outputs(
        out, cfg.velocity.attenuation, cfg.velocity.model_3d, cfg.input_crs
    )
    usable = [i for i, n in enumerate(sens.stations) if n not in cfg.design.exclude]
    unknown = sorted(set(cfg.detection.sites) - set(sens.stations))
    if unknown:
        raise ConfigError(f"detection.sites: unknown station(s) {unknown}")
    return station_xyz, profiles, eta, fref, usable


def run_export(cfg: Config, out: Path | None = None) -> list[Path]:
    """Shapefiles of the candidates, target boxes and design results in ``out/gis`` (see
    :mod:`netdesign.gis`); exports whatever stages have been run."""
    out = cfg.output_dir if out is None else out
    if cfg.stations is None:
        raise ConfigError("export needs the 'stations' config section")
    candidates = Candidates.read(cfg.stations.candidates, cfg.input_crs)
    return export_gis(cfg, _frame(cfg, candidates), out)


def check_run(cfg: Config) -> None:
    """Fail fast, before the long sensitivity stage, on settings the later stages need.

    Raises
    ------
    ConfigError
        If a required section is missing or validation cannot run with these settings.
    """
    missing = [n for n in ("stations", "grid", "target") if getattr(cfg, n) is None]
    if missing:
        raise ConfigError(f"config is missing section(s) {missing}")
    if cfg.validation.enabled:
        if cfg.nll.keep_grids == "none":
            raise ConfigError(
                "validation.enabled needs the starting model's grids: set nll.keep_grids to "
                "base (or all), or set validation.enabled: false"
            )
        if not cfg.velocity.perturb.include_base:
            raise ConfigError(
                "validation.enabled needs the unperturbed base model: set "
                "velocity.perturb.include_base: true, or validation.enabled: false"
            )
