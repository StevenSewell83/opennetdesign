"""Command-line interface. All settings come from the YAML config; ``--out`` overrides
``output_dir``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from netdesign.config import Config, load_config
from netdesign.design import DesignResult
from netdesign.errors import ConfigError, NetDesignError
from netdesign.pipeline import (
    check_run,
    run_design,
    run_ensemble,
    run_export,
    run_sensitivity,
    run_traveltimes,
    run_validate,
    run_vel_grids,
)
from netdesign.qc.figures import sites_map
from netdesign.qc.report import build_report
from netdesign.sensitivity import Sensitivity
from netdesign.sites import build_sites
from netdesign.validate import ValidationResult

app = typer.Typer(no_args_is_help=True, help="Microseismic network design.")

ConfigArg = Annotated[
    Path | None,
    typer.Argument(help="YAML config file (or give it with -c before the command)"),
]
OutOpt = Annotated[
    Path | None, typer.Option("--out", "-o", help="override output_dir from the config")
]


def _progress(message: str) -> None:
    typer.echo(f"  {message}", err=True)


def _fail(exc: NetDesignError) -> typer.Exit:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    return typer.Exit(1)


def _inputs(ctx: typer.Context, config: Path | None, out: Path | None) -> tuple[Path, Path | None]:
    """The config and output override: the command's own, else ``-c``/``-o`` before it."""
    top = ctx.obj or {}
    config = config or top.get("config")
    if config is None:
        typer.secho(
            "error: give the config file: netdesign -c config.yaml [COMMAND], or "
            "netdesign COMMAND config.yaml",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    return config, out or top.get("out")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="YAML config file; with no command, run the whole pipeline",
        ),
    ] = None,
    out: OutOpt = None,
) -> None:
    """Microseismic network design.

    `netdesign -c config.yaml` runs the whole pipeline (as `run`);
    `netdesign -c config.yaml COMMAND` runs one stage.
    """
    ctx.obj = {"config": config, "out": out}
    if ctx.invoked_subcommand is None:
        if config is None:
            typer.echo(ctx.get_help())
            raise typer.Exit(0)
        _run_all(config, out)


@app.command()
def ensemble(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Build the velocity/attenuation ensemble; writes <output_dir>/models."""
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
        result = run_ensemble(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    typer.echo(f"wrote {len(result)} model(s) to {(out or cfg.output_dir) / 'models'}")


@app.command()
def grids(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Build the ensemble and NonLinLoc model grids; writes <output_dir>/models and /grids."""
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
        result = run_vel_grids(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    g = result.geometry
    typer.echo(
        f"built {'/'.join(cfg.phases)} grids for {len(result.roots)} model(s) in "
        f"{(out or cfg.output_dir) / 'grids'} ({g.mode}, {g.nx}x{g.ny}x{g.nz} nodes, dx={g.dx} km)"
    )


@app.command()
def traveltimes(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Build model grids and Grid2Time travel-time grids; writes <output_dir>/traveltimes."""
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
        result = run_traveltimes(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    n_models = len(result.roots)
    n_sta = len(next(iter(next(iter(result.roots.values())).values())))
    typer.echo(
        f"built {'/'.join(cfg.phases)} travel-time grids for {n_sta} station(s) x "
        f"{n_models} model(s) in {(out or cfg.output_dir) / 'traveltimes'}"
    )


@app.command()
def report(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Rebuild the QC report (<output_dir>/qc/report.html) from existing QC files."""
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    path = build_report(out or cfg.output_dir)
    typer.echo(f"report: {path}")


@app.command()
def sites(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="overwrite an existing candidates file")
    ] = False,
) -> None:
    """Generate the candidate grid (`sites`) with DEM elevations; writes stations.candidates.

    Map of the sites in <output_dir>/qc/sites_map.png.
    """
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
        if cfg.sites is None or cfg.stations is None:
            raise ConfigError("`netdesign sites` needs the 'sites' and 'stations' sections")
        result = build_sites(cfg.sites, cfg.input_crs, cfg.stations.candidates, force)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    g = result.grid
    xy = np.column_stack([g.x_km, g.y_km])
    pad = cfg.sites.dem.margin_m / 1000.0
    ex = np.linspace(g.x_km.min() - pad, g.x_km.max() + pad, 200)
    ey = np.linspace(g.y_km.min() - pad, g.y_km.max() + pad, 200)
    gx, gy = np.meshgrid(ex, ey)
    dem_z = result.dem.sample(*g.frame.to_geographic(gx, gy))
    fig = sites_map(
        xy,
        result.elev_m,
        result.keep,
        (gx, gy),
        dem_z,
        (out or cfg.output_dir) / "qc" / "sites_map.png",
        cfg.qc.dpi,
    )
    dropped = []
    if result.n_no_data:
        dropped.append(f"{result.n_no_data} without DEM data")
    if result.n_low:
        dropped.append(f"{result.n_low} below min_elevation_m")
    typer.echo(
        f"wrote {result.n_written} of {result.n_grid} grid site(s) to {result.path}"
        + (f" (dropped {', '.join(dropped)})" if dropped else "")
    )
    typer.echo(f"DEM: {result.dem.path}; map: {fig}")


@app.command()
def export(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Shapefiles of the candidates (input and design outcome), target boxes, final network
    and every network size; writes <output_dir>/gis."""
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
        paths = run_export(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    for p in paths:
        typer.echo(f"wrote {p}")


# ---------------------------------------------------------------- stage summaries


def _echo_sensitivity(cfg: Config, out: Path | None, result: Sensitivity) -> None:
    r, s, p, j = result.T.shape
    typer.echo(
        f"sensitivities for {j} source point(s) x {s} station(s) x {p} phase(s) x {r} model(s) "
        f"in {(out or cfg.output_dir) / 'sensitivity'}"
    )


def _echo_validation(cfg: Config, out: Path | None, result: ValidationResult) -> None:
    s = result.summary
    typer.echo(f"located {s['n_located']} of {s['n_events']} synthetic events")
    if s["n_located"]:
        typer.echo("axis  NLLoc rms  linear rms  predicted (achievable)  design   corr(NLL, lin)")
        for ax in ("x", "y", "z"):
            a = s[ax]
            corr = a["corr_nll_vs_linear"]
            typer.echo(
                f"{ax:>4}  {a['rms_nll_m']:7.1f} m  {a['rms_linear_m']:8.1f} m  "
                f"{a['rms_predicted_achievable_m']:14.1f} m  {a['rms_predicted_design_m']:7.1f} m  "
                + (f"{corr:9.2f}" if corr is not None else "      n/a")
            )
    if "fraction_within_achievable_68pct" in s:
        typer.echo(
            f"within predicted 68% / 95% ellipsoids: "
            f"{100 * s['fraction_within_achievable_68pct']:.0f}% / "
            f"{100 * s['fraction_within_achievable_95pct']:.0f}%"
        )
    typer.echo(f"details in {(out or cfg.output_dir) / 'validation'}")


def _echo_design(cfg: Config, out: Path | None, result: DesignResult) -> None:
    data = json.loads(((out or cfg.output_dir) / "design" / "design.json").read_text())
    names = data["recommended_stations"]
    locating = data.get("min_located_fraction") is not None
    head = f"{'n':>3}  {cfg.design.criterion} ({cfg.design.measure}, m)  horiz  depth"
    typer.echo(head + ("  located" if locating else "") + "  stations")
    for net in data["networks"]:
        mark = "*" if net["n"] == result.recommended_n else " "
        loc = f"  {100 * net['located_fraction']:6.0f}%" if locating else ""
        typer.echo(
            f"{net['n']:>3}{mark} {net['criterion_m']:10.1f}  {net['horizontal_m']:5.0f}  "
            f"{net['depth_m']:5.0f}{loc}  {' '.join(net['stations'])}"
        )
    typer.echo(f"recommended: {result.recommended_n} stations: {' '.join(names)}")
    for warning in data.get("warnings", []):
        typer.secho(f"WARNING: {warning}", fg=typer.colors.YELLOW, err=True)
    typer.echo(f"details in {(out or cfg.output_dir) / 'design'}; report in qc/report.html")


# ---------------------------------------------------------------- stage commands


@app.command()
def sensitivity(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Travel times and gradients at the target points; writes <output_dir>/sensitivity."""
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
        result = run_sensitivity(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    _echo_sensitivity(cfg, out, result)


@app.command()
def validate(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Locate synthetic events with NonLinLoc; compare with predictions (<output_dir>/validation).

    Run `sensitivity` first.
    """
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
        result = run_validate(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    _echo_validation(cfg, out, result)


@app.command()
def design(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Choose the network (run `sensitivity` first); writes <output_dir>/design."""
    config, out = _inputs(ctx, config, out)
    try:
        cfg = load_config(config)
        result = run_design(cfg, out, _progress)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    _echo_design(cfg, out, result)


@app.command()
def run(
    ctx: typer.Context,
    config: ConfigArg = None,
    out: OutOpt = None,
) -> None:
    """Run the whole pipeline: sensitivity, validate (if validation.enabled), design.

    Stops at the first stage that fails. The QC report is rebuilt after every stage.
    """
    _run_all(*_inputs(ctx, config, out))


def _run_all(config: Path, out: Path | None) -> None:
    try:
        cfg = load_config(config)
        check_run(cfg)
        start = time.perf_counter()
        typer.secho("== sensitivity", bold=True)
        _echo_sensitivity(cfg, out, run_sensitivity(cfg, out))
        if cfg.validation.enabled:
            typer.secho("== validate", bold=True)
            _echo_validation(cfg, out, run_validate(cfg, out))
        else:
            typer.echo("(validation skipped: validation.enabled is false)")
        typer.secho("== design", bold=True)
        _echo_design(cfg, out, run_design(cfg, out, _progress))
    except NetDesignError as exc:
        raise _fail(exc) from exc
    typer.echo(
        f"finished in {(time.perf_counter() - start) / 60:.1f} min; "
        f"report: {(out or cfg.output_dir) / 'qc' / 'report.html'}"
    )
