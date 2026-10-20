"""Command-line interface. All settings come from the YAML config; ``--out`` overrides
``output_dir``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from netdesign.config import Config, load_config
from netdesign.design import DesignResult
from netdesign.errors import NetDesignError
from netdesign.pipeline import (
    check_run,
    run_design,
    run_ensemble,
    run_sensitivity,
    run_traveltimes,
    run_validate,
    run_vel_grids,
)
from netdesign.qc.report import build_report
from netdesign.sensitivity import Sensitivity
from netdesign.validate import ValidationResult

app = typer.Typer(no_args_is_help=True, help="Microseismic network design.")

ConfigArg = Annotated[Path, typer.Argument(help="YAML config file")]
OutOpt = Annotated[
    Path | None, typer.Option("--out", "-o", help="override output_dir from the config")
]


def _fail(exc: NetDesignError) -> typer.Exit:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    return typer.Exit(1)


@app.callback()
def main() -> None:
    """Microseismic network design."""


@app.command()
def ensemble(config: ConfigArg, out: OutOpt = None) -> None:
    """Build the velocity/attenuation ensemble; writes <output_dir>/models."""
    try:
        cfg = load_config(config)
        result = run_ensemble(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    typer.echo(f"wrote {len(result)} model(s) to {(out or cfg.output_dir) / 'models'}")


@app.command()
def grids(config: ConfigArg, out: OutOpt = None) -> None:
    """Build the ensemble and NonLinLoc model grids; writes <output_dir>/models and /grids."""
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
def traveltimes(config: ConfigArg, out: OutOpt = None) -> None:
    """Build model grids and Grid2Time travel-time grids; writes <output_dir>/traveltimes."""
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
def report(config: ConfigArg, out: OutOpt = None) -> None:
    """Rebuild the QC report (<output_dir>/qc/report.html) from existing QC files."""
    try:
        cfg = load_config(config)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    path = build_report(out or cfg.output_dir)
    typer.echo(f"report: {path}")


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
    typer.echo(f"{'n':>3}  {cfg.design.criterion} ({cfg.design.measure}, m)  stations")
    for net in data["networks"]:
        mark = "*" if net["n"] == result.recommended_n else " "
        typer.echo(f"{net['n']:>3}{mark} {net['criterion_m']:10.1f}  {' '.join(net['stations'])}")
    typer.echo(f"recommended: {result.recommended_n} stations: {' '.join(names)}")
    typer.echo(f"details in {(out or cfg.output_dir) / 'design'}; report in qc/report.html")


# ---------------------------------------------------------------- stage commands


@app.command()
def sensitivity(config: ConfigArg, out: OutOpt = None) -> None:
    """Travel times and gradients at the target points; writes <output_dir>/sensitivity."""
    try:
        cfg = load_config(config)
        result = run_sensitivity(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    _echo_sensitivity(cfg, out, result)


@app.command()
def validate(config: ConfigArg, out: OutOpt = None) -> None:
    """Locate synthetic events with NonLinLoc; compare with predictions (<output_dir>/validation).

    Run `sensitivity` first.
    """
    try:
        cfg = load_config(config)
        result = run_validate(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    _echo_validation(cfg, out, result)


@app.command()
def design(config: ConfigArg, out: OutOpt = None) -> None:
    """Choose the network (run `sensitivity` first); writes <output_dir>/design."""
    try:
        cfg = load_config(config)
        result = run_design(cfg, out)
    except NetDesignError as exc:
        raise _fail(exc) from exc
    _echo_design(cfg, out, result)


@app.command()
def run(config: ConfigArg, out: OutOpt = None) -> None:
    """Run the whole pipeline: sensitivity, validate (if validation.enabled), design.

    Stops at the first stage that fails. The QC report is rebuilt after every stage.
    """
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
        _echo_design(cfg, out, run_design(cfg, out))
    except NetDesignError as exc:
        raise _fail(exc) from exc
    typer.echo(
        f"finished in {(time.perf_counter() - start) / 60:.1f} min; "
        f"report: {(out or cfg.output_dir) / 'qc' / 'report.html'}"
    )
