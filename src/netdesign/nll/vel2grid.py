"""Build NonLinLoc P and S model grids for every ensemble realization with ``Vel2Grid``.

Output layout under ``outdir``::

    model_NNN/vel2grid.in          control file
    model_NNN/model.P.mod.{hdr,buf}
    model_NNN/model.S.mod.{hdr,buf}
"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from netdesign.errors import ConfigError, DataError
from netdesign.frame import GridGeometry, LocalFrame
from netdesign.nll.modelgrid import write_model_grid
from netdesign.velocity import Ensemble, Realization
from netdesign.velocity.heterogeneity import HeterogeneitySpec, build_model_grid, perturb_grid
from netdesign.velocity.model3d import SampledModel

__all__ = ["NLLSpec", "Phase", "VelGridSet", "build_vel_grids", "find_program", "model_grids"]

Phase = Literal["P", "S"]


class NLLSpec(BaseModel):
    """``nll`` config section."""

    model_config = ConfigDict(extra="forbid")

    bin_dir: Path | None = Field(None, description="directory with NonLinLoc programs")
    jobs: int = Field(default_factory=lambda: os.cpu_count() or 1, ge=1)
    # 1e-2 rather than NonLinLoc's customary 1e-3: in the analytic gradient test it cuts the
    # 3d near-station error from ~10 % to ~2.5 % and is also slightly better in 2d.
    plfd_eps: float = Field(1.0e-2, gt=0, description="Grid2Time GT_PLFD hs_eps_init")
    angles: bool = Field(False, description="Grid2Time takeoff-angle grids (ANGLES_YES)")
    keep_grids: Literal["none", "base", "all"] = Field(
        "base", description="grids kept after the sensitivity step"
    )


@dataclass(frozen=True)
class VelGridSet:
    """Grid file roots (without ``.hdr``/``.buf``) per realization index and phase."""

    geometry: GridGeometry
    roots: dict[int, dict[Phase, Path]]


def find_program(name: str, spec: NLLSpec) -> Path:
    """Locate a NonLinLoc program in ``spec.bin_dir`` or on ``PATH``.

    Raises
    ------
    ConfigError
        If the program is not found or not executable.
    """
    if spec.bin_dir is not None:
        path = spec.bin_dir / name
        if not (path.is_file() and os.access(path, os.X_OK)):
            raise ConfigError(f"nll.bin_dir: {path} is not an executable file")
        return path
    found = shutil.which(name)
    if found is None:
        raise ConfigError(f"{name} not found on PATH; set nll.bin_dir")
    return Path(found)


def control_file(
    r: Realization, frame: LocalFrame, geom: GridGeometry, root: Path, phases: tuple[Phase, ...]
) -> str:
    """Vel2Grid control file text for one realization."""
    return (
        "CONTROL 1 54321\n"
        f"{frame.nll_trans()}"
        f"VGOUT {root}\n"
        + "".join(f"VGTYPE {p}\n" for p in phases)
        + geom.vggrid("SLOW_LEN")
        + r.model.to_nll()
    )


def _run(exe: Path, ctrl: Path, roots: dict[Phase, Path]) -> None:
    proc = subprocess.run(
        [str(exe), ctrl.name], cwd=ctrl.parent, capture_output=True, text=True, check=False
    )
    log = proc.stdout + proc.stderr
    (ctrl.parent / "vel2grid.log").write_text(log)
    missing = [p for p, root in roots.items() if not root.with_name(root.name + ".buf").is_file()]
    if proc.returncode != 0 or "ERROR" in log or missing:
        tail = "\n".join(log.strip().splitlines()[-5:])
        raise DataError(f"Vel2Grid failed for {ctrl} (exit {proc.returncode}):\n{tail}")


def model_grids(
    r: Realization,
    frame: LocalFrame,
    geom: GridGeometry,
    rdir: Path,
    nll: NLLSpec,
    phases: tuple[Phase, ...],
    heterogeneity: HeterogeneitySpec | None = None,
    base3d: SampledModel | None = None,
) -> dict[Phase, Path]:
    """Model grids (``rdir/model.<phase>.mod``) for one realization.

    With a 3D starting model (``base3d``) the grids are its global perturbation
    (``r.params`` ``vp_factor``/``vpvs_delta``) plus heterogeneity. With a layered model and
    no heterogeneity Vel2Grid builds them; with heterogeneity they are written directly.
    The base realization never gets heterogeneity.

    Raises
    ------
    ConfigError
        If Vel2Grid cannot be found.
    DataError
        If Vel2Grid fails; the message includes the end of its log.
    """
    rdir.mkdir(parents=True, exist_ok=True)
    roots: dict[Phase, Path] = {p: rdir / f"model.{p}.mod" for p in phases}
    if base3d is not None:
        spec = None if r.is_base else heterogeneity
        grid = perturb_grid(
            base3d.vp,
            base3d.vs,
            geom,
            spec,
            r.index,
            vp_factor=r.params.get("vp_factor", [1.0])[0],
            vpvs_delta=r.params.get("vpvs_delta", [0.0])[0],
        )
        for p, root in roots.items():
            write_model_grid(root, grid.vp if p == "P" else grid.vs, geom, frame)
        return roots
    if heterogeneity is not None and heterogeneity.enabled:
        spec = None if r.is_base else heterogeneity
        grid = build_model_grid(r.model, geom, spec, r.index)
        for p, root in roots.items():
            write_model_grid(root, grid.vp if p == "P" else grid.vs, geom, frame)
        return roots
    ctrl = rdir / "vel2grid.in"
    ctrl.write_text(control_file(r, frame, geom, rdir / "model", phases))
    _run(find_program("Vel2Grid", nll), ctrl, roots)
    return roots


def build_vel_grids(
    ensemble: Ensemble,
    frame: LocalFrame,
    geom: GridGeometry,
    outdir: str | Path,
    nll: NLLSpec,
    phases: tuple[Phase, ...] = ("P", "S"),
    heterogeneity: HeterogeneitySpec | None = None,
    base3d: SampledModel | None = None,
) -> VelGridSet:
    """Model grids for every realization (in parallel), all kept under ``outdir``."""
    outdir = Path(outdir).resolve()
    with ThreadPoolExecutor(max_workers=nll.jobs) as pool:
        futures = {
            r.index: pool.submit(
                model_grids,
                r,
                frame,
                geom,
                outdir / f"model_{r.index:03d}",
                nll,
                phases,
                heterogeneity,
                base3d,
            )
            for r in ensemble.realizations
        }
        return VelGridSet(geom, {i: f.result() for i, f in futures.items()})
