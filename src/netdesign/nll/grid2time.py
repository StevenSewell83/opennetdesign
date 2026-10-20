"""Travel-time grids for every candidate, phase and realization with NonLinLoc ``Grid2Time``.

One Grid2Time run per (realization, phase) computes all stations (by reciprocity each station
grid holds the travel time from every source node to that station). Output layout under
``outdir``::

    model_NNN/grid2time.<phase>.in      control file (+ .log)
    model_NNN/time.<phase>.<STA>.time.{hdr,buf}
    model_NNN/time.<phase>.<STA>.angle.{hdr,buf}     if nll.angles

Grid2Time also copies the model grid next to its output; those copies are deleted. For 2D
grids it writes two x-sheets although the header says ``nx = 1``; buffers are truncated to
the header size (NonLinLoc itself reads only that much).

Accuracy (tests/unit/test_grid2time.py, linear-gradient medium, dx = 50 m, plfd_eps = 0.01,
excluding the first 300 m): Grid2Time's Podvin-Lecomte times are always slightly slow.
``2d``: bias about +0.3 %, at most about 1.3 % within 1 km of the station and 1.0 % beyond.
``3d`` is worse at the same spacing: bias about +0.6 %, up to about 2.5 % within 1 km and
1.6 % beyond.

Why 3d differs: both modes run the same solver (``time_3d``); ``2d`` runs it on a two-sheet
grid, so waves only travel in a vertical plane along a grid axis. The scheme's error depends
on direction relative to the grid axes (worst around 20-25 deg from an axis: mean +0.9 % at
50 m, +0.6 % at 25 m, +0.3 % at 12.5 m), and 3d sees all directions. 3d at 12.5 m is about
as accurate as 2d at 50 m. Off-node stations (the normal case in 3d) also need the larger
plfd_eps: with NonLinLoc's customary 1e-3 the 3d near-station error reaches about 8-10 %.

For design the absolute error matters little: it is systematic, so it largely cancels in the
differences between ensemble models (scaling velocity by 1.05 changes the grid times by the
exact amount to within the same relative error, i.e. ~2 % of the change). Prefer ``2d`` for
layered models.
"""

from __future__ import annotations

import math
import os
import subprocess
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from netdesign.errors import DataError
from netdesign.frame import GridGeometry, LocalFrame
from netdesign.nll.vel2grid import NLLSpec, Phase, VelGridSet, find_program
from netdesign.stations import Candidates

__all__ = ["TimeGridSet", "build_time_grids", "grid2time_control", "time_grids"]


@dataclass(frozen=True)
class TimeGridSet:
    """Time-grid roots (without ``.hdr``/``.buf``) by realization, phase and station."""

    roots: dict[int, dict[Phase, dict[str, Path]]]
    geometry: GridGeometry


def grid2time_control(
    model_root: Path,
    out_root: Path,
    phase: Phase,
    frame: LocalFrame,
    candidates: Candidates,
    mode: str,
    nll: NLLSpec,
) -> str:
    """Grid2Time control file text for one realization and phase."""
    gtmode = "GRID2D" if mode == "2d" else "GRID3D"
    angles = "ANGLES_YES" if nll.angles else "ANGLES_NO"
    return (
        "CONTROL 1 54321\n"
        f"{frame.nll_trans()}"
        f"GTFILES {model_root} {out_root} {phase}\n"
        f"GTMODE {gtmode} {angles}\n"
        f"{candidates.to_gtsrce()}"
        f"GT_PLFD {nll.plfd_eps:g} 0\n"
    )


def _run(exe: Path, ctrl: Path, expected: list[Path], model_copy: Path) -> None:
    proc = subprocess.run(
        [str(exe), ctrl.name], cwd=ctrl.parent, capture_output=True, text=True, check=False
    )
    log = proc.stdout + proc.stderr
    ctrl.with_suffix(".log").write_text(log)
    for ext in (".hdr", ".buf"):
        model_copy.with_name(model_copy.name + ext).unlink(missing_ok=True)
    missing = [r.name for r in expected if not r.with_name(r.name + ".buf").is_file()]
    if proc.returncode != 0 or "ERROR" in log or missing:
        tail = "\n".join(log.strip().splitlines()[-5:])
        raise DataError(
            f"Grid2Time failed for {ctrl} (exit {proc.returncode}, missing {missing}):\n{tail}"
        )
    for root in expected:
        _truncate_to_header(root)


def _truncate_to_header(root: Path) -> None:
    """Drop data beyond what the header describes (Grid2Time's extra 2D sheet)."""
    tokens = root.with_name(root.name + ".hdr").read_text().split()
    nbytes = int(tokens[0]) * int(tokens[1]) * int(tokens[2])
    nbytes *= 8 if len(tokens) > 10 and tokens[10] == "DOUBLE" else 4
    buf = root.with_name(root.name + ".buf")
    if buf.stat().st_size > nbytes:
        os.truncate(buf, nbytes)


def time_grids(
    model_roots: dict[Phase, Path],
    candidates: Candidates,
    frame: LocalFrame,
    mode: str,
    rdir: Path,
    nll: NLLSpec,
    pool: Executor,
) -> dict[Phase, dict[str, Path]]:
    """Travel-time grids for one realization, run in parallel on ``pool``.

    Stations are split into chunks (one Grid2Time run each) so that a single realization uses
    up to ``nll.jobs`` processes; each chunk writes to its own sub-directory of ``rdir``.

    Raises
    ------
    ConfigError
        If Grid2Time cannot be found.
    DataError
        If any run fails or leaves a station grid missing.
    """
    exe = find_program("Grid2Time", nll)
    n_s = len(candidates)
    n_chunks = min(n_s, max(1, math.ceil(nll.jobs / len(model_roots))))
    size = math.ceil(n_s / n_chunks)
    chunks = [range(i, min(i + size, n_s)) for i in range(0, n_s, size)]
    roots: dict[Phase, dict[str, Path]] = {}
    futures = []
    for phase, mod_root in model_roots.items():
        # GTFILES takes the model root without ".<phase>.mod".
        model_root = mod_root.with_name(mod_root.name.removesuffix(f".{phase}.mod"))
        roots[phase] = {}
        for c, idx in enumerate(chunks):
            cdir = rdir / f"{phase}_{c:02d}" if len(chunks) > 1 else rdir
            cdir.mkdir(parents=True, exist_ok=True)
            sub = candidates.subset(idx)
            ctrl = cdir / f"grid2time.{phase}.in"
            ctrl.write_text(
                grid2time_control(model_root, cdir / "time", phase, frame, sub, mode, nll)
            )
            sta = {n: cdir / f"time.{phase}.{n}.time" for n in sub.names}
            roots[phase].update(sta)
            futures.append(
                pool.submit(_run, exe, ctrl, list(sta.values()), cdir / f"time.{phase}.mod")
            )
    for f in futures:
        f.result()
    roots = {p: {n: roots[p][n] for n in candidates.names} for p in roots}  # station order
    return roots


def build_time_grids(
    vel: VelGridSet,
    candidates: Candidates,
    frame: LocalFrame,
    outdir: str | Path,
    nll: NLLSpec,
) -> TimeGridSet:
    """Travel-time grids for every realization, all kept under ``outdir``."""
    outdir = Path(outdir).resolve()
    roots: dict[int, dict[Phase, dict[str, Path]]] = {}
    with ThreadPoolExecutor(max_workers=nll.jobs) as pool:
        for index, model_roots in vel.roots.items():
            rdir = outdir / f"model_{index:03d}"
            roots[index] = time_grids(
                model_roots, candidates, frame, vel.geometry.mode, rdir, nll, pool
            )
    return TimeGridSet(roots, vel.geometry)
