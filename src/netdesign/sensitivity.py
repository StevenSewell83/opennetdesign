"""Travel times and their spatial derivatives at every source point.

For realization ``r``, station ``s``, phase ``p`` and source ``j`` this stores the travel time
``T[r, s, p, j]`` (s) and its gradient with respect to source position
``G[r, s, p, j, :] = (dT/dx, dT/dy, dT/dz)`` (s/km), by central differences of the interpolated
NonLinLoc grid with step ``h``. The derivative with respect to origin time is 1 and is not
stored. These are the rows of the linearised location problem used by
:mod:`netdesign.information`.

Accuracy (tests/unit/test_sensitivity.py, linear-gradient medium, dx = h = 50 m, sources more
than 0.5 km from the station): gradient vector error relative to the slowness is median
1.4-1.9 %, 95th percentile <= 2.8 %, worst ~3.7 % (3d slightly worse than 2d).

Saved as ``sensitivity.npz`` (arrays) with the station, phase and realization labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from netdesign.errors import ConfigError, DataError
from netdesign.nll import Phase, TimeGridSet, TravelTimeGrid
from netdesign.target import Sources

__all__ = ["Sensitivity", "compute_sensitivity", "realization_sensitivity"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Sensitivity:
    """Travel times ``T`` (R, S, P, J) and gradients ``G`` (R, S, P, J, 3) with labels."""

    realizations: tuple[int, ...]
    stations: tuple[str, ...]
    phases: tuple[str, ...]
    sources: Sources
    T: npt.NDArray[np.float32]
    G: npt.NDArray[np.float32]
    step_km: float

    def save(self, path: str | Path) -> Path:
        """Write a compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            realizations=np.array(self.realizations),
            stations=np.array(self.stations),
            phases=np.array(self.phases),
            xyz=self.sources.xyz,
            long=self.sources.long,
            lat=self.sources.lat,
            weights=self.sources.weights,
            T=self.T,
            G=self.G,
            step_km=np.array(self.step_km),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Sensitivity:
        """Read a file written by :meth:`save`.

        Raises
        ------
        DataError
            If the file is missing or lacks an expected array.
        """
        try:
            with np.load(path) as f:
                return cls(
                    tuple(int(i) for i in f["realizations"]),
                    tuple(str(s) for s in f["stations"]),
                    tuple(str(p) for p in f["phases"]),
                    Sources(f["xyz"], f["long"], f["lat"], f["weights"]),
                    f["T"],
                    f["G"],
                    float(f["step_km"]),
                )
        except (OSError, KeyError) as exc:
            raise DataError(f"{path}: cannot read sensitivity file ({exc})") from exc


def realization_sensitivity(
    roots: dict[Phase, dict[str, Path]], sources: Sources, step_km: float
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Travel times (S, P, J) and gradients (S, P, J, 3) for one realization.

    ``roots`` maps phase -> station -> time-grid root; phase and station order are kept.

    Raises
    ------
    ConfigError
        If any source point (or its finite-difference stencil) lies outside a travel-time grid.
    """
    phases = list(roots)
    stations = list(roots[phases[0]])
    n_j = len(sources)
    T = np.empty((len(stations), len(phases), n_j), dtype=np.float32)
    G = np.empty((len(stations), len(phases), n_j, 3), dtype=np.float32)
    x, y, z = sources.xyz.T
    h = step_km
    for ip, p in enumerate(phases):
        for is_, s in enumerate(stations):
            tt = TravelTimeGrid.read(roots[p][s])
            t0 = tt.at(x, y, z)
            gx = (tt.at(x + h, y, z) - tt.at(x - h, y, z)) / (2 * h)
            gy = (tt.at(x, y + h, z) - tt.at(x, y - h, z)) / (2 * h)
            gz = (tt.at(x, y, z + h) - tt.at(x, y, z - h)) / (2 * h)
            finite = np.isfinite(t0 + gx + gy + gz)
            if not finite.all():
                raise ConfigError(
                    f"{int((~finite).sum())} target point(s) lie outside the travel-time grid of "
                    f"station {s} (within one fd step); increase grid.padding_km or "
                    "grid.depth_max_km, or shrink the target"
                )
            T[is_, ip] = t0
            G[is_, ip] = np.stack([gx, gy, gz], axis=-1)
    if not np.all(T >= 0):
        raise DataError("negative travel times in the grids")
    return T, G


def compute_sensitivity(times: TimeGridSet, sources: Sources, step_km: float) -> Sensitivity:
    """Travel times and gradients for every realization, station and phase in ``times``."""
    realizations = tuple(sorted(times.roots))
    first = times.roots[realizations[0]]
    phases = tuple(first)
    stations = tuple(first[phases[0]])
    parts = [realization_sensitivity(times.roots[r], sources, step_km) for r in realizations]
    T = np.stack([t for t, _ in parts])
    G = np.stack([g for _, g in parts])
    return Sensitivity(realizations, stations, phases, sources, T, G, step_km)
