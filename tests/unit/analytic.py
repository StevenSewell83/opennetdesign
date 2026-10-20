"""Linear-gradient test medium with an exact travel-time solution, and a config for it."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

V0, GRAD, TOP, VPVS = 3.0, 0.6, -0.6, 1.8
"""Vp at the model top (km/s), dVp/dz (1/s), model top (km), Vp/Vs (constant)."""

PHASE_PARAMS = {"P": (V0, GRAD), "S": (V0 / VPVS, GRAD / VPVS)}
"""phase -> (velocity at the model top, gradient)."""


def gradient_time(
    v_top: float, grad: float, xs: np.ndarray, zs: np.ndarray, x: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Exact time in v(z) = v_top + grad * (z - TOP) between (xs, zs) and (x, z).

    ``xs`` and ``x`` hold horizontal coordinates in their last axis.
    """
    v1 = v_top + grad * (zs - TOP)
    v2 = v_top + grad * (z - TOP)
    r2 = np.sum((x - xs) ** 2, axis=-1) + (z - zs) ** 2
    return np.arccosh(1 + grad**2 * r2 / (2 * v1 * v2)) / grad


def write_case(tmp_path: Path, nll_bin: Path, mode: str, **sections: Any) -> Path:
    """Write model, two candidates and a config; ``sections`` override/add config sections."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "m.layer").write_text(f"LAYER {TOP} {V0} {GRAD} {V0 / VPVS} {GRAD / VPVS} 2.5 0\n")
    (tmp_path / "c.csv").write_text(
        "station,long,lat,z\nA,176.10,-38.60,300\nB,176.13,-38.62,-200\n"
    )
    cfg: dict[str, Any] = {
        "output_dir": "out",
        "stations": {"candidates": "c.csv"},
        "grid": {"mode": mode, "dx_km": 0.05, "depth_max_km": 3.0, "padding_km": 1.0},
        "nll": {"bin_dir": str(nll_bin), "jobs": 2},
        "velocity": {"base": "m.layer"},
    }
    cfg.update(sections)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path
