"""Build a small, cheap variant of examples/config.yaml for tests."""

import copy
from pathlib import Path
from typing import Any

import yaml

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

# Keep example-based tests cheap whatever the example's own settings are.
SMALL: dict[str, Any] = {
    "grid": {"dx_km": 0.2, "padding_km": 1.0},
    "nll": {"jobs": 4},
    "target": {"spacing_km": 0.5, "depth_spacing_km": 0.5},
    "velocity": {"perturb": {"n_realizations": 2}},
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def write_example(tmp_path: Path, nll_bin: Path | None = None, **over: Any) -> Path:
    """Write the example config with absolute paths, ``SMALL`` and ``over`` applied."""
    cfg = yaml.safe_load((EXAMPLES / "config.yaml").read_text())
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["stations"]["candidates"] = str(EXAMPLES / cfg["stations"]["candidates"])
    cfg["velocity"]["base"] = str(EXAMPLES / cfg["velocity"]["base"])
    if nll_bin is not None:
        cfg["nll"]["bin_dir"] = str(nll_bin)
    cfg = _merge(_merge(cfg, SMALL), over)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path
