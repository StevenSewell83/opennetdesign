"""Shared fixtures."""

import os
import shutil
from pathlib import Path

import pytest
import yaml

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config.yaml"
PROGRAMS = ("Vel2Grid", "Grid2Time", "NLLoc")


@pytest.fixture
def nll_bin() -> Path:
    """Directory holding the NonLinLoc programs.

    Looked up in order: ``NETDESIGN_NLL_BIN``, ``nll.bin_dir`` in examples/config.yaml, the
    directory of ``Vel2Grid`` on ``PATH``. Tests that need NonLinLoc are skipped otherwise.
    """
    candidates = []
    if env := os.environ.get("NETDESIGN_NLL_BIN"):
        candidates.append(Path(env))
    bin_dir = (yaml.safe_load(EXAMPLE_CONFIG.read_text()).get("nll") or {}).get("bin_dir")
    if bin_dir:
        path = Path(bin_dir)
        candidates.append(path if path.is_absolute() else EXAMPLE_CONFIG.parent / path)
    if on_path := shutil.which("Vel2Grid"):
        candidates.append(Path(on_path).parent)
    for d in candidates:
        if all((d / prog).is_file() for prog in PROGRAMS):
            return d
    pytest.skip("NonLinLoc not found: set nll.bin_dir in examples/config.yaml")
