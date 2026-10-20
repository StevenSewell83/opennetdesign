"""Shared fixtures."""

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def nll_bin() -> Path:
    """Directory holding the NonLinLoc programs.

    Taken from ``NETDESIGN_NLL_BIN`` or the directory of ``Vel2Grid`` on ``PATH``; tests that
    need NonLinLoc are skipped otherwise.
    """
    env = os.environ.get("NETDESIGN_NLL_BIN")
    candidates = [Path(env)] if env else []
    on_path = shutil.which("Vel2Grid")
    if on_path:
        candidates.append(Path(on_path).parent)
    for d in candidates:
        if all((d / prog).is_file() for prog in ("Vel2Grid", "Grid2Time", "NLLoc")):
            return d
    pytest.skip("NonLinLoc not found: set NETDESIGN_NLL_BIN to its bin directory")
