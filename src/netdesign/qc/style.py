"""Figure style: palette roles and matplotlib defaults.

Colours follow the reference data-viz palette (validated: first three categorical slots pass
all-pairs CVD checks on the light surface; aqua is below 3:1 contrast, so series are always
labelled directly). Sequential = one blue ramp, light to dark; diverging = blue <-> red with a
neutral grey midpoint; status colours are reserved and always paired with an icon and label.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # files only, no display

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "DIVERGING",
    "GRID",
    "SEQUENTIAL",
    "SERIES",
    "STATUS",
    "SURFACE",
    "TEXT",
    "TEXT_MUTED",
    "apply_style",
]

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e3e2de"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua (fixed order, max three)
STATUS = {"pass": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b", "info": "#52514e"}
STATUS_ICON = {"pass": "✓", "warn": "!", "fail": "✗", "info": "i"}

_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("netdesign_seq", _BLUE)
DIVERGING = LinearSegmentedColormap.from_list(
    "netdesign_div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9f2526"]
)


def apply_style() -> None:
    """Recessive axes and grid, surface background, text in ink colours."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "font.size": 9,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
