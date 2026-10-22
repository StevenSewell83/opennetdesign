"""QC figures (PNG). Coordinates are local km (x east, y north, z depth below sea level)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import Normalize, TwoSlopeNorm

from netdesign.qc.style import DIVERGING, SEQUENTIAL, SERIES, TEXT, TEXT_MUTED, apply_style

__all__ = [
    "candidate_outcome_map",
    "eikonal_histogram",
    "heterogeneity_slices",
    "inputs_map",
    "mislocation_vectors",
    "profiles",
    "sites_map",
    "station_contribution",
    "uncertainty_maps",
    "validation_scatter",
]

FloatArray = npt.NDArray[np.float64]
GRID_LINE = "#a9a8a3"  # contours and grid outline: neutral, recessive
STATION_MARKER: dict[str, Any] = {
    "marker": "^",
    "s": 70,
    "edgecolors": TEXT,
    "linewidths": 0.8,
    "zorder": 5,
}


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _label_stations(ax: plt.Axes, xy: FloatArray, names: Sequence[str]) -> None:
    for (x, y), n in zip(xy, names, strict=True):
        ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=TEXT)


def _outline(ax: plt.Axes, poly: FloatArray, label: str | None = None) -> None:
    ax.plot(*np.vstack([poly, poly[:1]]).T, color=TEXT_MUTED, lw=1.2, ls="--", label=label)


def inputs_map(
    xy: FloatArray,
    names: Sequence[str],
    elev_m: FloatArray,
    target: Sequence[FloatArray],
    grid_rect: FloatArray,
    path: Path,
    dpi: int,
) -> Path:
    """Candidates (coloured by elevation), target box and grid extent."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=elev_m, cmap=SEQUENTIAL, **STATION_MARKER)
    _label_stations(ax, xy, names)
    for i, poly in enumerate(target):
        _outline(ax, poly, "target volume" if i == 0 else None)
    ax.plot(*np.vstack([grid_rect, grid_rect[:1]]).T, color=GRID_LINE, lw=1.0, label="grid extent")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="station elevation (m)")
    ax.set(xlabel="x (km east)", ylabel="y (km north)", title="Candidate stations")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    return _save(fig, path, dpi)


def profiles(
    z_km: FloatArray,
    base: dict[str, FloatArray],
    ensemble: dict[str, FloatArray],
    n_samples: int,
    path: Path,
    dpi: int,
) -> Path:
    """Vp, Vs, Vp/Vs against depth: base model, ensemble 5-95 % envelope and samples."""
    apply_style()
    names = [("vp", "Vp (km/s)"), ("vs", "Vs (km/s)"), ("vpvs", "Vp/Vs")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes, names, strict=True):
        ens = ensemble[key]
        lo, hi = np.percentile(ens, [5, 95], axis=0)
        ax.fill_betweenx(z_km, lo, hi, color=SERIES[0], alpha=0.18, lw=0, label="5-95 %")
        if n_samples:
            pick = rng.choice(len(ens), size=min(n_samples, len(ens)), replace=False)
            for i in pick:
                ax.plot(ens[i], z_km, color=SERIES[0], lw=0.5, alpha=0.35)
        ax.plot(base[key], z_km, color=TEXT, lw=2.0, label="starting model")
        ax.set(xlabel=label)
    axes[0].set(ylabel="depth (km below sea level)")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left")
    fig.suptitle("Velocity ensemble (1D part)", color=TEXT)
    return _save(fig, path, dpi)


def heterogeneity_slices(
    xc: FloatArray,
    yc: FloatArray,
    zc: FloatArray,
    perturbations: dict[int, FloatArray],
    depth_km: float,
    stations_xy: FloatArray,
    path: Path,
    dpi: int,
) -> Path:
    """Vp % perturbation: a depth slice and an E-W section per model, shared diverging scale."""
    apply_style()
    models = list(perturbations)
    k = int(np.argmin(np.abs(zc - depth_km)))
    j = len(yc) // 2
    vmax = max(float(np.nanpercentile(np.abs(p), 99)) for p in perturbations.values()) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 8), squeeze=False)
    for col, m in enumerate(models):
        p = perturbations[m]
        ax = axes[0, col]
        ax.grid(False)
        im = ax.pcolormesh(xc, yc, p[:, :, k].T, cmap=DIVERGING, norm=norm, shading="auto")
        ax.scatter(stations_xy[:, 0], stations_xy[:, 1], c="none", **STATION_MARKER)
        ax.axhline(yc[j], color=TEXT_MUTED, lw=0.8, ls="--")
        ax.set(title=f"model {m:03d}: depth {zc[k]:.2f} km", xlabel="x (km)", ylabel="y (km)")
        ax.set_aspect("equal")
        ax = axes[1, col]
        ax.grid(False)
        ax.pcolormesh(xc, zc, p[:, j, :].T, cmap=DIVERGING, norm=norm, shading="auto")
        ax.set(
            title=f"model {m:03d}: section y = {yc[j]:.2f} km", xlabel="x (km)", ylabel="depth (km)"
        )
        ax.invert_yaxis()
    fig.colorbar(im, ax=axes, shrink=0.6, label="Vp perturbation (%)")
    return _save(fig, path, dpi)


def eikonal_histogram(ratio: dict[str, FloatArray], path: Path, dpi: int) -> Path:
    """|grad T| x v at the target points (1 = exact); outliers piled into the end bins."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    allr = np.concatenate(list(ratio.values()))
    lo, hi = np.percentile(allr, [0.5, 99.5])
    half = max(abs(lo - 1), abs(hi - 1), 0.005)
    bins = np.linspace(1 - half, 1 + half, 61)
    for (phase, r), color in zip(ratio.items(), SERIES, strict=False):
        ax.hist(
            np.clip(r, bins[0], bins[-1]),
            bins=bins.tolist(),
            histtype="step",
            lw=2.0,
            color=color,
            label=phase,
        )
    ax.axvline(1.0, color=TEXT_MUTED, lw=1.0, ls="--")
    ax.set(
        xlabel="|grad T| x velocity (1 = exact; outliers in the end bins)",
        ylabel="count",
        title="Eikonal check",
    )
    ax.legend()
    return _save(fig, path, dpi)


def uncertainty_maps(
    xyz: FloatArray,
    values: dict[str, FloatArray],
    bands: list[tuple[float, float]],
    stations_xy: FloatArray,
    point_size: float,
    path: Path,
    dpi: int,
    title: str = "Predicted location uncertainty, all candidates (achievable)",
    limits: dict[str, tuple[float, float]] | None = None,
) -> Path:
    """Map values (e.g. predicted 1-sigma): quantities in columns (shared scale), depth
    bands in rows."""
    apply_style()
    keys = list(values)
    fig, axes = plt.subplots(
        len(bands),
        len(keys),
        figsize=(4.0 * len(keys), 3.6 * len(bands) + 1.0),
        squeeze=False,
        layout="constrained",
    )
    for c, key in enumerate(keys):
        v = values[key]
        if limits and key in limits:
            norm = Normalize(*limits[key])
        else:
            norm = Normalize(float(np.nanpercentile(v, 1)), float(np.nanpercentile(v, 99)))
        for r, (z0, z1) in enumerate(bands):
            ax = axes[r, c]
            ax.grid(False)
            sel = (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
            sc = ax.scatter(
                xyz[sel, 0],
                xyz[sel, 1],
                c=v[sel],
                cmap=SEQUENTIAL,
                norm=norm,
                s=point_size,
                marker="s",
                linewidths=0,
            )
            ax.scatter(stations_xy[:, 0], stations_xy[:, 1], c="none", **STATION_MARKER)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(key)
            if r == len(bands) - 1:
                ax.set_xlabel("x (km)")
            if c == 0:
                mid = 0.5 * (z0 + z1)
                ax.set_ylabel(f"depth {mid:.2f} km\ny (km)")
        fig.colorbar(
            sc, ax=axes[:, c], orientation="horizontal", location="bottom", shrink=0.85, label=key
        )
    fig.suptitle(title, color=TEXT)
    return _save(fig, path, dpi)


def station_contribution(
    names: Sequence[str], inc_h: FloatArray, inc_z: FloatArray, path: Path, dpi: int
) -> Path:
    """Increase in median 1-sigma when each station is removed (sorted by horizontal)."""
    apply_style()
    order = np.argsort(inc_h)[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.5))
    h = 0.38
    ax.barh(y - h / 2, inc_h[order], height=h, color=SERIES[0], label="horizontal")
    ax.barh(y + h / 2, inc_z[order], height=h, color=SERIES[1], label="depth")
    ax.set_yticks(y, [names[i] for i in order])
    ax.invert_yaxis()
    ax.set(xlabel="increase in median 1-sigma if removed (%)", title="Station contribution")
    ax.legend(loc="lower right")
    return _save(fig, path, dpi)


def candidate_outcome_map(
    xy: FloatArray,
    names: Sequence[str],
    contribution: FloatArray,
    background_xy: FloatArray,
    background_sd: FloatArray,
    target: Sequence[FloatArray],
    path: Path,
    dpi: int,
    title: str,
) -> Path:
    """Every candidate coloured by contribution, over contours of predicted horizontal 1-sigma."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if len(background_sd) >= 3:
        # matplotlib's triangulation emits a harmless cast warning for some point sets.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cs = ax.tricontour(
                background_xy[:, 0],
                background_xy[:, 1],
                background_sd,
                levels=8,
                colors=GRID_LINE,
                linewidths=0.8,
            )
            ax.clabel(cs, fontsize=7, fmt="%.0f m")
    for i, poly in enumerate(target):
        _outline(ax, poly, "target volume" if i == 0 else None)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=contribution, cmap=SEQUENTIAL, **STATION_MARKER)
    _label_stations(ax, xy, names)
    fig.colorbar(
        sc, ax=ax, shrink=0.7, label="contribution: horizontal 1-sigma increase if removed (%)"
    )
    ax.set(xlabel="x (km east)", ylabel="y (km north)", title=title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    return _save(fig, path, dpi)


def validation_scatter(nll: FloatArray, lin: FloatArray, path: Path, dpi: int) -> Path:
    """NonLinLoc mislocation against the linear prediction, per axis (m)."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for i, (ax, axis) in enumerate(zip(axes, "xyz", strict=True)):
        a, b = lin[:, i], nll[:, i]
        lim = float(np.nanmax(np.abs(np.concatenate([a, b])))) * 1.05 or 1.0
        ax.plot([-lim, lim], [-lim, lim], color=TEXT_MUTED, lw=1.0, ls="--")
        ax.scatter(a, b, s=12, color=SERIES[0], alpha=0.7, linewidths=0)
        ax.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            title=f"{axis}",
            xlabel="linear prediction (m)",
            ylabel="NonLinLoc (m)" if i == 0 else None,
        )
        ax.set_aspect("equal")
    fig.suptitle("Mislocation: NonLinLoc vs linear prediction", color=TEXT)
    return _save(fig, path, dpi)


def mislocation_vectors(
    true_xyz: FloatArray, located_xyz: FloatArray, stations_xy: FloatArray, path: Path, dpi: int
) -> Path:
    """Arrows from true to NonLinLoc locations: map and E-W section."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    d = located_xyz - true_xyz
    kw = {"angles": "xy", "scale_units": "xy", "scale": 1, "width": 0.003, "color": SERIES[0]}
    ax1.quiver(true_xyz[:, 0], true_xyz[:, 1], d[:, 0], d[:, 1], **kw)
    ax1.scatter(stations_xy[:, 0], stations_xy[:, 1], c="none", **STATION_MARKER)
    ax1.set(xlabel="x (km)", ylabel="y (km)", title="Map: true → located")
    ax1.set_aspect("equal")
    ax2.quiver(true_xyz[:, 0], true_xyz[:, 2], d[:, 0], d[:, 2], **kw)
    both = np.concatenate([true_xyz, located_xyz])
    pad = 0.05 * np.ptp(both, axis=0) + 1e-3
    ax2.set_xlim(both[:, 0].min() - pad[0], both[:, 0].max() + pad[0])
    ax2.set_ylim(both[:, 2].max() + pad[2], both[:, 2].min() - pad[2])  # depth down
    ax2.set(
        xlabel="x (km)", ylabel="depth (km)", title="Section (all events projected): true → located"
    )
    return _save(fig, path, dpi)


def criterion_vs_n(
    ns: Sequence[int],
    best: Sequence[float],
    greedy: Sequence[float],
    random_pct: Sequence[tuple[float, float, float] | None],
    recommended: int,
    target_m: float | None,
    label: str,
    path: Path,
    dpi: int,
) -> Path:
    """Best criterion per network size, greedy, and the random-network 5-95 % envelope."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rn = [n for n, p in zip(ns, random_pct, strict=True) if p is not None]
    if rn:
        rp = np.array([p for p in random_pct if p is not None])
        ax.fill_between(
            rn,
            rp[:, 0],
            rp[:, 2],
            color=SERIES[1],
            alpha=0.15,
            lw=0,
            label="random networks 5-95 %",
        )
        ax.plot(rn, rp[:, 1], color=SERIES[1], lw=1.2, ls="--", label="random median")
    ax.plot(ns, greedy, color=TEXT_MUTED, lw=1.2, ls=":", marker="o", ms=4, label="greedy")
    ax.plot(ns, best, color=SERIES[0], marker="o", ms=6, label="optimised")
    i = list(ns).index(recommended)
    ax.scatter(
        [recommended], [best[i]], s=160, facecolors="none", edgecolors=TEXT, lw=1.5, zorder=6
    )
    ax.annotate(
        f"recommended n = {recommended}\n{best[i]:.0f} m",
        (recommended, best[i]),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=8,
        color=TEXT,
    )
    if target_m is not None:
        ax.axhline(target_m, color=TEXT_MUTED, lw=1.0, ls="--")
        ax.annotate(
            f"target {target_m:.0f} m",
            (ns[0], target_m),
            xytext=(2, 4),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_MUTED,
        )
    ax.set_yscale("log")  # small networks are orders of magnitude worse
    ax.set(
        xlabel="number of stations",
        ylabel=f"{label}, log scale",
        title="Design criterion vs network size",
    )
    ax.set_xticks(list(ns))
    ax.legend(loc="upper right")
    return _save(fig, path, dpi)


def robustness(
    ns: Sequence[int],
    best: Sequence[float],
    worst_loss: Sequence[float],
    median_loss: Sequence[float],
    label: str,
    path: Path,
    dpi: int,
) -> Path:
    """Criterion of each best network, and with its worst / median single station lost."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(ns, best, color=SERIES[0], marker="o", ms=5, label="all stations working")
    ax.plot(ns, median_loss, color=SERIES[2], marker="o", ms=5, label="median station lost")
    ax.plot(ns, worst_loss, color=SERIES[1], marker="o", ms=5, label="worst station lost")
    ax.set_yscale("log")
    ax.set(
        xlabel="number of stations",
        ylabel=f"{label}, log scale",
        title="Robustness to one station failing",
    )
    ax.set_xticks(list(ns))
    ax.legend(loc="upper right")
    return _save(fig, path, dpi)


def design_map(
    xy: FloatArray,
    names: Sequence[str],
    status: Sequence[str],
    rank: Sequence[int | None],
    gain_pct: Sequence[float | None],
    background_xy: FloatArray,
    background_sd: FloatArray,
    target: Sequence[FloatArray],
    path: Path,
    dpi: int,
    title: str,
) -> Path:
    """Every candidate coloured by outcome: selected (with rank), not selected (with the
    improvement it would bring), excluded; contours of the design's horizontal 1-sigma."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if len(background_sd) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cs = ax.tricontour(
                background_xy[:, 0],
                background_xy[:, 1],
                background_sd,
                levels=8,
                colors=GRID_LINE,
                linewidths=0.8,
            )
            ax.clabel(cs, fontsize=7, fmt="%.0f m")
    for i, poly in enumerate(target):
        _outline(ax, poly, "target volume" if i == 0 else None)
    st = np.array(status)
    styles = {
        "selected": {"c": SERIES[0], "label": "selected (rank: 1 = most important)"},
        "fixed": {"c": TEXT, "label": "fixed (existing)"},
        "candidate": {"c": "#d9d8d4", "label": "not selected (% improvement if added; - = worse)"},
        "excluded": {"c": "#ffffff", "label": "excluded"},
    }
    for key, sty in styles.items():
        sel = st == key
        if sel.any():
            ax.scatter(xy[sel, 0], xy[sel, 1], c=sty["c"], label=sty["label"], **STATION_MARKER)
    for i, n in enumerate(names):
        if status[i] in ("selected", "fixed") and rank[i] is not None:
            text = f"{n} ({rank[i]})"
        elif status[i] == "candidate" and gain_pct[i] is not None:
            text = f"{n} {gain_pct[i]:+.1f}%"
        else:
            text = n
        ax.annotate(
            text,
            (xy[i, 0], xy[i, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=TEXT,
        )
    ax.set(xlabel="x (km east)", ylabel="y (km north)", title=title)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return _save(fig, path, dpi)


def detection_by_station(
    names: Sequence[str], frac: FloatArray, phases: Sequence[str], path: Path, dpi: int
) -> Path:
    """Fraction of target points where each station's pick is more likely than not."""
    apply_style()
    y = np.arange(len(names))
    h = 0.8 / len(phases)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.5))
    for k, (phase, color) in enumerate(zip(phases, SERIES, strict=False)):
        ax.barh(y - 0.4 + h * (k + 0.5), 100 * frac[:, k], height=h, color=color, label=phase)
    ax.set_yticks(y, list(names))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set(
        xlabel="target points with detection probability > 50 % (%)",
        title="Relative detectability by station",
    )
    ax.legend(loc="lower right")
    return _save(fig, path, dpi)


def sites_map(
    xy: FloatArray,
    elev_m: FloatArray,
    keep: npt.NDArray[np.bool_],
    dem_xy: tuple[FloatArray, FloatArray],
    dem_z: FloatArray,
    path: Path,
    dpi: int,
) -> Path:
    """Generated candidate sites (coloured by elevation) over DEM contours; dropped sites
    (no DEM data or too low) as crosses."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    if np.isfinite(dem_z).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no contours in a flat DEM
            ax.contour(*dem_xy, dem_z, levels=15, colors=GRID_LINE, linewidths=0.5)
    sc = ax.scatter(xy[keep, 0], xy[keep, 1], c=elev_m[keep], cmap=SEQUENTIAL, **STATION_MARKER)
    if (~keep).any():
        ax.scatter(*xy[~keep].T, marker="x", color=TEXT_MUTED, label="dropped")
        ax.legend(loc="upper right")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="site elevation (m)")
    ax.set(xlabel="x (km east)", ylabel="y (km north)", title="Generated candidate sites")
    ax.set_aspect("equal")
    return _save(fig, path, dpi)
