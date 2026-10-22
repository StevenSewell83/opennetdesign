"""Shapefile export (``netdesign export``): candidates, target boxes and design results.

Written to ``<output_dir>/gis/`` in the input ``crs`` (with a ``.prj``), each layer only when
its inputs exist:

- ``candidates_input``: every candidate as given (``stations.candidates``) with its
  elevation and detection site type;
- ``target_boxes``: the source regions, with name, depth range and share of the events;
- ``candidates_outcome``: every candidate after the design (``qc/candidates_outcome.csv``):
  status, rank in the recommended network, greedy order, gain and contribution;
- ``network_final``: the recommended network, by rank;
- ``networks_all``: the best network of every size (one point per station and size), with
  its criterion, located fraction and whether it is the recommended one.

DBF field names are limited to 10 characters; the mapping is listed in each function.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import shapefile
from pyproj import CRS
from pyproj.enums import WktVersion

from netdesign.config import Config
from netdesign.crs import InputCRS
from netdesign.errors import ConfigError
from netdesign.frame import LocalFrame
from netdesign.stations import Candidates

__all__ = ["export_gis"]


def _write_prj(path: Path, crs: InputCRS) -> None:
    wkt = CRS.from_epsg(crs.epsg).to_wkt(WktVersion.WKT1_ESRI)
    path.with_suffix(".prj").write_text(wkt)


def _xy(crs: InputCRS, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """WGS-84 to the input crs."""
    a, b = crs.from_wgs84(np.atleast_1d(lon), np.atleast_1d(lat))
    return np.asarray(a), np.asarray(b)


def _num(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _candidates_input(cfg: Config, cand: Candidates, path: Path) -> Path:
    """Fields: STATION, ELEV_M, SITE (detection site type)."""
    d = cfg.detection
    a, b = _xy(cand.crs, cand.long, cand.lat)
    with shapefile.Writer(str(path), shapeType=shapefile.POINT) as w:
        w.field("STATION", "C", 32)
        w.field("ELEV_M", "N", 12, 2)
        w.field("SITE", "C", 32)
        for i, name in enumerate(cand.names):
            w.point(float(a[i]), float(b[i]))
            w.record(name, float(cand.elev_m[i]), d.sites.get(name, d.default_site))
    _write_prj(path, cand.crs)
    return path


def _clockwise(ring: np.ndarray) -> np.ndarray:
    """Close the ring and make it clockwise (shapefile outer ring)."""
    x, y = ring[:, 0], ring[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    ring = ring[::-1] if area2 > 0 else ring
    return np.vstack([ring, ring[:1]])


def _target_boxes(cfg: Config, frame: LocalFrame, crs: InputCRS, path: Path) -> Path:
    """Fields: NAME, DEPTH_TOP, DEPTH_BOT (km below sea level), WEIGHT (as given, or blank),
    SHARE (fraction of the design events)."""
    assert cfg.target is not None
    boxes = cfg.target.all_boxes()
    polys = cfg.target.polygons(frame, crs)
    shares = cfg.target.shares(frame, crs)
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as w:
        w.field("NAME", "C", 40)
        w.field("DEPTH_TOP", "N", 10, 3)
        w.field("DEPTH_BOT", "N", 10, 3)
        w.field("WEIGHT", "N", 12, 4)
        w.field("SHARE", "N", 8, 4)
        for i, (box, poly, share) in enumerate(zip(boxes, polys, shares, strict=True)):
            lon, lat = frame.to_geographic(poly[:, 0], poly[:, 1])
            a, b = _xy(crs, lon, lat)
            ring = _clockwise(np.column_stack([a, b]))
            w.poly([ring.tolist()])
            depth = box.depth_km or cfg.target.depth_km
            assert depth is not None
            w.record(box.name or f"box{i + 1}", depth[0], depth[1], box.weight, share)
    _write_prj(path, crs)
    return path


_OUTCOME_FIELDS = [
    # csv column, dbf name, type, size, decimals
    ("station", "STATION", "C", 32, 0),
    ("elev_m", "ELEV_M", "N", 12, 2),
    ("status", "STATUS", "C", 16, 0),
    ("selection_order", "SEL_ORDER", "N", 6, 0),
    ("greedy_order", "GREEDY_ORD", "N", 6, 0),
    ("gain_if_added_pct", "GAIN_PCT", "N", 10, 2),
    ("contribution_horizontal_pct", "CONTR_H", "N", 10, 2),
    ("contribution_depth_pct", "CONTR_Z", "N", 10, 2),
]


def _candidates_outcome(src: Path, crs: InputCRS, path: Path) -> Path:
    """Fields (from ``candidates_outcome.csv``): STATION, ELEV_M, STATUS, SEL_ORDER (rank in
    the recommended network), GREEDY_ORD, GAIN_PCT (gain if added), CONTR_H, CONTR_Z
    (contribution to the full candidate network, %)."""
    with src.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    with shapefile.Writer(str(path), shapeType=shapefile.POINT) as w:
        for _, name, kind, size, dec in _OUTCOME_FIELDS:
            w.field(name, kind, size, dec)
        for r in rows:
            a, b = _xy(crs, float(r["long"]), float(r["lat"]))
            w.point(float(a[0]), float(b[0]))
            w.record(
                *(r[col] if kind == "C" else _num(r[col]) for col, _, kind, _, _ in _OUTCOME_FIELDS)
            )
    _write_prj(path, crs)
    return path


def _networks(
    design: dict[str, Any],
    cand: Candidates,
    rank: dict[str, int],
    final: Path,
    everything: Path,
) -> list[Path]:
    """``network_final`` fields: STATION, RANK (1 = most important, from the design QC;
    blank without it), ELEV_M, N.
    ``networks_all`` fields: N (network size), STATION, CRIT_M (criterion, m), LOC_FRAC
    (located fraction), RECOMMEND (1 for the recommended size)."""
    index = {n: i for i, n in enumerate(cand.names)}
    a, b = _xy(cand.crs, cand.long, cand.lat)
    rec_n = design["recommended_n"]
    with shapefile.Writer(str(final), shapeType=shapefile.POINT) as w:
        w.field("STATION", "C", 32)
        w.field("RANK", "N", 6, 0)
        w.field("ELEV_M", "N", 12, 2)
        w.field("N", "N", 6, 0)
        for name in sorted(design["recommended_stations"], key=lambda n: rank.get(n, 1 << 30)):
            i = index[name]
            w.point(float(a[i]), float(b[i]))
            w.record(name, rank.get(name), float(cand.elev_m[i]), rec_n)
    with shapefile.Writer(str(everything), shapeType=shapefile.POINT) as w:
        w.field("N", "N", 6, 0)
        w.field("STATION", "C", 32)
        w.field("CRIT_M", "N", 14, 2)
        w.field("LOC_FRAC", "N", 8, 4)
        w.field("RECOMMEND", "N", 1, 0)
        for net in design["networks"]:
            for name in net["stations"]:
                i = index[name]
                w.point(float(a[i]), float(b[i]))
                w.record(
                    net["n"],
                    name,
                    net["criterion_m"],
                    net.get("located_fraction", 1.0),
                    int(net["n"] == rec_n),
                )
    for p in (final, everything):
        _write_prj(p, cand.crs)
    return [final, everything]


def export_gis(cfg: Config, frame: LocalFrame, out: Path) -> list[Path]:
    """Write the shapefiles described in the module docstring to ``out/gis``; returns the
    ``.shp`` paths written.

    Raises
    ------
    ConfigError
        If there is no ``stations`` section (nothing to export).
    """
    if cfg.stations is None:
        raise ConfigError("export needs the 'stations' config section")
    crs = cfg.input_crs
    cand = Candidates.read(cfg.stations.candidates, crs)
    gdir = out / "gis"
    gdir.mkdir(parents=True, exist_ok=True)
    written = [_candidates_input(cfg, cand, gdir / "candidates_input")]
    if cfg.target is not None:
        written.append(_target_boxes(cfg, frame, crs, gdir / "target_boxes"))
    outcome = out / "qc" / "candidates_outcome.csv"
    rank: dict[str, int] = {}
    if outcome.is_file():
        written.append(_candidates_outcome(outcome, crs, gdir / "candidates_outcome"))
        with outcome.open(newline="") as fh:
            rank = {
                r["station"]: int(r["selection_order"])
                for r in csv.DictReader(fh)
                if r["selection_order"]
            }
    design = out / "design" / "design.json"
    if design.is_file():
        data = json.loads(design.read_text())
        written += _networks(data, cand, rank, gdir / "network_final", gdir / "networks_all")
    return [p.with_suffix(".shp") for p in written]
