"""Shapefile export."""

import json
from pathlib import Path

import numpy as np
import shapefile
from example_config import write_example
from typer.testing import CliRunner

from netdesign.cli import app
from netdesign.config import load_config
from netdesign.stations import Candidates


def _fake_design(cfg_path: Path) -> list[str]:
    """Minimal design outputs (as `design` writes them) for the example candidates."""
    cfg = load_config(cfg_path)
    assert cfg.stations is not None
    cand = Candidates.read(cfg.stations.candidates)
    names = list(cand.names)
    chosen = [names[4], names[1], names[7]]  # rank order: 1, 2, 3
    qc = cfg.output_dir / "qc"
    qc.mkdir(parents=True)
    rows = [
        "station,long,lat,elev_m,status,selection_order,greedy_order,gain_if_added_pct,"
        "contribution_horizontal_pct,contribution_depth_pct"
    ]
    for i, n in enumerate(names):
        order = str(chosen.index(n) + 1) if n in chosen else ""
        status = "selected" if n in chosen else "candidate"
        rows.append(
            f"{n},{cand.long[i]},{cand.lat[i]},{cand.elev_m[i]},{status},{order},{i + 1},,1.5,2.5"
        )
    (qc / "candidates_outcome.csv").write_text("\n".join(rows) + "\n")
    design = cfg.output_dir / "design"
    design.mkdir()
    nets = [
        {"n": 2, "stations": sorted(chosen[:2]), "criterion_m": 300.0, "located_fraction": 0.8},
        {"n": 3, "stations": sorted(chosen), "criterion_m": 200.0, "located_fraction": 0.95},
    ]
    (design / "design.json").write_text(
        json.dumps({"recommended_n": 3, "recommended_stations": sorted(chosen), "networks": nets})
    )
    return chosen


def test_export_all_layers(tmp_path: Path) -> None:
    cfg_path = write_example(tmp_path)
    chosen = _fake_design(cfg_path)
    result = CliRunner().invoke(app, ["export", str(cfg_path)])
    assert result.exit_code == 0, result.output
    gis = tmp_path / "out" / "gis"
    layers = {p.stem for p in gis.glob("*.shp")}
    assert layers == {
        "candidates_input",
        "target_boxes",
        "candidates_outcome",
        "network_final",
        "networks_all",
    }
    assert all((gis / f"{name}.prj").read_text().startswith("GEOGCS") for name in layers)

    cand = shapefile.Reader(str(gis / "candidates_input.shp"))
    assert len(cand) == 10 and cand.shapeTypeName == "POINT"
    rec = cand.record(9).as_dict()
    assert rec["STATION"] == "C10" and rec["SITE"] == "borehole" and rec["ELEV_M"] == -150

    final = shapefile.Reader(str(gis / "network_final.shp"))
    assert [r["STATION"] for r in final.iterRecords()] == chosen  # ordered by rank
    assert [r["RANK"] for r in final.iterRecords()] == [1, 2, 3]

    everything = shapefile.Reader(str(gis / "networks_all.shp"))
    recs = [r.as_dict() for r in everything.iterRecords()]
    assert len(recs) == 5 and sum(r["RECOMMEND"] for r in recs) == 3
    assert {r["LOC_FRAC"] for r in recs if r["N"] == 2} == {0.8}

    outcome = shapefile.Reader(str(gis / "candidates_outcome.shp"))
    by_name = {r["STATION"]: r for r in outcome.iterRecords()}
    assert by_name[chosen[0]]["SEL_ORDER"] == 1 and by_name[chosen[0]]["STATUS"] == "selected"

    boxes = shapefile.Reader(str(gis / "target_boxes.shp"))
    assert len(boxes) == 1 and boxes.record(0)["SHARE"] == 1.0
    ring = np.array(boxes.shape(0).points)
    assert np.allclose(ring[0], ring[-1])  # closed
    x, y = ring[:, 0], ring[:, 1]
    assert np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]) < 0  # clockwise outer ring
    assert ring[:, 0].min() >= 176.11 - 1e-6 and ring[:, 0].max() <= 176.15 + 1e-6


def test_export_before_any_run(tmp_path: Path) -> None:
    """Only the inputs exist: candidates and target boxes are still exported."""
    result = CliRunner().invoke(app, ["export", str(write_example(tmp_path))])
    assert result.exit_code == 0, result.output
    assert {p.stem for p in (tmp_path / "out" / "gis").glob("*.shp")} == {
        "candidates_input",
        "target_boxes",
    }
