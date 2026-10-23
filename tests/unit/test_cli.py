"""CLI and config loading against the shipped example."""

import json
import re
from pathlib import Path
from typing import get_args

from pydantic import BaseModel
from typer.testing import CliRunner

from netdesign.cli import app
from netdesign.config import Config, load_config

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "config.yaml"


def test_example_config_loads() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.velocity.base.is_absolute()
    assert cfg.velocity.perturb.enabled


def test_ensemble_command(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["ensemble", str(EXAMPLE), "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    meta = json.loads((tmp_path / "models" / "ensemble.json").read_text())
    assert len(meta["realizations"]) == 101


def test_bad_config_reports_field(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("velocity:\n  base: x.layer\n  perturb: {n_realizations: 0}\n")
    result = CliRunner().invoke(app, ["ensemble", str(bad), "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "n_realizations" in result.output


def _field_names(model: type[BaseModel]) -> set[str]:
    names = set()
    for name, info in model.model_fields.items():
        names.add(name)
        for arg in (info.annotation, *get_args(info.annotation)):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                names |= _field_names(arg)
    return names


def test_example_config_documents_every_option() -> None:
    text = EXAMPLE.read_text()
    missing = {n for n in _field_names(Config) if not re.search(rf"\b{n}:", text)}
    assert not missing, f"add to examples/config.yaml: {sorted(missing)}"


def test_output_dir_from_config(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    model = EXAMPLE.parent / "start_1d.layer"
    cfg.write_text(f"output_dir: results\nvelocity:\n  base: {model}\n")
    result = CliRunner().invoke(app, ["ensemble", str(cfg)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "models" / "model_000.layer").is_file()


def test_user_template_is_valid() -> None:
    """templates/config.yaml must load (every key known, values in range)."""
    template = EXAMPLE.parents[1] / "templates" / "config.yaml"
    cfg = load_config(template)
    assert cfg.velocity.heterogeneity.enabled and cfg.detection.enabled
    assert cfg.grid is not None and cfg.grid.mode == "3d"


def test_run_command_end_to_end(tmp_path: Path, nll_bin: Path) -> None:
    from example_config import write_example

    path = write_example(
        tmp_path,
        nll_bin,
        grid={"dx_km": 0.2},
        velocity={"perturb": {"n_realizations": 2}},
        validation={"n_events": 10},
        design={"n_random": 5, "n_restarts": 1},
    )
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == 0, result.output
    for stage in ("== sensitivity", "== validate", "== design", "recommended:", "finished in"):
        assert stage in result.output
    out = tmp_path / "out"
    assert (out / "design" / "design.json").is_file()
    assert (out / "validation" / "summary.json").is_file()
    assert "<h2>Network design</h2>" in (out / "qc" / "report.html").read_text()


def test_run_skips_validation_when_disabled(tmp_path: Path, nll_bin: Path) -> None:
    from example_config import write_example

    path = write_example(
        tmp_path,
        nll_bin,
        grid={"dx_km": 0.2},
        velocity={"perturb": {"n_realizations": 2}},
        validation={"enabled": False},
        design={"n_random": 5, "n_restarts": 1},
    )
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == 0, result.output
    assert "validation skipped" in result.output
    assert not (tmp_path / "out" / "validation").exists()


def test_run_fails_fast_on_settings_validation_cannot_use(tmp_path: Path) -> None:
    from example_config import write_example

    path = write_example(tmp_path, None, nll={"keep_grids": "none"})
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == 1
    assert "keep_grids" in result.output
    assert not (tmp_path / "out").exists()  # failed before doing any work


def test_config_option_before_command(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["-c", str(EXAMPLE), "-o", str(tmp_path), "ensemble"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "models" / "ensemble.json").is_file()


def test_config_option_without_command_runs_pipeline(tmp_path: Path) -> None:
    from example_config import write_example

    path = write_example(tmp_path, None, nll={"keep_grids": "none"})
    result = CliRunner().invoke(app, ["-c", str(path)])
    assert result.exit_code == 1  # the run's fail-fast check: the pipeline was started
    assert "keep_grids" in result.output


def test_command_without_config_is_a_usage_error() -> None:
    result = CliRunner().invoke(app, ["design"])
    assert result.exit_code == 2
    assert "netdesign -c config.yaml" in result.output


def test_no_arguments_shows_help() -> None:
    result = CliRunner().invoke(app, [])
    assert "--config" in result.output
