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
