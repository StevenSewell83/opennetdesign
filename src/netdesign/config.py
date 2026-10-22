"""YAML configuration. Relative paths resolve against the config file's directory."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from netdesign.crs import CRSSpec, InputCRS
from netdesign.design import DesignSpec
from netdesign.detection import DetectionSpec
from netdesign.errors import ConfigError
from netdesign.frame import GridSpec
from netdesign.information import PickSpec
from netdesign.nll import NLLSpec, Phase
from netdesign.qc.checks import QCSpec
from netdesign.sites import SitesSpec
from netdesign.target import TargetSpec
from netdesign.validate import ValidateSpec
from netdesign.velocity import AttenuationSpec, PerturbSpec
from netdesign.velocity.heterogeneity import HeterogeneitySpec

__all__ = ["Config", "StationsConfig", "VelocityConfig", "load_config"]


class VelocityConfig(BaseModel):
    """``velocity`` section: starting model, optional Q and optional perturbation.

    The starting model is either a layered model (``base``) or a 3D node model
    (``model_3d``, see :mod:`netdesign.velocity.model3d`); give exactly one.
    """

    model_config = ConfigDict(extra="forbid")

    base: Path | None = None
    model_3d: Path | None = None
    attenuation: AttenuationSpec | None = None
    perturb: PerturbSpec = PerturbSpec()
    heterogeneity: HeterogeneitySpec = HeterogeneitySpec()

    @model_validator(mode="after")
    def _one_model(self) -> Self:
        if (self.base is None) == (self.model_3d is None):
            raise ConfigError("velocity: give exactly one of 'base' (layered) or 'model_3d'")
        if self.model_3d is not None:
            if self.perturb.enabled and self.perturb.interface_sd_km > 0:
                raise ConfigError(
                    "velocity.perturb.interface_sd_km: a 3D starting model has no interfaces"
                )
            if self.attenuation is not None and self.attenuation.model == "layered":
                raise ConfigError(
                    "velocity.attenuation.model 'layered' needs a layered model; use vs_scaled"
                )
            if self.heterogeneity.enabled:
                self.heterogeneity.check_unlayered()
        return self


class StationsConfig(BaseModel):
    """``stations`` section."""

    model_config = ConfigDict(extra="forbid")

    candidates: Path


class Config(BaseModel):
    """Top-level configuration. ``stations`` and ``grid`` are needed to build grids, ``target``
    for sensitivities."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("out")
    crs: CRSSpec = CRSSpec()
    phases: tuple[Phase, ...] = Field(("P", "S"), min_length=1)
    velocity: VelocityConfig
    stations: StationsConfig | None = None
    sites: SitesSpec | None = None  # generates stations.candidates
    grid: GridSpec | None = None
    nll: NLLSpec = NLLSpec()
    target: TargetSpec | None = None
    picks: PickSpec = PickSpec()
    validation: ValidateSpec = ValidateSpec()
    design: DesignSpec = DesignSpec()
    detection: DetectionSpec = DetectionSpec()
    qc: QCSpec = QCSpec()

    @property
    def input_crs(self) -> InputCRS:
        """The input coordinate system (all horizontal input coordinates)."""
        return InputCRS(self.crs.epsg)

    @field_validator("phases")
    @classmethod
    def _unique_phases(cls, v: tuple[Phase, ...]) -> tuple[Phase, ...]:
        if len(set(v)) != len(v):
            raise ValueError("phases must not repeat")
        return v


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config.

    Raises
    ------
    ConfigError
        If the file cannot be read or fails validation; the message names the field.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{path}: {details}") from exc

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else (path.parent / p).resolve()

    config.output_dir = resolve(config.output_dir)
    if config.velocity.base is not None:
        config.velocity.base = resolve(config.velocity.base)
    if config.velocity.model_3d is not None:
        config.velocity.model_3d = resolve(config.velocity.model_3d)
    if config.stations is not None:
        config.stations.candidates = resolve(config.stations.candidates)
    if config.nll.bin_dir is not None:
        config.nll.bin_dir = resolve(config.nll.bin_dir)
    if config.sites is not None:
        config.sites.dem.cache_dir = resolve(config.sites.dem.cache_dir)
        if config.sites.dem.path is not None:
            config.sites.dem.path = resolve(config.sites.dem.path)
    # Check now that every coordinate input matches the declared crs.
    crs = config.input_crs
    if config.target is not None:
        config.target.check_crs(crs)
    if config.grid is not None and config.grid.origin is not None:
        config.grid.origin.to_wgs84(crs)
    if config.sites is not None:
        config.sites.centre.to_wgs84(crs, where="sites.centre")
    return config
