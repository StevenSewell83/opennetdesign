"""QC check records and thresholds (``qc`` config section)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Check", "QCSpec", "Status", "write_checks"]

Status = Literal["pass", "warn", "fail", "info"]


class QCSpec(BaseModel):
    """``qc`` config section: what to produce and the pass/warn thresholds."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    dpi: int = Field(150, ge=50)
    profile_samples: int = Field(20, ge=0, description="individual models drawn on profiles")
    slice_models: list[int] = Field(default_factory=lambda: [1, 2])
    map_depths_km: list[float] | None = Field(None, description="default: top/mid/bottom")
    station_min_spacing_km: float = Field(0.05, ge=0)
    min_model_coverage: float = Field(0.8, ge=0, le=1)
    het_sd_tolerance: float = Field(0.25, gt=0, description="relative")
    het_corr_len_tolerance: float = Field(0.35, gt=0, description="relative")
    eikonal_median_tol: float = Field(0.03, gt=0)
    eikonal_p99_tol: float = Field(0.10, gt=0)
    validation_min_located: float = Field(0.95, ge=0, le=1)
    validation_rms_ratio: tuple[float, float] = (0.7, 1.4)
    validation_min_corr: float = Field(0.9, ge=-1, le=1)
    validation_68_range: tuple[float, float] = (0.55, 0.80)
    validation_95_range: tuple[float, float] = (0.85, 0.99)
    min_expected_p_picks: float = Field(4.0, ge=0, description="detection coverage check")
    min_expected_s_picks: float = Field(2.0, ge=0)
    min_detection_coverage: float = Field(0.95, ge=0, le=1)


@dataclass(frozen=True)
class Check:
    """One QC result. ``value`` is what was measured, ``expected`` the criterion."""

    stage: str
    name: str
    status: Status
    value: Any
    expected: str
    message: str = ""


def write_checks(path: Path, checks: list[Check], extra: dict[str, Any] | None = None) -> None:
    """Write checks (and any extra data for the report) as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checks": [asdict(c) for c in checks], **(extra or {})}
    path.write_text(json.dumps(payload, indent=2, default=_default))


def _default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def within(value: float, lo: float, hi: float) -> Status:
    """``pass`` inside ``[lo, hi]``, else ``warn``."""
    return "pass" if lo <= value <= hi else "warn"
