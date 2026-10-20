"""Target source volume: where events are expected, sampled into weighted source points.

The volume is a box in the input ``crs`` (``long``/``lat`` ranges for a geographic CRS,
``easting``/``northing`` for a projected one) over a depth range (``depth_km``, km below sea
level). Points are sampled in the local frame, either on a regular grid of cell centres or
uniformly at random, and kept if they fall inside the box.
Weights (summing to 1) express the expected distribution of events; ``uniform`` weights all
points equally, ``normal`` weights by a Gaussian in depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from netdesign.crs import WGS84, InputCRS
from netdesign.errors import ConfigError
from netdesign.frame import LocalFrame

__all__ = ["DepthWeight", "Sources", "TargetSpec", "sample_sources"]

FloatArray = npt.NDArray[np.float64]


class DepthWeight(BaseModel):
    """``target.depth_weight``: relative event likelihood with depth."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["uniform", "normal"] = "uniform"
    mean_km: float | None = None
    sd_km: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.model == "normal" and (self.mean_km is None or self.sd_km is None):
            raise ConfigError("target.depth_weight: model 'normal' needs mean_km and sd_km")
        return self


class TargetSpec(BaseModel):
    """``target`` config section."""

    model_config = ConfigDict(extra="forbid")

    long: tuple[float, float] | None = None
    lat: tuple[float, float] | None = None
    easting: tuple[float, float] | None = None
    northing: tuple[float, float] | None = None
    depth_km: tuple[float, float]
    sampling: Literal["grid", "random"] = "grid"
    spacing_km: float = Field(0.25, gt=0, description="horizontal spacing (grid sampling)")
    depth_spacing_km: float | None = Field(None, gt=0, description="default: spacing_km")
    n_points: int = Field(2000, ge=1, description="number of points (random sampling)")
    seed: int = 0
    depth_weight: DepthWeight = DepthWeight()
    fd_step_km: float | None = Field(
        None, gt=0, description="finite-difference step for gradients; default grid.dx_km"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        geo = self.long is not None and self.lat is not None
        proj = self.easting is not None and self.northing is not None
        if (
            geo == proj
            or (geo and (self.easting or self.northing))
            or (proj and (self.long or self.lat))
        ):
            raise ConfigError("target: give either long and lat, or easting and northing")
        for name in ("long", "lat", "easting", "northing", "depth_km"):
            rng = getattr(self, name)
            if rng is not None and not rng[0] < rng[1]:
                raise ConfigError(f"target.{name}: need [min, max] with min < max")
        return self

    def box(self, crs: InputCRS) -> tuple[tuple[float, float], tuple[float, float]]:
        """The horizontal ranges in ``crs`` units.

        Raises
        ------
        ConfigError
            If the box is given in the other kind of coordinates than ``crs``.
        """
        a, b = (self.long, self.lat) if crs.geographic else (self.easting, self.northing)
        if a is None or b is None:
            want = "long/lat" if crs.geographic else "easting/northing"
            raise ConfigError(f"target: {crs.name} needs the box as {want}")
        return a, b


@dataclass(frozen=True)
class Sources:
    """Source points: local ``xyz`` (J, 3) km, geographic ``long``/``lat`` and ``weights``."""

    xyz: FloatArray
    long: FloatArray
    lat: FloatArray
    weights: FloatArray

    def __len__(self) -> int:
        return len(self.weights)


def _local_bounds(
    box: tuple[tuple[float, float], tuple[float, float]], frame: LocalFrame, crs: InputCRS
) -> tuple[float, float, float, float]:
    """Local rectangle enclosing the box (perimeter sampled; edges may be curved)."""
    (a0, a1), (b0, b1) = box
    t = np.linspace(0.0, 1.0, 21)
    av = a0 + t * (a1 - a0)
    bv = b0 + t * (b1 - b0)
    ones = np.ones_like(t)
    aa = np.concatenate([av, av, a0 * ones, a1 * ones])
    bb = np.concatenate([b0 * ones, b1 * ones, bv, bv])
    x, y = frame.to_local(*crs.to_wgs84(aa, bb))
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _inside(
    box: tuple[tuple[float, float], tuple[float, float]],
    frame: LocalFrame,
    crs: InputCRS,
    x: FloatArray,
    y: FloatArray,
) -> npt.NDArray:
    (a0, a1), (b0, b1) = box
    a, b = crs.from_wgs84(*frame.to_geographic(x, y))
    return (a >= a0) & (a <= a1) & (b >= b0) & (b <= b1)


def sample_sources(spec: TargetSpec, frame: LocalFrame, crs: InputCRS = WGS84) -> Sources:
    """Sample the target volume.

    Raises
    ------
    ConfigError
        If no point falls inside the volume (e.g. spacing larger than the box).
    """
    box = spec.box(crs)
    xmin, xmax, ymin, ymax = _local_bounds(box, frame, crs)
    zmin, zmax = spec.depth_km
    if spec.sampling == "grid":
        s = spec.spacing_km
        sz = spec.depth_spacing_km or s
        xs = np.arange(xmin + s / 2, xmax, s)
        ys = np.arange(ymin + s / 2, ymax, s)
        zs = np.arange(zmin + sz / 2, zmax, sz)
        hx, hy = np.meshgrid(xs, ys, indexing="ij")
        keep = _inside(box, frame, crs, hx.ravel(), hy.ravel())
        hx, hy = hx.ravel()[keep], hy.ravel()[keep]
        x = np.repeat(hx, len(zs))
        y = np.repeat(hy, len(zs))
        z = np.tile(zs, len(hx))
    else:
        rng = np.random.default_rng(spec.seed)
        xs_, ys_ = [], []
        n_have = 0
        while n_have < spec.n_points:
            cx = rng.uniform(xmin, xmax, spec.n_points)
            cy = rng.uniform(ymin, ymax, spec.n_points)
            keep = _inside(box, frame, crs, cx, cy)
            xs_.append(cx[keep])
            ys_.append(cy[keep])
            n_have += int(keep.sum())
        x = np.concatenate(xs_)[: spec.n_points]
        y = np.concatenate(ys_)[: spec.n_points]
        z = rng.uniform(zmin, zmax, spec.n_points)
    if len(x) == 0:
        raise ConfigError("target: no source points inside the volume; reduce spacing_km")

    w = spec.depth_weight
    if w.model == "normal":
        assert w.mean_km is not None and w.sd_km is not None
        weights = np.exp(-0.5 * ((z - w.mean_km) / w.sd_km) ** 2)
    else:
        weights = np.ones_like(z)
    if weights.sum() <= 0:
        raise ConfigError("target.depth_weight gives zero weight to every point")
    lon, lat = frame.to_geographic(x, y)
    return Sources(np.stack([x, y, z], axis=1), lon, lat, weights / weights.sum())
