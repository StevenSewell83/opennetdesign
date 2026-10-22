"""Target source volume: where events are expected, sampled into weighted source points.

The volume is one or more boxes, each over a depth range (``depth_km``, km below sea level):

- a box given by ranges in the input ``crs`` (``long``/``lat`` for a geographic CRS,
  ``easting``/``northing`` for a projected one), either directly in ``target`` (one box) or
  as an entry of ``target.boxes``;
- a rotated rectangle in ``target.boxes``: ``centre`` (in the input ``crs``), ``size_km``
  ``[across, along]`` and ``rotation_deg``, the 'along' axis clockwise from north (as for
  ``sites``).

Points are sampled in the local frame, either on a regular grid of cell centres (aligned with
a rotated box's axes) or uniformly at random, and kept if they fall inside the box. Boxes are
sampled independently, so overlapping boxes count their overlap twice.

Weights (summing to 1) express the expected distribution of events. Within a box, ``uniform``
weights all points equally and ``normal`` by a Gaussian in depth. Between boxes, events are
spread with uniform density (each box's share proportional to its volume), unless every box
gives a ``weight``: its relative share of the events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from netdesign.crs import WGS84, InputCRS
from netdesign.errors import ConfigError
from netdesign.frame import LocalFrame, Origin

__all__ = ["DepthWeight", "Sources", "TargetBox", "TargetSpec", "sample_sources"]

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


Range = tuple[tuple[float, float], tuple[float, float]]


def _check_ranges(where: str, values: dict[str, tuple[float, float] | None]) -> bool:
    """Validate ``long``/``lat``/``easting``/``northing``; True if a range box is given."""
    geo = values["long"] is not None and values["lat"] is not None
    proj = values["easting"] is not None and values["northing"] is not None
    given = [k for k, v in values.items() if v is not None]
    if given and (geo == proj or len(given) != 2):
        raise ConfigError(f"{where}: give either long and lat, or easting and northing")
    for name in given:
        rng = values[name]
        assert rng is not None
        if not rng[0] < rng[1]:
            raise ConfigError(f"{where}.{name}: need [min, max] with min < max")
    return bool(given)


def _ranges_in(where: str, values: dict[str, tuple[float, float] | None], crs: InputCRS) -> Range:
    a, b = (
        (values["long"], values["lat"])
        if crs.geographic
        else (values["easting"], values["northing"])
    )
    if a is None or b is None:
        want = "long/lat" if crs.geographic else "easting/northing"
        raise ConfigError(f"{where}: {crs.name} needs the box as {want}")
    return a, b


class TargetBox(BaseModel):
    """One entry of ``target.boxes``: ranges in the input ``crs``, or a rotated rectangle."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    long: tuple[float, float] | None = None
    lat: tuple[float, float] | None = None
    easting: tuple[float, float] | None = None
    northing: tuple[float, float] | None = None
    centre: Origin | None = None
    size_km: tuple[float, float] | None = Field(None, description="[across, along], km")
    rotation_deg: float = Field(0.0, description="'along' axis, clockwise from north")
    depth_km: tuple[float, float] | None = Field(None, description="default: target.depth_km")
    weight: float | None = Field(None, gt=0, description="relative share of the events")

    def _ranges(self) -> dict[str, tuple[float, float] | None]:
        return {k: getattr(self, k) for k in ("long", "lat", "easting", "northing")}

    @model_validator(mode="after")
    def _check(self) -> Self:
        ranges = _check_ranges("target.boxes", self._ranges())
        rotated = self.centre is not None or self.size_km is not None
        if ranges == rotated:
            raise ConfigError(
                "target.boxes: give each box either long/lat (easting/northing) ranges, "
                "or centre and size_km"
            )
        if rotated:
            if self.centre is None or self.size_km is None:
                raise ConfigError("target.boxes: a rotated box needs both centre and size_km")
            if min(self.size_km) <= 0:
                raise ConfigError("target.boxes.size_km must be positive")
        elif self.rotation_deg:
            raise ConfigError("target.boxes: rotation_deg needs centre and size_km")
        if self.depth_km is not None and not self.depth_km[0] < self.depth_km[1]:
            raise ConfigError("target.boxes.depth_km: need [min, max] with min < max")
        return self


class TargetSpec(BaseModel):
    """``target`` config section."""

    model_config = ConfigDict(extra="forbid")

    long: tuple[float, float] | None = None
    lat: tuple[float, float] | None = None
    easting: tuple[float, float] | None = None
    northing: tuple[float, float] | None = None
    boxes: list[TargetBox] | None = Field(None, min_length=1)
    depth_km: tuple[float, float] | None = None
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
        ranges = _check_ranges(
            "target", {k: getattr(self, k) for k in ("long", "lat", "easting", "northing")}
        )
        if ranges == (self.boxes is not None):
            raise ConfigError(
                "target: give either one box (long and lat, or easting and northing) or boxes"
            )
        if self.depth_km is not None and not self.depth_km[0] < self.depth_km[1]:
            raise ConfigError("target.depth_km: need [min, max] with min < max")
        for box in self.all_boxes():
            if box.depth_km is None and self.depth_km is None:
                raise ConfigError("target: give depth_km (for every box, or as the default)")
        weights = [b.weight is not None for b in self.all_boxes()]
        if any(weights) and not all(weights):
            raise ConfigError("target.boxes: give a weight for every box or for none")
        return self

    def all_boxes(self) -> list[TargetBox]:
        """The boxes, with the single top-level box as a one-element list."""
        if self.boxes is not None:
            return self.boxes
        return [
            TargetBox(long=self.long, lat=self.lat, easting=self.easting, northing=self.northing)
        ]

    def check_crs(self, crs: InputCRS) -> None:
        """Raise :class:`ConfigError` if a box uses the other kind of coordinates than ``crs``."""
        for i, box in enumerate(self.all_boxes()):
            where = "target" if self.boxes is None else f"target.boxes[{i}]"
            if box.centre is not None:
                box.centre.to_wgs84(crs, where=f"{where}.centre")
            else:
                _ranges_in(where, box._ranges(), crs)

    def polygons(self, frame: LocalFrame, crs: InputCRS) -> list[FloatArray]:
        """Outline of each box in local ``(x, y)`` km, (n, 2)."""
        return [_Region(b, frame, crs).polygon() for b in self.all_boxes()]

    def shares(self, frame: LocalFrame, crs: InputCRS) -> list[float]:
        """Each box's share of the events: its ``weight`` relative to the others, or (no
        weights) its volume share. With depth weighting the share within a box moves in
        depth, not between boxes."""
        boxes = self.all_boxes()
        if all(b.weight is not None for b in boxes):
            raw = [float(b.weight or 0.0) for b in boxes]
        else:
            raw = []
            for b in boxes:
                depth = b.depth_km or self.depth_km
                assert depth is not None
                raw.append(_Region(b, frame, crs).area_km2() * (depth[1] - depth[0]))
        total = sum(raw)
        return [r / total for r in raw]


class _Region:
    """A box in the local frame: sampling, inside test and outline."""

    def __init__(self, box: TargetBox, frame: LocalFrame, crs: InputCRS) -> None:
        self.box, self.frame, self.crs = box, frame, crs
        if box.centre is not None:
            assert box.size_km is not None
            lat, lon = box.centre.to_wgs84(crs, where="target.boxes.centre")
            cx, cy = frame.to_local(lon, lat)
            self.centre = (float(cx), float(cy))
            t = np.radians(box.rotation_deg)
            self.across = np.array([np.cos(t), -np.sin(t)])
            self.along = np.array([np.sin(t), np.cos(t)])
            self.half = (box.size_km[0] / 2, box.size_km[1] / 2)
            self.ranges: Range | None = None
        else:
            self.ranges = _ranges_in("target", box._ranges(), crs)

    def _uv_to_xy(self, u: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
        x = self.centre[0] + u * self.across[0] + v * self.along[0]
        y = self.centre[1] + u * self.across[1] + v * self.along[1]
        return x, y

    def polygon(self) -> FloatArray:
        if self.ranges is None:
            hu, hv = self.half
            u = np.array([-hu, hu, hu, -hu])
            v = np.array([-hv, -hv, hv, hv])
            return np.stack(self._uv_to_xy(u, v), axis=1)
        (a0, a1), (b0, b1) = self.ranges
        t = np.linspace(0.0, 1.0, 11)
        aa = np.concatenate(
            [a0 + t * (a1 - a0), np.full_like(t, a1), a1 - t * (a1 - a0), np.full_like(t, a0)]
        )
        bb = np.concatenate(
            [np.full_like(t, b0), b0 + t * (b1 - b0), np.full_like(t, b1), b1 - t * (b1 - b0)]
        )
        x, y = self.frame.to_local(*self.crs.to_wgs84(aa, bb))
        return np.stack([x, y], axis=1)

    def area_km2(self) -> float:
        x, y = self.polygon().T
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)

    def grid(self, s: float) -> tuple[FloatArray, FloatArray]:
        """Horizontal cell centres inside the box."""
        if self.ranges is None:
            hu, hv = self.half
            u = np.arange(-hu + s / 2, hu, s)
            v = np.arange(-hv + s / 2, hv, s)
            uu, vv = np.meshgrid(u, v, indexing="ij")
            return self._uv_to_xy(uu.ravel(), vv.ravel())
        xmin, xmax, ymin, ymax = _local_bounds(self.ranges, self.frame, self.crs)
        hx, hy = np.meshgrid(
            np.arange(xmin + s / 2, xmax, s), np.arange(ymin + s / 2, ymax, s), indexing="ij"
        )
        keep = _inside(self.ranges, self.frame, self.crs, hx.ravel(), hy.ravel())
        return hx.ravel()[keep], hy.ravel()[keep]

    def random(self, n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        """``n`` uniformly random horizontal points inside the box."""
        if self.ranges is None:
            hu, hv = self.half
            return self._uv_to_xy(rng.uniform(-hu, hu, n), rng.uniform(-hv, hv, n))
        xmin, xmax, ymin, ymax = _local_bounds(self.ranges, self.frame, self.crs)
        xs_, ys_ = [], []
        n_have = 0
        while n_have < n:
            cx = rng.uniform(xmin, xmax, n)
            cy = rng.uniform(ymin, ymax, n)
            keep = _inside(self.ranges, self.frame, self.crs, cx, cy)
            xs_.append(cx[keep])
            ys_.append(cy[keep])
            n_have += int(keep.sum())
        return np.concatenate(xs_)[:n], np.concatenate(ys_)[:n]


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
    box: Range, frame: LocalFrame, crs: InputCRS
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
    box: Range,
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
        If no point falls inside a box (e.g. spacing larger than the box).
    """
    boxes = spec.all_boxes()
    regions = [_Region(b, frame, crs) for b in boxes]
    depths = [b.depth_km or spec.depth_km for b in boxes]
    volumes = [
        r.area_km2() * (d[1] - d[0]) if d else 0.0 for r, d in zip(regions, depths, strict=True)
    ]
    rng = np.random.default_rng(spec.seed)
    xs, ys, zs, ws = [], [], [], []
    for i, (region, depth) in enumerate(zip(regions, depths, strict=True)):
        assert depth is not None
        zmin, zmax = depth
        if spec.sampling == "grid":
            sz = spec.depth_spacing_km or spec.spacing_km
            hx, hy = region.grid(spec.spacing_km)
            levels = np.arange(zmin + sz / 2, zmax, sz)
            x, y = np.repeat(hx, len(levels)), np.repeat(hy, len(levels))
            z = np.tile(levels, len(hx))
        else:
            n = (
                spec.n_points
                if len(boxes) == 1
                else round(spec.n_points * volumes[i] / sum(volumes))
            )
            x, y = region.random(max(n, 1), rng)
            z = rng.uniform(zmin, zmax, len(x))
        if len(x) == 0:
            where = "target" if spec.boxes is None else f"target.boxes[{i}]"
            raise ConfigError(f"{where}: no source points inside the volume; reduce spacing_km")
        w = _depth_weights(spec.depth_weight, z)
        share = boxes[i].weight
        if share is not None:
            if w.sum() <= 0:
                raise ConfigError(f"target.depth_weight gives zero weight in target.boxes[{i}]")
            # this box's share of the events, spread over its points by the depth weighting
            w = share * w / w.sum()
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ws.append(w)
    weights = np.concatenate(ws)
    if weights.sum() <= 0:
        raise ConfigError("target.depth_weight gives zero weight to every point")
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    lon, lat = frame.to_geographic(x, y)
    return Sources(np.stack([x, y, z], axis=1), lon, lat, weights / weights.sum())


def _depth_weights(model: DepthWeight, z: FloatArray) -> FloatArray:
    if model.model == "normal":
        assert model.mean_km is not None and model.sd_km is not None
        return np.asarray(np.exp(-0.5 * ((z - model.mean_km) / model.sd_km) ** 2))
    return np.ones_like(z)
