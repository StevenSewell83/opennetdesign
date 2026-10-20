"""1D layered velocity/attenuation model with NonLinLoc ``LAYER`` and CSV I/O.

Conventions
-----------
- Depth ``z`` is km, positive down; the first layer's top is the model top (may be negative,
  i.e. above sea level, as in NonLinLoc).
- Velocities km/s, density g/cm³, gradients per km of depth. Q is dimensionless.
- Within layer ``i``: ``v(z) = v_top + v_grad * (z - top_km)`` for ``top_i <= z < top_{i+1}``;
  the last layer is a half-space. Q follows the same rule.
- Q is optional: either every layer has ``qp``/``qs`` or none does (see
  :mod:`netdesign.velocity.attenuation` for filling it from Vs).

NonLinLoc line format (Vel2Grid; Q is not representable and is dropped)::

    LAYER  depth  VpTop VpGrad  VsTop VsGrad  rhoTop rhoGrad

CSV format: header ``top_km,vp,vs`` plus optional ``rho,vp_grad,vs_grad,rho_grad,qp,qs,
qp_grad,qs_grad``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from netdesign.errors import DataError, PhysicsError

__all__ = ["MIN_VPVS", "Layer", "LayeredModel"]

FloatArray = npt.NDArray[np.float64]

MIN_VPVS = math.sqrt(4.0 / 3.0)
"""Smallest physical Vp/Vs (bulk modulus >= 0)."""


@dataclass(frozen=True)
class Layer:
    """One layer: values at its top plus linear gradients with depth."""

    top_km: float
    vp: float
    vs: float
    rho: float = 2.7
    vp_grad: float = 0.0
    vs_grad: float = 0.0
    rho_grad: float = 0.0
    qp: float | None = None
    qs: float | None = None
    qp_grad: float = 0.0
    qs_grad: float = 0.0

    @property
    def vpvs(self) -> float:
        """Vp/Vs at the layer top."""
        return self.vp / self.vs

    @property
    def has_q(self) -> bool:
        """True if both ``qp`` and ``qs`` are set."""
        return self.qp is not None and self.qs is not None


_REQUIRED_CSV = ("top_km", "vp", "vs")
_FIELD_NAMES = tuple(f.name for f in fields(Layer))


@dataclass(frozen=True)
class LayeredModel:
    """Ordered stack of layers; validated on construction.

    Raises
    ------
    PhysicsError
        If tops are not strictly increasing, a velocity or Q is non-positive, Q is set on some
        layers but not others, or Vp/Vs falls below :data:`MIN_VPVS`. Values are checked at the
        top and (for finite layers) the base of every layer.
    """

    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise PhysicsError("layered model has no layers")
        tops = self.tops_km
        if np.any(np.diff(tops) <= 0):
            raise PhysicsError(f"layer tops must be strictly increasing, got {tops.tolist()}")
        with_q = [la.has_q for la in self.layers]
        if any(with_q) and not all(with_q):
            raise PhysicsError("qp/qs must be set on every layer or on none")
        bases = [*tops[1:].tolist(), None]
        for i, (la, base) in enumerate(zip(self.layers, bases, strict=True)):
            for z in [la.top_km] if base is None else [la.top_km, base]:
                dz = z - la.top_km
                vp = la.vp + la.vp_grad * dz
                vs = la.vs + la.vs_grad * dz
                if vp <= 0 or vs <= 0:
                    raise PhysicsError(f"layer {i}: non-positive velocity at z={z:g} km")
                if vp / vs < MIN_VPVS:
                    raise PhysicsError(
                        f"layer {i}: Vp/Vs {vp / vs:.3f} < {MIN_VPVS:.3f} at z={z:g} km"
                    )
                if (
                    la.qp is not None
                    and la.qs is not None
                    and (la.qp + la.qp_grad * dz <= 0 or la.qs + la.qs_grad * dz <= 0)
                ):
                    raise PhysicsError(f"layer {i}: non-positive Q at z={z:g} km")

    @property
    def n_layers(self) -> int:
        """Number of layers (including the half-space)."""
        return len(self.layers)

    @property
    def has_q(self) -> bool:
        """True if the model carries Q."""
        return self.layers[0].has_q

    @property
    def tops_km(self) -> FloatArray:
        """Layer top depths, km."""
        return np.array([la.top_km for la in self.layers], dtype=np.float64)

    @property
    def vpvs(self) -> FloatArray:
        """Vp/Vs at each layer top."""
        return np.array([la.vpvs for la in self.layers], dtype=np.float64)

    def _profile(self, top: str, grad: str, z_km: npt.ArrayLike) -> FloatArray:
        z = np.asarray(z_km, dtype=np.float64)
        idx = np.clip(np.searchsorted(self.tops_km, z, side="right") - 1, 0, None)
        v0 = np.array([getattr(la, top) for la in self.layers], dtype=np.float64)
        g = np.array([getattr(la, grad) for la in self.layers], dtype=np.float64)
        result: FloatArray = v0[idx] + g[idx] * (z - self.tops_km[idx])
        return result

    def velocity_at(self, z_km: npt.ArrayLike, phase: Literal["P", "S"] = "P") -> FloatArray:
        """Velocity (km/s) at depths ``z_km``. Depths above the model top use the first layer."""
        name = "vp" if phase == "P" else "vs"
        return self._profile(name, f"{name}_grad", z_km)

    def q_at(self, z_km: npt.ArrayLike, phase: Literal["P", "S"] = "P") -> FloatArray:
        """Q at depths ``z_km``.

        Raises
        ------
        PhysicsError
            If the model has no Q.
        """
        if not self.has_q:
            raise PhysicsError("model has no Q; set velocity.attenuation in the config")
        name = "qp" if phase == "P" else "qs"
        return self._profile(name, f"{name}_grad", z_km)

    # ------------------------------------------------------------------ I/O

    def to_nll(self) -> str:
        """NonLinLoc ``LAYER`` lines, one per layer (Q is dropped)."""
        return "".join(
            f"LAYER {la.top_km:11.6f} {la.vp:9.6f} {la.vp_grad:9.6f} "
            f"{la.vs:9.6f} {la.vs_grad:9.6f} {la.rho:8.5f} {la.rho_grad:8.5f}\n"
            for la in self.layers
        )

    @classmethod
    def from_nll(cls, text: str, source: str = "<string>") -> LayeredModel:
        """Parse ``LAYER`` lines; all other lines (e.g. a full control file) are ignored."""
        layers: list[Layer] = []
        for lineno, raw in enumerate(text.splitlines(), start=1):
            tokens = raw.split("#", 1)[0].split()
            if not tokens or tokens[0] != "LAYER":
                continue
            try:
                z, vp, vpg, vs, vsg, rho, rhog = (float(t) for t in tokens[1:8])
            except ValueError as exc:
                raise DataError(f"{source}:{lineno}: bad LAYER line {raw.strip()!r}") from exc
            layers.append(Layer(z, vp, vs, rho, vpg, vsg, rhog))
        if not layers:
            raise DataError(f"{source}: no LAYER lines found")
        return cls(tuple(layers))

    @classmethod
    def from_csv(cls, path: Path) -> LayeredModel:
        """Read the CSV format described in the module docstring."""
        try:
            with path.open(newline="") as fh:
                rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
        except OSError as exc:
            raise DataError(f"{path}: cannot read velocity CSV ({exc})") from exc
        if not rows:
            raise DataError(f"{path}: no layers")
        columns = set(rows[0])
        missing = set(_REQUIRED_CSV) - columns
        if missing:
            raise DataError(f"{path}: missing columns {sorted(missing)}")
        unknown = columns - set(_FIELD_NAMES)
        if unknown:
            raise DataError(f"{path}: unknown columns {sorted(unknown)}")
        layers = []
        for lineno, row in enumerate(rows, start=2):
            try:
                values = {k: float(v) for k, v in row.items() if v not in ("", None)}
            except ValueError as exc:
                raise DataError(f"{path}: row {lineno}: {exc}") from exc
            layers.append(Layer(**values))
        return cls(tuple(layers))

    def to_csv(self, path: str | Path) -> None:
        """Write all fields (including Q, if present) as CSV."""
        names = [n for n in _FIELD_NAMES if self.has_q or not n.startswith("q")]
        with Path(path).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for la in self.layers:
                writer.writerow({k: f"{v:.10g}" for k, v in asdict(la).items() if k in names})

    @classmethod
    def read(cls, path: str | Path) -> LayeredModel:
        """Read a ``.csv`` model, or any other file as NonLinLoc ``LAYER`` lines."""
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return cls.from_csv(path)
        try:
            text = path.read_text()
        except OSError as exc:
            raise DataError(f"{path}: cannot read velocity model ({exc})") from exc
        return cls.from_nll(text, source=str(path))

    def write(self, path: str | Path) -> None:
        """Write ``.csv`` (all fields) or, for any other suffix, NonLinLoc ``LAYER`` lines."""
        path = Path(path)
        if path.suffix.lower() == ".csv":
            self.to_csv(path)
        else:
            path.write_text(self.to_nll())
