"""3D, spatially correlated, geologically structured perturbations of a layered model.

Each realization's (1D-perturbed) layered model is turned into 3D Vp and Vs on the grid's
cell centres (as NonLinLoc's Vel2Grid samples) and then perturbed by:

1. **Interface relief**: each interface below the model top is displaced by a correlated 2D
   surface (sd ``interface_sd_km``, horizontal correlation from ``structure``), clipped to
   ``interface_clip`` of the thinner adjacent layer so layers never cross. Velocity within a
   layer is referenced to its local top, i.e. layers drape over the relief.
2. **Vp heterogeneity**: ``Vp *= 1 + vp_pct/100 * f`` with ``f`` a correlated 3D field.
3. **Vp/Vs heterogeneity**: ``Vp/Vs += vpvs_sd * g`` with ``g`` an independent field; Vs then
   follows from the perturbed Vp and Vp/Vs, so Vs is not just a scaled copy of Vp.

``per_layer`` uses an independent field in every layer, so properties change abruptly across
lithological boundaries; otherwise one field spans all layers. Amplitudes may be one value or
one per layer (``vp_pct``, ``vpvs_sd``) or per interface (``interface_sd_km``), e.g. to make
shallow volcanic units more heterogeneous than basement. Fields are truncated at
``truncate_sd`` standard deviations and Vp/Vs is kept above ``MIN_VPVS + 0.01``.

Fields are drawn from ``numpy.random.default_rng([seed, realization_index])`` so any
realization can be regenerated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from netdesign.errors import ConfigError
from netdesign.frame import GridGeometry
from netdesign.velocity.layered import MIN_VPVS, LayeredModel
from netdesign.velocity.random_field import CorrelationSpec, correlated_field

__all__ = ["HeterogeneitySpec", "ModelGrid", "build_model_grid", "cell_centres", "perturb_grid"]

FloatArray = npt.NDArray[np.float64]


class HeterogeneitySpec(BaseModel):
    """``velocity.heterogeneity`` config section (requires ``grid.mode: 3d``)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    seed: int = 0
    structure: CorrelationSpec = CorrelationSpec()
    per_layer: bool = True
    vp_pct: float | list[float] = Field(0.0, description="Vp sd, %; scalar or per layer")
    vpvs_sd: float | list[float] = Field(0.0, description="Vp/Vs sd; scalar or per layer")
    interface_sd_km: float | list[float] = Field(
        0.0, description="interface relief sd, km; scalar or per interface"
    )
    interface_clip: float = Field(0.45, gt=0, lt=0.5)
    truncate_sd: float = Field(3.0, gt=0)
    pad_km: float | None = Field(None, gt=0, description="FFT padding; default 2x corr length")

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name in ("vp_pct", "vpvs_sd", "interface_sd_km"):
            values = getattr(self, name)
            values = values if isinstance(values, list) else [values]
            if any(v < 0 for v in values):
                raise ConfigError(f"velocity.heterogeneity.{name} must be >= 0")
        if self.enabled and not any(
            np.any(np.asarray(getattr(self, n)) > 0)
            for n in ("vp_pct", "vpvs_sd", "interface_sd_km")
        ):
            raise ConfigError("velocity.heterogeneity.enabled is true but every sd is zero")
        vp_max = np.max(np.asarray(self.vp_pct))
        if vp_max * self.truncate_sd >= 100:
            raise ConfigError("velocity.heterogeneity: vp_pct * truncate_sd must be < 100")
        return self

    def per_layer_values(self, name: str, n: int) -> FloatArray:
        """Amplitude ``name`` expanded to ``n`` entries (layers or interfaces).

        Raises
        ------
        ConfigError
            If a list has the wrong length.
        """
        v = getattr(self, name)
        if isinstance(v, list):
            if len(v) != n:
                raise ConfigError(
                    f"velocity.heterogeneity.{name} has {len(v)} values, model needs {n}"
                )
            return np.asarray(v, dtype=np.float64)
        return np.full(n, float(v))

    def check_unlayered(self) -> None:
        """Raise ``ConfigError`` for settings that need layers (used with a 3D model)."""
        for name in ("vp_pct", "vpvs_sd"):
            if isinstance(getattr(self, name), list):
                raise ConfigError(
                    f"velocity.heterogeneity.{name}: a 3D starting model has no layers; "
                    "give a single value"
                )
        if np.any(np.asarray(self.interface_sd_km) > 0):
            raise ConfigError(
                "velocity.heterogeneity.interface_sd_km: a 3D starting model has no interfaces"
            )

    def check_grid(self, geom: GridGeometry) -> None:
        """Raise ``ConfigError`` unless the grid is 3D and resolves the correlation lengths."""
        if geom.mode != "3d":
            raise ConfigError("velocity.heterogeneity needs grid.mode: 3d")
        shortest = min(self.structure.corr_len_km)
        if shortest < 2 * geom.dx:
            raise ConfigError(
                f"velocity.heterogeneity.structure.corr_len_km: {shortest} km is under two grid "
                f"cells ({geom.dx} km); use longer lengths or a finer grid"
            )


@dataclass(frozen=True)
class ModelGrid:
    """Vp and Vs (km/s) at cell centres, shape (nx, ny, nz), each cell's layer index and the
    background Vp (after interface relief and global scaling, before the random fields)."""

    vp: FloatArray
    vs: FloatArray
    layer: npt.NDArray[np.intp] | None = None
    vp_background: FloatArray | None = None


def cell_centres(geom: GridGeometry) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cell-centre coordinates (km) along x, y and z, as Vel2Grid samples them."""
    d = geom.dx

    def axis(origin: float, n: int) -> FloatArray:
        return origin + (np.arange(n, dtype=np.float64) + 0.5) * d

    return axis(geom.x0, geom.nx), axis(geom.y0, geom.ny), axis(geom.z0, geom.nz)


def build_model_grid(
    model: LayeredModel,
    geom: GridGeometry,
    spec: HeterogeneitySpec | None,
    index: int,
) -> ModelGrid:
    """3D Vp/Vs of ``model`` on ``geom``, with heterogeneity for realization ``index``.

    With ``spec`` ``None`` or disabled this is the plain layered model sampled exactly as
    Vel2Grid does (the caller passes ``None`` for the unperturbed base realization).
    """
    _, _, zc = cell_centres(geom)
    n_l = model.n_layers
    nx, ny, nz = geom.nx, geom.ny, geom.nz
    active = spec is not None and spec.enabled
    rng = np.random.default_rng([spec.seed if spec else 0, index])

    # Interface depths per column, shape (n_l, nx, ny); the model top never moves.
    tops = np.broadcast_to(model.tops_km[:, None, None], (n_l, nx, ny)).copy()
    if active and n_l > 1:
        assert spec is not None
        sds = spec.per_layer_values("interface_sd_km", n_l - 1)
        thick = np.diff(model.tops_km)
        for k in range(1, n_l):
            if sds[k - 1] == 0:
                continue
            relief = correlated_field((nx, ny), geom.dx, spec.structure, rng, spec.pad_km)
            relief = np.clip(relief, -spec.truncate_sd, spec.truncate_sd) * sds[k - 1]
            adjacent = thick[k - 1] if k == n_l - 1 else min(thick[k - 1], thick[k])
            limit = spec.interface_clip * adjacent
            tops[k] += np.clip(relief, -limit, limit)

    # Layer index of every cell and depth below its layer's (local) top.
    z = zc[None, None, :]
    layer = np.zeros((nx, ny, nz), dtype=np.intp)
    for k in range(1, n_l):
        layer += z >= tops[k][:, :, None]
    top_here = np.take_along_axis(tops.transpose(1, 2, 0), layer, axis=2)
    dz = z - top_here

    def prop(name: str) -> FloatArray:
        values = np.array([getattr(la, name) for la in model.layers])
        result: FloatArray = values[layer]
        return result

    vp = prop("vp") + prop("vp_grad") * dz
    vs = prop("vs") + prop("vs_grad") * dz
    if not active:
        return ModelGrid(vp, vs, layer)
    assert spec is not None
    background = vp.copy()
    grid = _apply_fields(vp, vs, layer, n_l, geom, spec, rng)
    return ModelGrid(grid.vp, grid.vs, layer, background)


def perturb_grid(
    vp: FloatArray,
    vs: FloatArray,
    geom: GridGeometry,
    spec: HeterogeneitySpec | None,
    index: int,
    vp_factor: float = 1.0,
    vpvs_delta: float = 0.0,
) -> ModelGrid:
    """Perturb a 3D starting model (no layers) for realization ``index``.

    ``vp_factor`` and ``vpvs_delta`` are the realization's global (1D-section) perturbation;
    heterogeneity, if enabled, uses a single field over the whole volume with scalar
    amplitudes. ``spec`` ``None`` (the base realization) applies only the global terms.

    Raises
    ------
    ConfigError
        If the spec uses layer-specific settings (lists, interface relief).
    """
    vp_new = vp * vp_factor
    vpvs = vp / vs + vpvs_delta
    if spec is None or not spec.enabled:
        np.maximum(vpvs, MIN_VPVS + 0.01, out=vpvs)
        return ModelGrid(vp_new, vp_new / vpvs)
    spec.check_unlayered()
    rng = np.random.default_rng([spec.seed, index])
    layer = np.zeros(vp.shape, dtype=np.intp)
    background = vp_new.copy()
    grid = _apply_fields(vp_new, vp_new / vpvs, layer, 1, geom, spec, rng)
    return ModelGrid(grid.vp, grid.vs, layer, background)


def _apply_fields(
    vp: FloatArray,
    vs: FloatArray,
    layer: npt.NDArray[np.intp],
    n_l: int,
    geom: GridGeometry,
    spec: HeterogeneitySpec,
    rng: np.random.Generator,
) -> ModelGrid:
    """Multiply Vp by ``1 + sd f`` and add ``sd g`` to Vp/Vs, per layer group (in place)."""
    nx, ny, _ = vp.shape
    vpvs = vp / vs
    vp_sd = spec.per_layer_values("vp_pct", n_l) / 100.0
    vpvs_sd = spec.per_layer_values("vpvs_sd", n_l)
    for sd, target, additive in ((vp_sd, vp, False), (vpvs_sd, vpvs, True)):
        if not np.any(sd > 0):
            continue
        groups = [[k] for k in range(n_l)] if spec.per_layer else [list(range(n_l))]
        for ks in groups:
            mask = np.isin(layer, ks)
            levels = np.flatnonzero(mask.any(axis=(0, 1)))
            if not np.any(sd[ks] > 0) or levels.size == 0:
                continue
            # Generate the field only over the depth slab this group occupies.
            z0, z1 = int(levels[0]), int(levels[-1]) + 1
            f = correlated_field((nx, ny, z1 - z0), geom.dx, spec.structure, rng, spec.pad_km)
            f = np.clip(f, -spec.truncate_sd, spec.truncate_sd)
            slab = mask[:, :, z0:z1]
            amp = sd[layer[:, :, z0:z1][slab]]
            view = target[:, :, z0:z1]
            if additive:
                view[slab] += amp * f[slab]
            else:
                view[slab] *= 1.0 + amp * f[slab]
    np.maximum(vpvs, MIN_VPVS + 0.01, out=vpvs)
    return ModelGrid(vp, vp / vpvs)
