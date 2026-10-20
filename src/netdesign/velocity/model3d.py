"""3D starting velocity model given as a CSV of nodes on a regular grid.

CSV header (any column order), horizontal columns as for the config's ``crs``::

    long,lat,depth_km,vp,vs            (geographic crs; or vpvs instead of vs)
    easting,northing,depth_km,vp,vs    (projected crs)

- ``depth_km``: km below mean sea level (negative above).
- One row per node of a full regular grid in the input crs (every combination of the unique
  horizontal and ``depth_km`` values, in any row order), as most tomography codes can export.

The model is sampled at the design grid's cell centres by trilinear interpolation in the
input crs coordinates and depth. Outside the model's extent the nearest edge value is used
(common for tomography models that resolve only the core of a field); :func:`sample_model`
reports the fraction of cells that lie inside the extent.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from netdesign.crs import WGS84, InputCRS
from netdesign.errors import DataError
from netdesign.frame import GridGeometry, LocalFrame
from netdesign.velocity.heterogeneity import cell_centres
from netdesign.velocity.layered import MIN_VPVS

__all__ = ["NodeModel", "SampledModel", "sample_model"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class NodeModel:
    """Vp, Vs (km/s) of shape (n_x, n_y, n_depth) on sorted axes; ``x``/``y`` are the
    horizontal coordinates in ``crs`` (long/lat or easting/northing)."""

    x: FloatArray
    y: FloatArray
    depth_km: FloatArray
    vp: FloatArray
    vs: FloatArray
    crs: InputCRS = WGS84

    @classmethod
    def read(cls, path: str | Path, crs: InputCRS = WGS84) -> NodeModel:
        """Read and validate the node CSV.

        Raises
        ------
        DataError
            On missing columns, unparseable values, an incomplete or duplicated grid, or
            non-physical velocities.
        """
        path = Path(path)
        try:
            with path.open(newline="") as fh:
                reader = csv.DictReader(line for line in fh if not line.startswith("#"))
                columns = {c.strip() for c in reader.fieldnames or ()}
                rows = list(reader)
        except OSError as exc:
            raise DataError(f"{path}: cannot read 3D model ({exc})") from exc
        ax, ay = crs.axes
        need = {ax, ay, "depth_km", "vp"}
        if not need <= columns or not ({"vs", "vpvs"} & columns):
            raise DataError(
                f"{path}: need columns {ax},{ay},depth_km,vp and vs (or vpvs) for {crs.name}; "
                f"got {sorted(columns)}"
            )
        second = "vs" if "vs" in columns else "vpvs"
        try:
            data = np.array(
                [[float(r[k].strip()) for k in (ax, ay, "depth_km", "vp", second)] for r in rows]
            )
        except (ValueError, AttributeError) as exc:
            raise DataError(f"{path}: bad value ({exc})") from exc
        if data.size == 0:
            raise DataError(f"{path}: no nodes")

        axes = [np.unique(data[:, i]) for i in range(3)]
        shape = tuple(len(a) for a in axes)
        if min(shape) < 2:
            raise DataError(f"{path}: need at least 2 nodes along {ax}, {ay} and depth")
        if len(data) != shape[0] * shape[1] * shape[2]:
            raise DataError(
                f"{path}: {len(data)} rows but the unique long/lat/depth values imply "
                f"{shape[0]}x{shape[1]}x{shape[2]} nodes; the nodes must form a full regular grid"
            )
        idx = tuple(np.searchsorted(a, data[:, i]) for i, a in enumerate(axes))
        filled = np.zeros(shape, dtype=bool)
        filled[idx] = True
        if not filled.all():
            raise DataError(f"{path}: duplicate nodes (the grid is incomplete)")
        vp = np.empty(shape)
        other = np.empty(shape)
        vp[idx] = data[:, 3]
        other[idx] = data[:, 4]
        vs = other if second == "vs" else vp / other
        if np.any(vp <= 0) or np.any(vs <= 0):
            raise DataError(f"{path}: velocities must be positive")
        if np.any(vp / vs < MIN_VPVS):
            raise DataError(f"{path}: Vp/Vs below {MIN_VPVS:.3f} at some nodes")
        return cls(axes[0], axes[1], axes[2], vp, vs, crs)


@dataclass(frozen=True)
class SampledModel:
    """The model at the design grid's cell centres and the fraction inside its extent."""

    vp: FloatArray
    vs: FloatArray
    coverage: float


def sample_model(model: NodeModel, frame: LocalFrame, geom: GridGeometry) -> SampledModel:
    """Interpolate ``model`` at the cell centres of ``geom`` (nearest edge outside)."""
    xc, yc, zc = cell_centres(geom)
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    ma, mb = model.crs.from_wgs84(*frame.to_geographic(xx, yy))  # cell centres in model crs
    inside_h = (ma >= model.x[0]) & (ma <= model.x[-1]) & (mb >= model.y[0]) & (mb <= model.y[-1])
    inside_z = (zc >= model.depth_km[0]) & (zc <= model.depth_km[-1])
    coverage = float(inside_h.mean() * inside_z.mean())

    a_c = np.clip(ma, model.x[0], model.x[-1])
    b_c = np.clip(mb, model.y[0], model.y[-1])
    z_c = np.clip(zc, model.depth_km[0], model.depth_km[-1])
    pts = np.empty((geom.nx, geom.ny, geom.nz, 3))
    pts[..., 0] = a_c[:, :, None]
    pts[..., 1] = b_c[:, :, None]
    pts[..., 2] = z_c[None, None, :]
    axes = (model.x, model.y, model.depth_km)
    vp = RegularGridInterpolator(axes, model.vp)(pts)
    vs = RegularGridInterpolator(axes, model.vs)(pts)
    return SampledModel(vp, vs, coverage)
