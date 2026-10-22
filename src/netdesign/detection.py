"""Relative detectability: how likely each station is to pick each phase, compared with the
best candidate site for the same event.

No absolute magnitude or noise level is needed. For realization ``r``, station ``s``, phase
``p`` and target point ``j`` the relative amplitude is::

    A = c_p * amp_s / R * mean_f exp(-pi f (t* + kappa_s))

- ``R``: straight-line hypocentral distance (geometric spreading ``1/R``);
- ``t* = T * <1/Q>``: the NonLinLoc travel time times the slowness-weighted mean of
  ``1/Q(f)`` along the straight line, from that realization's 1D velocity/Q profile
  (``Q(f) = Q0 (f / fref) ** eta``); 3D heterogeneity of Q is not represented;
- ``kappa_s``, ``amp_s``: near-surface attenuation and amplification of the site type;
- ``c_p``: 1 for P, ``s_to_p_amplitude`` for S;
- ``f``: ``n_frequencies`` log-spaced frequencies across ``frequency_band_hz``; the source
  spectrum is taken as flat across the band (small events have high corner frequencies).

The relative signal-to-noise ``q = 20 log10(A) - noise_db_s`` is measured from a reference
arrival for that event: the ``reference_rank``-th strongest among the usable candidates
(default 4, the picks a location needs). The detection probability is
``Phi((q + margin_db) / width_db)``: 50 % when an arrival is ``margin_db`` below the reference.
``margin_db`` therefore plays the role of "how small an event are we designing for". Using a
rank rather than the single best keeps one exceptional site (e.g. a quiet borehole) from
setting the scale for every other station; arrivals stronger than the reference have
``q > 0``.

``reference: per_phase`` (default) takes the maximum separately for each phase, since P and
S are picked against their own noise (vertical vs horizontal components) and P onsets are
usually as pickable as the larger S arrivals. ``all_phases`` takes one maximum over both
phases, with S scaled by ``s_to_p_amplitude``; P is then typically ~14 dB down.

Magnitude mode (``mode: magnitude``)
------------------------------------
Absolute signal-to-noise for an event of moment magnitude ``magnitude``, so the design is
for the smallest event that should be located. The far-field ground-velocity Fourier
amplitude is (Brune source; Aki & Richards; Boore 2003)::

    V(f) = 2 pi f * rad_p M0 amp_s / (4 pi rho v_src^3 R) * 1 / (1 + (f / fc)^2)
           * exp(-pi f (t*(f) + kappa_s))

with ``M0 = 10^(1.5 Mw + 9.1)`` N m, ``fc = 0.4906 beta_src (stress_drop / M0)^(1/3)``
(``beta_src``, ``v_src`` the S and phase velocity at the source in the realization's
profile), ``rho = density_kg_m3`` and ``rad_p`` the average radiation coefficient (0.52 P,
0.63 S). The signal is compared with the site's noise (``noise_psd_db``, acceleration PSD,
dB rel. 1 (m/s^2)^2/Hz, flat across the band) over ``signal_window_s``::

    SNR = (2 int |V|^2 df / signal_window_s) / int noise_psd / (2 pi f)^2 df

both integrals over ``frequency_band_hz`` (trapezoid on ``n_frequencies`` log-spaced points).
The detection probability is ``Phi((10 log10 SNR - snr_threshold_db) / width_db)``. Same
straight-line ``1/R`` spreading and ``t*`` as the relative mode; no source directivity or
site resonance.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import ndtr

from netdesign.crs import WGS84, InputCRS
from netdesign.errors import ConfigError
from netdesign.sensitivity import Sensitivity
from netdesign.velocity import AttenuationSpec, LayeredModel

__all__ = ["DetectionSpec", "Profile", "SiteType", "profiles_from_outputs", "relative_detection"]

FloatArray = npt.NDArray[np.float64]

# depth (km) -> (vp, vs, qp, qs) on a depth grid
Profile = Callable[[FloatArray], tuple[FloatArray, FloatArray, FloatArray, FloatArray]]


class SiteType(BaseModel):
    """Site properties relative to other sites (placeholders unless measured)."""

    model_config = ConfigDict(extra="forbid")

    kappa_s: float = Field(0.03, ge=0, description="near-surface attenuation kappa0, s")
    amplification: float = Field(2.0, gt=0, description="site amplification (incl. free surface)")
    noise_db: float = Field(0.0, description="noise level relative to other site types, dB")
    noise_psd_db: float | None = Field(
        None, description="magnitude mode: noise acceleration PSD, dB rel. 1 (m/s^2)^2/Hz"
    )


def _default_site_types() -> dict[str, SiteType]:
    return {
        "surface": SiteType(noise_psd_db=-130.0),
        "borehole": SiteType(kappa_s=0.005, amplification=1.0, noise_db=-20.0, noise_psd_db=-150.0),
    }


class DetectionSpec(BaseModel):
    """``detection`` config section."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal["relative", "magnitude"] = "relative"
    frequency_band_hz: tuple[float, float] = (5.0, 30.0)
    n_frequencies: int = Field(5, ge=1)
    reference: Literal["per_phase", "all_phases"] = "per_phase"
    reference_rank: int = Field(4, ge=1, description="reference = k-th strongest candidate")
    s_to_p_amplitude: float = Field(
        5.0, gt=0, description="source S/P amplitude ratio (reference: all_phases only)"
    )
    margin_db: float = Field(20.0, gt=0, description="50 % detection this far below the best")
    width_db: float = Field(5.0, gt=0, description="transition width (1 sd), dB")
    site_types: dict[str, SiteType] = Field(default_factory=_default_site_types)
    default_site: str = "surface"
    sites: dict[str, str] = Field(default_factory=dict, description="station -> site type")
    margin_sensitivity_db: list[float] = Field(
        default_factory=lambda: [5.0, 10.0, 15.0], description="extra margins to test"
    )
    magnitude: float = Field(0.0, description="magnitude mode: smallest event to locate, Mw")
    magnitude_sensitivity: list[float] = Field(
        default_factory=lambda: [-0.5, 0.5, 1.0], description="extra magnitudes to test"
    )
    stress_drop_mpa: float = Field(1.0, gt=0)
    density_kg_m3: float = Field(2700.0, gt=0)
    radiation: dict[str, float] = Field(
        default_factory=lambda: {"P": 0.52, "S": 0.63}, description="average radiation"
    )
    snr_threshold_db: float = Field(10.0, description="50 % picking probability at this SNR")
    signal_window_s: float = Field(0.5, gt=0, description="window the SNR is measured over")

    @model_validator(mode="after")
    def _check(self) -> Self:
        lo, hi = self.frequency_band_hz
        if not 0 < lo <= hi:
            raise ConfigError(
                "detection.frequency_band_hz must be [low, high] with 0 < low <= high"
            )
        types = {self.default_site, *self.sites.values()}
        missing = sorted(types - set(self.site_types))
        if missing:
            raise ConfigError(f"detection: undefined site type(s) {missing}")
        if self.mode == "magnitude":
            no_psd = sorted(t for t in types if self.site_types[t].noise_psd_db is None)
            if no_psd:
                raise ConfigError(
                    f"detection.site_types {no_psd}: magnitude mode needs noise_psd_db"
                )
        return self

    @property
    def sensitivity_values(self) -> tuple[str, list[float], float]:
        """The design-event knob of this mode: key, values to test and the design value."""
        if self.mode == "magnitude":
            values = sorted({*self.magnitude_sensitivity, self.magnitude})
            return "magnitude", values, self.magnitude
        return "margin_db", sorted({*self.margin_sensitivity_db, self.margin_db}), self.margin_db


def relative_detection(
    sens: Sensitivity,
    station_xyz: FloatArray,
    profiles: list[Profile],
    eta: list[float],
    fref_hz: float,
    spec: DetectionSpec,
    usable: list[int] | None = None,
    margin_db: float | None = None,
    magnitude: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Detection probability and SNR (dB), both shaped (R, S, P, J).

    The SNR is relative to the reference arrival (relative mode) or to the 50 % threshold
    ``snr_threshold_db`` (magnitude mode).

    Parameters
    ----------
    station_xyz : (S, 3) local km, in ``sens.stations`` order.
    profiles : one per realization; returns ``(vp, vs, qp, qs)`` at the given depths.
    eta : frequency exponent per realization.
    usable : candidates the best-arrival reference is taken over (default all).
    margin_db : overrides ``spec.margin_db`` (relative mode).
    magnitude : overrides ``spec.magnitude`` (magnitude mode).
    """
    n_r, n_s, n_p, n_j = sens.T.shape
    if len(profiles) != n_r or len(eta) != n_r:
        raise ConfigError("detection: need one velocity/Q profile per realization")
    usable = list(range(n_s)) if usable is None else usable
    src = sens.sources.xyz
    d = src[None, :, :] - station_xyz[:, None, :]  # (S, J, 3)
    dist = np.maximum(np.linalg.norm(d, axis=-1), 0.05)

    # Straight-line sample depths (S, J, N) for the slowness-weighted mean of 1/Q.
    n_seg = 24
    frac = (np.arange(n_seg) + 0.5) / n_seg
    z_path = (
        station_xyz[:, None, None, 2]
        + (src[None, :, None, 2] - station_xyz[:, None, None, 2]) * frac
    )
    freqs = np.geomspace(*spec.frequency_band_hz, spec.n_frequencies)

    names = sens.stations
    site = [spec.site_types[spec.sites.get(n, spec.default_site)] for n in names]
    kappa = np.array([s.kappa_s for s in site])
    amp = np.array([s.amplification for s in site])
    absolute = spec.mode == "magnitude"
    if absolute:
        mw = spec.magnitude if magnitude is None else magnitude
        m0 = 10 ** (1.5 * mw + 9.1)
        df = _trapezoid_weights(freqs, spec.frequency_band_hz)
        psd = np.array([10 ** (s.noise_psd_db / 10) for s in site if s.noise_psd_db is not None])
        noise_power = psd * np.sum(df / (2 * np.pi * freqs) ** 2)  # (S,), (m/s)^2
        noise = 10 * np.log10(noise_power)
    else:
        noise = np.array([s.noise_db for s in site])
    phase_c = np.array(
        [
            1.0 if p == "P" or spec.reference == "per_phase" else spec.s_to_p_amplitude
            for p in sens.phases
        ]
    )

    level = np.empty((n_r, n_s, n_p, n_j))
    flat = z_path.ravel()
    for r in range(n_r):
        vp, vs, qp, qs = profiles[r](flat)
        if absolute:
            vp_src, vs_src, _, _ = profiles[r](src[:, 2])
            fc = 0.4906 * vs_src * 1000 * (spec.stress_drop_mpa * 1e6 / m0) ** (1 / 3)  # (J,)
        for ip, phase in enumerate(sens.phases):
            v = (vp if phase == "P" else vs).reshape(z_path.shape)
            q0 = (qp if phase == "P" else qs).reshape(z_path.shape)
            slow = 1.0 / v
            mean_inv_q0 = (slow / q0).sum(axis=-1) / slow.sum(axis=-1)  # (S, J), at fref
            t = sens.T[r, :, ip].astype(np.float64)  # (S, J)
            spectrum = np.zeros((n_s, n_j))
            if absolute:
                v_src = (vp_src if phase == "P" else vs_src) * 1000  # m/s
                rad = spec.radiation.get(phase, 0.6)
                scale = rad * m0 / (4 * np.pi * spec.density_kg_m3 * v_src**3)  # (J,)
                for f, w in zip(freqs, df, strict=True):
                    t_star = t * mean_inv_q0 * (f / fref_hz) ** (-eta[r])
                    u = (
                        scale
                        * amp[:, None]
                        / (dist * 1000)
                        / (1 + (f / fc) ** 2)
                        * np.exp(-np.pi * f * (t_star + kappa[:, None]))
                    )
                    spectrum += w * (2 * np.pi * f * u) ** 2
                signal_power = 2 * spectrum / spec.signal_window_s  # mean-square velocity
                level[r, :, ip] = 10 * np.log10(signal_power) - noise[:, None]
                continue
            for f in freqs:
                t_star = t * mean_inv_q0 * (f / fref_hz) ** (-eta[r])
                spectrum += np.exp(-np.pi * f * (t_star + kappa[:, None]))
            a = phase_c[ip] * amp[:, None] * (spectrum / len(freqs)) / dist
            level[r, :, ip] = 20 * np.log10(a) - noise[:, None]
    if absolute:
        q = level - spec.snr_threshold_db
        return ndtr(q / spec.width_db), q
    sub = level[:, usable]  # (R, S', P, J)
    if spec.reference == "per_phase":
        ranked = -np.sort(-sub, axis=1)  # strongest first, per realization, phase, event
        k = min(spec.reference_rank, ranked.shape[1]) - 1
        ref = ranked[:, k : k + 1]
    else:
        merged = sub.reshape(n_r, -1, n_j)
        ranked = -np.sort(-merged, axis=1)
        k = min(spec.reference_rank, ranked.shape[1]) - 1
        ref = ranked[:, k : k + 1][:, :, None, :]
    q = level - ref
    m = spec.margin_db if margin_db is None else margin_db
    return ndtr((q + m) / spec.width_db), q


def _trapezoid_weights(freqs: FloatArray, band: tuple[float, float]) -> FloatArray:
    """Integration weights over ``freqs`` (a single frequency gets the band width)."""
    if len(freqs) == 1:
        return np.array([max(band[1] - band[0], 1.0)])
    gaps = np.diff(freqs)
    w = np.zeros_like(freqs)
    w[:-1] += gaps / 2
    w[1:] += gaps / 2
    return w


def _layered_profile(model: LayeredModel) -> Profile:
    if not model.has_q:
        raise ConfigError("detection needs Q: set velocity.attenuation")

    def profile(z: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        return (
            model.velocity_at(z, "P"),
            model.velocity_at(z, "S"),
            model.q_at(z, "P"),
            model.q_at(z, "S"),
        )

    return profile


def profiles_from_outputs(
    out: Path,
    attenuation: AttenuationSpec | None,
    model_3d: Path | None,
    crs: InputCRS = WGS84,
) -> tuple[list[Profile], list[float], float]:
    """Per-realization profiles, eta and fref from ``out/models`` (and the 3D model)."""
    meta = json.loads((out / "models" / "ensemble.json").read_text())
    reals = meta["realizations"]
    fref = attenuation.fref_hz if attenuation is not None else 1.0
    eta = [float(r["eta"]) if r.get("eta") is not None else 0.0 for r in reals]
    if model_3d is None:
        profiles = [
            _layered_profile(LayeredModel.read(out / "models" / r["files"]["csv"])) for r in reals
        ]
        return profiles, eta, fref
    if attenuation is None or attenuation.model != "vs_scaled":
        raise ConfigError("detection with a 3D model needs velocity.attenuation (vs_scaled)")
    att = attenuation
    from netdesign.velocity.model3d import NodeModel

    nodes = NodeModel.read(model_3d, crs)
    z0 = nodes.depth_km
    vp0 = nodes.vp.mean(axis=(0, 1))
    r0 = (nodes.vp / nodes.vs).mean(axis=(0, 1))
    profiles = []
    for r in reals:
        f = r["params"].get("vp_factor", [1.0])[0]
        dv = r["params"].get("vpvs_delta", [0.0])[0]
        qf = r["params"].get("q_factor", [1.0])[0]
        qr = r["params"].get("qpqs_factor", [1.0])[0]

        def profile(
            z: FloatArray, f: float = f, dv: float = dv, qf: float = qf, qr: float = qr
        ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
            vp = f * np.interp(z, z0, vp0)
            vs = vp / (np.interp(z, z0, r0) + dv)
            qs = att.qs_per_vs * vs * qf
            return vp, vs, att.qp_qs_ratio * qs * qr, qs

        profiles.append(profile)
    return profiles, eta, fref
