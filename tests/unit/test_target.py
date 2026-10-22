"""Target volume sampling."""

import numpy as np
import pytest

from netdesign.crs import WGS84, InputCRS
from netdesign.errors import ConfigError
from netdesign.frame import LocalFrame
from netdesign.target import TargetSpec, sample_sources

FRAME = LocalFrame(-38.6, 176.13)
BOX = {"long": [176.11, 176.15], "lat": [-38.62, -38.59], "depth_km": [0.5, 4.0]}


def test_frame_round_trip() -> None:
    x, y = FRAME.to_local([176.11, 176.2], [-38.62, -38.5])
    lon, lat = FRAME.to_geographic(x, y)
    np.testing.assert_allclose(lon, [176.11, 176.2], atol=1e-9)
    np.testing.assert_allclose(lat, [-38.62, -38.5], atol=1e-9)


def test_grid_sampling_inside_box_and_regular() -> None:
    src = sample_sources(TargetSpec.model_validate({**BOX, "spacing_km": 0.25}), FRAME)
    assert np.all((src.long >= 176.11) & (src.long <= 176.15))
    assert np.all((src.lat >= -38.62) & (src.lat <= -38.59))
    z = np.unique(src.xyz[:, 2])
    np.testing.assert_allclose(np.diff(z), 0.25)
    assert z.min() == pytest.approx(0.625) and z.max() < 4.0
    # 0.04 deg long x 0.03 deg lat ~ 3.48 km x 3.33 km -> about 13-14 x 13 columns, 14 levels
    assert 13 * 13 * 14 <= len(src) <= 14 * 14 * 14
    np.testing.assert_allclose(src.weights.sum(), 1.0)
    np.testing.assert_allclose(src.weights, 1 / len(src))


def test_random_sampling_count_and_seed() -> None:
    spec = TargetSpec.model_validate({**BOX, "sampling": "random", "n_points": 500, "seed": 3})
    a, b = sample_sources(spec, FRAME), sample_sources(spec, FRAME)
    assert len(a) == 500
    np.testing.assert_array_equal(a.xyz, b.xyz)
    assert np.all((a.xyz[:, 2] >= 0.5) & (a.xyz[:, 2] <= 4.0))


def test_normal_depth_weight() -> None:
    spec = TargetSpec.model_validate(
        {**BOX, "depth_weight": {"model": "normal", "mean_km": 2.5, "sd_km": 0.5}}
    )
    src = sample_sources(spec, FRAME)
    z_best = src.xyz[np.argmax(src.weights), 2]
    assert abs(z_best - 2.5) <= 0.125
    np.testing.assert_allclose(src.weights.sum(), 1.0)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"long": [176.15, 176.11]}, "target.long"),
        ({"depth_weight": {"model": "normal"}}, "mean_km"),
    ],
)
def test_bad_target(changes: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        TargetSpec.model_validate({**BOX, **changes})


def test_spacing_larger_than_box() -> None:
    spec = TargetSpec.model_validate({**BOX, "spacing_km": 50.0})
    with pytest.raises(ConfigError, match="no source points"):
        sample_sources(spec, FRAME)


def _rotated(**kw: object) -> dict:
    box = {"centre": {"lat": -38.6, "long": 176.13}, "size_km": [1.0, 2.0], "depth_km": [1, 3]}
    return box | kw


def test_rotated_box_points_inside_and_aligned() -> None:
    spec = TargetSpec.model_validate(
        {"boxes": [_rotated(rotation_deg=30.0)], "spacing_km": 0.25, "depth_spacing_km": 0.5}
    )
    src = sample_sources(spec, FRAME)
    assert len(src) == 4 * 8 * 4  # 1 km x 2 km at 0.25 km, 2 km depth at 0.5 km
    t = np.radians(30.0)
    u = src.xyz[:, 0] * np.cos(t) - src.xyz[:, 1] * np.sin(t)  # across
    v = src.xyz[:, 0] * np.sin(t) + src.xyz[:, 1] * np.cos(t)  # along
    assert np.abs(u).max() < 0.5 and np.abs(v).max() < 1.0
    np.testing.assert_allclose(np.unique(np.round(v, 6)), np.arange(-0.875, 1.0, 0.25))
    poly = spec.polygons(FRAME, WGS84)[0]
    assert poly.shape == (4, 2)


def test_boxes_share_events_by_volume_or_weight() -> None:
    small = _rotated(size_km=[1.0, 1.0])
    big = _rotated(size_km=[2.0, 2.0], centre={"lat": -38.65, "long": 176.13})
    by_volume = sample_sources(TargetSpec.model_validate({"boxes": [small, big]}), FRAME)
    near = by_volume.xyz[:, 1] > -2.5
    assert by_volume.weights[near].sum() == pytest.approx(0.2)  # 1 : 4 in area
    weighted = TargetSpec.model_validate(
        {"boxes": [small | {"weight": 3.0}, big | {"weight": 1.0}]}
    )
    src = sample_sources(weighted, FRAME)
    assert src.weights[src.xyz[:, 1] > -2.5].sum() == pytest.approx(0.75)


def test_boxes_mix_ranges_and_default_depth() -> None:
    spec = TargetSpec.model_validate(
        {
            "boxes": [{"long": [176.11, 176.12], "lat": [-38.62, -38.61]}, _rotated()],
            "depth_km": [0.5, 1.5],
            "sampling": "random",
            "n_points": 300,
        }
    )
    src = sample_sources(spec, FRAME)
    assert abs(len(src) - 300) <= 2
    assert src.xyz[:, 2].max() <= 3.0 and src.xyz[:, 2].min() >= 0.5
    assert len(spec.polygons(FRAME, WGS84)) == 2


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({**BOX, "boxes": [_rotated()]}, "either one box"),
        ({"boxes": [_rotated(depth_km=None)]}, "depth_km"),
        ({"boxes": [_rotated(weight=1.0), _rotated()]}, "weight for every box"),
        (
            {
                "boxes": [{"long": [176.1, 176.2], "lat": [-38.7, -38.6], "rotation_deg": 10}],
                "depth_km": [1, 2],
            },
            "rotation_deg",
        ),
        ({"boxes": [_rotated(size_km=None)]}, "centre and size_km"),
    ],
)
def test_bad_boxes(raw: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        TargetSpec.model_validate(raw)


def test_box_crs_checked() -> None:
    spec = TargetSpec.model_validate({"boxes": [_rotated()]})
    with pytest.raises(ConfigError, match=r"boxes\[0\]\.centre"):
        spec.check_crs(InputCRS(2193))
