"""Target volume sampling."""

import numpy as np
import pytest

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
