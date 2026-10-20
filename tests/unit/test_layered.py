"""Layered model: validation, profiles and I/O."""

from pathlib import Path

import numpy as np
import pytest

from netdesign.errors import DataError, PhysicsError
from netdesign.velocity import AttenuationSpec, Layer, LayeredModel, apply_attenuation

NLL_TEXT = """\
CONTROL 1 54321
# comment
LAYER  -0.5 2.0 0.8   1.1 0.45  2.2 0.0
LAYER   1.0 4.0 0.0   2.3 0.0   2.6 0.0  # trailing comment
VGGRID 10 10 10 0 0 0 0.1 0.1 0.1 SLOW_LEN
"""


def test_from_nll_ignores_other_lines() -> None:
    model = LayeredModel.from_nll(NLL_TEXT)
    assert model.n_layers == 2
    assert model.layers[0] == Layer(-0.5, 2.0, 1.1, 2.2, 0.8, 0.45, 0.0)
    assert not model.has_q


def test_nll_round_trip() -> None:
    model = LayeredModel.from_nll(NLL_TEXT)
    assert LayeredModel.from_nll(model.to_nll()) == model


def test_bad_layer_line_names_line() -> None:
    with pytest.raises(DataError, match=":2:"):
        LayeredModel.from_nll("LAYER 0 4 0 2.3 0 2.6 0\nLAYER 1 x 0 2 0 2 0\n")


def test_no_layers() -> None:
    with pytest.raises(DataError, match="no LAYER"):
        LayeredModel.from_nll("CONTROL 1 1\n")


def test_tops_must_increase() -> None:
    with pytest.raises(PhysicsError, match="increasing"):
        LayeredModel((Layer(0.0, 4.0, 2.3), Layer(0.0, 5.0, 2.9)))


def test_vpvs_checked_at_layer_base() -> None:
    # Vp/Vs = 1.6 at top but Vs gradient drives it below sqrt(4/3) by the base at z=1.
    with pytest.raises(PhysicsError, match="Vp/Vs"):
        LayeredModel((Layer(0.0, 4.0, 2.5, vs_grad=1.0), Layer(1.0, 5.0, 2.9)))


def test_partial_q_rejected() -> None:
    with pytest.raises(PhysicsError, match="every layer"):
        LayeredModel((Layer(0.0, 4.0, 2.3, qp=100, qs=50), Layer(1.0, 5.0, 2.9)))


def test_velocity_profile_with_gradient() -> None:
    model = LayeredModel.from_nll(NLL_TEXT)
    v = model.velocity_at([-1.0, -0.5, 0.5, 1.0, 5.0], "P")
    np.testing.assert_allclose(v, [1.6, 2.0, 2.8, 4.0, 4.0])
    np.testing.assert_allclose(model.velocity_at(0.5, "S"), 1.1 + 0.45)


def test_q_at_requires_q() -> None:
    with pytest.raises(PhysicsError, match="no Q"):
        LayeredModel.from_nll(NLL_TEXT).q_at(0.0)


def test_csv_round_trip_with_q(tmp_path: Path) -> None:
    model = apply_attenuation(LayeredModel.from_nll(NLL_TEXT), AttenuationSpec())
    path = tmp_path / "m.csv"
    model.write(path)
    back = LayeredModel.read(path)
    assert back.has_q
    for a, b in zip(back.layers, model.layers, strict=True):
        assert a.qs == pytest.approx(b.qs)
        assert a.vp_grad == pytest.approx(b.vp_grad)


def test_csv_minimal_columns(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text("# header comment\ntop_km,vp,vs\n0,4.0,2.3\n1,5.0,2.9\n")
    model = LayeredModel.read(path)
    assert model.layers[1].rho == 2.7
    assert not model.has_q


def test_csv_unknown_column(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text("top_km,vp,vs,qq\n0,4.0,2.3,1\n")
    with pytest.raises(DataError, match="unknown columns"):
        LayeredModel.read(path)
