"""
Tests for the observer → local solar vertical frame transformation.

Covers the algebraic invariants of the rotation, the sign/ordering conventions
that are easy to get silently wrong, agreement with sunpy-derived geometry, and
the atmosphere integration.
"""

import numpy as np
import pytest
import torch

from neural_me.core.geometry import (
    apply_local_vertical,
    heliocentric_angles,
    local_vertical_matrix,
    local_vertical_matrix_from_map,
    solar_north_axis,
)
from neural_me.core.atmospheres.stereoscopic import StereoscopicAtmosphere


sunpy = pytest.importorskip("sunpy", reason="sunpy needed for WCS-based tests")


@pytest.fixture
def angles():
    """A spread of image-aligned heliocentric angles, avoiding the poles."""
    rng = np.random.default_rng(1234)
    theta = rng.uniform(-1.2, 1.2, size=(7, 5))
    phi = rng.uniform(-1.2, 1.2, size=(7, 5))
    return theta, phi


# --------------------------------------------------------------------- #
#  Algebraic invariants
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("heliographic", [False, True])
def test_rotation_is_orthonormal_and_proper(angles, heliographic):
    """R R^T = I and det(R) = +1: a rotation, not a reflection."""
    theta, phi = angles
    north = np.array([0.0, np.cos(-0.115), np.sin(-0.115)]) if heliographic else None
    rot = local_vertical_matrix(theta, phi, north_axis=north)

    eye = np.einsum("...ij,...kj->...ik", rot, rot)
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_matches_closed_form_matrix(angles):
    """Image-aligned case reproduces the classical analytic matrix exactly."""
    theta, phi = angles
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    zero = np.zeros_like(theta)

    expected = np.stack(
        [
            np.stack([ct * cp, st, ct * sp], axis=-1),
            np.stack([-st * cp, ct, -st * sp], axis=-1),
            np.stack([-sp, zero, cp], axis=-1),
        ],
        axis=-2,
    )
    np.testing.assert_allclose(
        local_vertical_matrix(theta, phi), expected, atol=1e-12
    )


def test_disk_centre_is_identity():
    """At Theta = Phi = 0 the local vertical frame is the observer frame."""
    rot = local_vertical_matrix(np.array(0.0), np.array(0.0))
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)


def test_disk_centre_radial_sign():
    """B_r = +B_los at disk centre, pinning the LOS sign convention.

    A positive B_los means field pointing toward the observer, which at disk
    centre is outward along the local vertical.
    """
    rot = torch.eye(3)
    B_r, B_tn, B_tw = apply_local_vertical(
        torch.tensor(500.0), torch.tensor(0.0), torch.tensor(0.0), rot
    )
    assert B_r.item() == pytest.approx(500.0)
    assert B_tn.item() == pytest.approx(0.0)
    assert B_tw.item() == pytest.approx(0.0)


@pytest.mark.parametrize("heliographic", [False, True])
def test_norm_is_preserved(angles, heliographic, hinode_sp_map):
    """An orthonormal change of basis cannot change |B|."""
    theta, phi = angles
    north = solar_north_axis(hinode_sp_map((8, 8))) if heliographic else None
    rot = torch.as_tensor(local_vertical_matrix(theta, phi, north_axis=north))

    rng = np.random.default_rng(7)
    B = torch.as_tensor(rng.normal(0, 800, size=theta.shape + (3,)))
    B_r, B_tn, B_tw = apply_local_vertical(B[..., 0], B[..., 1], B[..., 2], rot)

    before = torch.linalg.norm(B, dim=-1)
    after = torch.sqrt(B_r**2 + B_tn**2 + B_tw**2)
    torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("heliographic", [False, True])
def test_purely_radial_field_has_no_tangential_part(angles, heliographic, hinode_sp_map):
    """A field along r-hat must rotate to (|B|, 0, 0)."""
    theta, phi = angles
    north = solar_north_axis(hinode_sp_map((8, 8))) if heliographic else None
    rot = local_vertical_matrix(theta, phi, north_axis=north)

    # Row 0 of R is r-hat expressed in (los, n, w); a field along it is radial.
    amplitude = 1234.0
    B = amplitude * rot[..., 0, :]
    B_r, B_tn, B_tw = apply_local_vertical(
        *[torch.as_tensor(B[..., i]) for i in range(3)], torch.as_tensor(rot)
    )

    torch.testing.assert_close(
        B_r, torch.full_like(B_r, amplitude), rtol=1e-10, atol=1e-8
    )
    torch.testing.assert_close(B_tn, torch.zeros_like(B_tn), rtol=0, atol=1e-8)
    torch.testing.assert_close(B_tw, torch.zeros_like(B_tw), rtol=0, atol=1e-8)


def test_round_trip_through_transpose(angles):
    """R^T undoes R, so the original LOS vector is recoverable."""
    theta, phi = angles
    rot = torch.as_tensor(local_vertical_matrix(theta, phi))

    rng = np.random.default_rng(11)
    B = torch.as_tensor(rng.normal(0, 500, size=theta.shape + (3,)))
    local = apply_local_vertical(B[..., 0], B[..., 1], B[..., 2], rot)
    back = apply_local_vertical(*local, rot.transpose(-1, -2))

    for i in range(3):
        torch.testing.assert_close(back[i], B[..., i], rtol=1e-10, atol=1e-8)


def test_transformation_is_differentiable(angles):
    """Gradients flow through the rotation (needed for a B_r regularizer)."""
    theta, phi = angles
    rot = torch.as_tensor(local_vertical_matrix(theta[:2, :2], phi[:2, :2]))
    B = [
        torch.randn(2, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)
    ]

    assert torch.autograd.gradcheck(
        lambda a, b, c: torch.stack(apply_local_vertical(a, b, c, rot), dim=-1),
        tuple(B),
    )


# --------------------------------------------------------------------- #
#  Geometry derived from the WCS
# --------------------------------------------------------------------- #


def test_hinode_map_reproduces_expected_pointing(hinode_sp_map):
    smap = hinode_sp_map((64, 64))
    assert smap.data.shape == (64, 64)
    assert smap.meta["CRVAL1"] == pytest.approx(511.0)
    assert smap.meta["CRVAL2"] == pytest.approx(329.0)
    # The Hinode SP field of view for this scan is well off disk centre.
    assert 0.6 < np.cos(np.deg2rad(40.0)) < 0.9


def test_heliocentric_angles_are_image_aligned_not_stonyhurst(hinode_sp_map):
    """Guards against the B0 trap: image-aligned latitude != Stonyhurst latitude."""
    import astropy.units as u
    from sunpy.coordinates import frames

    smap = hinode_sp_map((32, 32))
    theta, phi, on_disk = heliocentric_angles(smap)
    assert on_disk.all(), "this FOV should be entirely on disk"
    assert np.isfinite(theta).all() and np.isfinite(phi).all()

    xi, yi = np.meshgrid(np.arange(32), np.arange(32))
    hgs = smap.pixel_to_world(xi * u.pix, yi * u.pix).transform_to(
        frames.HeliographicStonyhurst
    )
    b0 = smap.observer_coordinate.lat.to_value(u.deg)

    # They differ by roughly the B0 angle, which is ~ -6.6 deg for this date.
    offset = np.rad2deg(theta) - hgs.lat.to_value(u.deg)
    assert b0 == pytest.approx(-6.57, abs=0.1)
    assert np.abs(np.mean(offset) - abs(b0)) < 1.5


def test_radial_row_matches_sunpy_position_vector(hinode_sp_map):
    """Row 0 of R must be the true unit position vector of each pixel."""
    import astropy.units as u
    from sunpy.coordinates import frames

    smap = hinode_sp_map((16, 16))
    rot, on_disk = local_vertical_matrix_from_map(smap, heliographic=True)

    xi, yi = np.meshgrid(np.arange(16), np.arange(16))
    hc = smap.pixel_to_world(xi * u.pix, yi * u.pix).transform_to(
        frames.Heliocentric(observer=smap.observer_coordinate, obstime=smap.date)
    )
    xyz = np.stack(
        [hc.x.to_value(u.R_sun), hc.y.to_value(u.R_sun), hc.z.to_value(u.R_sun)],
        axis=-1,
    )
    r_hat_xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    # rot columns are ordered (los, n, w) = xyz reversed.
    np.testing.assert_allclose(rot[..., 0, :], r_hat_xyz[..., ::-1], atol=1e-9)


def test_radial_field_identical_for_both_tangential_bases(hinode_sp_map):
    """The pole choice changes the transverse split but never B_r."""
    smap = hinode_sp_map((16, 16))
    rot_img, _ = local_vertical_matrix_from_map(smap, heliographic=False)
    rot_hgs, _ = local_vertical_matrix_from_map(smap, heliographic=True)

    np.testing.assert_allclose(rot_img[..., 0, :], rot_hgs[..., 0, :], atol=1e-12)
    # ...and the tangential rows genuinely differ (by roughly the B0 angle).
    assert np.abs(rot_img[..., 1, :] - rot_hgs[..., 1, :]).max() > 1e-3


def test_solar_north_axis_is_unit_and_tilted_by_b0(hinode_sp_map):
    import astropy.units as u

    smap = hinode_sp_map((8, 8))
    axis = solar_north_axis(smap)
    b0 = smap.observer_coordinate.lat.to_value(u.rad)

    assert np.linalg.norm(axis) == pytest.approx(1.0)
    np.testing.assert_allclose(
        axis, [0.0, np.cos(b0), np.sin(b0)], atol=1e-6
    )


def test_offdisk_pixels_are_masked_and_finite(hinode_sp_map):
    """Off-disk geometry is undefined; it must not leak NaN into the rotation."""
    smap = hinode_sp_map((32, 32), crval=(950.0, 0.0), cdelt=2.0)
    rot, on_disk = local_vertical_matrix_from_map(smap)

    assert not on_disk.all(), "this pointing should straddle the limb"
    assert np.isfinite(rot).all()
    offdisk = rot[~on_disk]
    np.testing.assert_allclose(
        offdisk, np.broadcast_to(np.eye(3), offdisk.shape), atol=1e-12
    )


# --------------------------------------------------------------------- #
#  Atmosphere integration
# --------------------------------------------------------------------- #


def test_atmosphere_raises_before_geometry_is_installed():
    atm = StereoscopicAtmosphere(grid_shape=(4, 4))
    assert not atm.has_local_vertical_geometry
    with pytest.raises(RuntimeError, match="No local vertical geometry"):
        atm.local_vertical_field()


def test_atmosphere_rejects_mismatched_grid(hinode_sp_map):
    atm = StereoscopicAtmosphere(grid_shape=(4, 4))
    with pytest.raises(ValueError, match="does not match"):
        atm.set_local_vertical_geometry_from_map(hinode_sp_map((8, 8)))


def test_atmosphere_local_vertical_field(hinode_sp_map):
    grid_shape = (8, 6)
    atm = StereoscopicAtmosphere(grid_shape=grid_shape)
    atm.set_local_vertical_geometry_from_map(hinode_sp_map(grid_shape))

    atm.set_parameters(
        {
            "global_Blos": np.full(grid_shape, 800.0),
            "global_Bn": np.full(grid_shape, -300.0),
            "global_Bw": np.full(grid_shape, 200.0),
        }
    )

    B_r, B_tn, B_tw = atm.local_vertical_field()
    assert B_r.shape == grid_shape

    # |B| is unchanged by the rotation.
    expected = np.sqrt(800.0**2 + 300.0**2 + 200.0**2)
    total = torch.sqrt(B_r**2 + B_tn**2 + B_tw**2)
    torch.testing.assert_close(
        total, torch.full(grid_shape, expected), rtol=1e-5, atol=1e-2
    )

    # Well off disk centre, B_r must differ appreciably from B_los.
    assert abs(B_r.mean().item() - 800.0) > 50.0

    stack = atm.to_local_vertical_numpy()
    assert stack.shape == grid_shape + (3,)


def test_geometry_is_a_buffer_not_a_parameter(hinode_sp_map):
    """Buffers follow .to(device) and are checkpointed, but are not optimized."""
    grid_shape = (4, 4)
    atm = StereoscopicAtmosphere(grid_shape=grid_shape)
    atm.set_local_vertical_geometry_from_map(hinode_sp_map(grid_shape))

    names = dict(atm.named_buffers())
    assert "lv_rotation" in names and "lv_on_disk" in names
    assert "lv_rotation" not in dict(atm.named_parameters())
    assert "lv_rotation" in atm.state_dict()
    assert not names["lv_rotation"].requires_grad


def test_gradients_reach_parameters_through_the_rotation(hinode_sp_map):
    """A loss on B_r must backpropagate to the raw Cartesian parameters."""
    grid_shape = (4, 4)
    atm = StereoscopicAtmosphere(grid_shape=grid_shape)
    atm.set_local_vertical_geometry_from_map(hinode_sp_map(grid_shape))

    B_r, _, _ = atm.local_vertical_field()
    B_r.square().mean().backward()

    for p in (atm.global_Blos_raw, atm.global_Bn_raw, atm.global_Bw_raw):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_reinstalling_geometry_overwrites_buffer(hinode_sp_map):
    grid_shape = (4, 4)
    atm = StereoscopicAtmosphere(grid_shape=grid_shape)
    atm.set_local_vertical_geometry_from_map(hinode_sp_map(grid_shape))
    first = atm.lv_rotation.clone()

    atm.set_local_vertical_geometry(
        np.broadcast_to(np.eye(3), grid_shape + (3, 3)).copy()
    )
    assert sorted(k for k in dict(atm.named_buffers()) if k.startswith("lv_")) == ["lv_on_disk", "lv_rotation"]
    assert not torch.allclose(first, atm.lv_rotation)
