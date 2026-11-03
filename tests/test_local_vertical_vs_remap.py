"""
Cross-validation of :mod:`neural_me.core.geometry` against :mod:`ISPy.util.remap`.

``ISPy.util.remap.vector_transformation`` is the Gary & Hagyard (1990) matrix as coded in
Sun (2013), Eq. 7 (https://arxiv.org/pdf/1309.2392.pdf) — the same
transformation the SHARP pipeline uses.  It is an independent, long-trusted
implementation, so agreeing with it is the strongest available check on the new
code.

The two parameterise the pixel differently: ``remap`` takes heliographic
``(BB, LL)`` plus ``B0``, while ``geometry`` takes the image-aligned
``(theta, phi)``.  These tests bridge them through the radial unit vector and
compare the resulting rotation matrices element by element.

Conventions established here (and relied on by the rest of the code):

* ``B_r``   == ``remap``'s ``field_z_h``  (radial, positive outward)
* ``B_tW``  == ``remap``'s ``field_x_h``  (zonal, positive west)
* ``B_tN``  == **minus** ``remap``'s ``field_y_h``

The last one is not a bug in either implementation: ``remap`` returns the
SHARP/Sun ``B_theta`` convention, which is positive toward the **south**.
``B_tN`` is positive toward the north, as its name says.  Anything comparing
these outputs against SHARP ``Bt`` products must flip the sign.
"""

import numpy as np
import pytest

from neural_me.core.geometry import local_vertical_matrix

remap = pytest.importorskip(
    "ISPy.util.remap", reason="cross-validation against Gary & Hagyard needs ISPy"
)


# Heliographic (latitude, longitude-from-central-meridian) sample points, in
# degrees.  Includes disk centre, the actual Hinode/SP target for this dataset
# (~14.5N, 33.4W), and points either side of the meridian and equator.
SAMPLE_POINTS = [
    (0.0, 0.0),
    (14.5, 33.4),
    (-25.0, -40.0),
    (60.0, 10.0),
    (-5.0, 65.0),
]

B0_2022_03_31 = np.deg2rad(-6.57)


def gary_hagyard_matrix(lat, lon, b0, peff):
    """Recover ``remap``'s 3x3 matrix by pushing unit vectors through it.

    Returns the matrix in ``xyz`` = (West, North, toward-observer) ordering for
    both input and output, with rows ``(field_x_h, field_y_h, field_z_h)`` —
    i.e. including the internal ``field_y_h *= -1`` flip.
    """
    lat_arr = np.array([[lat]])
    lon_arr = np.array([[lon]])
    columns = []
    for unit in np.eye(3):
        fx, fy, fz = remap.vector_transformation(
            peff, lat_arr, lon_arr, b0, unit[0], unit[1], unit[2], lat_in_rad=True
        )
        columns.append([np.ravel(fx)[0], np.ravel(fy)[0], np.ravel(fz)[0]])
    return np.array(columns).T


def radial_unit_vector(lat, lon, b0):
    """Heliographic (lat, lon) → unit position vector in observer ``xyz``."""
    return np.array(
        [
            np.cos(lat) * np.sin(lon),
            np.sin(lat) * np.cos(b0) - np.cos(lat) * np.cos(lon) * np.sin(b0),
            np.sin(lat) * np.sin(b0) + np.cos(lat) * np.cos(lon) * np.cos(b0),
        ]
    )


def geometry_matrix_xyz(lat, lon, b0, roll=0.0):
    """Our matrix for the same pixel, re-expressed in ``xyz`` column ordering."""
    r_hat = radial_unit_vector(lat, lon, b0)
    theta = np.arcsin(np.clip(r_hat[1], -1.0, 1.0))
    phi = np.arctan2(r_hat[0], r_hat[2])
    north = np.array([0.0, np.cos(b0), np.sin(b0)])
    rot = local_vertical_matrix(
        np.array(theta), np.array(phi), north_axis=north, roll=roll
    )
    return rot[..., ::-1]  # columns lnw -> xyz


@pytest.mark.parametrize("lat_deg,lon_deg", SAMPLE_POINTS)
def test_radial_row_matches_remap(lat_deg, lon_deg):
    """B_r is identical to remap's field_z_h — no convention ambiguity."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    reference = gary_hagyard_matrix(lat, lon, B0_2022_03_31, 0.0)
    ours = geometry_matrix_xyz(lat, lon, B0_2022_03_31)
    np.testing.assert_allclose(ours[0], reference[2], atol=1e-12)


@pytest.mark.parametrize("lat_deg,lon_deg", SAMPLE_POINTS)
def test_zonal_row_matches_remap(lat_deg, lon_deg):
    """B_tW is identical to remap's field_x_h (both positive west)."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    reference = gary_hagyard_matrix(lat, lon, B0_2022_03_31, 0.0)
    ours = geometry_matrix_xyz(lat, lon, B0_2022_03_31)
    np.testing.assert_allclose(ours[2], reference[0], atol=1e-12)


@pytest.mark.parametrize("lat_deg,lon_deg", SAMPLE_POINTS)
def test_meridional_row_is_remap_negated(lat_deg, lon_deg):
    """B_tN is north-positive; remap's field_y_h is south-positive (SHARP Bt)."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    reference = gary_hagyard_matrix(lat, lon, B0_2022_03_31, 0.0)
    ours = geometry_matrix_xyz(lat, lon, B0_2022_03_31)
    np.testing.assert_allclose(ours[1], -reference[1], atol=1e-12)

    # And the sign really is the only difference: never accidentally equal.
    assert not np.allclose(ours[1], reference[1], atol=1e-6)


@pytest.mark.parametrize("b0_deg", [-7.2, 0.0, 3.5, 7.2])
@pytest.mark.parametrize("lat_deg,lon_deg", SAMPLE_POINTS)
def test_agreement_holds_across_b0(lat_deg, lon_deg, b0_deg):
    """The bridge between the two parameterisations holds for any B0."""
    lat, lon, b0 = np.deg2rad(lat_deg), np.deg2rad(lon_deg), np.deg2rad(b0_deg)
    reference = gary_hagyard_matrix(lat, lon, b0, 0.0)
    ours = geometry_matrix_xyz(lat, lon, b0)
    expected = np.stack([reference[2], -reference[1], reference[0]])
    np.testing.assert_allclose(ours, expected, atol=1e-12)


@pytest.mark.parametrize("roll_deg", [-12.0, 0.0, 3.8, 45.0])
def test_roll_matches_remap_p_angle(roll_deg):
    """Our `roll` is exactly Gary & Hagyard's p-angle (`peff`)."""
    lat, lon = np.deg2rad(14.5), np.deg2rad(33.4)
    roll = np.deg2rad(roll_deg)
    reference = gary_hagyard_matrix(lat, lon, B0_2022_03_31, roll)
    ours = geometry_matrix_xyz(lat, lon, B0_2022_03_31, roll=roll)
    expected = np.stack([reference[2], -reference[1], reference[0]])
    np.testing.assert_allclose(ours, expected, atol=1e-12)


@pytest.mark.parametrize("roll_deg", [-12.0, 3.8, 45.0])
def test_roll_leaves_radial_field_invariant(roll_deg):
    """Rolling the transverse reference direction cannot change B_r.

    A rotation about the LOS preserves the LOS component, and B_r depends on
    the transverse components only through their projection onto r-hat, which
    the roll rotates within the transverse plane.  So B_r changes -- unless the
    field itself is rolled with the frame.  This checks the physical statement:
    the same *physical* field gives the same B_r however the frame is labelled.
    """
    lat, lon = np.deg2rad(14.5), np.deg2rad(33.4)
    roll = np.deg2rad(roll_deg)

    rng = np.random.default_rng(3)
    b_solar = rng.normal(0, 700, size=3)  # physical field in the lnw solar frame

    # Express the same field in a detector frame rolled by `roll`, then let the
    # transform undo it.
    c, s = np.cos(roll), np.sin(roll)
    b_det = np.array(
        [b_solar[0], c * b_solar[1] + s * b_solar[2], -s * b_solar[1] + c * b_solar[2]]
    )

    plain = local_vertical_matrix(
        *_angles(lat, lon, B0_2022_03_31), north_axis=_north(B0_2022_03_31)
    )
    rolled = local_vertical_matrix(
        *_angles(lat, lon, B0_2022_03_31),
        north_axis=_north(B0_2022_03_31),
        roll=roll,
    )
    np.testing.assert_allclose(rolled @ b_det, plain @ b_solar, atol=1e-9)


def _angles(lat, lon, b0):
    r_hat = radial_unit_vector(lat, lon, b0)
    return (
        np.array(np.arcsin(np.clip(r_hat[1], -1.0, 1.0))),
        np.array(np.arctan2(r_hat[0], r_hat[2])),
    )


def _north(b0):
    return np.array([0.0, np.cos(b0), np.sin(b0)])


def test_image_aligned_basis_equals_heliographic_at_zero_b0():
    """With B0 = 0 the rotation axis is projected north, so both agree."""
    lat, lon = np.deg2rad(14.5), np.deg2rad(33.4)
    theta, phi = _angles(lat, lon, 0.0)
    image_aligned = local_vertical_matrix(theta, phi)
    heliographic = local_vertical_matrix(theta, phi, north_axis=_north(0.0))
    np.testing.assert_allclose(image_aligned, heliographic, atol=1e-12)


def test_remap_disagrees_with_image_aligned_basis_at_nonzero_b0():
    """Guards the B0 trap: the image-aligned split is *not* the SHARP split."""
    lat, lon = np.deg2rad(14.5), np.deg2rad(33.4)
    theta, phi = _angles(lat, lon, B0_2022_03_31)
    image_aligned = local_vertical_matrix(theta, phi)
    reference = gary_hagyard_matrix(lat, lon, B0_2022_03_31, 0.0)

    # Radial row still agrees exactly...
    np.testing.assert_allclose(
        image_aligned[..., ::-1][0], reference[2], atol=1e-12
    )
    # ...but the tangential rows are rotated by roughly B0.
    assert np.abs(image_aligned[..., ::-1][2] - reference[0]).max() > 1e-3
