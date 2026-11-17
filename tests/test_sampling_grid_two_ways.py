"""The two ways to the sampling grid of SparseWarpOperator: from the two WCS (sampling_grid_from_wcs) and, without WCS,
from a mapping fitted to two images (sampling_grid_from_images).

The synthetic image tests build the instrument image by sampling a textured model image at a known transform of its
pixels (similar images), so the fitted mapping, the grid and the operator can be checked against the truth. The last
test runs the default and the 'nodes' refinement on the 2022-03-31 SO/PHI-HRT and Hinode/SP data (different views,
~100 deg apart) against the WCS mapping in use."""
import contextlib
import io
import os

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

cv2 = pytest.importorskip("cv2")
from neural_me.core.coregistration import sampling_grid_from_images, save_sampling_grid
from neural_me.optimization import SparseWarpOperator


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def textured(shape, seed=0):
    """Structure at several scales (granulation-like texture, a few dark blobs) for ORB to find corners."""
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.normal(size=shape), 2.0) + 0.5 * gaussian_filter(rng.normal(size=shape), 6.0)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    for cy, cx, r in rng.uniform([0, 0, 6], [shape[0], shape[1], 14], (8, 3)):
        img -= 1.5 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r ** 2))
    return img


def apply(T, x, y):
    w = T[2, 0] * x + T[2, 1] * y + T[2, 2]
    return (T[0, 0] * x + T[0, 1] * y + T[0, 2]) / w, (T[1, 0] * x + T[1, 1] * y + T[1, 2]) / w


def transform(scale_x, scale_y, angle_deg, shift, persp=(0.0, 0.0)):
    """Instrument pixels -> model pixels: rotation, anisotropic scale (foreshortening), shift, perspective terms."""
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([scale_x, scale_y])
    return np.array([[R[0, 0], R[0, 1], shift[0]], [R[1, 0], R[1, 1], shift[1]], [persp[0], persp[1], 1.0]])


def instrument_image(model_img, T, shape):
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    mx, my = apply(T, x, y)
    return map_coordinates(model_img, [my, mx], order=3, mode="constant", cval=np.nan)


PROJ = transform(1.6, 1.0, 12.0, (40.0, 30.0), persp=(2e-4, -1e-4))


def distance_to(T, g, shape_model):
    y, x = np.mgrid[0:g["grid"].shape[0], 0:g["grid"].shape[1]].astype(float)
    tx, ty = apply(T, x, y)
    px = (g["grid"][..., 0] + 1) / 2 * (shape_model[1] - 1)
    py = (g["grid"][..., 1] + 1) / 2 * (shape_model[0] - 1)
    return np.hypot(px - tx, py - ty)[g["valid"]]


@pytest.mark.parametrize("model, T", [("projective", PROJ), ("affine", transform(1.4, 0.9, -8.0, (35.0, 50.0)))])
def test_similar_images_ecc_recovers_the_transform(model, T):
    model_img = textured((320, 320))
    inst = instrument_image(model_img, T, (150, 140))
    g = sampling_grid_from_images(inst, model_img, model=model, stretches=[(1, 0)])        # refine='ecc' (default)
    d = distance_to(T, g, (320, 320))
    assert g["valid"].mean() > 0.9 and g["model_shape"] == (320, 320)
    assert d.max() < 0.05                                                # model pixels; measured 0.016 / 0.015
    assert np.allclose(g["footprint"], np.linalg.svd(T[:2, :2], compute_uv=False), rtol=0.05)


def test_default_with_the_stretch_search():
    model_img = textured((320, 320))
    inst = instrument_image(model_img, PROJ, (150, 140))
    g = sampling_grid_from_images(inst, model_img)
    assert distance_to(PROJ, g, (320, 320)).max() < 0.05                # measured 0.026
    assert g["refine"] == "ecc" and len(g["stretches"]) == 4 and g["n_inliers"] >= 20


def test_operator_reproduces_the_instrument_image(tmp_path):
    model_img = textured((320, 320), seed=1)
    inst = instrument_image(model_img, PROJ, (150, 140))
    g = sampling_grid_from_images(inst, model_img, stretches=[(1, 0)])
    op = quiet(SparseWarpOperator.from_sampling, g, footprint=False)       # one bilinear sample per pixel
    out = op((torch.from_numpy(model_img.reshape(-1, 1)),))[0].numpy().reshape(150, 140)
    v = g["valid"]
    assert np.sqrt(np.mean((out - inst)[v] ** 2)) < 0.03 * np.nanstd(inst[v])   # bilinear vs cubic; measured 0.014
    save_sampling_grid(tmp_path / "w.npz", g)
    S = torch.randn(320 * 320, 2, dtype=torch.float64)
    a = quiet(SparseWarpOperator.from_sampling, g)((S,))[0]
    b = quiet(SparseWarpOperator.from_file, tmp_path / "w.npz", input_shape=(320, 320))((S,))[0]
    assert torch.equal(a, b)


def test_bad_inputs_raise():
    flat = np.ones((64, 64))
    with pytest.raises(ValueError):
        sampling_grid_from_images(flat, flat)
    img = textured((64, 64))
    with pytest.raises(ValueError):
        sampling_grid_from_images(img, img, model="poly2")
    with pytest.raises(ValueError):
        sampling_grid_from_images(img, img, refine="tps")


def test_wcs_version_is_the_geometry_of_instrument_sampling_grid():
    pytest.importorskip("sunpy.map")
    from test_coregistration import wcs_map
    from neural_me.core.coregistration import sampling_grid_from_wcs
    from neural_me.core.geometry import instrument_sampling_grid

    model = wcs_map((24, 24), (100.0, 50.0), 1.0)
    inst = wcs_map((12, 12), (100.0, 50.0), 1.5)
    a = sampling_grid_from_wcs(inst, model.wcs, (24, 24), n_sub=2)
    b = instrument_sampling_grid(inst, model.wcs, (24, 24), n_sub=2)
    for k in ("grid", "grid_sub", "valid", "footprint"):
        assert np.array_equal(a[k], b[k])
    assert a["model_shape"] == (24, 24)


# --- different views: SO/PHI-HRT against Hinode/SP, 2022-03-31 ------------------------------------------------------

def _ptot(d):
    """Total polarization over the line times the continuum, (ny, nx, 4, nw) -> (ny, nx)."""
    return np.sqrt((d[:, :, 1:] ** 2).sum(2)).sum(-1) / d.shape[-1] * d[:, :, 0, 0]


def test_phi_on_hinode_without_wcs(data_path):
    from astropy.io import fits

    hin = fits.getdata(data_path("sp_hinode_data/SP3D20220331_193959_warped.fits")).astype(np.float64)
    hin = np.transpose(hin, (2, 1, 0, 3))[..., 55:]                     # (ns, nx, ny, nw) -> (ny, nx, ns, nw), Fe I 6302
    phi = fits.getdata(data_path("phi_solo_data/solo_L2_phi-hrt-stokes_20220331T195608_V202305161958_0243310805_"
                                 "native_crop.fits")).astype(np.float64)
    phi = np.transpose(phi, (2, 3, 1, 0))                                # (nw, ns, ny, nx) -> (ny, nx, ns, nw)
    w = np.load(data_path("phi_solo_data/phi_native_warp.npz"))
    valid = w["valid"]
    gx, gy = (w["grid"][..., 0] + 1) / 2 * 511, (w["grid"][..., 1] + 1) / 2 * 511
    img_phi, img_hin = np.where(valid, _ptot(phi), np.nan), _ptot(hin)
    out = {}
    for refine in ("nodes", "ecc"):
        g = sampling_grid_from_images(img_phi, img_hin, refine=refine)
        px = np.stack([(g["grid"][..., 0] + 1) / 2 * 511, (g["grid"][..., 1] + 1) / 2 * 511], -1)
        out[refine] = np.hypot(px[..., 0] - gx, px[..., 1] - gy)[valid & g["valid"]]
    # measured 2.44 / 22.7 ('nodes') and 5.6 / 42 ('ecc'), model pixels
    assert np.median(out["nodes"]) < 3.0 and out["nodes"].max() < 25.0
    assert np.median(out["nodes"]) < np.median(out["ecc"])
