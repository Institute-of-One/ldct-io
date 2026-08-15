"""The edge estimator must refuse what it cannot measure, and be right about what it can."""

from __future__ import annotations

import numpy as np
import pytest

from ldct_io import disk_sinogram, fan_beam_fbp, fit_edge_circle, radial_mtf, to_hu

RADIUS = 99.939
MU = 0.0192


@pytest.fixture(scope="module")
def disk_image(flash):
    angles = np.linspace(0.0, 2.0 * np.pi, 576, endpoint=False)
    sinogram = disk_sinogram(flash, RADIUS, MU, angles, n_subsamples=7)
    result = fan_beam_fbp(sinogram, flash, angles, fov=260.0, n_pixels=512)
    return to_hu(result.image, flash.water_attenuation), result.spacing


def test_a_sloped_background_is_refused(disk_image, flash):
    """A pedestal survives differentiation and, through the DC normalisation, lowers the MTF.

    Real CT images have one — residual beam hardening, scatter — so the estimator must not
    quietly return the number anyway.
    """
    hu, spacing = disk_image
    n = hu.shape[0]
    axis = (np.arange(n) - (n - 1) / 2.0) * spacing
    X, Y = np.meshgrid(axis, axis)
    r = np.hypot(X, Y)

    tilted = hu.copy()
    inside = r < RADIUS
    tilted[inside] += 5.0 * (r[inside] - RADIUS)  # 5 HU/mm of cupping, zero at the edge

    circle = fit_edge_circle(tilted, spacing, search_range=(85.0, 112.0))
    with pytest.raises(ValueError, match="not flat"):
        radial_mtf(tilted, spacing, circle, band=6.0)

    relaxed = radial_mtf(tilted, spacing, circle, band=6.0, background="asymptote")
    clean = radial_mtf(
        hu,
        spacing,
        fit_edge_circle(hu, spacing, search_range=(85.0, 112.0)),
        band=6.0,
        background="asymptote",
    )
    assert relaxed.mtf50 == pytest.approx(clean.mtf50, rel=0.10), (
        "removing a known gradient must recover the ungradiented answer"
    )


def test_measured_tail_slopes_are_reported(disk_image):
    hu, spacing = disk_image
    circle = fit_edge_circle(hu, spacing, search_range=(85.0, 112.0))
    mtf = radial_mtf(hu, spacing, circle, band=6.0)
    assert abs(mtf.meta["tail_slope_inner_fraction"]) < 0.005
    assert abs(mtf.meta["tail_slope_outer_fraction"]) < 0.005
    assert mtf.meta["n_bins"] > 100


def test_jitter_correction_changes_the_answer(disk_image):
    """Bias A is not cosmetic: switching it off must move the MTF."""
    hu, spacing = disk_image
    circle = fit_edge_circle(hu, spacing, search_range=(85.0, 112.0))
    with_corr = radial_mtf(hu, spacing, circle, band=6.0)
    without = radial_mtf(hu, spacing, circle, band=6.0, jitter_correction=False)
    assert not np.allclose(with_corr.mtf, without.mtf)


def test_mtf_is_normalised_and_finite(disk_image):
    hu, spacing = disk_image
    circle = fit_edge_circle(hu, spacing, search_range=(85.0, 112.0))
    mtf = radial_mtf(hu, spacing, circle, band=6.0)
    assert mtf.mtf[0] == pytest.approx(1.0, rel=1e-9)
    assert np.all(np.isfinite(mtf.mtf))
    assert mtf.frequency.max() <= 0.5 / spacing + 1e-12
    assert mtf.contrast > 900.0


def test_circle_fit_reports_a_tilted_cylinder(disk_image):
    """An elliptical cross-section smears a radial ESF; the fit must say so rather than hide it."""
    hu, spacing = disk_image
    n = hu.shape[0]
    axis = (np.arange(n) - (n - 1) / 2.0) * spacing
    # Squash the image by 2 % along x: a circle becomes an ellipse.
    squashed = np.stack(
        [np.interp(axis, axis * 0.98, row, left=-1000.0, right=-1000.0) for row in hu]
    )
    round_fit = fit_edge_circle(hu, spacing, search_range=(85.0, 112.0))
    oval_fit = fit_edge_circle(squashed, spacing, search_range=(85.0, 114.0))
    assert round_fit.ellipticity < 0.02
    assert oval_fit.ellipticity > 0.5


def test_a_uniform_image_has_no_edge(flash):
    flat = np.full((128, 128), 42.0)
    with pytest.raises(ValueError):
        fit_edge_circle(flat, 0.5, search_range=(10.0, 20.0))


def test_nonfinite_input_is_refused():
    img = np.zeros((64, 64))
    img[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_edge_circle(img, 0.5, search_range=(5.0, 10.0))
