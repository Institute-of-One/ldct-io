"""The reconstruction is held to a closed form, not to its own past output.

A uniform disk has an analytic fan-beam projection, so its reconstruction has a known answer
everywhere: its own attenuation inside, zero outside, at its stated radius. These tests fail
if the geometry, the equiangular filter, the backprojection weight or the HU scale drifts.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldct_io import (
    disk_sinogram,
    fan_beam_fbp,
    fit_edge_circle,
    parker_weights,
    radial_mtf,
    ramp_kernel,
    sampling_chain_mtf,
    to_hu,
)

RADIUS = 99.939
MU = 0.0192
FOV = 260.0


def _reconstruct(flash, angles, *, n_pixels: int, n_subsamples: int):
    sinogram = disk_sinogram(flash, RADIUS, MU, angles, n_subsamples=n_subsamples)
    result = fan_beam_fbp(sinogram, flash, angles, fov=FOV, n_pixels=n_pixels)
    return result, to_hu(result.image, flash.water_attenuation)


def test_disk_reconstructs_to_its_own_attenuation(flash, full_rotation):
    result, hu = _reconstruct(flash, full_rotation, n_pixels=256, n_subsamples=1)
    X, Y = result.pixel_coordinates()
    r = np.hypot(X, Y)

    inside = hu[r < 20.0]
    assert abs(inside.mean()) < 1.0, f"centre reads {inside.mean():+.3f} HU, truth 0"
    assert inside.std() < 1.0, "a noiseless disk must reconstruct flat"

    outside = hu[(r > 115.0) & (r < 125.0)]
    assert abs(outside.mean() + 1000.0) < 1.0, f"air reads {outside.mean():+.2f} HU, truth -1000"


def test_reconstructed_radius_matches_the_disk(flash, full_rotation):
    result, hu = _reconstruct(flash, full_rotation, n_pixels=512, n_subsamples=1)
    circle = fit_edge_circle(hu, result.spacing, search_range=(85.0, 112.0))
    assert abs(circle.radius - RADIUS) < 0.1, (
        f"fitted radius {circle.radius:.3f} mm, truth {RADIUS} mm"
    )
    assert circle.ellipticity < 0.02, "a disk must not reconstruct elliptical"


def test_mtf_is_bounded_by_the_sampling_chain(flash, full_rotation):
    """A reconstruction can be no sharper than its sampling, and no much blurrier either."""
    result, hu = _reconstruct(flash, full_rotation, n_pixels=512, n_subsamples=7)
    circle = fit_edge_circle(hu, result.spacing, search_range=(85.0, 112.0))
    measured = radial_mtf(hu, result.spacing, circle, band=6.0)

    f = np.linspace(1e-6, 1.0, 500)
    chain = sampling_chain_mtf(flash, f, result.spacing)
    chain50 = float(np.interp(0.5, chain[::-1], f[::-1]))

    assert measured.mtf.max() <= 1.05, "an MTF above 1 is not a transfer function"
    assert 0.7 * chain50 < measured.mtf50 < 1.3 * chain50, (
        f"MTF50 {measured.mtf50:.3f} is not consistent with the sampling chain {chain50:.3f}"
    )


def test_a_point_detector_is_sharper_than_an_averaged_one(flash, full_rotation):
    """The detector aperture is a real blur: modelling it must lower the MTF."""
    out = {}
    for n_sub in (1, 7):
        result, hu = _reconstruct(flash, full_rotation, n_pixels=512, n_subsamples=n_sub)
        circle = fit_edge_circle(hu, result.spacing, search_range=(85.0, 112.0))
        out[n_sub] = radial_mtf(hu, result.spacing, circle, band=6.0).mtf50
    assert out[1] > out[7], f"point-sampled {out[1]:.3f} should exceed averaged {out[7]:.3f}"


def test_short_scan_without_parker_weights_is_refused(flash):
    """Under-covered views reconstruct to a plausible image with the wrong values."""
    angles = np.linspace(0.0, np.pi, 300, endpoint=False)
    sinogram = disk_sinogram(flash, RADIUS, MU, angles)
    with pytest.raises(ValueError, match="less than a full rotation"):
        fan_beam_fbp(sinogram, flash, angles, fov=FOV, n_pixels=64)


def test_a_weighted_short_scan_agrees_with_a_full_rotation(flash):
    """Parker weighting is what makes half a rotation give the same numbers as a whole one.

    This is the check that matters for helical data: a short scan halves how far the source
    travels in z, and is only worth using if it costs nothing in the reconstructed values.
    """
    full_angles = np.linspace(0.0, 2.0 * np.pi, 576, endpoint=False)
    full = fan_beam_fbp(
        disk_sinogram(flash, RADIUS, MU, full_angles),
        flash,
        full_angles,
        fov=FOV,
        n_pixels=256,
    )

    span = flash.short_scan_range
    n_short = int(np.ceil(576 * span / (2 * np.pi))) + 1
    short_angles = np.linspace(0.0, span, n_short, endpoint=False)
    short = fan_beam_fbp(
        disk_sinogram(flash, RADIUS, MU, short_angles),
        flash,
        short_angles,
        fov=FOV,
        n_pixels=256,
        short_scan=True,
    )

    hu_full = to_hu(full.image, flash.water_attenuation)
    hu_short = to_hu(short.image, flash.water_attenuation)
    X, Y = full.pixel_coordinates()
    r = np.hypot(X, Y)

    assert abs(hu_short[r < 20].mean()) < 3.0, (
        f"short scan reads {hu_short[r < 20].mean():+.2f} HU at the centre, truth 0"
    )
    assert abs(hu_short[r < 20].mean() - hu_full[r < 20].mean()) < 3.0
    assert abs(hu_short[(r > 115) & (r < 125)].mean() + 1000.0) < 5.0


def test_parker_weights_sum_to_one_over_conjugate_rays(flash):
    """The point of the weights: every line through the object is counted exactly once."""
    span = flash.short_scan_range
    angles = np.linspace(0.0, span, 900, endpoint=False)
    w = parker_weights(flash, angles)

    assert w.min() >= 0.0 and w.max() <= 1.0
    # The central channel: weights at beta and its conjugate beta + pi must sum to 1.
    centre = int(round(flash.central_channel))
    beta = angles
    for b in (0.05, 0.15, 0.25):
        i = int(np.argmin(np.abs(beta - b)))
        j = int(np.argmin(np.abs(beta - (b + np.pi))))
        assert w[i, centre] + w[j, centre] == pytest.approx(1.0, abs=0.02)


def test_parker_weights_refuse_too_little_coverage(flash):
    angles = np.linspace(0.0, np.pi * 0.8, 200, endpoint=False)
    with pytest.raises(ValueError, match="short scan needs at least"):
        parker_weights(flash, angles)


def test_ramp_kernel_matches_its_closed_form(flash):
    """The equiangular kernel is not the parallel-beam ramp, and the difference is silent."""
    d_gamma = flash.channel_angle_spacing
    g = ramp_kernel(flash.n_channels, d_gamma)
    n = np.arange(-flash.n_channels + 1, flash.n_channels)

    assert g[n == 0] == pytest.approx(1.0 / (8.0 * d_gamma**2), rel=1e-12)
    assert np.all(g[(n % 2 == 0) & (n != 0)] == 0.0)
    odd = n % 2 != 0
    expected = -1.0 / (2.0 * np.pi**2 * np.sin(n[odd] * d_gamma) ** 2)
    assert g[odd] == pytest.approx(expected, rel=1e-12)


def test_hu_scale_uses_the_recorded_water_value(flash):
    assert to_hu(np.array([flash.water_attenuation]), flash.water_attenuation)[0] == pytest.approx(
        0.0, abs=1e-9
    )
    assert to_hu(np.array([0.0]), flash.water_attenuation)[0] == pytest.approx(-1000.0)
    with pytest.raises(ValueError):
        to_hu(np.zeros(3), 0.0)
