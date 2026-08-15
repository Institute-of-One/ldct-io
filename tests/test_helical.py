"""The three-dimensional reconstruction, held to a sphere.

A cylinder cannot test a helical algorithm: it does not vary along z, so every ray sees the
same object and the hardest part of the problem disappears. A sphere does vary along z, and
its cone-beam line integrals are still exact for any ray from any source position — so every
plane through it has a known answer, a disk of known radius and known attenuation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldct_io import (
    ScanGeometry,
    fit_edge_circle,
    illuminating_views,
    row_window,
    sphere_projection,
    sphere_slice_truth,
    to_hu,
    wfbp_slice,
)

# A small scanner, so the test stays fast while keeping the real proportions: the same
# source distances, a ~29 deg fan, and a collimation slightly under the feed per rotation.
SMALL = dict(
    source_to_isocentre=595.0,
    source_to_detector=1085.6,
    transverse_spacing=4.3336,
    axial_spacing=2.18944,
    central_channel=63.5,
    central_row=7.5,
    n_channels=128,
    n_rows=16,
    detector_shape="CYLINDRICAL",
    trajectory="HELICAL",
    beam="FANBEAM",
    focal_spot_mode="NONE",
    water_attenuation=0.02,
)
RADIUS, MU = 80.0, 0.02
PITCH = 0.9
VIEWS_PER_ROTATION = 256


@pytest.fixture(scope="module")
def small() -> ScanGeometry:
    return ScanGeometry(**SMALL)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def helix(small):
    """A helical acquisition of a uniform sphere, generated analytically."""
    feed = PITCH * small.collimation_at_isocentre
    # Long enough that every plane the tests ask for is illuminated over a full rotation.
    n_views = int(round(VIEWS_PER_ROTATION * 9))
    angles = np.arange(n_views) * (2.0 * np.pi / VIEWS_PER_ROTATION)
    source_z = -0.5 * n_views * feed / VIEWS_PER_ROTATION + np.arange(n_views) * (
        feed / VIEWS_PER_ROTATION
    )
    frames = np.stack(
        [
            sphere_projection(
                small,
                np.array(
                    [
                        -small.source_to_isocentre * np.sin(a),
                        small.source_to_isocentre * np.cos(a),
                        zs,
                    ]
                ),
                float(a),
                RADIUS,
                MU,
            )
            for a, zs in zip(angles, source_z, strict=True)
        ]
    )
    return frames, angles, source_z


def _reconstruct(small, helix, z, *, n_pixels=128, fov=220.0, taper=0.35):
    frames, angles, source_z = helix
    result = wfbp_slice(frames, small, angles, source_z, z, fov=fov, n_pixels=n_pixels, taper=taper)
    return result, to_hu(result.image, small.water_attenuation)


@pytest.mark.parametrize("z", [0.0, 30.0, -45.0])
def test_a_sphere_reconstructs_to_its_own_attenuation(small, helix, z):
    """Every plane through the sphere must come back at mu inside and zero outside."""
    truth_radius, _ = sphere_slice_truth(RADIUS, MU, z)
    result, hu = _reconstruct(small, helix, z)
    X, Y = result.pixel_coordinates()
    r = np.hypot(X, Y)

    core = hu[r < 0.5 * truth_radius]
    assert abs(core.mean()) < 25.0, (
        f"z = {z}: inside the sphere reads {core.mean():+.1f} HU, truth 0 "
        f"(mu = {small.water_attenuation})"
    )

    outside = hu[(r > truth_radius + 12.0) & (r < 105.0)]
    assert abs(outside.mean() + 1000.0) < 25.0, (
        f"z = {z}: outside reads {outside.mean():+.1f} HU, truth -1000"
    )


def test_the_reconstructed_radius_follows_the_sphere(small, helix):
    """The plane's disk must shrink with z exactly as sqrt(a^2 - dz^2)."""
    for z in (0.0, 30.0, 50.0):
        truth_radius, _ = sphere_slice_truth(RADIUS, MU, z)
        result, hu = _reconstruct(small, helix, z)
        circle = fit_edge_circle(
            hu, result.spacing, search_range=(0.6 * truth_radius, truth_radius + 15.0)
        )
        assert abs(circle.radius - truth_radius) < 1.5, (
            f"z = {z}: reconstructed radius {circle.radius:.2f} mm, truth {truth_radius:.2f}"
        )


def test_every_voxel_is_reconstructed(small, helix):
    result, _ = _reconstruct(small, helix, 0.0)
    assert result.meta["unreconstructed_fraction"] == 0.0
    assert result.meta["redundancy"] > 1.0


def test_a_plane_outside_the_scan_is_refused(small, helix):
    frames, angles, source_z = helix
    with pytest.raises(ValueError, match="outside the scan"):
        wfbp_slice(frames, small, angles, source_z, 10_000.0, fov=200.0, n_pixels=32)


def test_illumination_reports_the_redundancy(small, helix):
    _, angles, source_z = helix
    illum = illuminating_views(small, source_z, angles, 0.0)
    # Collimation 19.2 mm at the isocentre against a 17.28 mm feed per rotation.
    assert illum.redundancy == pytest.approx(19.2 / 17.28, rel=0.05)
    assert illum.views.size > 200


def test_the_row_window_is_smooth_and_bounded():
    q = np.linspace(-1.4, 1.4, 501)
    w = row_window(q, taper=0.35)
    assert w.min() == 0.0 and w.max() == 1.0
    assert w[np.abs(q) > 1.0].max() == 0.0
    assert w[np.abs(q) < 0.6].min() == 1.0
    assert np.abs(np.diff(w)).max() < 0.05, "a step here would stripe the image along z"
    with pytest.raises(ValueError):
        row_window(q, taper=0.0)
