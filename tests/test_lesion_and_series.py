"""Hybrid lesions, and the guards on reading a reconstructed series."""

from __future__ import annotations

import numpy as np
import pytest

from ldct_io import (
    SIMULATED_DOSE_FRACTION,
    ImageSeries,
    disk_lesion,
    homogeneous_sites,
    make_trials,
    noise_only,
)


def test_a_disk_lesion_carries_the_contrast_it_was_asked_for():
    spacing = 0.7
    lesion = disk_lesion(64, spacing, diameter_mm=8.0, contrast_hu=-25.0)
    axis = (np.arange(64) - 63 / 2.0) * spacing
    X, Y = np.meshgrid(axis, axis)
    core = lesion[np.hypot(X, Y) < 2.5]
    assert core.mean() == pytest.approx(-25.0, rel=1e-6)
    assert lesion[np.hypot(X, Y) > 6.0].max() == pytest.approx(0.0, abs=1e-9)

    # Area is what sets detectability, so it must be the disk's own, not the pixel grid's.
    area = lesion.sum() * spacing**2 / -25.0
    assert area == pytest.approx(np.pi * 4.0**2, rel=0.01)


def test_a_blurred_lesion_conserves_its_integral():
    """Blurring the edge must not change how much signal is there."""
    sharp = disk_lesion(64, 0.7, 8.0, -25.0)
    blurred = disk_lesion(64, 0.7, 8.0, -25.0, edge_sigma_mm=0.8)
    assert blurred.sum() == pytest.approx(sharp.sum(), rel=1e-6)
    assert blurred.min() > sharp.min(), "blurring must reduce the peak"


def test_a_lesion_too_big_for_the_roi_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        disk_lesion(32, 0.7, diameter_mm=40.0, contrast_hu=-25.0)


@pytest.fixture
def fake_liver():
    """Textured tissue with a bright organ boundary running through part of it."""
    rng = np.random.default_rng(0)
    vol = 60.0 + 18.0 * rng.standard_normal((6, 256, 256))
    vol[:, :, :60] = -900.0  # air, which no ROI should be taken from
    vol[:, 200:, :] += 400.0  # a bright structure
    return vol.astype(np.float32), 0.7


def test_sites_avoid_air_and_boundaries(fake_liver):
    vol, spacing = fake_liver
    sites = homogeneous_sites(vol, spacing, 64, hu_range=(0.0, 150.0), max_sd=60.0)
    assert len(sites) > 20
    for s, r, c in sites:
        roi = vol[s, r - 32 : r + 32, c - 32 : c + 32]
        assert 0.0 <= roi.mean() <= 150.0
        assert c - 32 >= 60 or roi.mean() > 0.0


def test_present_and_absent_backgrounds_are_disjoint(fake_liver):
    """Sharing a background between the two classes makes detectability infinite."""
    vol, spacing = fake_liver
    trials = make_trials(vol, spacing, size=64, diameter_mm=8.0, contrast_hu=-25.0, seed=1)
    n = trials.meta["n_per_class"]
    present_sites = {tuple(x) for x in trials.locations[:n]}
    absent_sites = {tuple(x) for x in trials.locations[n:]}
    assert present_sites.isdisjoint(absent_sites)
    assert trials.present.shape == trials.absent.shape == (n, 64, 64)


def test_the_lesion_is_actually_in_the_present_trials(fake_liver):
    vol, spacing = fake_liver
    trials = make_trials(vol, spacing, size=64, diameter_mm=10.0, contrast_hu=-40.0, seed=2)
    axis = (np.arange(64) - 63 / 2.0) * spacing
    X, Y = np.meshgrid(axis, axis)
    core = np.hypot(X, Y) < 3.0
    difference = trials.present[:, core].mean() - trials.absent[:, core].mean()
    assert difference == pytest.approx(-40.0, abs=6.0)


def test_asking_for_more_trials_than_the_split_allows_is_refused(fake_liver):
    vol, spacing = fake_liver
    sites = homogeneous_sites(vol, spacing, 64)
    with pytest.raises(ValueError, match="disjoint sites"):
        make_trials(vol, spacing, size=64, n_trials=len(sites), sites=sites)


def test_trials_are_deterministic(fake_liver):
    vol, spacing = fake_liver
    a = make_trials(vol, spacing, size=64, seed=7)
    b = make_trials(vol, spacing, size=64, seed=7)
    assert np.array_equal(a.present, b.present)
    assert np.array_equal(a.locations, b.locations)
    c = make_trials(vol, spacing, size=64, seed=8)
    assert not np.array_equal(a.locations, c.locations)


def _series(volume, z, spacing=0.8):
    return ImageSeries(volume=volume, z=z, spacing=spacing, slice_thickness=5.0, kernel="B30f")


def test_the_difference_of_a_dose_pair_is_a_noise_realisation():
    rng = np.random.default_rng(0)
    anatomy = 60.0 + 200.0 * rng.standard_normal((4, 64, 64))
    z = np.arange(4) * 3.0
    inserted = 25.0 * rng.standard_normal((4, 64, 64))
    full = _series(anatomy, z)
    low = _series(anatomy + inserted, z)

    difference, factor = noise_only(full, low, dose_fraction=0.25)
    assert np.allclose(difference, inserted)
    assert factor == pytest.approx(3.0)
    # Anatomy, 8x larger than the noise, has to cancel exactly rather than approximately.
    assert difference.std() == pytest.approx(inserted.std(), rel=1e-6)


def test_a_mismatched_pair_is_refused():
    z = np.arange(4) * 3.0
    full = _series(np.zeros((4, 64, 64)), z)
    low = _series(np.zeros((4, 32, 32)), z)
    with pytest.raises(ValueError, match="not on the same grid"):
        noise_only(full, low, dose_fraction=0.25)


def test_an_impossible_dose_fraction_is_refused():
    z = np.arange(2) * 3.0
    s = _series(np.zeros((2, 8, 8)), z)
    with pytest.raises(ValueError, match="dose_fraction"):
        noise_only(s, s, dose_fraction=1.0)


def test_the_documented_dose_fractions_are_the_collection_s():
    assert SIMULATED_DOSE_FRACTION["CHEST"] == 0.10
    assert SIMULATED_DOSE_FRACTION["ABDOMEN"] == 0.25
