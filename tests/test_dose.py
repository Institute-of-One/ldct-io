"""Dose reduction in the projection domain, checked against the physics it is meant to obey.

The point of doing the reduction on the counts rather than on the image is that the square-root
law stops being an assumption and becomes a consequence. These tests therefore check the law as
an *output*: nothing in :func:`insert_quantum_noise` references it, so if the Poisson draw or the
logarithm were wrong the law would come out wrong too.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldct_io import calibrate_incident_counts, insert_quantum_noise, noise_scale_factor


def _flat(value: float = 2.0, shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """A sinogram of constant line integral: the noise in it is the noise the draw adds."""
    return np.full(shape, float(value))


def test_the_noise_is_the_square_root_law_without_being_told_it():
    """The law is an output of the Poisson draw, not a parameter of it."""
    p = _flat()
    full = insert_quantum_noise(p, dose_fraction=1.0, incident_counts=1e5, seed=0)
    for beta in (0.5, 0.25, 0.1, 0.05):
        low = insert_quantum_noise(p, dose_fraction=beta, incident_counts=1e5, seed=0)
        measured = low.std() / full.std()
        assert measured == pytest.approx(noise_scale_factor(beta), rel=0.05), beta


def test_lower_dose_is_noisier_monotonically():
    p = _flat()
    sds = [
        insert_quantum_noise(p, dose_fraction=b, incident_counts=1e5, seed=1).std()
        for b in (1.0, 0.5, 0.25, 0.1)
    ]
    assert sds == sorted(sds)


def test_more_photons_means_less_noise():
    p = _flat()
    sds = [
        insert_quantum_noise(p, dose_fraction=1.0, incident_counts=i0, seed=2).std()
        for i0 in (1e4, 1e5, 1e6)
    ]
    assert sds[0] > sds[1] > sds[2]


def test_the_line_integral_is_preserved_on_average():
    """The draw adds noise; it must not add a bias large enough to shift the HU scale."""
    p = _flat(2.0, (256, 256))
    out = insert_quantum_noise(p, dose_fraction=1.0, incident_counts=1e5, seed=3)
    # The log of a Poisson mean is biased by about 1/(2N); at 1e5 * exp(-2) counts that is tiny.
    assert out.mean() == pytest.approx(2.0, abs=1e-3)


def test_the_bias_grows_as_the_counts_fall():
    """Stated rather than hidden: the logarithm of a small count is biased upward."""
    p = _flat(2.0, (256, 256))
    rich = insert_quantum_noise(p, dose_fraction=1.0, incident_counts=1e6, seed=4).mean() - 2.0
    poor = insert_quantum_noise(p, dose_fraction=0.01, incident_counts=1e4, seed=4).mean() - 2.0
    assert abs(poor) > abs(rich)


def test_the_draw_is_seeded():
    p = _flat()
    a = insert_quantum_noise(p, dose_fraction=0.25, incident_counts=1e5, seed=7)
    b = insert_quantum_noise(p, dose_fraction=0.25, incident_counts=1e5, seed=7)
    c = insert_quantum_noise(p, dose_fraction=0.25, incident_counts=1e5, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_realisations_at_different_doses_are_not_the_same_pattern_rescaled():
    """The reason for doing this at all: the dose points must be independent draws.

    Rescaling one realisation would make the normalised patterns identical. They are not.
    """
    p = _flat()
    a = insert_quantum_noise(p, dose_fraction=1.0, incident_counts=1e5, seed=0)
    b = insert_quantum_noise(p, dose_fraction=0.25, incident_counts=1e5, seed=0)
    correlation = np.corrcoef((a - a.mean()).ravel(), (b - b.mean()).ravel())[0, 1]
    assert abs(correlation) < 0.2, correlation


def test_a_thicker_object_is_noisier_at_the_same_exposure():
    """Beam hardening aside, fewer photons arrive through more attenuation."""
    thin = insert_quantum_noise(_flat(1.0), dose_fraction=1.0, incident_counts=1e5, seed=5)
    thick = insert_quantum_noise(_flat(4.0), dose_fraction=1.0, incident_counts=1e5, seed=5)
    assert thick.std() > thin.std()


@pytest.mark.parametrize("beta", [0.0, -0.5, 1.5, float("nan")])
def test_it_refuses_an_impossible_dose_fraction(beta):
    with pytest.raises(ValueError):
        insert_quantum_noise(_flat(), dose_fraction=beta, incident_counts=1e5)
    with pytest.raises(ValueError):
        noise_scale_factor(beta)


@pytest.mark.parametrize("i0", [0.0, -1.0, float("nan")])
def test_it_refuses_an_impossible_flux(i0):
    with pytest.raises(ValueError):
        insert_quantum_noise(_flat(), dose_fraction=1.0, incident_counts=i0)


def test_it_refuses_a_sinogram_that_is_not_line_integrals():
    bad = _flat()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        insert_quantum_noise(bad, dose_fraction=1.0, incident_counts=1e5)


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------


def _toy_reconstruct(sinogram: np.ndarray) -> np.ndarray:
    """Stand-in for a reconstructor: scales line-integral noise into HU-like numbers."""
    return 1000.0 * (sinogram - 2.0)


def test_calibration_finds_a_flux_that_reproduces_the_target_noise():
    p = _flat(2.0, (256, 256))
    target = 10.0
    result = calibrate_incident_counts(p, _toy_reconstruct, target, seed=0, tolerance=0.02)
    assert abs(result["relative_error"]) <= 0.02
    assert result["achieved_noise_sd"] == pytest.approx(target, rel=0.02)
    assert result["incident_counts"] > 0.0


def test_calibration_is_monotone_in_the_target():
    """A quieter target needs more photons; if this inverted, every dose would be wrong."""
    p = _flat(2.0, (256, 256))
    quiet = calibrate_incident_counts(p, _toy_reconstruct, 5.0, seed=0, tolerance=0.02)
    loud = calibrate_incident_counts(p, _toy_reconstruct, 20.0, seed=0, tolerance=0.02)
    assert quiet["incident_counts"] > loud["incident_counts"]


def test_calibration_refuses_a_target_outside_its_bracket():
    p = _flat(2.0, (128, 128))
    with pytest.raises(ValueError):
        calibrate_incident_counts(p, _toy_reconstruct, 1e-9, seed=0, bracket=(1e2, 1e4))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_calibration_refuses_an_impossible_target(bad):
    with pytest.raises(ValueError):
        calibrate_incident_counts(_flat(), _toy_reconstruct, bad)
