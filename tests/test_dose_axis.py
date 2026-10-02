"""The dose axis is a rescaled measurement, so the rescaling has to be exact.

``examples/dose_decision.py`` does not re-read the archive at every dose. It builds the trials
once at the fraction the collection simulated, and recombines them linearly, relying on the fact
that ``make_paired_trials`` composes ``present = background + lesion + noise`` and
``absent = background + noise`` and on nothing else. If that recombination were even slightly
wrong, every dose except the nominal one would be quietly fictitious, and the run would look
perfectly healthy: the numbers would still be smooth, monotone and plausible. So it is tested
against the thing it stands in for -- trials built directly from noise that was scaled first.

The closed forms that convert between dose, detectability and apparent exposure are tested
against their own definitions rather than against remembered values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from ldct_io.lesion import make_paired_trials  # noqa: E402

dose_decision = pytest.importorskip(
    "dose_decision", reason="needs denoiq_core and taskiq_core on the path"
)

SPACING = 0.7
SIZE = 32
N_SITES = 40


def _volume_and_sites(seed: int = 0):
    """A small volume with structure, a noise realisation, and sites away from the edges."""
    rng = np.random.default_rng(seed)
    shape = (6, 96, 96)
    background = 40.0 * rng.normal(size=shape).cumsum(axis=1) / np.sqrt(shape[1])
    noise = 25.0 * rng.normal(size=shape)
    half = SIZE // 2
    sites = np.array(
        [
            (
                int(rng.integers(0, shape[0])),
                int(rng.integers(half + 1, shape[1] - half - 1)),
                int(rng.integers(half + 1, shape[2] - half - 1)),
            )
            for _ in range(N_SITES)
        ]
    )
    return background, noise, sites


def _trials(background, noise, sites, **extra):
    return make_paired_trials(
        background,
        noise,
        SPACING,
        sites,
        size=SIZE,
        diameter_mm=8.0,
        contrast_hu=-25.0,
        edge_sigma_mm=0.5,
        n_trials=N_SITES,
        seed=0,
        **extra,
    )


@pytest.mark.parametrize("beta", [0.5, 0.25, 0.1, 0.0625])
def test_recombining_equals_rebuilding_at_every_dose(beta):
    """The shortcut and the long way round give the same trials, to floating point."""
    alpha = 0.25
    background, noise, sites = _volume_and_sites()
    k = dose_decision.noise_scale(beta, alpha)

    nominal = _trials(background, noise, sites)
    clean = _trials(background, np.zeros_like(noise), sites)
    recombined_present = clean.present + k * (nominal.present - clean.present)
    recombined_absent = clean.absent + k * (nominal.absent - clean.absent)

    rebuilt = _trials(background, k * noise, sites)
    assert np.allclose(recombined_present, rebuilt.present, atol=1e-9, rtol=0)
    assert np.allclose(recombined_absent, rebuilt.absent, atol=1e-9, rtol=0)


def test_the_noise_free_pass_is_the_background_and_the_lesion():
    """The premise the recombination rests on, asserted rather than assumed."""
    background, noise, sites = _volume_and_sites()
    clean = _trials(background, np.zeros_like(noise), sites)
    # absent is the background alone, so present minus absent is exactly the lesion.
    difference = clean.present - clean.absent
    for plane in difference:
        assert np.allclose(plane, clean.signal, atol=1e-9, rtol=0)


def test_the_scale_is_one_at_the_fraction_the_noise_was_measured_at():
    """Necessary, and on its own not sufficient.

    Every plausible way of getting this wrong -- dividing the powers instead of their square
    roots, using the ratio of the doses rather than of the inserted noise -- also returns one
    here. It is the test below that separates them, and this one that pins the anchor.
    """
    assert dose_decision.noise_scale(0.25, 0.25) == pytest.approx(1.0)
    assert dose_decision.noise_scale(0.1, 0.1) == pytest.approx(1.0)


def test_the_scale_follows_the_inserted_noise_power():
    """k^2 is the ratio of (1/beta - 1), which is what the inserted noise's power tracks."""
    alpha, beta = 0.25, 0.0625
    k = dose_decision.noise_scale(beta, alpha)
    assert k**2 == pytest.approx((1 / beta - 1) / (1 / alpha - 1))
    assert dose_decision.noise_scale(0.0625, 0.25) == pytest.approx(np.sqrt(5.0))


@pytest.mark.parametrize("beta", [0.5, 0.25, 0.1])
def test_lower_dose_means_more_noise(beta):
    assert dose_decision.noise_scale(beta / 2.0, 0.25) > dose_decision.noise_scale(beta, 0.25)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_the_scale_refuses_a_fraction_that_is_not_one(bad):
    with pytest.raises(ValueError):
        dose_decision.noise_scale(bad, 0.25)
    with pytest.raises(ValueError):
        dose_decision.noise_scale(0.25, bad)


def test_a_method_already_at_the_requirement_needs_the_dose_it_has():
    assert dose_decision.dose_for_requirement(0.25, 5.0, 5.0) == pytest.approx(0.25)


def test_the_required_dose_inverts_the_scaling_law():
    """Solve for the dose, scale the noise to it, and the detectability must be the requirement."""
    beta, measured, requirement = 0.25, 6.025, 5.0
    target = dose_decision.dose_for_requirement(beta, measured, requirement)
    predicted = measured * dose_decision.noise_scale(beta, 0.25) / dose_decision.noise_scale(
        target, 0.25
    )
    assert predicted == pytest.approx(requirement)


def test_a_method_below_the_requirement_needs_more_dose_and_above_it_needs_less():
    assert dose_decision.dose_for_requirement(0.25, 4.0, 5.0) > 0.25
    assert dose_decision.dose_for_requirement(0.25, 7.0, 5.0) < 0.25


def test_an_observer_that_sees_nothing_never_reaches_the_requirement():
    assert np.isnan(dose_decision.dose_for_requirement(0.25, 0.0, 5.0))
    assert np.isnan(dose_decision.dose_for_requirement(0.25, float("nan"), 5.0))


def test_no_fidelity_gain_means_the_image_looks_like_its_own_exposure():
    for beta in (0.5, 0.25, 0.0625):
        assert dose_decision.apparent_dose(beta, 0.0) == pytest.approx(beta)


def test_a_fidelity_gain_makes_the_image_claim_a_larger_exposure():
    """And the claim is exactly the exposure whose noise power is that much lower."""
    beta, gain_db = 0.25, 4.26
    claimed = dose_decision.apparent_dose(beta, gain_db)
    assert claimed > beta
    assert (1 / claimed - 1) == pytest.approx((1 / beta - 1) * 10 ** (-gain_db / 10))


def test_three_decibels_is_half_the_inserted_noise_power():
    beta = 0.25
    claimed = dose_decision.apparent_dose(beta, 10.0 * np.log10(2.0))
    assert (1 / claimed - 1) == pytest.approx((1 / beta - 1) / 2.0)
