"""The fabrication detector has to fire on a fabricator, or a null result means nothing.

The adversarial arm exists to be caught. If it is not caught, the paper says so -- and that
sentence is only worth writing if the detector would have caught a denoiser that did fabricate.
A clean run and a blind detector look identical from the outside, so the detector is aimed at
known fabricators here and required to fire on the one it claims to catch, and shown not to
catch the one it cannot.

The prespecified criterion, and why it is not enough here
--------------------------------------------------------
The prespecified criterion -- lesion-shaped structure in a signal-absent image, at half the
lesion's contrast -- is the whole story on a uniform phantom, where the background is flat and
every structure in it was therefore put there by the noise or by the processing. On real
parenchyma it is not: the liver contains disc-like structure of its own at around twice this
lesion's contrast, so the rate is already near one before anything is processed and cannot rise.
That is not a reason to retune the threshold after the fact. It is a reason to measure what the
processing *added*, which this design can do because it knows the background exactly: run the
lesion-free, noise-free anatomy through the same method, subtract, and the anatomy cancels.

What the added measure can and cannot see
-----------------------------------------
Anatomy cancels, and so does anything the method invents from the anatomy alone. A generator
that renders one particular parenchymal feature as a crisp disc every time it sees it does so
identically in both classes, and this measure is blind to it. That blindness is not incidental:
structure identical in both classes carries no information about the hypothesis, so it cannot
produce an apparent breach of the ceiling, which is the claim under test. It can still mislead a
reader, and the paper has to say so. Both facts are tested below, so that neither can be
misremembered as the other.
"""

from __future__ import annotations

import numpy as np
import pytest

redlamp = pytest.importorskip("denoiq_core.redlamp")
ndimage = pytest.importorskip("scipy.ndimage")

from ldct_io.lesion import disk_lesion  # noqa: E402

SIZE = 48
SPACING = 0.7
LESION_MM = 8.0
LESION_HU = -25.0
NOISE_HU = 25.0
N_TRIALS = 120

#: Half the lesion's diameter in pixels: the scale at which parenchyma is confusable with this
#: lesion, and the scale the backgrounds are given structure at.
RADIUS_PX = 0.5 * LESION_MM / SPACING


def _blur(stack: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([ndimage.gaussian_filter(plane, sigma, mode="nearest") for plane in stack])


def _structured_background(rng: np.random.Generator, n: int) -> np.ndarray:
    """Backgrounds with structure at the lesion's own scale, like parenchyma.

    Smoothing white noise at the lesion's radius reproduces the property that matters:
    disc-shaped features of the lesion's size and several times its contrast, present before
    anything is processed.
    """
    rough = _blur(rng.normal(0.0, 1.0, size=(n, SIZE, SIZE)), RADIUS_PX)
    return 60.0 * rough / rough.std()


#: Band limits of the test's noise, in pixels. Reconstructed CT noise is neither white nor
#: low-pass: the ramp filter suppresses the lowest frequencies and the detector aperture the
#: highest, and how much power sits at the lesion's own scale is exactly what decides how often
#: noise alone makes a lesion-shaped peak. These two numbers put the test at the real run's
#: operating point -- unprocessed noise with a mean peak amplitude near half a lesion, which is
#: what the liver data gives (0.47 here against 0.49 measured) -- so that the honest arms have
#: room to fall and a fabricator has room to rise. A low-pass noise saturates the added measure
#: too, and white noise leaves it at zero for everything; neither would test anything.
NOISE_BAND_PX = (0.8, 4.0)


def _correlated_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Band-pass noise, as reconstruction gives it, scaled to the dose penalty in HU."""
    low, high = NOISE_BAND_PX
    white = rng.normal(0.0, 1.0, size=shape)
    band = _blur(white, low) - _blur(white, high)
    return NOISE_HU * band / band.std()


def _smooth(stack: np.ndarray) -> np.ndarray:
    """An honest denoiser: it removes contrast and cannot invent any."""
    return _blur(stack, 1.0)


def _stamp_at(plane: np.ndarray, signal: np.ndarray, row: int, col: int, amplitude: float):
    shifted = np.roll(signal, (row - SIZE // 2, col - SIZE // 2), axis=(0, 1))
    return plane + amplitude * shifted


#: The stamp is placed inside this margin, so that a fabricated lesion is whole rather than
#: wrapped around the edge.
MARGIN = SIZE // 4


def _peak_of(plane: np.ndarray, sigma: float) -> tuple[int, int]:
    """Where a feature at the lesion's scale is strongest, searched inside the margin.

    The interior is *searched*, not clipped into. Clipping collapses every peak near an edge
    onto the same border pixel, which made an earlier version of this injection stamp the noisy
    and the noise-free pass at the same place two thirds of the time -- so the stamps cancelled
    in the subtraction and the fabricator looked undetectable. That was a defect in the
    injection, and it is the reason the injection is written down rather than assumed.
    """
    inner = np.abs(ndimage.gaussian_filter(plane, sigma, mode="nearest"))[
        MARGIN:-MARGIN, MARGIN:-MARGIN
    ]
    row, col = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return int(row) + MARGIN, int(col) + MARGIN


def anatomy_driven_fabricator(
    stack: np.ndarray, signal: np.ndarray, *, amplitude: float = 0.8
) -> np.ndarray:
    """Smooths, then stamps a lesion onto the strongest *anatomical* feature.

    Deterministic in its input, as the bound requires. Its fabrication is a function of the
    anatomy, which both classes share, so it is the same in both and carries no information
    about the hypothesis.
    """
    out = _smooth(stack)
    return np.stack(
        [_stamp_at(p, signal, *_peak_of(p, RADIUS_PX), amplitude) for p in out]
    )


def noise_driven_fabricator(
    stack: np.ndarray, signal: np.ndarray, *, amplitude: float = 0.8
) -> np.ndarray:
    """Smooths, then stamps a lesion onto whichever noise blob looks most like one.

    This is the failure the ceiling argument is about: the structure follows the noise, so it
    differs between the two classes and between the noisy and the noise-free pass.
    """
    out = _smooth(stack)
    detail = stack - _blur(stack, RADIUS_PX)
    return np.stack(
        [
            _stamp_at(p, signal, *_peak_of(d, 1.0), amplitude)
            for p, d in zip(out, detail)
        ]
    )


@pytest.fixture(scope="module")
def scene() -> dict[str, np.ndarray]:
    """One background per trial, one noise realisation, and the lesion."""
    rng = np.random.default_rng(0)
    background = _structured_background(rng, N_TRIALS)
    noise = _correlated_noise(rng, background.shape)
    signal = disk_lesion(SIZE, SPACING, LESION_MM, LESION_HU, edge_sigma_mm=0.5)
    return {"background": background, "absent": background + noise, "signal": signal}


def _rates(method, scene: dict[str, np.ndarray]) -> tuple[float, float]:
    """The prespecified rate and the anatomy-cancelled rate, as liver_cnn.py computes them."""
    signal = scene["signal"]
    processed = method(scene["absent"])
    added = processed - method(scene["background"])
    return (
        redlamp.false_structure_rate(processed, signal)["rate"],
        redlamp.false_structure_rate(added, signal)["rate"],
    )


def test_the_prespecified_criterion_is_saturated_by_the_anatomy(scene):
    """Near one before anything is processed, so it has nowhere to rise to."""
    measured = redlamp.false_structure_rate(scene["absent"], scene["signal"])
    assert measured["rate"] > 0.9, measured
    assert measured["mean_max_amplitude"] > 1.0, measured


def test_the_prespecified_criterion_cannot_tell_a_fabricator_apart(scene):
    """The reason the second measure exists, as a test rather than as a paragraph."""
    honest, _ = _rates(_smooth, scene)
    faked, _ = _rates(lambda s: noise_driven_fabricator(s, scene["signal"]), scene)
    assert honest > 0.9 and faked > 0.9, (honest, faked)
    assert abs(faked - honest) < 0.1, (honest, faked)


def test_the_added_rate_fires_on_a_fabricator_that_follows_the_noise(scene):
    """The measurement the arm is judged by, on a denoiser that does fabricate.

    Two requirements, neither of them the observed number. A fabricator stamping eight tenths
    of a lesion has to be caught in the majority of images, and it has to stand clear of the
    honest arms by more than the whole spread those arms occupy -- unprocessed noise to a
    two-pixel Gaussian is about 0.15 wide here -- or the measure could not separate a
    fabricating network from an ordinary filter on this data.
    """
    _, honest = _rates(_smooth, scene)
    _, faked = _rates(lambda s: noise_driven_fabricator(s, scene["signal"]), scene)
    assert faked > 0.5, faked
    assert faked - honest > 0.4, (honest, faked)


def test_the_added_rate_scales_with_how_much_is_fabricated(scene):
    """Monotone in the fabricated amplitude: it measures the thing it is named after."""
    rates = [
        _rates(
            lambda s, a=amplitude: noise_driven_fabricator(s, scene["signal"], amplitude=a),
            scene,
        )[1]
        for amplitude in (0.0, 0.4, 1.0)
    ]
    assert rates[0] <= rates[1] <= rates[2], rates
    assert rates[2] - rates[0] > 0.4, rates


def test_the_added_rate_is_blind_to_fabrication_driven_by_the_anatomy(scene):
    """The limit of the measure, recorded so it cannot be forgotten or overclaimed.

    Structure invented from the anatomy is identical in both classes and cancels. It carries no
    information about the hypothesis and cannot breach the ceiling, which is why the measure is
    still the right one for the claim -- but a paper that reports a null here has to say this.
    """
    _, honest = _rates(_smooth, scene)
    _, faked = _rates(lambda s: anatomy_driven_fabricator(s, scene["signal"], amplitude=1.0), scene)
    assert abs(faked - honest) < 0.1, (honest, faked)
    # And the prespecified criterion does not rescue it either: it is saturated.
    total, _ = _rates(lambda s: anatomy_driven_fabricator(s, scene["signal"], amplitude=1.0), scene)
    assert total > 0.9, total


def test_a_smoother_adds_less_structure_the_harder_it_smooths(scene):
    """The honest direction. Removing contrast cannot look like adding it."""
    rates = [_rates(lambda s, g=sigma: _blur(s, g), scene)[1] for sigma in (0.5, 1.0, 2.0)]
    assert rates[0] > rates[1] > rates[2], rates
