"""Reduce the dose of a measured scan where the dose actually lives: in the photon counts.

Why this exists
---------------
The liver arm's dose axis rescales a measured noise realisation by a factor derived from a noise
model. That keeps the spectrum and the anatomy real, and it makes the amplitude at every dose
fraction but one a modelled quantity: the same noise pattern is enlarged rather than redrawn, so
the dose points are perfectly correlated and the scaling law cannot be checked against itself.

With projections in hand the reduction can be done in the domain where it physically happens.
A line integral is :math:`p = -\\ln(I/I_0)`; at a fraction :math:`\\beta` of the exposure the
detector collects :math:`\\beta I` on average, and what it actually collects is a Poisson draw
about that mean. Reconstructing from those draws gives an *independent* realisation at every
dose, with the noise correlated by the same reconstruction that correlates the real noise, and
with the :math:`\\sqrt{\\beta}` behaviour an emergent property of the simulation rather than an
assumption put into it.

What has to be supplied, and what it costs
------------------------------------------
:math:`I_0` is not in the data. The stored projections are line integrals; the flux that produced
them has been divided out. So the incident count per ray is a calibration parameter, fixed by
:func:`calibrate_incident_counts` against the noise the real reconstruction actually shows, and
everything downstream inherits whatever that calibration is worth. A single scalar cannot
represent a bowtie filter, tube-current modulation, electronic noise or a detector's own gain
map, so what this models is a quantum-limited scan with a uniform flux. At very low dose, where
electronic noise stops being negligible, it will be optimistic.
"""

from __future__ import annotations

import numpy as np

#: Counts below this are treated as this, because the logarithm of zero is not a line integral.
#: A detector that collects no photons at all is a photon-starvation artefact rather than noise,
#: and this module does not model one.
MINIMUM_COUNTS = 1.0


def insert_quantum_noise(
    sinogram: np.ndarray,
    *,
    dose_fraction: float,
    incident_counts: float,
    seed: int = 0,
) -> np.ndarray:
    r"""A Poisson realisation of the same scan taken at ``dose_fraction`` of the exposure.

    Parameters
    ----------
    sinogram:
        Line integrals, any shape. Values are :math:`-\ln(I/I_0)` and must be finite.
    dose_fraction:
        Exposure relative to the one that produced ``sinogram``, in ``(0, 1]``. One returns a
        fresh Poisson realisation at the original exposure rather than the input itself, which
        is deliberate: the input carries the scan's own noise already, and adding a draw at
        :math:`\beta = 1` would double it. Use this function for the *simulated* arm at every
        fraction including one, and compare against the measurement separately.
    incident_counts:
        :math:`I_0`, the mean photons per ray at full exposure, from
        :func:`calibrate_incident_counts`.
    seed:
        Seeds the draw.

    Returns
    -------
    ndarray
        Line integrals of the lower-dose scan, same shape and dtype float64.

    """
    if not 0.0 < float(dose_fraction) <= 1.0:
        raise ValueError(f"dose_fraction must be in (0, 1], got {dose_fraction!r}")
    if not np.isfinite(incident_counts) or incident_counts <= 0.0:
        raise ValueError(f"incident_counts must be finite and > 0, got {incident_counts!r}")
    p = np.asarray(sinogram, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise ValueError("the sinogram contains non-finite line integrals")

    rng = np.random.default_rng(int(seed))
    expected = float(dose_fraction) * float(incident_counts) * np.exp(-p)
    counts = rng.poisson(expected).astype(np.float64)
    counts = np.maximum(counts, MINIMUM_COUNTS)
    return -np.log(counts / (float(dose_fraction) * float(incident_counts)))


def noise_scale_factor(dose_fraction: float) -> float:
    r"""The factor by which the line-integral noise grows at ``dose_fraction``.

    For a Poisson count :math:`N` the variance of :math:`-\ln N` is :math:`1/N` to first order,
    so the line-integral noise standard deviation goes as :math:`1/\sqrt{\beta}`. This is the
    closed form the simulation should reproduce, and it is given here so that the simulation can
    be checked against it rather than assumed to obey it.
    """
    if not 0.0 < float(dose_fraction) <= 1.0:
        raise ValueError(f"dose_fraction must be in (0, 1], got {dose_fraction!r}")
    return float(dose_fraction) ** -0.5


def calibrate_incident_counts(
    sinogram: np.ndarray,
    reconstruct,
    target_noise_sd: float,
    *,
    region: tuple[slice, slice] | None = None,
    seed: int = 0,
    bracket: tuple[float, float] = (1e2, 1e8),
    tolerance: float = 0.01,
    max_iterations: int = 40,
) -> dict[str, float]:
    r"""Find the :math:`I_0` whose simulated scan has the noise the real one has.

    The stored projections are line integrals with the flux divided out, so :math:`I_0` has to
    come from somewhere; here it comes from matching one measurable consequence of it. Noise
    standard deviation falls monotonically with :math:`I_0`, so the search is a bisection and
    its result is unique.

    Parameters
    ----------
    sinogram:
        Line integrals at full exposure.
    reconstruct:
        Callable taking a sinogram and returning an image in HU.
    target_noise_sd:
        The noise standard deviation the real reconstruction shows, in HU, measured in the same
        region ``region`` selects.
    region:
        Rows and columns of a uniform part of the image. ``None`` uses the central quarter.
    seed, bracket, tolerance, max_iterations:
        Seed for the draws, the search bracket on :math:`I_0`, the relative agreement required,
        and the iteration cap.

    Returns
    -------
    dict
        ``incident_counts``, the ``achieved_noise_sd`` it gives, the ``target_noise_sd`` asked
        for, the ``relative_error`` between them and the number of ``iterations``.

    """
    p = np.asarray(sinogram, dtype=np.float64)
    target = float(target_noise_sd)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(f"target_noise_sd must be finite and > 0, got {target_noise_sd!r}")

    def noise_at(i0: float) -> float:
        image = reconstruct(
            insert_quantum_noise(p, dose_fraction=1.0, incident_counts=i0, seed=seed)
        )
        if region is None:
            n, m = image.shape
            patch = image[n // 4 : 3 * n // 4, m // 4 : 3 * m // 4]
        else:
            patch = image[region]
        return float(np.std(patch))

    low, high = float(bracket[0]), float(bracket[1])
    # More photons, less noise: the function is decreasing, so the bracket is checked that way.
    if noise_at(low) < target or noise_at(high) > target:
        raise ValueError(
            f"the target noise {target:.3f} HU is not inside the bracket "
            f"[{low:g}, {high:g}] photons; widen it"
        )

    achieved = float("nan")
    for iteration in range(1, int(max_iterations) + 1):
        middle = float(np.sqrt(low * high))  # geometric, because I0 spans decades
        achieved = noise_at(middle)
        error = (achieved - target) / target
        if abs(error) <= float(tolerance):
            return {
                "incident_counts": middle,
                "achieved_noise_sd": achieved,
                "target_noise_sd": target,
                "relative_error": error,
                "iterations": iteration,
            }
        if achieved > target:
            low = middle
        else:
            high = middle
    return {
        "incident_counts": float(np.sqrt(low * high)),
        "achieved_noise_sd": achieved,
        "target_noise_sd": target,
        "relative_error": (achieved - target) / target,
        "iterations": int(max_iterations),
    }
