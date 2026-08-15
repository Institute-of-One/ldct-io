"""Put a lesion of known size and contrast into a real image, so a task has ground truth.

A patient scan has no ground truth: the lesions in it were found by a reader, at a contrast
nobody measured, and half the interesting ones are the ones nobody found. Inserting a lesion
of known amplitude into real anatomy — a hybrid image — keeps the background that makes the
task hard and the truth that makes it measurable. It is the standard construction for
observer studies on patient data, and its one real assumption is stated here: the inserted
lesion is added *after* reconstruction, so it does not carry the beam hardening, the scatter
or the noise correlation that a real lesion of that density would have imposed on the
projections. It is a signal in the image, not an object in the patient.

The trials this module builds are signal-known-exactly with a background known only
statistically (SKE/BKS): present and absent trials draw on **disjoint** locations, because
inserting a lesion into a background and then also scoring that same background as the
absent trial makes the difference exactly the lesion and the detectability infinite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class LesionTrials:
    """Signal-present and signal-absent images drawn from real anatomy.

    Attributes
    ----------
    present, absent:
        ``(n, size, size)`` stacks in HU. Their backgrounds are disjoint.
    signal:
        The noise-free lesion that was added, ``(size, size)`` in HU — the template an
        observer that knows the signal exactly is entitled to.
    spacing:
        Pixel pitch [mm].
    locations:
        ``(slice, row, col)`` of each ROI used, present then absent, for provenance.
    meta:
        Lesion description and the selection criteria.

    """

    present: np.ndarray
    absent: np.ndarray
    signal: np.ndarray
    spacing: float
    locations: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def disk_lesion(
    size: int,
    spacing: float,
    diameter_mm: float,
    contrast_hu: float,
    *,
    edge_sigma_mm: float = 0.0,
    supersample: int = 4,
) -> np.ndarray:
    """A centred disk of the given diameter and contrast, optionally with a blurred edge.

    ``edge_sigma_mm`` should be set to the system's own blur if the lesion is meant to look
    like something the scanner produced; leaving it at zero inserts an object sharper than
    anything the scanner can image, which flatters every observer that uses high frequencies.
    """
    if size < 4 or spacing <= 0.0:
        raise ValueError(f"implausible ROI: size={size}, spacing={spacing}")
    if diameter_mm <= 0.0 or diameter_mm > size * spacing:
        raise ValueError(f"a {diameter_mm} mm lesion does not fit in a {size * spacing:.1f} mm ROI")
    fine = (np.arange(size * supersample) - (size * supersample - 1) / 2.0) * (
        spacing / supersample
    )
    XF, YF = np.meshgrid(fine, fine)
    disk = (np.hypot(XF, YF) <= diameter_mm / 2.0).astype(np.float64)
    disk = disk.reshape(size, supersample, size, supersample).mean((1, 3)) * contrast_hu

    if edge_sigma_mm > 0.0:
        fx = np.fft.fftfreq(size, spacing)
        FX, FY = np.meshgrid(fx, fx)
        gaussian = np.exp(-2.0 * np.pi**2 * edge_sigma_mm**2 * (FX**2 + FY**2))
        disk = np.real(np.fft.ifft2(np.fft.fft2(disk) * gaussian))
    return disk


def homogeneous_sites(
    volume: np.ndarray,
    spacing: float,
    size: int,
    *,
    hu_range: tuple[float, float] = (0.0, 150.0),
    max_sd: float = 60.0,
    max_gradient: float = 40.0,
    stride: int | None = None,
    slices: range | None = None,
) -> np.ndarray:
    """Find ROIs of uniform-looking tissue: ``(n, 3)`` of ``(slice, row, col)`` centres.

    Uniform *looking*, not uniform: liver parenchyma has vessels and texture, and a criterion
    strict enough to exclude all of it would leave nothing and would also remove exactly the
    structure that makes detection hard. The criteria here reject ROIs that straddle an organ
    boundary — a mean outside the tissue window, a large spread, or a strong overall gradient
    — and keep the rest.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    half = size // 2
    stride = stride or half
    slice_range = slices if slices is not None else range(volume.shape[0])

    found = []
    for s in slice_range:
        plane = volume[s]
        for r in range(half, plane.shape[0] - half, stride):
            for c in range(half, plane.shape[1] - half, stride):
                roi = plane[r - half : r + half, c - half : c + half]
                mean = float(roi.mean())
                if not hu_range[0] <= mean <= hu_range[1]:
                    continue
                if float(roi.std()) > max_sd:
                    continue
                gy, gx = np.gradient(roi.astype(np.float64))
                if abs(float(gy.mean())) * size > max_gradient:
                    continue
                if abs(float(gx.mean())) * size > max_gradient:
                    continue
                found.append((s, r, c))
    if not found:
        raise ValueError(
            "no homogeneous ROI met the criteria; loosen hu_range, max_sd or max_gradient"
        )
    return np.array(found, dtype=np.int64)


def make_trials(
    volume: np.ndarray,
    spacing: float,
    *,
    size: int = 64,
    diameter_mm: float = 8.0,
    contrast_hu: float = -25.0,
    edge_sigma_mm: float = 0.0,
    n_trials: int | None = None,
    seed: int = 0,
    sites: np.ndarray | None = None,
    **site_kwargs: Any,
) -> LesionTrials:
    """Build SKE/BKS trials from real anatomy, with the lesion inserted into half of them.

    Parameters
    ----------
    volume, spacing:
        The patient volume in HU and its pixel pitch.
    size:
        ROI side in pixels.
    diameter_mm, contrast_hu:
        The lesion. A hypodense liver metastasis is negative contrast; the default is a
        conservative 8 mm at −25 HU.
    edge_sigma_mm:
        Blur applied to the inserted lesion. See :func:`disk_lesion`.
    n_trials:
        Trials per class. Defaults to as many as the disjoint split allows.
    seed:
        Chooses which sites become present and which absent.
    sites:
        Pre-computed sites from :func:`homogeneous_sites`; computed here if omitted.
    **site_kwargs:
        Selection criteria forwarded to :func:`homogeneous_sites`. Passing both these and
        ``sites`` is an error rather than a silent preference for one of them.

    """
    if sites is None:
        sites = homogeneous_sites(volume, spacing, size, **site_kwargs)
    elif site_kwargs:
        raise ValueError("pass site selection criteria or `sites`, not both")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sites))
    per_class = len(sites) // 2
    if n_trials is not None:
        if n_trials > per_class:
            raise ValueError(
                f"{n_trials} trials per class asked for, but only {len(sites)} disjoint sites "
                f"were found, which allows {per_class}"
            )
        per_class = n_trials
    present_idx = sites[order[:per_class]]
    absent_idx = sites[order[per_class : 2 * per_class]]

    half = size // 2
    signal = disk_lesion(size, spacing, diameter_mm, contrast_hu, edge_sigma_mm=edge_sigma_mm)

    def crop(index: np.ndarray) -> np.ndarray:
        return np.stack(
            [volume[s, r - half : r + half, c - half : c + half] for s, r, c in index]
        ).astype(np.float64)

    absent = crop(absent_idx)
    present = crop(present_idx) + signal[None, :, :]

    return LesionTrials(
        present=present,
        absent=absent,
        signal=signal,
        spacing=spacing,
        locations=np.concatenate([present_idx, absent_idx]),
        meta={
            "size": size,
            "diameter_mm": diameter_mm,
            "contrast_hu": contrast_hu,
            "edge_sigma_mm": edge_sigma_mm,
            "n_per_class": int(per_class),
            "n_sites_found": int(len(sites)),
            "seed": seed,
            "construction": "SKE/BKS, disjoint backgrounds, lesion added after reconstruction",
        },
    )


def make_paired_trials(
    background: np.ndarray,
    noise: np.ndarray,
    spacing: float,
    sites: np.ndarray,
    *,
    size: int = 48,
    diameter_mm: float = 8.0,
    contrast_hu: float = -25.0,
    edge_sigma_mm: float = 0.0,
    n_trials: int | None = None,
    seed: int = 0,
) -> LesionTrials:
    """Background-known-exactly trials: one anatomy, two independent noise realisations.

    Why this exists
    ---------------
    On a uniform phantom the only thing an observer has to see through is quantum noise:
    stationary, Gaussian, and estimable, so the prewhitening ideal observer is available in
    closed form and makes an exact ceiling. In a liver it is not. The variation between two
    parenchymal ROIs is mostly *anatomy* — here about 35 HU against 25 HU of quantum noise —
    and anatomy is neither stationary nor Gaussian. A prewhitening template estimated against
    it is so inefficient that it comes out below a channelised observer, which is an upper
    bound that is not one.

    This collection makes a way out available. Its low-dose series is the *same* projections
    with noise inserted, so the difference of the two reconstructions is a measured
    realisation of exactly the noise that dose reduction costs, on real anatomy, with the
    anatomy cancelled. Holding the background fixed and drawing two independent noise patches
    for the two classes gives a task whose only stochastic component is that noise. The
    ceiling is then a closed form again, and it is a ceiling for the question actually being
    asked: what dose reduction takes away, and whether processing can put any of it back.

    What it costs
    -------------
    The background is treated as known, and it is only known up to the full-dose scan's own
    noise. The resulting detectability is therefore an idealisation — the task a reader
    familiar with this patient's anatomy would face, not the task of reading the scan cold.
    It is the right ceiling for a data-processing question and the wrong one for a claim about
    clinical detectability.

    Parameters
    ----------
    background:
        The known anatomy, ``(n_slices, ny, nx)`` in HU — the full-dose reconstruction.
    noise:
        A measured noise realisation on the same grid, from
        :func:`~ldct_io.series.noise_only`.
    spacing, sites, size, diameter_mm, contrast_hu, edge_sigma_mm, seed:
        As :func:`make_trials`.
    n_trials:
        Number of pairs. Defaults to the number of sites.

    Returns
    -------
    LesionTrials
        ``present[i]`` and ``absent[i]`` share the background of site ``i`` and carry noise
        patches drawn from two different sites, so the pair may be scored together and the
        background contribution cancels.

    """
    background = np.asarray(background)
    noise = np.asarray(noise)
    if background.shape != noise.shape:
        raise ValueError(
            f"background {background.shape} and noise {noise.shape} must be the same volume"
        )
    if len(sites) < 2:
        raise ValueError("at least two sites are needed to draw independent noise patches")

    rng = np.random.default_rng(seed)
    n = len(sites) if n_trials is None else int(n_trials)
    if n > len(sites):
        raise ValueError(f"{n} pairs asked for but only {len(sites)} sites are available")
    order = rng.permutation(len(sites))[:n]

    # Two noise draws per trial, from sites other than the one supplying the background, so
    # that neither noise patch is the one that actually accompanied this anatomy.
    shift_a = rng.integers(1, len(sites), size=n)
    shift_b = rng.integers(1, len(sites), size=n)
    same = shift_a == shift_b
    shift_b[same] = (shift_b[same] % (len(sites) - 1)) + 1
    still_same = shift_a == shift_b
    shift_b[still_same] = (shift_b[still_same] + 1) % len(sites)

    half = size // 2
    signal = disk_lesion(size, spacing, diameter_mm, contrast_hu, edge_sigma_mm=edge_sigma_mm)

    def patch(volume: np.ndarray, index: np.ndarray) -> np.ndarray:
        return np.stack(
            [volume[s, r - half : r + half, c - half : c + half] for s, r, c in index]
        ).astype(np.float64)

    base = patch(background, sites[order])
    noise_a = patch(noise, sites[(order + shift_a) % len(sites)])
    noise_b = patch(noise, sites[(order + shift_b) % len(sites)])

    return LesionTrials(
        present=base + signal[None, :, :] + noise_a,
        absent=base + noise_b,
        signal=signal,
        spacing=spacing,
        locations=sites[order],
        meta={
            "size": size,
            "diameter_mm": diameter_mm,
            "contrast_hu": contrast_hu,
            "edge_sigma_mm": edge_sigma_mm,
            "n_per_class": int(n),
            "n_sites_found": int(len(sites)),
            "seed": seed,
            "construction": (
                "SKE/BKE, paired: one background per trial, two independent measured noise "
                "patches, so the pair can be scored together and the anatomy cancels"
            ),
            "paired": True,
        },
    )


def paired_d_prime(present_scores: np.ndarray, absent_scores: np.ndarray) -> float:
    r"""Detectability from paired scores, where the two members share a background.

    With the background common to the pair, the per-trial difference
    :math:`\Delta = s_1 - s_0` removes it, leaving the signal against *two* independent noise
    draws. Its variance is therefore twice that of a single image, so the detectability of one
    image is

    .. math:: d' = \sqrt{2}\,\frac{\overline{\Delta}}{\mathrm{sd}(\Delta)}

    Forgetting the :math:`\sqrt{2}` understates ``d'`` by 29 %, which is exactly the size of
    difference this kind of study is trying to resolve.
    """
    present_scores = np.asarray(present_scores, dtype=np.float64)
    absent_scores = np.asarray(absent_scores, dtype=np.float64)
    if present_scores.shape != absent_scores.shape:
        raise ValueError("paired scores must come in equal numbers")
    delta = present_scores - absent_scores
    sd = float(delta.std(ddof=1))
    if sd == 0.0:
        raise ValueError("the paired differences have no spread; this is not a measurement")
    return float(np.sqrt(2.0) * delta.mean() / sd)
