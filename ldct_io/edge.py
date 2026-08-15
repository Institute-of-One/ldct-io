r"""MTF from a circular edge, via a radial edge-spread function.

A patient has no known edge, and the slanted-edge estimators written for a *straight* edge do
not apply to a cylindrical phantom. But a circle is as good an oversampling device as a slant:
pixels fall at a continuum of distances from the centre, so binning them by radius builds an
edge-spread function sampled far finer than the pixel pitch.

The three biases a binned ESF carries are the same ones a slanted edge carries, and they are
corrected the same way — they are consequences of the estimator, not fudge factors:

A. **Bin-position jitter.** The mean sample radius inside a bin is not the bin centre. Treating
   the bin average as a sample *at* the centre evaluates the ESF in the wrong place. Each bin
   is shifted back with its own measured mean radius.
B. **The bin average.** Averaging over a bin of width ``h`` is a boxcar: divide by ``sinc(f h)``.
C. **The central difference.** A difference over ``2h`` is not a derivative: divide by
   ``sinc(2 f h)``.

What this estimator will *not* do is measure an edge whose background is not flat. A sloped
pedestal survives differentiation, and because the MTF is normalised at DC the slope quietly
drags the whole curve down. Real CT has such slopes — residual beam hardening and scatter —
so the default refuses, and removing them is something the caller asks for explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class CircleFit:
    """A circle fitted to a high-contrast boundary.

    Attributes
    ----------
    centre_x, centre_y, radius:
        Fitted circle [mm], in the image's coordinate frame.
    residual_sd, residual_ptp:
        Spread of the detected boundary about the fitted circle [mm]. A large *systematic*
        residual means the boundary is not a circle and a radial ESF would smear it; check
        :attr:`ellipticity` before blaming the scanner.
    ellipticity:
        Amplitude of the two-cycle component of the residual [mm] — the signature of an
        elliptical cross-section, i.e. a tilted cylinder.
    n_points:
        Boundary points used.
    meta:
        Settings used.

    """

    centre_x: float
    centre_y: float
    radius: float
    residual_sd: float
    residual_ptp: float
    ellipticity: float
    n_points: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class RadialMTF:
    """MTF measured from a circular edge.

    Attributes
    ----------
    frequency, mtf:
        Radial spatial frequency [cycles/mm] up to the image Nyquist, and the MTF there.
    mtf50, mtf10:
        Frequencies where the MTF crosses 0.5 and 0.1 [cycles/mm]; ``nan`` if not crossed.
    esf_radius, esf:
        The oversampled edge-spread function: radius relative to the fitted circle [mm], and
        its value.
    lsf:
        The line-spread function the MTF was computed from.
    contrast:
        Edge contrast in the image's units.
    meta:
        Bin width, sample counts, the measured tail slopes, and the background treatment.

    """

    frequency: np.ndarray
    mtf: np.ndarray
    mtf50: float
    mtf10: float
    esf_radius: np.ndarray
    esf: np.ndarray
    lsf: np.ndarray
    contrast: float
    meta: dict[str, Any] = field(default_factory=dict)


def _coordinates(image: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    n = image.shape[0]
    axis = (np.arange(n) - (n - 1) / 2.0) * spacing
    return np.meshgrid(axis, axis)


def fit_edge_circle(
    image: np.ndarray,
    spacing: float,
    *,
    search_range: tuple[float, float],
    inside_level: float | None = None,
    outside_level: float | None = None,
    n_azimuth: int = 720,
) -> CircleFit:
    """Fit a circle to a high-contrast boundary by half-height crossing along radial rays.

    Parameters
    ----------
    image:
        Square 2-D image.
    spacing:
        Pixel pitch [mm].
    search_range:
        ``(r_min, r_max)`` [mm] to search for the boundary along each ray.
    inside_level, outside_level:
        Values either side of the boundary. Default: the median inside ``r_min`` and the
        median outside ``r_max``.
    n_azimuth:
        Rays to cast.

    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"image must be square 2-D, got shape {img.shape}")
    if not np.all(np.isfinite(img)):
        raise ValueError("image contains a non-finite value")
    r_min, r_max = search_range
    if not 0.0 < r_min < r_max:
        raise ValueError(f"search_range must satisfy 0 < r_min < r_max, got {search_range}")

    X, Y = _coordinates(img, spacing)
    r_grid = np.hypot(X, Y)
    if inside_level is None:
        inside_level = float(np.median(img[r_grid < r_min]))
    if outside_level is None:
        outside_level = float(np.median(img[r_grid > r_max]))
    if inside_level == outside_level:
        raise ValueError("no contrast across the boundary: inside and outside levels are equal")
    level = 0.5 * (inside_level + outside_level)
    descending = inside_level > outside_level

    mask = img > level if descending else img < level
    if not mask.any():
        raise ValueError("nothing on the inside of the boundary; check search_range")
    x0, y0 = float(X[mask].mean()), float(Y[mask].mean())

    n = img.shape[0]
    axis0 = -(n - 1) / 2.0 * spacing
    radii = np.arange(r_min, r_max, spacing / 8.0)
    points = []
    for a in np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False):
        xs, ys = x0 + radii * np.cos(a), y0 + radii * np.sin(a)
        ix = np.clip(((xs - axis0) / spacing).astype(int), 0, n - 1)
        iy = np.clip(((ys - axis0) / spacing).astype(int), 0, n - 1)
        profile = img[iy, ix]
        crossed = np.flatnonzero(profile < level if descending else profile > level)
        if crossed.size and crossed[0] > 0:
            i = crossed[0]
            t = (profile[i - 1] - level) / (profile[i - 1] - profile[i])
            r_edge = radii[i - 1] + t * (radii[i] - radii[i - 1])
            points.append((x0 + r_edge * np.cos(a), y0 + r_edge * np.sin(a)))

    if len(points) < 32:
        raise ValueError(
            f"only {len(points)} boundary crossings found of {n_azimuth} rays; the boundary is "
            f"not where search_range says it is"
        )
    P = np.asarray(points)
    A = np.column_stack([2.0 * P[:, 0], 2.0 * P[:, 1], np.ones(len(P))])
    cx, cy, c = np.linalg.lstsq(A, (P**2).sum(axis=1), rcond=None)[0]
    radius = float(np.sqrt(c + cx**2 + cy**2))

    r_fit = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    resid = r_fit - radius
    theta = np.arctan2(P[:, 1] - cy, P[:, 0] - cx)
    two_cycle = float(
        np.hypot(2 * np.mean(resid * np.cos(2 * theta)), 2 * np.mean(resid * np.sin(2 * theta)))
    )
    return CircleFit(
        centre_x=float(cx),
        centre_y=float(cy),
        radius=radius,
        residual_sd=float(resid.std()),
        residual_ptp=float(np.ptp(resid)),
        ellipticity=two_cycle,
        n_points=len(P),
        meta={"level": level, "inside": inside_level, "outside": outside_level},
    )


def radial_mtf(
    image: np.ndarray,
    spacing: float,
    circle: CircleFit,
    *,
    band: float = 6.0,
    bin_subsample: int = 10,
    min_bin_count: int = 3,
    background: str = "flat",
    tail_fraction: float = 0.45,
    tail_tolerance: float = 0.005,
    jitter_correction: bool = True,
) -> RadialMTF:
    """Presampled MTF from the circular edge described by ``circle``.

    Parameters
    ----------
    image:
        Square 2-D image the edge lives in.
    spacing:
        Pixel pitch [mm].
    circle:
        The fitted edge, from :func:`fit_edge_circle`.
    band:
        Half-width of the radial window about the edge [mm]. Wide enough that the LSF has
        decayed, narrow enough to stay clear of other structure.
    bin_subsample:
        ESF bins per pixel; the bin width is ``spacing / bin_subsample``.
    min_bin_count:
        Samples a bin needs to be used.
    background:
        ``"flat"`` (default) requires the ESF to be flat on both sides and raises if it is
        not. ``"asymptote"`` fits a straight line to each side and normalises the ESF onto a
        0..1 step between them — use it when a residual gradient is known to be present and
        the measurement is intended to be of the edge alone, and say so in the write-up: it
        removes a real property of the image.
    tail_fraction:
        Fraction of ``band`` at each end treated as tail.
    tail_tolerance:
        Largest tail slope tolerated in ``"flat"`` mode, as a fraction of the edge contrast
        over the tail's own length.
    jitter_correction:
        Apply bias A.

    Raises
    ------
    ValueError
        Too few usable bins; no contrast; or, in ``"flat"`` mode, a tail that is not flat.

    """
    img = np.asarray(image, dtype=np.float64)
    if background not in ("flat", "asymptote"):
        raise ValueError(f"background must be 'flat' or 'asymptote', got {background!r}")
    if not 0.0 < tail_fraction < 0.9:
        raise ValueError(f"tail_fraction must be in (0, 0.9), got {tail_fraction}")

    X, Y = _coordinates(img, spacing)
    r = np.hypot(X - circle.centre_x, Y - circle.centre_y)
    sel = np.abs(r - circle.radius) <= band
    if sel.sum() < 500:
        raise ValueError(f"only {int(sel.sum())} pixels in the edge band; widen `band`")

    h = spacing / bin_subsample
    edges = np.arange(circle.radius - band, circle.radius + band + h, h)
    which = np.digitize(r[sel], edges) - 1
    n_bins = edges.size - 1
    count = np.bincount(which, minlength=n_bins)[:n_bins]
    sum_v = np.bincount(which, weights=img[sel], minlength=n_bins)[:n_bins]
    sum_r = np.bincount(which, weights=r[sel], minlength=n_bins)[:n_bins]
    good = count >= min_bin_count
    if good.sum() < 64:
        raise ValueError(
            f"only {int(good.sum())} ESF bins have {min_bin_count}+ samples; reduce "
            f"bin_subsample or widen band"
        )

    esf = sum_v[good] / count[good]
    r_bar = sum_r[good] / count[good]
    r_centre = (0.5 * (edges[:-1] + edges[1:]))[good]
    u = r_centre - circle.radius

    if jitter_correction:
        esf = esf + (r_centre - r_bar) * np.gradient(esf, r_bar)

    tail = tail_fraction * band
    inner = u < -(band - tail)
    outer = u > (band - tail)
    if inner.sum() < 8 or outer.sum() < 8:
        raise ValueError("not enough tail samples to characterise the background")
    p_in = np.polyfit(u[inner], esf[inner], 1)
    p_out = np.polyfit(u[outer], esf[outer], 1)
    contrast = float(np.polyval(p_in, 0.0) - np.polyval(p_out, 0.0))
    if contrast == 0.0:
        raise ValueError("no contrast across the edge")

    slope_in = float(p_in[0] * tail / contrast)
    slope_out = float(p_out[0] * tail / contrast)
    if background == "flat":
        worst = max(abs(slope_in), abs(slope_out))
        if worst > tail_tolerance:
            raise ValueError(
                f"the ESF background is not flat: the tails drift by {worst * 100:.2f} % of "
                f"the edge contrast over {tail:.1f} mm (tolerance {tail_tolerance * 100:.2f} %). "
                f"A sloped background survives differentiation and, through the DC "
                f"normalisation, lowers the whole MTF. Pass background='asymptote' to fit and "
                f"remove it deliberately, and report that you did."
            )
        work = esf
    else:
        lo = np.polyval(p_out, u)
        hi = np.polyval(p_in, u)
        work = (esf - lo) / (hi - lo)

    lsf = (work[2:] - work[:-2]) / (2.0 * h)
    lsf = lsf - np.median(np.r_[lsf[:15], lsf[-15:]])
    frequency = np.fft.rfftfreq(lsf.size, h)
    spectrum = np.abs(np.fft.rfft(lsf))
    if spectrum[0] == 0.0:
        raise ValueError("the LSF integrates to zero; this is not an edge")
    mtf = spectrum / spectrum[0]
    mtf = mtf / (np.sinc(frequency * h) * np.sinc(2.0 * frequency * h))  # biases B and C

    nyquist = 0.5 / spacing
    keep = frequency <= nyquist
    frequency, mtf = frequency[keep], mtf[keep]

    def crossing(target: float) -> float:
        below = np.flatnonzero(mtf < target)
        if not below.size or below[0] == 0:
            return float("nan")
        i = below[0]
        t = (mtf[i - 1] - target) / (mtf[i - 1] - mtf[i])
        return float(frequency[i - 1] + t * (frequency[i] - frequency[i - 1]))

    return RadialMTF(
        frequency=frequency,
        mtf=mtf,
        mtf50=crossing(0.5),
        mtf10=crossing(0.1),
        esf_radius=u,
        esf=work,
        lsf=lsf,
        contrast=contrast,
        meta={
            "bin_width_mm": h,
            "n_bins": int(good.sum()),
            "min_samples_per_bin": int(count[good].min()),
            "max_samples_per_bin": int(count[good].max()),
            "tail_slope_inner_fraction": slope_in,
            "tail_slope_outer_fraction": slope_out,
            "background": background,
            "jitter_correction": jitter_correction,
            "image_nyquist": nyquist,
            "radius_mm": circle.radius,
        },
    )
