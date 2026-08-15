r"""Equiangular fan-beam filtered backprojection.

Written out rather than delegated, for two reasons. The projectors that could be delegated to
cannot express a source position that moves from view to view, which is exactly what a flying
focal spot does; and a reconstruction that a reviewer cannot run without a CUDA build is not
reproducible. This runs on the CPU in seconds per slice and is checked against a closed form
in the test suite — the analytic fan-beam projection of a uniform disk, reconstructed and
compared against its own known attenuation, radius and edge response.

Convention
----------
Source at angle :math:`\beta` sits at :math:`(-D\sin\beta,\; D\cos\beta)` with
:math:`D` the source-to-isocentre distance; :math:`\gamma` is the fan angle of a ray,
positive towards increasing channel index. The reconstruction returns attenuation in the
units of the projections (1/mm for DICOM-CT-PD), which :func:`to_hu` converts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ldct_io.geometry import ScanGeometry


@dataclass(frozen=True, eq=False)
class ReconResult:
    """A reconstructed slice.

    Attributes
    ----------
    image:
        Attenuation ``(n_pixels, n_pixels)`` in the units of the input projections.
    spacing:
        Pixel pitch [mm].
    fov:
        Field of view [mm]; ``spacing * n_pixels``.
    meta:
        Settings used, including the filter and the number of views.

    """

    image: np.ndarray
    spacing: float
    fov: float
    meta: dict[str, Any] = field(default_factory=dict)

    def pixel_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, Y)`` coordinate grids [mm], centred on the isocentre."""
        n = self.image.shape[0]
        axis = (np.arange(n) - (n - 1) / 2.0) * self.spacing
        return np.meshgrid(axis, axis)


def ramp_kernel(n_channels: int, d_gamma: float) -> np.ndarray:
    r"""The equiangular ramp filter, sampled on the channel grid.

    For an equiangular detector the reconstruction filter is not the parallel-beam ramp: the
    discrete kernel is

    .. math::
        g(0) = \frac{1}{8\,\Delta\gamma^2},\quad g(2k) = 0,\quad
        g(2k+1) = \frac{-1}{2\pi^2 \sin^2\!\big((2k+1)\Delta\gamma\big)}

    (Kak & Slaney, *Principles of Computerized Tomographic Imaging*, §3.4). Using the
    parallel-beam kernel here is a common and quiet error: it reconstructs, and the image
    looks right, but the scale and the high-frequency response are wrong.
    """
    if n_channels < 2:
        raise ValueError(f"n_channels must be at least 2, got {n_channels}")
    if not np.isfinite(d_gamma) or d_gamma <= 0.0:
        raise ValueError(f"d_gamma must be finite and positive, got {d_gamma!r}")
    n = np.arange(-n_channels + 1, n_channels)
    g = np.zeros(n.size, dtype=np.float64)
    g[n == 0] = 1.0 / (8.0 * d_gamma**2)
    odd = n % 2 != 0
    g[odd] = -0.5 / (np.pi * np.sin(n[odd] * d_gamma)) ** 2
    return g


def parker_weights(geometry: ScanGeometry, angles: np.ndarray) -> np.ndarray:
    r"""Parker weights for a short scan, ``(n_views, n_channels)``.

    Over :math:`\pi + 2\Gamma` of source rotation every line through the object is measured
    once, except in two wedges at the ends of the range where it is measured twice. Weighting
    those wedges so that each pair of conjugate rays sums to one is what makes a short scan
    reconstruct to the same values as a full rotation; without it the doubly-measured
    directions are counted twice and the image gains a smooth, plausible-looking bias.

    The advantage of paying that price is in z: a short scan spans half the axial travel of a
    full rotation, so in a helical acquisition it halves how far each ray strays from the
    plane being reconstructed.
    """
    angles = np.asarray(angles, dtype=np.float64)
    gamma = geometry.channel_angles()
    half_fan = geometry.half_fan_angle

    beta = np.unwrap(angles)
    beta = np.abs(beta - beta[0])  # measured from the start, whichever way the gantry turns
    # One view covers its own angular step, so the range spanned is the last angle plus a step.
    span = float(beta[-1] + np.abs(np.diff(beta)).mean())
    required = geometry.short_scan_range
    if span < required - 1e-3:
        raise ValueError(
            f"a short scan needs at least pi + 2*half-fan = {np.degrees(required):.1f} deg, "
            f"got {np.degrees(span):.1f} deg"
        )

    b = beta[:, None]
    g = gamma[None, :]
    w = np.ones((angles.size, geometry.n_channels), dtype=np.float64)

    lead = 2.0 * (half_fan - g)
    rising = b < lead
    w[rising] = (np.sin(np.pi / 4.0 * np.divide(b, np.maximum(lead, 1e-12)) * 2.0) ** 2)[rising]

    trail_start = np.pi - 2.0 * g
    falling = b > trail_start
    arg = (np.pi + 2.0 * half_fan - b) / np.maximum(2.0 * (half_fan + g), 1e-12)
    w[falling] = (np.sin(np.pi / 4.0 * arg * 2.0) ** 2)[falling]

    return np.clip(w, 0.0, 1.0)


def filter_projections(
    sinogram: np.ndarray, geometry: ScanGeometry, *, apodisation: str = "none"
) -> np.ndarray:
    """Cosine-weight and ramp-filter a fan-beam sinogram.

    Parameters
    ----------
    sinogram:
        ``(n_views, n_channels)`` of line integrals.
    geometry:
        Supplies the fan angles and the channel spacing.
    apodisation:
        ``"none"`` (default) is the bare ramp — the sharpest reconstruction the sampling
        allows, and the one whose transfer function is known exactly. ``"hann"`` applies a
        Hann window, which is what a clinical "smooth" kernel resembles; use it when the point
        is to mimic a vendor kernel, not when the point is to measure the system.

    """
    sino = np.asarray(sinogram, dtype=np.float64)
    if sino.ndim != 2:
        raise ValueError(f"sinogram must be 2-D (n_views, n_channels), got {sino.shape}")
    if sino.shape[1] != geometry.n_channels:
        raise ValueError(
            f"sinogram has {sino.shape[1]} channels but the geometry says {geometry.n_channels}"
        )
    if not np.all(np.isfinite(sino)):
        raise ValueError("sinogram contains a non-finite value")

    d_gamma = geometry.channel_angle_spacing
    weighted = sino * (geometry.source_to_isocentre * np.cos(geometry.channel_angles()))[None, :]

    g = ramp_kernel(geometry.n_channels, d_gamma)
    n_fft = 1 << int(np.ceil(np.log2(geometry.n_channels + g.size)))
    kernel = np.fft.rfft(g, n_fft)
    if apodisation == "hann":
        f = np.fft.rfftfreq(n_fft)
        kernel = kernel * (0.5 + 0.5 * np.cos(np.pi * f / f.max()))
    elif apodisation != "none":
        raise ValueError(f"unknown apodisation {apodisation!r}; use 'none' or 'hann'")

    spectrum = np.fft.rfft(weighted, n_fft, axis=1)
    start = geometry.n_channels - 1
    out = np.fft.irfft(spectrum * kernel[None, :], n_fft, axis=1)
    return out[:, start : start + geometry.n_channels] * d_gamma


def backproject(
    filtered: np.ndarray,
    geometry: ScanGeometry,
    angles: np.ndarray,
    *,
    fov: float,
    n_pixels: int,
    source_radius: np.ndarray | float | None = None,
) -> np.ndarray:
    """Weighted backprojection of filtered fan-beam data onto a square grid.

    Parameters
    ----------
    filtered:
        ``(n_views, n_channels)`` from :func:`filter_projections`.
    geometry:
        Supplies the channel spacing and the central channel the rays are indexed against.
    angles:
        Source angle of every view [rad].
    fov, n_pixels:
        Field of view [mm] and grid size; the pixel pitch is ``fov / n_pixels``.
    source_radius:
        Source-to-isocentre distance per view, if the focal spot is displaced radially.
        Scalar or one value per view. Defaults to the geometry's nominal distance.

    """
    filtered = np.asarray(filtered, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if filtered.shape[0] != angles.size:
        raise ValueError(f"{filtered.shape[0]} views but {angles.size} angles")
    if n_pixels < 2 or fov <= 0.0:
        raise ValueError(f"implausible grid: fov={fov}, n_pixels={n_pixels}")

    if source_radius is None:
        radius = np.full(angles.size, geometry.source_to_isocentre)
    else:
        radius = np.broadcast_to(np.asarray(source_radius, dtype=np.float64), angles.shape).copy()

    spacing = fov / n_pixels
    axis = (np.arange(n_pixels) - (n_pixels - 1) / 2.0) * spacing
    X, Y = np.meshgrid(axis, axis)
    image = np.zeros((n_pixels, n_pixels), dtype=np.float64)

    d_gamma = geometry.channel_angle_spacing
    u0 = geometry.central_channel
    n_ch = geometry.n_channels

    for k in range(angles.size):
        d = radius[k]
        sx, sy = -d * np.sin(angles[k]), d * np.cos(angles[k])
        dx, dy = X - sx, Y - sy
        inv_l2 = 1.0 / (dx * dx + dy * dy)
        cx, cy = -sx / d, -sy / d  # unit vector from source towards the isocentre
        gamma = np.arctan2(cx * dy - cy * dx, cx * dx + cy * dy)
        pos = gamma / d_gamma + u0
        j = np.clip(np.floor(pos).astype(np.int32), 0, n_ch - 2)
        w = np.clip(pos - j, 0.0, 1.0)
        row = filtered[k]
        image += (row[j] * (1.0 - w) + row[j + 1] * w) * inv_l2

    step = float(np.abs(np.diff(np.unwrap(angles))).mean())
    return image * step


def fan_beam_fbp(
    sinogram: np.ndarray,
    geometry: ScanGeometry,
    angles: np.ndarray,
    *,
    fov: float = 260.0,
    n_pixels: int = 512,
    apodisation: str = "none",
    source_radius: np.ndarray | float | None = None,
    short_scan: bool = False,
) -> ReconResult:
    """Reconstruct one slice from an equiangular fan-beam sinogram.

    Parameters
    ----------
    sinogram:
        ``(n_views, n_channels)`` of line integrals.
    geometry:
        The acquisition geometry.
    angles:
        Source angle of every view [rad].
    fov, n_pixels:
        Field of view [mm] and grid size.
    apodisation:
        Passed to :func:`filter_projections`.
    source_radius:
        Per-view source-to-isocentre distance, if the focal spot is displaced radially.
    short_scan:
        ``False`` (default) requires a full rotation. ``True`` accepts
        :attr:`~ldct_io.geometry.ScanGeometry.short_scan_range` and applies
        :func:`parker_weights`. Reconstructing an under-covered set *without* those weights
        produces a plausible image with the wrong values, so the choice is explicit rather
        than inferred from the data.

    """
    angles = np.asarray(angles, dtype=np.float64)
    covered = float(np.abs(np.unwrap(angles)[-1] - np.unwrap(angles)[0]))
    step = float(np.abs(np.diff(np.unwrap(angles))).mean())
    if not short_scan and covered + step < 2.0 * np.pi - 1e-3:
        raise ValueError(
            f"views cover {np.degrees(covered + step):.1f} deg, less than a full rotation. "
            f"Pass short_scan=True to apply Parker weighting; without it the result would be "
            f"quietly wrong rather than obviously wrong."
        )

    sinogram = np.asarray(sinogram, dtype=np.float64)
    if short_scan:
        sinogram = sinogram * parker_weights(geometry, angles)

    filtered = filter_projections(sinogram, geometry, apodisation=apodisation)
    image = backproject(
        filtered, geometry, angles, fov=fov, n_pixels=n_pixels, source_radius=source_radius
    )

    # A full rotation measures every line through the object twice, once from each side, and
    # the equiangular kernel used here carries that redundancy in its normalisation. Parker
    # weights make a short scan count every line exactly once instead, so the same kernel
    # returns half the attenuation — an error that looks entirely plausible in an image and is
    # caught only by comparing against a known value. Undo it explicitly.
    if short_scan:
        image = image * 2.0

    return ReconResult(
        image=image,
        spacing=fov / n_pixels,
        fov=fov,
        meta={
            "n_views": int(angles.size),
            "angular_coverage_deg": float(np.degrees(covered + step)),
            "apodisation": apodisation,
            "short_scan": short_scan,
            "filter": "equiangular ramp (Kak & Slaney 3.4)",
            "interpolation": "linear in the channel direction",
        },
    )


def to_hu(image: np.ndarray, water_attenuation: float) -> np.ndarray:
    """Convert attenuation to Hounsfield units.

    ``water_attenuation`` must be the value the scanner recorded for this acquisition, not a
    textbook constant: it encodes the effective beam energy the vendor calibrated against.
    """
    if not np.isfinite(water_attenuation) or water_attenuation <= 0.0:
        raise ValueError(
            f"water_attenuation must be finite and positive, got {water_attenuation!r}"
        )
    return (np.asarray(image, dtype=np.float64) - water_attenuation) / water_attenuation * 1000.0
