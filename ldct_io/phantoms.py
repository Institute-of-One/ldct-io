r"""Analytic fan-beam projections — the closed-form answers the pipeline is checked against.

A reconstruction that is only compared against its own past output cannot be shown to be
right. These generators produce sinograms whose reconstruction is known in advance: a uniform
disk of attenuation :math:`\mu` must come back at :math:`\mu` inside, zero outside, with its
stated radius, and with an edge response set only by the sampling.

A uniform disk is also the exact model of a cylindrical phantom in a *helical* scan, because
the object does not vary along z: every ray sees the same object regardless of its axial
position, so single-slice rebinning of a helical acquisition and a circular scan of the same
object give the same sinogram. That is what lets a circular-scan closed form validate a
helical pipeline.
"""

from __future__ import annotations

import numpy as np

from ldct_io.geometry import ScanGeometry


def disk_projection(
    geometry: ScanGeometry,
    radius: float,
    attenuation: float,
    *,
    centre: tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    n_subsamples: int = 1,
) -> np.ndarray:
    r"""Line integrals through a uniform disk, one view, ``(n_channels,)``.

    A ray whose perpendicular distance from the disk centre is :math:`t` has path length
    :math:`2\sqrt{a^2 - t^2}` inside a disk of radius :math:`a`, so the projection is
    :math:`2\mu\sqrt{a^2 - t^2}` — exact, with no discretisation anywhere.

    Parameters
    ----------
    geometry:
        Supplies the fan angles and the source distance.
    radius, attenuation:
        Disk radius [mm] and attenuation [1/mm].
    centre:
        Disk centre ``(x, y)`` [mm] in the fixed frame.
    angle:
        Source angle [rad] of this view.
    n_subsamples:
        Sub-samples across each detector element, averaged. ``1`` gives ideal line integrals
        (a point-sampled detector); a larger value models the element's finite aperture, which
        is the dominant blur of the real acquisition. Use ``1`` when validating the
        reconstruction alone, and a larger value when comparing against a real edge.

    """
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError(f"radius must be finite and positive, got {radius!r}")
    if n_subsamples < 1:
        raise ValueError(f"n_subsamples must be at least 1, got {n_subsamples}")

    d = geometry.source_to_isocentre
    d_gamma = geometry.channel_angle_spacing
    j = np.arange(geometry.n_channels)

    # Perpendicular distance from the disk centre to the ray, in the source frame.
    sx, sy = -d * np.sin(angle), d * np.cos(angle)
    cx, cy = -sx / d, -sy / d  # unit vector source -> isocentre
    ox, oy = centre[0] - sx, centre[1] - sy  # source -> disk centre
    # Signed angle of the disk centre from the central ray, and its distance.
    centre_gamma = np.arctan2(cx * oy - cy * ox, cx * ox + cy * oy)
    centre_distance = float(np.hypot(ox, oy))

    offsets = (
        (np.arange(n_subsamples) - (n_subsamples - 1) / 2.0) / n_subsamples
        if n_subsamples > 1
        else np.zeros(1)
    )
    total = np.zeros(geometry.n_channels, dtype=np.float64)
    for off in offsets:
        gamma = (j + off - geometry.central_channel) * d_gamma
        t = centre_distance * np.sin(gamma - centre_gamma)
        inside = np.abs(t) < radius
        p = np.zeros(geometry.n_channels, dtype=np.float64)
        p[inside] = 2.0 * attenuation * np.sqrt(radius**2 - t[inside] ** 2)
        total += p
    return total / offsets.size


def disk_sinogram(
    geometry: ScanGeometry,
    radius: float,
    attenuation: float,
    angles: np.ndarray,
    *,
    centre: tuple[float, float] = (0.0, 0.0),
    n_subsamples: int = 1,
) -> np.ndarray:
    """A full sinogram of a uniform disk, ``(n_views, n_channels)``."""
    angles = np.asarray(angles, dtype=np.float64)
    return np.stack(
        [
            disk_projection(
                geometry,
                radius,
                attenuation,
                centre=centre,
                angle=float(a),
                n_subsamples=n_subsamples,
            )
            for a in angles
        ]
    )


def sampling_chain_mtf(
    geometry: ScanGeometry, frequency: np.ndarray, pixel_spacing: float
) -> np.ndarray:
    r"""The transfer function the reconstruction's own sampling imposes, in closed form.

    Three boxcars in series, none of them adjustable:

    * the detector element, width ``transverse_spacing_at_isocentre``  ->  ``sinc(f w)``
    * linear interpolation in the backprojector, a triangle of the same width  ->  ``sinc^2(f w)``
    * the image pixel  ->  ``sinc(f p)``

    A reconstruction of an ideal step must not be sharper than this, and — with a
    point-sampled detector — must not be much blurrier either. It is the reference the test
    suite holds :func:`~ldct_io.recon.fan_beam_fbp` to, and the floor against which a *real*
    edge measurement is judged: anything blurrier than this in real data is the scanner, not
    the code.
    """
    f = np.asarray(frequency, dtype=np.float64)
    w = geometry.transverse_spacing_at_isocentre
    return np.sinc(f * w) * np.sinc(f * w) ** 2 * np.sinc(f * pixel_spacing)
