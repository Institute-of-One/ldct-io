r"""Reconstruct a helical acquisition by backprojecting along the rays as measured.

Single-slice rebinning (:mod:`ldct_io.rebin`) has to pretend that a measured ray lies in the
plane being reconstructed. It does not: at radius :math:`r` the ray has already strayed
:math:`r\,(z_0 - z_s)/R` out of the plane, several millimetres at the edge of a body. On an
object that does not vary along z that costs nothing, and on a chest it is ruinous — measured
against the vendor's own reconstruction of the same projections, single-slice rebinning put
air at :math:`-815 \pm 666` HU where the vendor had :math:`-1004 \pm 22`.

The fix is not a better choice of row. It is to stop choosing one: for every voxel and every
view, use the ray that actually passes through that voxel, wherever on the detector it landed.
That is a three-dimensional backprojection, and it is what this module does, in the weighted
form of Stierstorfer et al. (*Phys. Med. Biol.* **49** (2004) 2209).

The weighting, and what it is for
---------------------------------
A voxel is illuminated for as long as it stays inside the beam — here about 1.1 rotations,
because the collimation (38.4 mm at the isocentre) slightly exceeds the table feed per
rotation (34.4 mm). Backprojecting all of it would count some directions more than once. Each
contribution is therefore weighted by a smooth window in the normalised row coordinate
:math:`q = (v - v_0)/(N_{\text{rows}}/2)`, which tapers to zero at the edge of the beam so that
a voxel does not enter and leave the sum abruptly, and the total is renormalised per voxel so
that the angular measure integrates to exactly :math:`2\pi`.

That renormalisation is exact when the redundancy is spread evenly over direction, and
approximate when it is not. Rather than argue about the size of the residual, the test suite
measures it: a uniform sphere has an analytic cone-beam projection, so the reconstruction of
any plane through it has a known answer — a disk of known radius and known attenuation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ldct_io.dicomctpd import ProjectionSeries
from ldct_io.geometry import ScanGeometry
from ldct_io.recon import ReconResult, ramp_kernel


@dataclass(frozen=True, eq=False)
class Illumination:
    r"""Which views can see a plane, and how completely.

    Attributes
    ----------
    views:
        Indices of the views whose beam reaches the plane at the isocentre.
    angular_range:
        Source rotation those views span [rad]. Below :math:`2\pi` the plane is not fully
        sampled at the isocentre and the reconstruction will be incomplete.
    redundancy:
        ``angular_range / 2pi``. Above 1 the extra is what the weighting has to normalise away.
    meta:
        Provenance.

    """

    views: np.ndarray
    angular_range: float
    redundancy: float
    meta: dict[str, Any] = field(default_factory=dict)


def illuminating_views(
    geometry: ScanGeometry,
    source_z: np.ndarray,
    angles: np.ndarray,
    z: float,
    *,
    margin: float = 1.0,
) -> Illumination:
    """The views whose beam covers the plane ``z``, and how much rotation they span."""
    half_beam = 0.5 * geometry.collimation_at_isocentre * margin
    inside = np.flatnonzero(np.abs(source_z - z) <= half_beam)
    if inside.size < 8:
        raise ValueError(
            f"only {inside.size} views illuminate z = {z} mm; the plane is outside the scan"
        )

    # A helix passes a plane once, so the illuminating views are one run. They need not each
    # satisfy the test, though: a z-flying focal spot moves the source by a fraction of a
    # millimetre from view to view, so at the very edge of the beam the alternation flickers
    # in and out of the window. Take the whole run — the row window zeroes anything that
    # really has fallen off the detector — but insist that the run is mostly inside, because a
    # run that is not means the series is broken rather than merely dithered.
    views = np.arange(inside[0], inside[-1] + 1)
    fraction_inside = inside.size / views.size
    if fraction_inside < 0.9:
        raise ValueError(
            f"only {fraction_inside:.0%} of the views between the first and last that "
            f"illuminate z = {z} mm are themselves within the beam; this is not one pass of a "
            f"helix past the plane"
        )

    unwrapped = np.unwrap(angles[views])
    span = float(abs(unwrapped[-1] - unwrapped[0]))
    return Illumination(
        views=views,
        angular_range=span,
        redundancy=span / (2.0 * np.pi),
        meta={
            "half_beam_mm": half_beam,
            "n_views": int(views.size),
            "fraction_within_beam": float(fraction_inside),
        },
    )


def row_window(q: np.ndarray, taper: float = 0.35) -> np.ndarray:
    """A smooth window on the normalised row coordinate ``q`` in ``[-1, 1]``.

    Flat over the middle of the detector and tapering to zero at its edges with a raised
    cosine, so that a voxel fades in and out of the sum instead of switching. A hard edge
    would put a step in the angular weighting of every voxel and stripe the image along z.
    """
    if not 0.0 < taper <= 1.0:
        raise ValueError(f"taper must be in (0, 1], got {taper}")
    a = np.abs(np.asarray(q, dtype=np.float64))
    w = np.zeros_like(a)
    flat = a <= 1.0 - taper
    w[flat] = 1.0
    edge = (a > 1.0 - taper) & (a < 1.0)
    w[edge] = 0.5 * (1.0 + np.cos(np.pi * (a[edge] - (1.0 - taper)) / taper))
    return w


def wfbp_slice(
    frames: Callable[[int], np.ndarray] | np.ndarray,
    geometry: ScanGeometry,
    angles: np.ndarray,
    source_z: np.ndarray,
    z: float,
    *,
    fov: float = 400.0,
    n_pixels: int = 512,
    source_radius: np.ndarray | float | None = None,
    taper: float = 0.35,
    margin: float = 1.0,
) -> ReconResult:
    """Reconstruct the plane ``z`` by weighted three-dimensional backprojection.

    Parameters
    ----------
    frames:
        ``(n_views, n_channels, n_rows)`` of line integrals, or a callable taking a view index
        and returning one such frame. A callable keeps a 40 GB series off the heap.
    geometry, angles, source_z:
        The acquisition. ``angles`` and ``source_z`` have one entry per view.
    z:
        Plane to reconstruct, in the frame of ``source_z``.
    fov, n_pixels:
        Field of view [mm] and grid size.
    source_radius:
        Per-view source-to-isocentre distance, if the focal spot is displaced radially.
    taper:
        Width of the raised-cosine roll-off at the edges of the detector, as a fraction of the
        half-detector. See :func:`row_window`.
    margin:
        Fraction of the collimation to accept when choosing illuminating views.

    """
    angles = np.asarray(angles, dtype=np.float64)
    source_z = np.asarray(source_z, dtype=np.float64)
    if angles.size != source_z.size:
        raise ValueError(f"{angles.size} angles but {source_z.size} source positions")

    illum = illuminating_views(geometry, source_z, angles, z, margin=margin)
    if illum.redundancy < 0.999:
        raise ValueError(
            f"the views that illuminate z = {z} mm span only "
            f"{np.degrees(illum.angular_range):.1f} deg. Less than a full rotation cannot be "
            f"reconstructed without a short-scan weighting this function does not apply."
        )

    if source_radius is None:
        radius = np.full(angles.size, geometry.source_to_isocentre)
    else:
        radius = np.broadcast_to(np.asarray(source_radius, dtype=np.float64), angles.shape)

    spacing = fov / n_pixels
    axis = (np.arange(n_pixels) - (n_pixels - 1) / 2.0) * spacing
    X, Y = np.meshgrid(axis, axis)

    kernel = ramp_kernel(geometry.n_channels, geometry.channel_angle_spacing)
    n_fft = 1 << int(np.ceil(np.log2(geometry.n_channels + kernel.size)))
    kernel_f = np.fft.rfft(kernel, n_fft)[:, None]

    d_gamma = geometry.channel_angle_spacing
    u0, v0 = geometry.central_channel, geometry.central_row
    n_ch, n_rows = geometry.n_channels, geometry.n_rows
    half_rows = 0.5 * n_rows
    rows_mm = (np.arange(n_rows) - v0) * geometry.axial_spacing
    cos_cone = geometry.source_to_detector / np.hypot(geometry.source_to_detector, rows_mm)
    cos_gamma = np.cos(geometry.channel_angles())

    accum = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    weight = np.zeros((n_pixels, n_pixels), dtype=np.float64)

    for k_np in illum.views:
        k = int(k_np)
        frame = frames(k) if callable(frames) else np.asarray(frames[k], dtype=np.float64)
        if frame.shape != (n_ch, n_rows):
            raise ValueError(f"view {k}: frame shape {frame.shape}, expected {(n_ch, n_rows)}")

        weighted = frame * (geometry.source_to_isocentre * cos_gamma)[:, None] * cos_cone[None, :]
        spectrum = np.fft.rfft(weighted, n_fft, axis=0)
        filtered = np.fft.irfft(spectrum * kernel_f, n_fft, axis=0)
        filtered = filtered[n_ch - 1 : 2 * n_ch - 1] * d_gamma

        d = radius[k]
        sx, sy = -d * np.sin(angles[k]), d * np.cos(angles[k])
        dx, dy = X - sx, Y - sy
        d_xy2 = dx * dx + dy * dy
        d_xy = np.sqrt(d_xy2)
        cx, cy = -sx / d, -sy / d
        gamma = np.arctan2(cx * dy - cy * dx, cx * dx + cy * dy)
        channel = gamma / d_gamma + u0

        # The row the ray through this voxel actually landed on: it rises (z - z_s) over an
        # in-plane run of d_xy, and the detector is SDD away in that same in-plane sense.
        row = v0 + (z - source_z[k]) * geometry.source_to_detector / (d_xy * geometry.axial_spacing)

        w = row_window((row - v0) / half_rows, taper=taper)
        live = (w > 0.0) & (channel >= 0.0) & (channel <= n_ch - 1)
        if not live.any():
            continue

        c = np.clip(channel[live], 0.0, n_ch - 1 - 1e-9)
        r_ = np.clip(row[live], 0.0, n_rows - 1 - 1e-9)
        j0 = c.astype(np.int32)
        i0 = r_.astype(np.int32)
        fj = c - j0
        fi = r_ - i0
        j1 = np.minimum(j0 + 1, n_ch - 1)
        i1 = np.minimum(i0 + 1, n_rows - 1)
        value = (
            filtered[j0, i0] * (1 - fj) * (1 - fi)
            + filtered[j1, i0] * fj * (1 - fi)
            + filtered[j0, i1] * (1 - fj) * fi
            + filtered[j1, i1] * fj * fi
        )
        ww = w[live]
        accum[live] += ww * value / d_xy2[live]
        weight[live] += ww

    step = float(np.abs(np.diff(np.unwrap(angles[illum.views]))).mean())
    empty = weight <= 0.0
    # Renormalise so that every voxel integrates over exactly 2 pi of source rotation.
    scale = np.zeros_like(weight)
    scale[~empty] = 2.0 * np.pi / (weight[~empty] * step)
    image = accum * step * scale

    return ReconResult(
        image=image,
        spacing=spacing,
        fov=fov,
        meta={
            "method": "weighted 3-D backprojection (Stierstorfer-type)",
            "z": float(z),
            "n_views": int(illum.views.size),
            "redundancy": illum.redundancy,
            "taper": taper,
            "unreconstructed_fraction": float(empty.mean()),
            "filter": "equiangular ramp per detector row",
        },
    )


def wfbp_from_series(
    series: ProjectionSeries,
    z: float,
    *,
    fov: float = 400.0,
    n_pixels: int = 512,
    taper: float = 0.35,
    margin: float = 1.0,
) -> ReconResult:
    """:func:`wfbp_slice` on an indexed series, reading frames as it goes."""
    views = series.views
    return wfbp_slice(
        lambda k: series.read_frame(k).astype(np.float64),
        series.geometry,
        views.angle,
        views.axial_position + views.ffs_dz,
        z,
        fov=fov,
        n_pixels=n_pixels,
        source_radius=series.geometry.source_to_isocentre + views.ffs_drho,
        taper=taper,
        margin=margin,
    )
