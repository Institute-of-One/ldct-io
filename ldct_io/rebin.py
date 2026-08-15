"""Turn a helical acquisition into the circular fan-beam sinogram of one slice.

Single-slice rebinning assigns to each output ray one measured ray, chosen by its detector
row. Two choices of row are offered here: the naive one, which puts the ray in the target
plane at the isocentre, and the ``cos(gamma)`` one, which puts it there where it comes closest
to the rotation axis. Neither can do anything about the fact that a measured ray is *tilted*
with respect to the plane: at radius ``r`` it has already strayed
``r * (z0 - z_source) / source_to_isocentre`` out of it. That is a property of the ray, not of
which row is picked.

How much that matters was measured against the vendor's own reconstruction of the same
projections (a chest scan, 64 x 0.6 mm, pitch 0.9):

===============================  ==============  =========
configuration                    air [HU]        r vs vendor
===============================  ==============  =========
naive row, full rotation         -815 +- 666     0.855
cos(gamma) row, full rotation    refused: the plane is not covered at the fan edges
naive row, short scan            -757 +- 1041    0.736
cos(gamma) row, short scan       -758 +- 1039    0.735
vendor                          -1004 +-   22    1
===============================  ==============  =========

So: the row correction changes nothing measurable, the short scan is worse (fewer views, and
Parker weights pair conjugate rays that a helix has placed at different z), and all of them
streak badly. The one thing the correction did do was refuse a slice the naive choice had
silently accepted — asking for the right rows revealed that a full rotation does not in fact
cover that plane out at the fan edges.

The conclusion is not that the rebinning needs tuning. It is that no single-slice rebinning
reconstructs patient anatomy from this acquisition, and what is needed is a backprojection
along the rays as measured, in three dimensions (Stierstorfer et al.'s weighted FBP is the
published algorithm for this scanner family). On an object that does not vary along z —
a cylindrical phantom — this module remains exact, which is why the ACR phantom reconstructs
cleanly through it and a chest does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ldct_io.dicomctpd import ProjectionSeries


@dataclass(frozen=True, eq=False)
class SliceSinogram:
    """One slice's fan-beam sinogram, rebinned from a helical acquisition.

    Attributes
    ----------
    sinogram:
        ``(n_views, n_channels)`` of line integrals.
    angles:
        Source angle of each view [rad].
    source_radius:
        Source-to-isocentre distance of each view [mm], including any radial focal-spot
        deflection — pass it to the backprojector, do not assume the nominal value.
    z:
        The plane reconstructed [mm], in the same frame as the series' axial positions.
    views:
        Indices into the series.
    axial_drift_at:
        How far the sampled ray drifts in z, per mm of radius [mm/mm]. Multiply by the radius
        of interest to get the z blur there.
    meta:
        Provenance.

    """

    sinogram: np.ndarray
    angles: np.ndarray
    source_radius: np.ndarray
    z: float
    views: np.ndarray
    axial_drift_at: float
    meta: dict[str, Any] = field(default_factory=dict)

    def drift_at_radius(self, radius: float) -> float:
        """Peak axial drift of the sampled rays at ``radius`` mm from the isocentre [mm]."""
        return float(self.axial_drift_at * radius)


def single_slice_rebin(
    series: ProjectionSeries,
    z: float,
    *,
    focal_spot_class: int | None = 0,
    max_rotations: float = 1.0,
    cos_gamma_correction: bool = True,
    coverage: str = "full",
) -> SliceSinogram:
    """Rebin a helical series to the circular sinogram of the plane at ``z``.

    Parameters
    ----------
    series:
        An indexed series.
    z:
        Target plane, in the frame of ``series.views.axial_position``.
    focal_spot_class:
        Which focal-spot position to use, as indexed by
        :meth:`~ldct_io.geometry.ViewTable.focal_spot_classes`. Using one class keeps the
        source geometry exactly constant across the views used, at the cost of half the data
        on a two-position flying focal spot. ``None`` uses every view and returns the per-view
        source radius so the backprojector can account for the deflection itself.
    max_rotations:
        Rotations of data to use, before ``coverage`` clamps it.
    cos_gamma_correction:
        Choose each ray's detector row as a function of its fan angle, so that every ray meets
        the reconstruction plane where it comes closest to the rotation axis. On by default;
        turning it off reproduces the naive choice, which places the outer channels in the
        wrong plane.
    coverage:
        ``"full"`` uses a whole rotation. ``"short"`` uses π plus the fan angle, which halves
        how far the source travels in z and therefore halves the cone-angle error — at the
        price of needing Parker weighting in the reconstruction (pass ``short_scan=True`` to
        :func:`~ldct_io.recon.fan_beam_fbp`).

    Raises
    ------
    ValueError
        The plane is not covered by the detector over the angular range asked for — the series
        does not contain enough data to reconstruct it.

    """
    views_all = series.views
    geometry = series.geometry

    if coverage not in ("full", "short"):
        raise ValueError(f"coverage must be 'full' or 'short', got {coverage!r}")

    if focal_spot_class is None:
        candidates = np.arange(len(views_all))
    else:
        classes = views_all.focal_spot_classes()
        if not 0 <= focal_spot_class < len(classes):
            raise ValueError(
                f"focal_spot_class {focal_spot_class} is out of range; the series has "
                f"{len(classes)} focal-spot position(s)"
            )
        candidates = classes[focal_spot_class]

    z_source = views_all.axial_position[candidates] + views_all.ffs_dz[candidates]
    radius = geometry.source_to_isocentre + views_all.ffs_drho[candidates]

    step = float(np.abs(np.diff(np.unwrap(views_all.angle[candidates]))).mean())
    per_rotation = int(round(2.0 * np.pi / step))
    if coverage == "short":
        # A little over the minimum, so that discretising to whole views cannot land under it.
        max_rotations = min(max_rotations, geometry.short_scan_range / (2.0 * np.pi) + 0.005)
    n_take = int(round(per_rotation * max_rotations))
    if n_take < 8:
        raise ValueError(f"only {n_take} views per rotation in this class; that is not a scan")

    centre = int(np.argmin(np.abs(z_source - z)))
    start = centre - n_take // 2
    if start < 0 or start + n_take > candidates.size:
        raise ValueError(
            f"z = {z} mm is too close to the end of the series to take {n_take} views "
            f"({max_rotations:.2f} rotation(s)) around it"
        )
    take = slice(start, start + n_take)
    chosen = candidates[take]

    # The row each ray needs, as a function of the fan angle as well as the view.
    #
    # A ray (beta, gamma) approaches the rotation axis most closely at a path length
    # R cos(gamma) from the source, and it is *there* that it should sit in the plane being
    # reconstructed -- that point is what the ray contributes to the 2-D sinogram. Requiring
    # it gives (v - v0) = (z0 - z_s) * SDD / (R cos(gamma) * dv), which reduces to the naive
    # choice only on the central ray. Ignoring the cos(gamma) puts the outer channels in the
    # wrong plane by up to (1/cos(Gamma) - 1) of the row offset, ~10 % at a 25 deg half-fan.
    gamma = geometry.channel_angles() if cos_gamma_correction else np.zeros(geometry.n_channels)
    scale = geometry.source_to_detector / geometry.axial_spacing
    rows = geometry.central_row + (
        (z - z_source[take])[:, None] * scale / (radius[take][:, None] * np.cos(gamma)[None, :])
    )

    if rows.min() < -0.5 or rows.max() > geometry.n_rows - 0.5:
        raise ValueError(
            f"the plane z = {z} mm needs detector rows {rows.min():.1f}..{rows.max():.1f} of "
            f"0..{geometry.n_rows - 1} over this angular range: it is not covered by the beam. "
            f"Choose a z further inside the scanned range, or coverage='short'."
        )
    rows = np.clip(rows, 0.0, geometry.n_rows - 1)

    sinogram = np.empty((chosen.size, geometry.n_channels), dtype=np.float64)
    channel = np.arange(geometry.n_channels)
    for k, view in enumerate(chosen):
        frame = series.read_frame(int(view))
        lo = np.floor(rows[k]).astype(np.int32)
        lo = np.clip(lo, 0, geometry.n_rows - 2)
        w = rows[k] - lo
        sinogram[k] = frame[channel, lo] * (1.0 - w) + frame[channel, lo + 1] * w

    max_row_offset = float(np.abs(rows - geometry.central_row).max())
    drift_per_mm = max_row_offset * geometry.axial_spacing / geometry.source_to_detector

    return SliceSinogram(
        sinogram=sinogram,
        angles=views_all.angle[chosen],
        source_radius=radius[take],
        z=float(z),
        views=chosen,
        axial_drift_at=drift_per_mm,
        meta={
            "method": "single-slice rebinning"
            + (" with the cos(gamma) row correction" if cos_gamma_correction else ""),
            "coverage": coverage,
            "focal_spot_class": focal_spot_class,
            "n_views": int(chosen.size),
            "views_per_rotation": per_rotation,
            "angular_coverage_deg": float(
                np.degrees(
                    abs(
                        np.unwrap(views_all.angle[chosen])[-1]
                        - np.unwrap(views_all.angle[chosen])[0]
                    )
                    + step
                )
            ),
            "detector_rows_used": (float(rows.min()), float(rows.max())),
            "source_offset_mm": (
                float((z - z_source[take]).min()),
                float((z - z_source[take]).max()),
            ),
            "note": (
                "exact only where the object does not vary along z; "
                f"rays drift up to {drift_per_mm:.4f} mm in z per mm of radius"
            ),
        },
    )
