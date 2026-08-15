"""Turn a helical acquisition into the circular fan-beam sinogram of one slice.

Single-slice rebinning, which is what this module implements, picks for every view the
detector row whose ray crosses the target plane *at the isocentre*. Away from the isocentre
that ray drifts in z, by up to ``r * (row offset) / source_to_detector``, so the approximation
is exact only where the object does not vary along z — a cylindrical phantom wall, a uniform
module — and degrades where it does. The degradation is visible, not subtle: slices near a
joint between dissimilar objects streak.

That is why this module reports the drift it will incur rather than only the sinogram: the
number tells you whether the slice you asked for is one this method can give you. Rebinning
that is correct for a varying object (Noo et al.) belongs here too and is not yet written.
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
        Rotations of data to use. One full rotation is the minimum this reconstruction
        supports.

    Raises
    ------
    ValueError
        The plane is not covered by the detector over a full rotation — the series does not
        contain enough data to reconstruct it.

    """
    views_all = series.views
    geometry = series.geometry

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
    row_step = geometry.axial_spacing_at_isocentre
    row_needed = geometry.central_row + (z - z_source) / row_step

    step = float(np.abs(np.diff(np.unwrap(views_all.angle[candidates]))).mean())
    per_rotation = int(round(2.0 * np.pi / step))
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
    rows = row_needed[take]
    if rows.min() < 0.0 or rows.max() > geometry.n_rows - 1:
        raise ValueError(
            f"the plane z = {z} mm needs detector rows {rows.min():.1f}..{rows.max():.1f} of "
            f"0..{geometry.n_rows - 1} over a full rotation: it is not covered by the beam. "
            f"Choose a z further inside the scanned range."
        )

    chosen = candidates[take]
    sinogram = np.stack(
        [series.read_row(int(v), float(row)) for v, row in zip(chosen, rows, strict=True)]
    )

    max_row_offset = float(np.abs(rows - geometry.central_row).max())
    drift_per_mm = max_row_offset * geometry.axial_spacing / geometry.source_to_detector

    return SliceSinogram(
        sinogram=sinogram,
        angles=views_all.angle[chosen],
        source_radius=geometry.source_to_isocentre + views_all.ffs_drho[chosen],
        z=float(z),
        views=chosen,
        axial_drift_at=drift_per_mm,
        meta={
            "method": "single-slice rebinning",
            "focal_spot_class": focal_spot_class,
            "n_views": int(chosen.size),
            "views_per_rotation": per_rotation,
            "detector_rows_used": (float(rows.min()), float(rows.max())),
            "note": (
                "exact only where the object does not vary along z; "
                f"rays drift {drift_per_mm:.4f} mm in z per mm of radius"
            ),
        },
    )
