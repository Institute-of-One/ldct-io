"""Read a reconstructed CT image series, in Hounsfield units, sorted along z.

The trap here is not the pixels; it is the dose. In LDCT-and-Projection-data the low-dose
images of a case are *simulated* — noise was added in the projection domain and the result
reconstructed — and the acquisition tags were carried over unchanged from the full-dose scan.
So a low-dose series reports the same ``Exposure`` as the full-dose one it came from, and
sometimes a larger ``XRayTubeCurrent``. Anything that reads dose from the header of these
series is wrong with no indication that it is. This module therefore refuses to report dose
from tags at all, and asks for the simulated fraction to be stated instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

#: Nominal dose fraction of the simulated low-dose series, by body part, as documented for
#: LDCT-and-Projection-data. Chest was simulated at 10 % of the routine dose, abdomen/liver
#: at 25 %. These are the *stated* fractions; nothing in the files records them.
SIMULATED_DOSE_FRACTION = {"CHEST": 0.10, "ABDOMEN": 0.25, "LIVER": 0.25}


@dataclass(frozen=True, eq=False)
class ImageSeries:
    """A reconstructed series as a volume in Hounsfield units.

    Attributes
    ----------
    volume:
        ``(n_slices, ny, nx)`` in HU, ordered by increasing z.
    z:
        Slice positions [mm], increasing.
    spacing:
        In-plane pixel pitch [mm]; isotropic in plane.
    slice_thickness:
        Nominal reconstructed slice thickness [mm]. Not the same as the z increment.
    kernel:
        Reconstruction kernel as recorded, e.g. ``B30f``.
    meta:
        Series identity and acquisition description. Deliberately carries no dose figure.

    """

    volume: np.ndarray
    z: np.ndarray
    spacing: float
    slice_thickness: float
    kernel: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of slices."""
        return int(self.volume.shape[0])

    def index_of(self, z: float) -> int:
        """The slice nearest ``z``."""
        return int(np.argmin(np.abs(self.z - z)))

    def matches(self, other: ImageSeries) -> bool:
        """Whether two series lie on the same grid, so they can be differenced."""
        return bool(
            self.volume.shape == other.volume.shape
            and np.allclose(self.z, other.z)
            and np.isclose(self.spacing, other.spacing)
        )


def read_image_series(directory: str | Path, *, pattern: str = "*.dcm") -> ImageSeries:
    """Read a directory of reconstructed slices into an :class:`ImageSeries`."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise ValueError(f"no files matching {pattern!r} in {directory}")

    datasets = [pydicom.dcmread(str(f)) for f in files]
    datasets.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    first = datasets[0]

    spacings = {tuple(float(v) for v in d.PixelSpacing) for d in datasets}
    if len(spacings) != 1:
        raise ValueError(f"the series mixes pixel spacings: {sorted(spacings)}")
    spacing_pair = spacings.pop()
    if not np.isclose(spacing_pair[0], spacing_pair[1]):
        raise ValueError(f"anisotropic in-plane pixels {spacing_pair}; not supported")

    shapes = {(int(d.Rows), int(d.Columns)) for d in datasets}
    if len(shapes) != 1:
        raise ValueError(f"the series mixes matrix sizes: {sorted(shapes)}")

    volume = np.stack(
        [
            d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept)
            for d in datasets
        ]
    )
    z = np.array([float(d.ImagePositionPatient[2]) for d in datasets], dtype=np.float64)
    steps = np.diff(z)
    if steps.size and (steps <= 0).any():
        raise ValueError("slice positions are not strictly increasing after sorting")

    return ImageSeries(
        volume=volume,
        z=z,
        spacing=float(spacing_pair[0]),
        slice_thickness=float(first.SliceThickness),
        kernel=str(first.get("ConvolutionKernel", "")),
        meta={
            "series_instance_uid": str(first.get("SeriesInstanceUID", "")),
            "patient_id": str(first.get("PatientID", "")),
            "series_description": str(first.get("SeriesDescription", "")),
            "body_part": str(first.get("BodyPartExamined", "")),
            "manufacturer": str(first.get("Manufacturer", "")),
            "model": str(first.get("ManufacturerModelName", "")),
            "kvp": str(first.get("KVP", "")),
            "z_increment_mm": float(np.median(steps)) if steps.size else 0.0,
            "n_slices": len(datasets),
            "directory": str(directory),
            "dose_note": (
                "acquisition tags in this collection are inherited by the simulated low-dose "
                "series and do not describe its dose; use SIMULATED_DOSE_FRACTION"
            ),
        },
    )


def noise_only(
    full: ImageSeries, low: ImageSeries, *, dose_fraction: float
) -> tuple[np.ndarray, float]:
    r"""The noise the dose reduction added, and the factor relating it to the full-dose noise.

    The low-dose series of this collection is a reconstruction of the *same* projections with
    noise inserted, so — filtered backprojection being linear — the difference of the two
    reconstructions is exactly the reconstruction of that inserted noise. Anatomy cancels
    identically, which is what makes this a clean noise realisation where a uniform ROI in a
    patient never is.

    What it is *not* is the full-dose noise. If the inserted noise is independent of the
    noise already there, and the dose fraction is :math:`\alpha`, then

    .. math::

        \mathrm{NPS}_{\text{diff}} =
            \left(\tfrac{1}{\alpha} - 1\right)\,\mathrm{NPS}_{\text{full}}

    so the full-dose NPS is the difference NPS divided by :math:`1/\alpha - 1` — a factor of 3
    for the abdomen at 25 %, 9 for the chest at 10 %. The function returns both the difference
    image and that factor rather than applying it, because the assumption behind it (the
    stated dose fraction, and independence) belongs in the caller's write-up.

    Returns
    -------
    difference, factor
        ``low - full`` in HU, and :math:`1/\alpha - 1`.

    """
    if not 0.0 < dose_fraction < 1.0:
        raise ValueError(f"dose_fraction must be in (0, 1), got {dose_fraction}")
    if not full.matches(low):
        raise ValueError(
            "the two series are not on the same grid, so their difference is not a noise "
            "realisation: check that they are the full- and low-dose reconstructions of one "
            "acquisition"
        )
    return low.volume - full.volume, 1.0 / dose_fraction - 1.0
