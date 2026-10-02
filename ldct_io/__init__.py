"""ldct_io — read, reconstruct and measure real CT projection data.

The bridge between a public projection dataset and the task-based image-quality framework in
``taskiq-core`` / ``denoiq-core``. Those two packages are deliberately free of DICOM and of
patient data; this one is where the archive, its private tags and its traps live, so that
neither of them has to change to meet real data.

What it does
------------
``geometry``    the DICOM-CT-PD acquisition geometry, validated rather than assumed
``dicomctpd``   read a projection series in *acquisition* order, not file-name order
``rebin``       helical acquisition -> the circular sinogram of one plane
``recon``       equiangular fan-beam FBP, on the CPU, checked against a closed form
``helical``     weighted 3-D backprojection, for objects that vary along z
``phantoms``    analytic sinograms — the closed-form answers the pipeline is held to
``edge``        MTF from a circular edge, with the biases of a binned ESF corrected
``series``      reconstructed image series in HU — and why their dose tags lie
``lesion``      hybrid lesions in real anatomy, so a task on patient data has ground truth
``manifest``    which series were used, under what licence

The discipline is the one the companion packages use: every estimator is checked against a
closed-form answer rather than against its own past output, and anything degenerate raises
instead of returning a plausible number.
"""

from __future__ import annotations

from ldct_io.dicomctpd import (
    INDEX_NAME,
    ProjectionSeries,
    directory_digest,
    index_series,
)
from ldct_io.edge import CircleFit, RadialMTF, fit_edge_circle, radial_mtf
from ldct_io.geometry import ScanGeometry, ViewTable
from ldct_io.helical import (
    Illumination,
    illuminating_views,
    row_window,
    wfbp_from_series,
    wfbp_slice,
)
from ldct_io.dose import (
    calibrate_incident_counts,
    insert_quantum_noise,
    noise_scale_factor,
)
from ldct_io.lesion import (
    LesionTrials,
    disk_lesion,
    homogeneous_sites,
    make_paired_trials,
    make_trials,
    paired_d_prime,
)
from ldct_io.manifest import Manifest, SeriesRecord
from ldct_io.phantoms import (
    disk_projection,
    disk_sinogram,
    ray_directions,
    sampling_chain_mtf,
    sphere_projection,
    sphere_slice_truth,
)
from ldct_io.rebin import SliceSinogram, single_slice_rebin
from ldct_io.recon import (
    ReconResult,
    backproject,
    fan_beam_fbp,
    filter_projections,
    parker_weights,
    ramp_kernel,
    to_hu,
)
from ldct_io.series import (
    SIMULATED_DOSE_FRACTION,
    ImageSeries,
    noise_only,
    read_image_series,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # geometry
    "ScanGeometry",
    "ViewTable",
    # reading
    "ProjectionSeries",
    "index_series",
    "directory_digest",
    "INDEX_NAME",
    # rebinning
    "SliceSinogram",
    "single_slice_rebin",
    # reconstruction
    "ReconResult",
    "fan_beam_fbp",
    "filter_projections",
    "backproject",
    "parker_weights",
    "ramp_kernel",
    "to_hu",
    # helical, three-dimensional
    "Illumination",
    "illuminating_views",
    "row_window",
    "wfbp_slice",
    "wfbp_from_series",
    # closed-form references
    "disk_projection",
    "disk_sinogram",
    "ray_directions",
    "sphere_projection",
    "sphere_slice_truth",
    "sampling_chain_mtf",
    # measurement
    "CircleFit",
    "RadialMTF",
    "fit_edge_circle",
    "radial_mtf",
    # reconstructed image series
    "ImageSeries",
    "read_image_series",
    "noise_only",
    "SIMULATED_DOSE_FRACTION",
    "insert_quantum_noise",
    "noise_scale_factor",
    "calibrate_incident_counts",
    # hybrid lesions
    "LesionTrials",
    "disk_lesion",
    "homogeneous_sites",
    "make_trials",
    "make_paired_trials",
    "paired_d_prime",
    # provenance
    "Manifest",
    "SeriesRecord",
]
