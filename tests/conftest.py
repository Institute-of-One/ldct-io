"""Fixtures: the geometry of the scanner the package was written against, and fake series."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from ldct_io.geometry import ScanGeometry

# Values read from a real LDCT-and-Projection-data series (Siemens SOMATOM Definition Flash).
FLASH = dict(
    source_to_isocentre=595.0,
    source_to_detector=1085.5999755859375,
    transverse_spacing=1.285839319229126,
    axial_spacing=1.0947227478027344,
    central_channel=369.625,
    central_row=32.5,
    n_channels=736,
    n_rows=64,
    detector_shape="CYLINDRICAL",
    trajectory="HELICAL",
    beam="FANBEAM",
    focal_spot_mode="FFSZ",
    water_attenuation=0.0192,
)


@pytest.fixture(scope="session")
def flash() -> ScanGeometry:
    """The real scanner geometry."""
    return ScanGeometry(**FLASH)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def full_rotation() -> np.ndarray:
    """Source angles for one full rotation, enough views to be well sampled."""
    return np.linspace(0.0, 2.0 * np.pi, 576, endpoint=False)


def _f32(value: float) -> bytes:
    return struct.pack("<f", float(value))


def write_fake_series(
    directory: Path,
    n_views: int = 24,
    *,
    shuffle_filenames: bool = True,
    n_channels: int = 16,
    n_rows: int = 4,
    seed: int = 0,
) -> dict[int, str]:
    """Write a tiny DICOM-CT-PD series, optionally with file names in a scrambled order.

    Returns ``{InstanceNumber: filename}`` so a test can assert what the true order was.
    This mirrors how an archive hands the data over: sequential file names assigned in an
    order unrelated to the scan.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_views) if shuffle_filenames else np.arange(n_views)

    mapping: dict[int, str] = {}
    for view in range(n_views):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66"
        ds.file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.{view + 1}"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66"
        ds.SOPInstanceUID = f"1.2.3.4.{view + 1}"
        ds.SeriesInstanceUID = "1.2.3.4"
        ds.Modality = "CT"
        ds.Manufacturer = "TEST"
        ds.InstanceNumber = view + 1

        ds.Rows, ds.Columns = n_channels, n_rows
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = "0.001"
        ds.RescaleIntercept = "0.0"
        # A value that identifies the view, so a test can tell which frame it received.
        ds.PixelData = np.full((n_channels, n_rows), view + 1, dtype=np.uint16).tobytes()

        block_geom = {
            (0x7031, 0x1001): _f32(2 * np.pi * view / n_views),  # angle
            (0x7031, 0x1002): _f32(-100.0 - 0.5 * view),  # axial position
            (0x7031, 0x1003): _f32(595.0),
            (0x7031, 0x1031): _f32(1085.6),
            (0x7031, 0x1033): struct.pack("<ff", n_channels / 2 - 0.5, n_rows / 2 - 0.5),
            (0x7029, 0x1002): _f32(1.285839319229126),
            (0x7029, 0x1006): _f32(1.0947227478027344),
            (0x7029, 0x100B): b"CYLINDRICAL ",
            (0x7029, 0x1010): struct.pack("<H", n_rows),
            (0x7029, 0x1011): struct.pack("<H", n_channels),
            (0x7033, 0x100B): _f32(0.000336),
            (0x7033, 0x100C): _f32(0.0 if view % 2 == 0 else -0.66),
            (0x7033, 0x100D): _f32(0.0 if view % 2 == 0 else 5.45),
            (0x7033, 0x100E): b"FFSZ",
            (0x7037, 0x1009): b"HELICAL ",
            (0x7037, 0x100A): b"FANBEAM ",
            (0x7041, 0x1001): b"0.0192",
        }
        for tag, value in block_geom.items():
            ds.add_new(tag, "UN", value)

        name = f"{order[view] + 1:08d}.dcm"
        mapping[view + 1] = name
        pydicom.dcmwrite(str(directory / name), ds, enforce_file_format=True)
    return mapping
