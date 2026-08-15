"""Geometry validation, and the licence boundary the manifest enforces."""

from __future__ import annotations

import numpy as np
import pytest

from ldct_io import Manifest, ScanGeometry, SeriesRecord, ViewTable
from tests.conftest import FLASH


def _geometry(**overrides):
    return ScanGeometry(**{**FLASH, **overrides})


def test_real_geometry_derives_the_documented_quantities(flash):
    """The numbers the rest of the package reasons with, pinned to the real scanner.

    A drift in any of these silently changes what a reconstruction means, so they are
    asserted rather than recomputed in prose. ``axial_spacing_at_isocentre`` coming out at
    exactly 0.600 mm and the collimation at 38.4 mm are the recognisable signature of a
    64 x 0.6 mm acquisition, which is the check that the tag mapping is the right way round.
    """
    assert flash.magnification == pytest.approx(1.8245377740940125, rel=1e-12)
    assert flash.transverse_spacing_at_isocentre == pytest.approx(0.70474798, rel=1e-6)
    assert flash.axial_spacing_at_isocentre == pytest.approx(0.600, abs=1e-6)
    assert flash.detector_nyquist == pytest.approx(0.70947347, rel=1e-6)
    assert np.degrees(flash.fan_angle) == pytest.approx(49.948, abs=0.01)
    assert flash.collimation_at_isocentre == pytest.approx(38.400, abs=1e-4)
    assert flash.channel_angles()[0] < 0 < flash.channel_angles()[-1]


def test_an_unsupported_detector_shape_is_refused():
    with pytest.raises(ValueError, match="detector shape"):
        _geometry(detector_shape="FLAT")


def test_in_plane_flying_focal_spot_is_refused():
    """FFSXY moves the source within the scan plane, so a focal-spot class is not fixed."""
    with pytest.raises(ValueError, match="flying-focal-spot mode"):
        _geometry(focal_spot_mode="FFSXY")
    _geometry(focal_spot_mode="FFSZ")  # supported
    _geometry(focal_spot_mode="NONE")


def test_impossible_distances_are_refused():
    with pytest.raises(ValueError, match="must exceed"):
        _geometry(source_to_detector=500.0)
    with pytest.raises(ValueError, match="finite and positive"):
        _geometry(source_to_isocentre=0.0)


def test_view_table_recovers_the_scan_description():
    n = 2304
    views = ViewTable(
        angle=np.linspace(0.0, 2.0 * np.pi, n, endpoint=False),
        axial_position=np.linspace(0.0, -30.662, n),
        ffs_dphi=np.full(n, 0.000336),
        ffs_dz=np.where(np.arange(n) % 2 == 0, 0.0, -0.66),
        ffs_drho=np.where(np.arange(n) % 2 == 0, 0.0, 5.45),
    )
    assert views.rotations == pytest.approx(1.0, rel=1e-3)
    assert views.views_per_rotation == pytest.approx(n, rel=1e-3)
    assert views.pitch(ScanGeometry(**FLASH)) == pytest.approx(0.7985, rel=0.02)

    classes = views.focal_spot_classes()
    assert len(classes) == 2
    assert {c.size for c in classes} == {n // 2}


def test_source_positions_include_the_focal_spot_deflection(flash):
    n = 8
    views = ViewTable(
        angle=np.linspace(0.0, 2.0 * np.pi, n, endpoint=False),
        axial_position=np.zeros(n),
        ffs_dphi=np.zeros(n),
        ffs_dz=np.where(np.arange(n) % 2 == 0, 0.0, -0.66),
        ffs_drho=np.where(np.arange(n) % 2 == 0, 0.0, 5.45),
    )
    pos = views.source_positions(flash)
    radius = np.hypot(pos[:, 0], pos[:, 1])
    assert radius[0] == pytest.approx(595.0)
    assert radius[1] == pytest.approx(600.45)
    assert pos[1, 2] == pytest.approx(-0.66)


def test_mismatched_view_arrays_are_refused():
    with pytest.raises(ValueError, match="entries"):
        ViewTable(
            angle=np.zeros(10),
            axial_position=np.zeros(9),
            ffs_dphi=np.zeros(10),
            ffs_dz=np.zeros(10),
            ffs_drho=np.zeros(10),
        )


def test_manifest_refuses_the_controlled_access_body_part():
    """Head cases stay under NIH Controlled Data Access; this project must not read them."""
    with pytest.raises(ValueError, match="controlled access"):
        SeriesRecord(
            series_instance_uid="1.2.3",
            patient_id="N001",
            body_part="HEAD",
            description="Full dose projections",
            manufacturer="SIEMENS",
        )


def test_manifest_round_trips(tmp_path):
    m = Manifest(notes="ACR phantom, full-dose projections")
    m.add(
        SeriesRecord(
            series_instance_uid="1.3.6.1.4.1.9590.100.1.2.79685099111720316617921522283988355980",
            patient_id="PatientID",
            body_part="ABDOMEN",
            description="ACR_Phantom projections",
            manufacturer="SIEMENS",
            n_files=18032,
            total_bytes=1798201356,
            role="MTF and NPS of the real system",
        )
    )
    path = m.write(tmp_path / "manifest.json")
    back = Manifest.read(path)
    assert len(back.records) == 1
    assert back.records[0].n_files == 18032
    assert back.doi.endswith("9npb-2637")


def test_a_duplicate_series_is_refused():
    m = Manifest()
    record = SeriesRecord(
        series_instance_uid="1.2.3",
        patient_id="L019",
        body_part="ABDOMEN",
        description="Full dose projections",
        manufacturer="SIEMENS",
    )
    m.add(record)
    with pytest.raises(ValueError, match="already in the manifest"):
        m.add(record)
