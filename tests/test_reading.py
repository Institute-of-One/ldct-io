"""Reading a series: acquisition order, and the traps that do not announce themselves."""

from __future__ import annotations

import numpy as np
import pytest

from ldct_io import directory_digest, index_series
from tests.conftest import write_fake_series


def test_views_come_back_in_acquisition_order_not_file_name_order(tmp_path):
    """The bug this package exists to prevent.

    An archive assigns sequential file names in an order unrelated to the scan. Reading in
    file-name order interleaves unrelated gantry angles, and because the geometry tags are
    read the same way and then sorted, the geometry still looks perfect — the error surfaces
    only as a sinogram that will not reconstruct.
    """
    mapping = write_fake_series(tmp_path, n_views=24, shuffle_filenames=True)
    assert [mapping[i] for i in sorted(mapping)] != sorted(mapping.values()), (
        "this test is pointless unless the file names really are scrambled"
    )

    series = index_series(tmp_path, cache=False)
    assert len(series) == 24
    assert list(series.filenames) == [mapping[i] for i in sorted(mapping)]

    # Each fake frame is filled with its own view number, so the frames prove the order.
    for view in range(len(series)):
        frame = series.read_frame(view)
        assert frame.mean() == pytest.approx((view + 1) * 0.001, rel=1e-6)


def test_angles_are_monotone_once_ordered(tmp_path):
    write_fake_series(tmp_path, n_views=32, shuffle_filenames=True)
    series = index_series(tmp_path, cache=False)
    steps = np.diff(np.unwrap(series.views.angle))
    assert np.all(steps > 0), "the gantry does not reverse; ordering must make the angle monotone"
    assert steps.std() < 1e-6 * abs(steps.mean()) + 1e-9


def test_missing_views_are_refused(tmp_path):
    """A gap is invisible to a reconstruction: it only streaks."""
    mapping = write_fake_series(tmp_path, n_views=16, shuffle_filenames=False)
    (tmp_path / mapping[8]).unlink()
    with pytest.raises(ValueError, match="not contiguous"):
        index_series(tmp_path, cache=False)
    series = index_series(tmp_path, cache=False, require_contiguous=False)
    assert len(series) == 15


def test_index_is_cached_and_reused(tmp_path):
    write_fake_series(tmp_path, n_views=12)
    first = index_series(tmp_path, cache=True)
    cache_file = tmp_path / "ldct_io_index.json"
    assert cache_file.exists()
    second = index_series(tmp_path, cache=True)
    assert second.filenames == first.filenames
    assert second.geometry.n_channels == first.geometry.n_channels
    assert np.allclose(second.views.angle, first.views.angle)


def test_flying_focal_spot_splits_into_fixed_geometry_classes(tmp_path):
    write_fake_series(tmp_path, n_views=24)
    series = index_series(tmp_path, cache=False)
    classes = series.views.focal_spot_classes()
    assert len(classes) == 2
    assert sum(c.size for c in classes) == len(series)
    for indices in classes:
        assert np.ptp(series.views.ffs_dz[indices]) == 0.0
        assert np.ptp(series.views.ffs_drho[indices]) == 0.0


def test_geometry_is_decoded_from_the_private_tags(tmp_path):
    write_fake_series(tmp_path, n_views=8, n_channels=16, n_rows=4)
    g = index_series(tmp_path, cache=False).geometry
    assert g.source_to_isocentre == pytest.approx(595.0, rel=1e-6)
    assert g.source_to_detector == pytest.approx(1085.6, rel=1e-6)
    assert g.n_channels == 16
    assert g.n_rows == 4
    assert g.detector_shape == "CYLINDRICAL"
    assert g.focal_spot_mode == "FFSZ"
    assert g.water_attenuation == pytest.approx(0.0192)
    assert g.magnification == pytest.approx(1085.6 / 595.0)


def test_empty_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no files matching"):
        index_series(tmp_path, cache=False)


def test_directory_digest_notices_a_changed_file_set(tmp_path):
    mapping = write_fake_series(tmp_path, n_views=10, shuffle_filenames=False)
    before = directory_digest(tmp_path)
    (tmp_path / mapping[3]).unlink()
    after = directory_digest(tmp_path)
    assert before["listing_sha256"] != after["listing_sha256"]
    assert after["n_files"] == before["n_files"] - 1
