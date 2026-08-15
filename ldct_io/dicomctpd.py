"""Read a DICOM-CT-PD projection series from disk.

One acquisition is thousands to tens of thousands of single-view files. Two things about
that are easy to get wrong and neither announces itself:

**Filename order is not acquisition order.** A series downloaded from TCIA arrives with
sequential filenames (``00000001.dcm`` …) assigned in an order unrelated to the scan. Reading
in filename order interleaves unrelated gantry angles; the per-view geometry tags, read the
same way and then sorted, still look perfect, so the error surfaces only as a sinogram that
will not reconstruct. Acquisition order is ``InstanceNumber``, and this module will not hand
out frames in any other order.

**The pixel values are not attenuation until rescaled.** ``RescaleSlope``/``RescaleIntercept``
map the stored integers to line integrals; the raw array is meaningless on its own.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from ldct_io.geometry import (
    TAG_ANGLE,
    TAG_AXIAL_POSITION,
    TAG_AXIAL_SPACING,
    TAG_BEAM,
    TAG_CENTRAL_ELEMENT,
    TAG_DETECTOR_SHAPE,
    TAG_FFS_DPHI,
    TAG_FFS_DRHO,
    TAG_FFS_DZ,
    TAG_FFS_MODE,
    TAG_N_CHANNELS,
    TAG_N_ROWS,
    TAG_SOURCE_TO_DETECTOR,
    TAG_SOURCE_TO_ISOCENTRE,
    TAG_TRAJECTORY,
    TAG_TRANSVERSE_SPACING,
    TAG_WATER_ATTENUATION,
    ScanGeometry,
    ViewTable,
    decode_float,
    decode_float_pair,
    decode_text,
    decode_uint16,
)

#: Version of the on-disk index. Bump when the layout changes so stale caches are rebuilt.
INDEX_VERSION = 1
#: Default name of the cached index inside the series directory.
INDEX_NAME = "ldct_io_index.json"


@dataclass(frozen=True, eq=False)
class ProjectionSeries:
    """An indexed DICOM-CT-PD series: geometry, per-view table, and ordered file names.

    Attributes
    ----------
    directory:
        Where the ``.dcm`` files live.
    filenames:
        File names **in acquisition order** — element ``i`` is view ``i``.
    geometry:
        The view-independent geometry.
    views:
        The per-view table, in the same order as :attr:`filenames`.
    meta:
        Provenance, including the index version and how the order was established.

    """

    directory: Path
    filenames: tuple[str, ...]
    geometry: ScanGeometry
    views: ViewTable
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of views."""
        return len(self.filenames)

    def path(self, view: int) -> Path:
        """Path of one view's file."""
        return self.directory / self.filenames[view]

    def read_frame(self, view: int) -> np.ndarray:
        """Line integrals of one view, shape ``(n_channels, n_rows)``.

        The stored integers are mapped through ``RescaleSlope``/``RescaleIntercept``; the
        result is attenuation path length in the units the scanner recorded (1/mm times mm).
        """
        ds = pydicom.dcmread(str(self.path(view)), force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.shape != (self.geometry.n_channels, self.geometry.n_rows):
            raise ValueError(
                f"view {view} has frame shape {arr.shape}, but the geometry says "
                f"{(self.geometry.n_channels, self.geometry.n_rows)}"
            )
        return arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    def read_row(self, view: int, row: float) -> np.ndarray:
        """One detector row of one view, ``(n_channels,)``, linearly interpolated in row.

        A fractional ``row`` is the normal case: a helical scan places the slice of interest
        between rows, and rounding to the nearest row throws away up to half a row of z.
        """
        n_rows = self.geometry.n_rows
        if not 0.0 <= row <= n_rows - 1:
            raise ValueError(f"row {row} is outside the detector (0..{n_rows - 1})")
        frame = self.read_frame(view)
        lo = int(np.floor(row))
        if lo == n_rows - 1:
            return frame[:, lo]
        w = row - lo
        return (1.0 - w) * frame[:, lo] + w * frame[:, lo + 1]

    def frames(self, views: np.ndarray | range) -> Iterator[np.ndarray]:
        """Iterate frames for the given view indices, in the order given."""
        for v in views:
            yield self.read_frame(int(v))


def _read_headers(path: Path) -> dict[str, Any]:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    if "InstanceNumber" not in ds:
        raise ValueError(f"{path.name} has no InstanceNumber, so its place in the scan is unknown")
    return {
        "instance": int(ds.InstanceNumber),
        "name": path.name,
        "angle": decode_float(ds[TAG_ANGLE].value),
        "axial": decode_float(ds[TAG_AXIAL_POSITION].value),
        "dphi": decode_float(ds[TAG_FFS_DPHI].value),
        "dz": decode_float(ds[TAG_FFS_DZ].value),
        "drho": decode_float(ds[TAG_FFS_DRHO].value),
    }


def _read_geometry(path: Path) -> ScanGeometry:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    central_channel, central_row = decode_float_pair(ds[TAG_CENTRAL_ELEMENT].value)
    ffs_mode = decode_text(ds[TAG_FFS_MODE].value) if TAG_FFS_MODE in ds else "NONE"
    return ScanGeometry(
        source_to_isocentre=decode_float(ds[TAG_SOURCE_TO_ISOCENTRE].value),
        source_to_detector=decode_float(ds[TAG_SOURCE_TO_DETECTOR].value),
        transverse_spacing=decode_float(ds[TAG_TRANSVERSE_SPACING].value),
        axial_spacing=decode_float(ds[TAG_AXIAL_SPACING].value),
        central_channel=central_channel,
        central_row=central_row,
        n_channels=decode_uint16(ds[TAG_N_CHANNELS].value),
        n_rows=decode_uint16(ds[TAG_N_ROWS].value),
        detector_shape=decode_text(ds[TAG_DETECTOR_SHAPE].value),
        trajectory=decode_text(ds[TAG_TRAJECTORY].value),
        beam=decode_text(ds[TAG_BEAM].value),
        focal_spot_mode=ffs_mode,
        water_attenuation=decode_float(ds[TAG_WATER_ATTENUATION].value),
        meta={
            "series_instance_uid": str(ds.get("SeriesInstanceUID", "")),
            "manufacturer": str(ds.get("Manufacturer", "")),
            "protocol_name": str(ds.get("ProtocolName", "")),
            "kvp": str(ds.get("KVP", "")),
            "tube_current": str(ds.get("XRayTubeCurrent", "")),
            "exposure_time": str(ds.get("ExposureTime", "")),
            "source_file": path.name,
        },
    )


def index_series(
    directory: str | Path,
    *,
    pattern: str = "*.dcm",
    cache: bool | str | Path = True,
    require_contiguous: bool = True,
) -> ProjectionSeries:
    """Index a DICOM-CT-PD series and return it in acquisition order.

    Every file's header is read once to recover its ``InstanceNumber`` and per-view geometry;
    the result is cached beside the data so the pass is not repeated.

    Parameters
    ----------
    directory:
        Directory of single-view ``.dcm`` files.
    pattern:
        Glob for the projection files.
    cache:
        ``True`` (default) reads and writes :data:`INDEX_NAME` in ``directory``; a path uses
        that file instead; ``False`` disables caching.
    require_contiguous:
        Require ``InstanceNumber`` to run 1..N with no gaps. A gap means views are missing,
        which a reconstruction cannot detect for itself — it just produces streaks. Set
        ``False`` only if you intend to handle an incomplete series deliberately.

    Raises
    ------
    ValueError
        No files matched; a file lacks ``InstanceNumber``; instance numbers are duplicated;
        or (with ``require_contiguous``) views are missing.

    """
    directory = Path(directory)
    if cache is True:
        cache_path: Path | None = directory / INDEX_NAME
    elif cache is False:
        cache_path = None
    else:
        cache_path = Path(cache)

    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        if payload.get("index_version") == INDEX_VERSION:
            return _from_payload(directory, payload)

    files = sorted(directory.glob(pattern))
    if not files:
        raise ValueError(f"no files matching {pattern!r} in {directory}")

    records = [_read_headers(p) for p in files]
    records.sort(key=lambda r: r["instance"])

    instances = [r["instance"] for r in records]
    if len(set(instances)) != len(instances):
        raise ValueError("duplicate InstanceNumber values: this is not one coherent series")
    if require_contiguous:
        expected = list(range(instances[0], instances[0] + len(instances)))
        if instances != expected:
            missing = sorted(set(expected) - set(instances))
            raise ValueError(
                f"InstanceNumber is not contiguous: {len(missing)} views missing, first "
                f"{missing[:5]}. A reconstruction cannot detect missing views; it only "
                f"streaks. Pass require_contiguous=False to proceed anyway."
            )

    geometry = _read_geometry(directory / records[0]["name"])
    views = ViewTable(
        angle=np.array([r["angle"] for r in records], dtype=np.float64),
        axial_position=np.array([r["axial"] for r in records], dtype=np.float64),
        ffs_dphi=np.array([r["dphi"] for r in records], dtype=np.float64),
        ffs_dz=np.array([r["dz"] for r in records], dtype=np.float64),
        ffs_drho=np.array([r["drho"] for r in records], dtype=np.float64),
        meta={"ordered_by": "InstanceNumber", "n_views": len(records)},
    )
    series = ProjectionSeries(
        directory=directory,
        filenames=tuple(r["name"] for r in records),
        geometry=geometry,
        views=views,
        meta={
            "index_version": INDEX_VERSION,
            "ordered_by": "InstanceNumber",
            "instance_range": (instances[0], instances[-1]),
            "n_files": len(records),
            "pattern": pattern,
        },
    )
    if cache_path is not None:
        cache_path.write_text(json.dumps(_to_payload(series)))
    return series


def _to_payload(series: ProjectionSeries) -> dict[str, Any]:
    g = series.geometry
    return {
        "index_version": INDEX_VERSION,
        "filenames": list(series.filenames),
        "geometry": {
            "source_to_isocentre": g.source_to_isocentre,
            "source_to_detector": g.source_to_detector,
            "transverse_spacing": g.transverse_spacing,
            "axial_spacing": g.axial_spacing,
            "central_channel": g.central_channel,
            "central_row": g.central_row,
            "n_channels": g.n_channels,
            "n_rows": g.n_rows,
            "detector_shape": g.detector_shape,
            "trajectory": g.trajectory,
            "beam": g.beam,
            "focal_spot_mode": g.focal_spot_mode,
            "water_attenuation": g.water_attenuation,
            "meta": g.meta,
        },
        "views": {
            "angle": series.views.angle.tolist(),
            "axial_position": series.views.axial_position.tolist(),
            "ffs_dphi": series.views.ffs_dphi.tolist(),
            "ffs_dz": series.views.ffs_dz.tolist(),
            "ffs_drho": series.views.ffs_drho.tolist(),
            "meta": series.views.meta,
        },
        "meta": series.meta,
    }


def _from_payload(directory: Path, payload: dict[str, Any]) -> ProjectionSeries:
    g = dict(payload["geometry"])
    v = payload["views"]
    return ProjectionSeries(
        directory=directory,
        filenames=tuple(payload["filenames"]),
        geometry=ScanGeometry(**g),
        views=ViewTable(
            angle=np.asarray(v["angle"], dtype=np.float64),
            axial_position=np.asarray(v["axial_position"], dtype=np.float64),
            ffs_dphi=np.asarray(v["ffs_dphi"], dtype=np.float64),
            ffs_dz=np.asarray(v["ffs_dz"], dtype=np.float64),
            ffs_drho=np.asarray(v["ffs_drho"], dtype=np.float64),
            meta=v.get("meta", {}),
        ),
        meta=payload.get("meta", {}),
    )


def directory_digest(directory: str | Path, *, pattern: str = "*.dcm") -> dict[str, Any]:
    """A content digest of a series directory, for provenance.

    Hashing tens of gigabytes on every run is not useful, so this records the file count, the
    total size, and the sha256 of the *sorted list of (name, size)* — enough to detect a
    partial download, a re-download in a different order, or an edited file count.
    """
    directory = Path(directory)
    entries = sorted((p.name, p.stat().st_size) for p in directory.glob(pattern))
    if not entries:
        raise ValueError(f"no files matching {pattern!r} in {directory}")
    h = hashlib.sha256()
    for name, size in entries:
        h.update(f"{name}:{size}\n".encode())
    return {
        "n_files": len(entries),
        "total_bytes": sum(size for _, size in entries),
        "listing_sha256": h.hexdigest(),
    }
