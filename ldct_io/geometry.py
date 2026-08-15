"""Acquisition geometry of a DICOM-CT-PD projection series.

The manufacturer-neutral DICOM-CT-PD format carries the scanner geometry in private tags.
This module names those tags, decodes them, and refuses anything it has not been validated
against rather than guessing — an unknown detector shape or focal-spot mode changes what a
reconstruction means, and a silently wrong geometry produces a plausible image.

Tag numbers follow the DICOM-CT-PD dictionary distributed with the dataset; the same mapping
is used by ``helix2fan`` (Apache-2.0). Every value below was checked against the bytes of a
real series before being trusted.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --- private tags -------------------------------------------------------------------------
#: Per-view gantry rotation angle [rad].
TAG_ANGLE = (0x7031, 0x1001)
#: Per-view axial position of the focal centre [mm] — the table feed, seen from the gantry.
TAG_AXIAL_POSITION = (0x7031, 0x1002)
#: Source-to-isocentre distance [mm].
TAG_SOURCE_TO_ISOCENTRE = (0x7031, 0x1003)
#: Source-to-detector distance [mm].
TAG_SOURCE_TO_DETECTOR = (0x7031, 0x1031)
#: Central detector element, ``(channel, row)`` in float indices.
TAG_CENTRAL_ELEMENT = (0x7031, 0x1033)
#: Detector element spacing, transverse [mm].
TAG_TRANSVERSE_SPACING = (0x7029, 0x1002)
#: Detector element spacing, axial [mm].
TAG_AXIAL_SPACING = (0x7029, 0x1006)
#: Detector shape, e.g. ``CYLINDRICAL``.
TAG_DETECTOR_SHAPE = (0x7029, 0x100B)
#: Number of detector rows.
TAG_N_ROWS = (0x7029, 0x1010)
#: Number of detector channels.
TAG_N_CHANNELS = (0x7029, 0x1011)
#: Per-view flying-focal-spot deflection: angular [rad], axial [mm], radial [mm].
TAG_FFS_DPHI = (0x7033, 0x100B)
TAG_FFS_DZ = (0x7033, 0x100C)
TAG_FFS_DRHO = (0x7033, 0x100D)
#: Flying-focal-spot mode, e.g. ``FFSZ``.
TAG_FFS_MODE = (0x7033, 0x100E)
#: Scan trajectory, e.g. ``HELICAL``.
TAG_TRAJECTORY = (0x7037, 0x1009)
#: Beam geometry, e.g. ``FANBEAM``.
TAG_BEAM = (0x7037, 0x100A)
#: Attenuation coefficient of water at the effective beam energy [1/mm], for the HU scale.
TAG_WATER_ATTENUATION = (0x7041, 0x1001)

#: Detector shapes this package's reconstruction is written for.
SUPPORTED_DETECTOR_SHAPES = frozenset({"CYLINDRICAL"})
#: Beam geometries this package's reconstruction is written for.
SUPPORTED_BEAM = frozenset({"FANBEAM"})
#: Flying-focal-spot modes whose per-view deflections this package handles exactly.
#:
#: ``NONE`` needs no handling. ``FFSZ`` deflects the spot in z (and, on a tilted anode,
#: radially with it) but leaves the in-plane *angular* deflection constant across views, so
#: the views split into focal-spot classes each of which has a fixed geometry — see
#: :meth:`ViewTable.focal_spot_classes`. In-plane deflection (``FFSXY``, ``FFSXYZ``) makes the
#: geometry vary within a class and is not handled.
SUPPORTED_FFS_MODES = frozenset({"NONE", "", "FFSZ"})


def decode_float(raw: Any) -> float:
    """One number from a private tag whose VR was lost to ``UN``.

    De-identification strips the private-tag VRs, so the same file carries some values as raw
    ``float32`` bytes and others as ASCII decimal strings — the water attenuation coefficient
    is written ``b"0.0192"``, not four bytes. Guessing by length is the only thing left, so it
    is done explicitly here rather than in five call sites.
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, bytes):
        if len(raw) == 4:
            return float(struct.unpack("<f", raw)[0])
        try:
            return float(raw.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"cannot read a number from {len(raw)} bytes {raw!r}: not a float32 and not "
                f"an ASCII decimal"
            ) from exc
    return float(str(raw).strip())


def decode_float_pair(raw: Any) -> tuple[float, float]:
    """Two little-endian ``float32`` values from one private tag."""
    if isinstance(raw, bytes):
        if len(raw) != 8:
            raise ValueError(f"expected 8 bytes for two float32, got {len(raw)}")
        a, b = struct.unpack("<ff", raw)
        return float(a), float(b)
    values = tuple(float(v) for v in raw)
    if len(values) != 2:
        raise ValueError(f"expected two values, got {values!r}")
    return values[0], values[1]


def decode_uint16(raw: Any) -> int:
    """One little-endian ``uint16`` from a private tag."""
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, bytes):
        if len(raw) != 2:
            raise ValueError(f"expected 2 bytes for a uint16, got {len(raw)}")
        return int(struct.unpack("<H", raw)[0])
    return int(str(raw).strip())


def decode_text(raw: Any) -> str:
    """A trimmed ASCII string from a private tag."""
    if isinstance(raw, bytes):
        return raw.decode("ascii", errors="replace").strip()
    return str(raw).strip()


@dataclass(frozen=True, eq=False)
class ScanGeometry:
    """The parts of the acquisition geometry that do not change between views.

    Attributes
    ----------
    source_to_isocentre, source_to_detector:
        Distances in mm. Their ratio is the magnification.
    transverse_spacing, axial_spacing:
        Detector element pitch in mm, **on the detector**, not at the isocentre.
    central_channel, central_row:
        Position of the central ray on the detector, in float element indices. A fractional
        channel is the quarter-detector offset and is not a rounding error.
    n_channels, n_rows:
        Detector size.
    detector_shape, trajectory, beam, focal_spot_mode:
        Strings as recorded. Validated on construction.
    water_attenuation:
        Attenuation coefficient of water [1/mm] at the effective beam energy, as recorded by
        the scanner. This sets the HU scale and is *not* a universal constant: it is the value
        the vendor's own calibration used, and a reconstruction that reproduces the vendor's
        HU must use it.
    meta:
        Provenance of the values.

    """

    source_to_isocentre: float
    source_to_detector: float
    transverse_spacing: float
    axial_spacing: float
    central_channel: float
    central_row: float
    n_channels: int
    n_rows: int
    detector_shape: str
    trajectory: str
    beam: str
    focal_spot_mode: str
    water_attenuation: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a geometry this package has not been validated against."""
        for name in (
            "source_to_isocentre",
            "source_to_detector",
            "transverse_spacing",
            "axial_spacing",
            "water_attenuation",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if self.source_to_detector <= self.source_to_isocentre:
            raise ValueError(
                f"source_to_detector ({self.source_to_detector}) must exceed "
                f"source_to_isocentre ({self.source_to_isocentre})"
            )
        if self.n_channels < 2 or self.n_rows < 1:
            raise ValueError(f"implausible detector size {self.n_channels}x{self.n_rows}")
        if self.detector_shape.upper() not in SUPPORTED_DETECTOR_SHAPES:
            raise ValueError(
                f"detector shape {self.detector_shape!r} is not supported; this package's "
                f"reconstruction assumes an equiangular (cylindrical) detector. "
                f"Supported: {sorted(SUPPORTED_DETECTOR_SHAPES)}"
            )
        if self.beam.upper() not in SUPPORTED_BEAM:
            raise ValueError(
                f"beam geometry {self.beam!r} is not supported. Supported: {sorted(SUPPORTED_BEAM)}"
            )
        if self.focal_spot_mode.upper() not in SUPPORTED_FFS_MODES:
            raise ValueError(
                f"flying-focal-spot mode {self.focal_spot_mode!r} deflects the spot in the "
                f"scan plane, so the geometry varies from view to view within a focal-spot "
                f"class and the exact treatment in this package does not apply. "
                f"Supported: {sorted(SUPPORTED_FFS_MODES)}"
            )

    @property
    def magnification(self) -> float:
        """``source_to_detector / source_to_isocentre``."""
        return self.source_to_detector / self.source_to_isocentre

    @property
    def channel_angle_spacing(self) -> float:
        """Angular spacing between detector channels [rad] — the equiangular sampling step."""
        return self.transverse_spacing / self.source_to_detector

    def channel_angles(self) -> np.ndarray:
        """Fan angle ``gamma`` of every channel [rad], zero on the central ray."""
        return (np.arange(self.n_channels) - self.central_channel) * self.channel_angle_spacing

    @property
    def fan_angle(self) -> float:
        """Full fan angle subtended by the detector [rad]."""
        return float(self.n_channels * self.channel_angle_spacing)

    @property
    def half_fan_angle(self) -> float:
        """Largest fan angle of any channel [rad].

        Not half of :attr:`fan_angle`: the central ray sits at a fractional channel (the
        quarter-detector offset), so the fan is slightly asymmetric and the short-scan range
        is set by the *larger* side. Using half the full fan instead leaves the short scan a
        fraction of a degree short of complete, which Parker weighting then cannot fix.
        """
        return float(np.abs(self.channel_angles()).max())

    @property
    def short_scan_range(self) -> float:
        """Source rotation a short scan needs, ``pi + 2 * half_fan_angle`` [rad]."""
        return float(np.pi + 2.0 * self.half_fan_angle)

    @property
    def transverse_spacing_at_isocentre(self) -> float:
        """Detector element pitch projected back to the isocentre [mm].

        This, not the image pixel, sets the frequency beyond which the acquisition carries
        no information.
        """
        return self.transverse_spacing / self.magnification

    @property
    def axial_spacing_at_isocentre(self) -> float:
        """Detector row pitch projected back to the isocentre [mm]."""
        return self.axial_spacing / self.magnification

    @property
    def detector_nyquist(self) -> float:
        """Nyquist frequency of the detector sampling at the isocentre [cycles/mm]."""
        return 0.5 / self.transverse_spacing_at_isocentre

    @property
    def collimation_at_isocentre(self) -> float:
        """Total beam width along z at the isocentre [mm]."""
        return self.n_rows * self.axial_spacing_at_isocentre


@dataclass(frozen=True, eq=False)
class ViewTable:
    """The per-view quantities of a projection series, in acquisition order.

    Attributes
    ----------
    angle:
        Gantry rotation angle [rad], one per view.
    axial_position:
        Axial position of the focal centre [mm], one per view. In a helical scan this is the
        table feed seen from the gantry, and it is what places a view in z.
    ffs_dphi, ffs_dz, ffs_drho:
        Flying-focal-spot deflection of each view: angular [rad], axial [mm], radial [mm].
    meta:
        Provenance.

    """

    angle: np.ndarray
    axial_position: np.ndarray
    ffs_dphi: np.ndarray
    ffs_dz: np.ndarray
    ffs_drho: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the arrays agree in length and carry no NaN."""
        n = self.angle.size
        for name in ("axial_position", "ffs_dphi", "ffs_dz", "ffs_drho"):
            arr = getattr(self, name)
            if arr.size != n:
                raise ValueError(f"{name} has {arr.size} entries, expected {n}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains a non-finite value")
        if n < 2:
            raise ValueError(f"a view table needs at least two views, got {n}")

    def __len__(self) -> int:
        """Number of views."""
        return int(self.angle.size)

    @property
    def unwrapped_angle(self) -> np.ndarray:
        """The gantry angle without the 2π wrap, so differences are meaningful."""
        return np.unwrap(self.angle)

    @property
    def rotations(self) -> float:
        """Number of gantry rotations covered by the series."""
        span = self.unwrapped_angle[-1] - self.unwrapped_angle[0]
        return float(abs(span) / (2.0 * np.pi))

    @property
    def views_per_rotation(self) -> float:
        """Views per full rotation, from the mean angular step."""
        step = float(np.abs(np.diff(self.unwrapped_angle)).mean())
        return float(2.0 * np.pi / step)

    def pitch(self, geometry: ScanGeometry) -> float:
        """Helical pitch: table feed per rotation divided by the collimation at the isocentre."""
        feed = float(np.abs(np.diff(self.axial_position)).mean()) * self.views_per_rotation
        return feed / geometry.collimation_at_isocentre

    def focal_spot_classes(self) -> list[np.ndarray]:
        """Split the views into groups that share one exact focal-spot position.

        A flying focal spot moves the source between views. Correcting that view by view
        would require a reconstruction whose source position varies, which most projectors
        cannot express. But when the deflection cycles through a *fixed* set of positions —
        which is what ``FFSZ`` does — each position defines a subset of views with a constant,
        exactly known geometry. Reconstructing per class is therefore exact rather than
        approximate, at the cost of dividing the views between classes.

        Returns
        -------
        list of ndarray
            View indices for each distinct deflection, ordered by first appearance.

        """
        key = np.stack(
            [
                np.round(self.ffs_dphi, 9),
                np.round(self.ffs_dz, 6),
                np.round(self.ffs_drho, 6),
            ],
            axis=1,
        )
        _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)
        return [np.flatnonzero(inverse == k) for k in order]

    def source_positions(self, geometry: ScanGeometry) -> np.ndarray:
        """Source position of every view in the gantry frame, ``(n_views, 3)`` in mm.

        The nominal source sits at radius ``source_to_isocentre`` and the focal-spot
        deflection is added to it: ``ffs_drho`` along the radius, ``ffs_dphi`` around the
        rotation axis, ``ffs_dz`` along it.
        """
        rho = geometry.source_to_isocentre + self.ffs_drho
        phi = self.angle + self.ffs_dphi
        return np.stack(
            [-rho * np.sin(phi), rho * np.cos(phi), self.axial_position + self.ffs_dz],
            axis=1,
        )
