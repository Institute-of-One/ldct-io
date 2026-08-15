"""Provenance: which series were used, from where, under what licence.

A study that claims to be reproducible has to say exactly which data it read. The series UID
is the only stable identifier — file names are assigned by whatever downloaded the data and
carry no meaning — so a manifest records UIDs, not paths, and travels in the repository while
the data itself stays outside it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: The subset of LDCT-and-Projection-data this project is permitted to use.
#:
#: Chest, liver/abdomen and the phantom object are CC BY 4.0. The head cases are *not*: they
#: remain under the NIH Controlled Data Access Policy because a face can be reconstructed from
#: them. Nothing in this project reads them.
PERMITTED_BODY_PARTS = frozenset({"CHEST", "ABDOMEN", "LIVER", "PHANTOM"})
EXCLUDED_BODY_PARTS = frozenset({"HEAD", "NEURO", "BRAIN"})

#: The collection this project draws on.
COLLECTION = "LDCT-and-Projection-data"
COLLECTION_DOI = "https://doi.org/10.7937/9npb-2637"
COLLECTION_LICENCE = "CC BY 4.0"


@dataclass(frozen=True)
class SeriesRecord:
    """One series used by the study.

    Attributes
    ----------
    series_instance_uid:
        The identifier a reader would use to fetch exactly this data.
    patient_id, body_part, description:
        As recorded by the archive.
    manufacturer, model:
        Scanner.
    n_files, total_bytes, listing_sha256:
        Content digest, from :func:`~ldct_io.dicomctpd.directory_digest`.
    role:
        What the study used it for, in the study's own words.
    licence:
        The licence this series is served under.

    """

    series_instance_uid: str
    patient_id: str
    body_part: str
    description: str
    manufacturer: str
    model: str = ""
    n_files: int = 0
    total_bytes: int = 0
    listing_sha256: str = ""
    role: str = ""
    licence: str = COLLECTION_LICENCE

    def __post_init__(self) -> None:
        """Refuse a body part this project is not permitted to use."""
        part = self.body_part.strip().upper()
        if part in EXCLUDED_BODY_PARTS:
            raise ValueError(
                f"body part {self.body_part!r} is under controlled access in {COLLECTION} and "
                f"is excluded from this project by design; it must not appear in a manifest"
            )
        if part and part not in PERMITTED_BODY_PARTS:
            raise ValueError(
                f"body part {self.body_part!r} is not one this project has established a "
                f"licence basis for. Permitted: {sorted(PERMITTED_BODY_PARTS)}"
            )
        if not self.series_instance_uid.strip():
            raise ValueError("a series record needs its SeriesInstanceUID")


@dataclass
class Manifest:
    """The set of series a study used, written next to the code that used them."""

    records: list[SeriesRecord] = field(default_factory=list)
    collection: str = COLLECTION
    doi: str = COLLECTION_DOI
    notes: str = ""

    def add(self, record: SeriesRecord) -> None:
        """Add a series, refusing a duplicate UID."""
        if any(r.series_instance_uid == record.series_instance_uid for r in self.records):
            raise ValueError(f"series {record.series_instance_uid} is already in the manifest")
        self.records.append(record)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dictionary."""
        return {
            "collection": self.collection,
            "doi": self.doi,
            "licence": COLLECTION_LICENCE,
            "excluded_by_design": sorted(EXCLUDED_BODY_PARTS),
            "notes": self.notes,
            "series": [asdict(r) for r in self.records],
        }

    def write(self, path: str | Path) -> Path:
        """Write the manifest as JSON."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        """Read a manifest back, re-validating every record."""
        payload = json.loads(Path(path).read_text())
        return cls(
            records=[SeriesRecord(**r) for r in payload.get("series", [])],
            collection=payload.get("collection", COLLECTION),
            doi=payload.get("doi", COLLECTION_DOI),
            notes=payload.get("notes", ""),
        )
