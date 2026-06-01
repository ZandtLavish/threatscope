"""Canonical normalized schema — the data contract for the transform stage.

The extractors each emit source-shaped records (NVD CVE dicts, OTX pulse dicts,
MITRE ATT&CK dataclasses). Everything downstream — the joiner, the encoder, the
SQLAlchemy loaders, and the ``tf.data`` pipeline — speaks instead in terms of
the source-agnostic types defined here:

* :class:`ThreatEvent` — one normalized observation (a CVE or an OTX pulse),
* :class:`CVSS`        — a normalized severity score + decomposed vector, and
* :class:`Severity` / :class:`SourceType` — the standardized enumerations.

This module is the single place that pins field names, units, and types, so it
deliberately holds *only* the contract: no parsing, no I/O, no source-specific
logic (that belongs in ``normalizer.py``). The few helpers here exist purely to
construct or validate the schema and to keep its invariants in one spot — most
importantly that every datetime is timezone-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


def to_utc(value: datetime) -> datetime:
    """Coerce a datetime to timezone-aware UTC (the schema's only convention).

    Naive datetimes are assumed to already be UTC, which matches how the source
    APIs report timestamps.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SourceType(str, Enum):
    """Provenance of a :class:`ThreatEvent`."""

    NVD = "nvd"
    OTX = "otx"
    MITRE = "mitre"


class Severity(str, Enum):
    """Standardized qualitative severity (CVSS v3 bands)."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        """Map a 0.0-10.0 CVSS base score onto its qualitative band."""
        if score <= 0.0:
            return cls.NONE
        if score < 4.0:
            return cls.LOW
        if score < 7.0:
            return cls.MEDIUM
        if score < 9.0:
            return cls.HIGH
        return cls.CRITICAL


@dataclass(frozen=True)
class CVSS:
    """A normalized CVSS score with the base-vector metrics broken out.

    The decomposed metrics (``attack_vector`` etc.) carry the full normalized
    words ("NETWORK", "LOCAL", ...) rather than the single-letter codes, so the
    encoder can map them to features without re-parsing the vector string.
    """

    version: str                       # "3.1", "3.0", "2.0"
    base_score: float                  # 0.0–10.0
    severity: Severity
    vector_string: str                 # raw vector, e.g. "CVSS:3.1/AV:N/AC:L/..."
    attack_vector: Optional[str] = None
    attack_complexity: Optional[str] = None
    privileges_required: Optional[str] = None
    user_interaction: Optional[str] = None
    scope: Optional[str] = None


@dataclass
class ThreatEvent:
    """One normalized threat observation, unified across sources.

    A CVE and an OTX pulse collapse onto the same shape so the rest of the
    pipeline never branches on provenance. ``technique_ids`` is the multi-label
    ATT&CK target; it may be empty after normalization and is populated/enriched
    later by the joiner. The record is intentionally mutable so the joiner can
    enrich it in place; collection fields are tuples to keep that enrichment
    explicit (reassign, don't mutate).
    """

    event_id: str                      # canonical id: "CVE-2021-44228" or pulse id
    source: SourceType
    title: str = ""
    description: str = ""
    published: Optional[datetime] = None
    modified: Optional[datetime] = None
    cvss: Optional[CVSS] = None
    platforms: Tuple[str, ...] = ()
    vendors: Tuple[str, ...] = ()
    ioc_count: int = 0
    tags: Tuple[str, ...] = ()
    technique_ids: Tuple[str, ...] = ()   # MITRE T-codes; multi-label target
    tactics: Tuple[str, ...] = ()         # ATT&CK tactics / attack phases
    actor: Optional[str] = None           # adversary / group (attribution use)
    # Original source record, retained for lineage; excluded from identity.
    raw: Optional[Mapping[str, Any]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Enforce the UTC-aware invariant at construction so every consumer can
        # rely on it without defensive checks.
        if self.published is not None:
            self.published = to_utc(self.published)
        if self.modified is not None:
            self.modified = to_utc(self.modified)

    def age_days(self, reference: datetime) -> Optional[int]:
        """Whole days between publication and ``reference`` (the outline's
        ``days_since_pub`` feature). Returns ``None`` when undated."""
        if self.published is None:
            return None
        return (to_utc(reference) - self.published).days

    def to_row(self) -> Dict[str, Any]:
        """Flatten to a JSON/DataFrame-friendly dict (drops ``raw``).

        CVSS is unrolled into scalar columns; collection fields become lists.
        This is the row shape the feature store and SQLAlchemy loaders persist.
        """
        cvss = self.cvss
        return {
            "event_id": self.event_id,
            "source": self.source.value,
            "title": self.title,
            "description": self.description,
            "published": self.published.isoformat() if self.published else None,
            "modified": self.modified.isoformat() if self.modified else None,
            "cvss_score": cvss.base_score if cvss else None,
            "cvss_severity": cvss.severity.value if cvss else None,
            "cvss_attack_vector": cvss.attack_vector if cvss else None,
            "cvss_attack_complexity": cvss.attack_complexity if cvss else None,
            "cvss_privileges_required": cvss.privileges_required if cvss else None,
            "cvss_user_interaction": cvss.user_interaction if cvss else None,
            "cvss_scope": cvss.scope if cvss else None,
            "platforms": list(self.platforms),
            "vendors": list(self.vendors),
            "ioc_count": self.ioc_count,
            "tags": list(self.tags),
            "technique_ids": list(self.technique_ids),
            "tactics": list(self.tactics),
            "actor": self.actor,
        }
