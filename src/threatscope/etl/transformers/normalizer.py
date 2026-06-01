"""Normalization: raw source records -> canonical :class:`ThreatEvent`.

This is the first transform stage and the only one that knows about source
quirks. It maps the heterogeneous outputs of the extractors onto the single
:class:`ThreatEvent` contract — standardizing timestamps and severity scales
and deduplicating along the way (per the outline) — so the joiner and encoder
never branch on provenance.

One normalizer per *event-producing* source:

* :class:`NVDNormalizer` — NVD CVE 2.0 records,
* :class:`OTXNormalizer` — AlienVault OTX pulses.

MITRE ATT&CK is intentionally absent: it yields reference/label data (techniques,
tactics, groups), not threat events, and is consumed by the joiner rather than
normalized into the event stream.
"""

from __future__ import annotations

import abc
import logging
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Tuple, TypeVar

from .base import BaseTransformer
from .schema import CVSS, Severity, SourceType, ThreatEvent

logger = logging.getLogger(__name__)

# CVSS metric blocks in the NVD response, in descending order of preference.
_CVSS_METRIC_KEYS = ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2")
# Fallback CVSS version when the metric block omits it.
_CVSS_VERSION_BY_KEY = {
    "cvssMetricV31": "3.1",
    "cvssMetricV30": "3.0",
    "cvssMetricV2": "2.0",
}

RecordT = TypeVar("RecordT", bound=Mapping[str, Any])


# --------------------------------------------------------------------------- #
# Shared field-level helpers
# --------------------------------------------------------------------------- #
def parse_timestamp(value: Any) -> Optional[datetime]:
    """Best-effort ISO-8601 parse. Returns ``None`` for missing/unparseable
    values; the schema coerces the result to UTC on construction."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):  # fromisoformat (<3.11) rejects a bare 'Z'
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        logger.debug("Unparseable timestamp: %r", value)
        return None


def clean_text(value: Any) -> str:
    """Collapse internal whitespace/newlines to single spaces; never None."""
    if not value:
        return ""
    return " ".join(str(value).split())


def _coerce_severity(name: Optional[str], score: float) -> Severity:
    """Use the source's qualitative severity if valid, else derive from score."""
    if name:
        try:
            return Severity(name.upper())
        except ValueError:
            pass
    return Severity.from_score(float(score))


def _english_description(descriptions: Optional[Sequence[Mapping[str, Any]]]) -> str:
    """Pick the English description, falling back to the first available one."""
    fallback = ""
    for entry in descriptions or []:
        value = entry.get("value", "")
        if entry.get("lang") == "en":
            return value
        fallback = fallback or value
    return fallback


def _select_cvss_metric(metrics: Mapping[str, Any]) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
    """Choose the best CVSS metric entry, preferring v3.1 and Primary scorers."""
    for key in _CVSS_METRIC_KEYS:
        entries = metrics.get(key)
        if entries:
            primary = next((e for e in entries if e.get("type") == "Primary"), None)
            return key, (primary or entries[0])
    return None, None


def _cvss_from_metrics(metrics: Optional[Mapping[str, Any]]) -> Optional[CVSS]:
    """Build a normalized :class:`CVSS` from an NVD ``metrics`` block.

    NVD already exposes the base-vector metrics as full words, so we read them
    directly. v2 uses different field names (``accessVector``/``accessComplexity``)
    and lacks PR/UI — those simply stay ``None``.
    """
    if not metrics:
        return None
    key, entry = _select_cvss_metric(metrics)
    if entry is None:
        return None
    data = entry.get("cvssData", {})
    base_score = data.get("baseScore")
    if base_score is None:
        return None
    severity = _coerce_severity(data.get("baseSeverity") or entry.get("baseSeverity"), base_score)
    return CVSS(
        version=str(data.get("version") or _CVSS_VERSION_BY_KEY.get(key, "")),
        base_score=float(base_score),
        severity=severity,
        vector_string=data.get("vectorString", ""),
        attack_vector=data.get("attackVector") or data.get("accessVector"),
        attack_complexity=data.get("attackComplexity") or data.get("accessComplexity"),
        privileges_required=data.get("privilegesRequired"),
        user_interaction=data.get("userInteraction"),
        scope=data.get("scope"),
    )


def _cpe_vendor(criteria: Optional[str]) -> Optional[str]:
    """Pull the vendor field out of a CPE 2.3 string (``cpe:2.3:part:vendor:...``)."""
    if not criteria:
        return None
    parts = criteria.split(":")
    if len(parts) > 4 and parts[0] == "cpe":
        vendor = parts[3]
        if vendor and vendor not in ("*", "-"):
            return vendor.replace("\\", "")
    return None


def _vendors_from_configurations(configurations: Optional[Sequence[Mapping[str, Any]]]) -> Tuple[str, ...]:
    """Collect distinct affected vendors from a CVE's CPE configurations."""
    vendors: list = []
    seen = set()
    for config in configurations or []:
        for node in config.get("nodes", []) or []:
            for match in node.get("cpeMatch", []) or []:
                vendor = _cpe_vendor(match.get("criteria") or match.get("cpe23Uri"))
                if vendor and vendor not in seen:
                    seen.add(vendor)
                    vendors.append(vendor)
    return tuple(vendors)


def _coerce_attack_ids(attack_ids: Any) -> Tuple[str, ...]:
    """Normalize OTX ``attack_ids`` (strings or ``{id, name}`` dicts) to T-codes."""
    out: list = []
    for item in attack_ids or []:
        if isinstance(item, str):
            technique_id = item
        elif isinstance(item, Mapping):
            technique_id = item.get("id") or item.get("display_name") or item.get("name")
        else:
            technique_id = None
        if technique_id:
            out.append(str(technique_id).upper())
    return tuple(dict.fromkeys(out))  # dedupe, preserving order


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #
class BaseNormalizer(BaseTransformer[RecordT, ThreatEvent], abc.ABC):
    """Drives the normalize loop shared by every source.

    Iterates raw records, delegates the per-record mapping to :meth:`_to_event`,
    isolates per-record failures (a bad record is logged and skipped, never
    aborting the stream), and optionally deduplicates by ``event_id`` with
    keep-first semantics so the stage stays streaming.
    """

    #: Provenance assigned to events from this source.
    source: SourceType

    def __init__(self, *, dedupe: bool = True) -> None:
        self.dedupe = dedupe

    def transform(self, records):
        seen: set = set()
        for record in records:
            try:
                event = self._to_event(record)
            except Exception:  # noqa: BLE001 - one bad record must not kill the run
                logger.exception("Failed to normalize a %s record; skipping", self.source.value)
                continue
            if event is None:
                continue
            if self.dedupe:
                if event.event_id in seen:
                    logger.debug("Dropping duplicate %s", event.event_id)
                    continue
                seen.add(event.event_id)
            yield event

    @abc.abstractmethod
    def _to_event(self, record: RecordT) -> Optional[ThreatEvent]:
        """Map one raw record to a :class:`ThreatEvent`, or ``None`` to drop it."""
        raise NotImplementedError


class NVDNormalizer(BaseNormalizer[Mapping[str, Any]]):
    """Normalizes NVD CVE 2.0 ``cve`` objects."""

    source = SourceType.NVD

    def _to_event(self, record: Mapping[str, Any]) -> Optional[ThreatEvent]:
        cve_id = record.get("id")
        if not cve_id:
            return None
        return ThreatEvent(
            event_id=cve_id,
            source=self.source,
            title=cve_id,
            description=clean_text(_english_description(record.get("descriptions"))),
            published=parse_timestamp(record.get("published")),
            modified=parse_timestamp(record.get("lastModified")),
            cvss=_cvss_from_metrics(record.get("metrics")),
            vendors=_vendors_from_configurations(record.get("configurations")),
            raw=record,
        )


class OTXNormalizer(BaseNormalizer[Mapping[str, Any]]):
    """Normalizes AlienVault OTX pulses."""

    source = SourceType.OTX

    def _to_event(self, record: Mapping[str, Any]) -> Optional[ThreatEvent]:
        pulse_id = record.get("id")
        if not pulse_id:
            return None
        adversary = clean_text(record.get("adversary")) or None
        return ThreatEvent(
            event_id=str(pulse_id),
            source=self.source,
            title=clean_text(record.get("name")),
            description=clean_text(record.get("description")),
            published=parse_timestamp(record.get("created")),
            modified=parse_timestamp(record.get("modified")),
            ioc_count=len(record.get("indicators") or []),
            tags=tuple(record.get("tags") or ()),
            technique_ids=_coerce_attack_ids(record.get("attack_ids")),
            actor=adversary,
            raw=record,
        )
