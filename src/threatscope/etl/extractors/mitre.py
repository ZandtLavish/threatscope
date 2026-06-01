"""Extractor / parser for MITRE ATT&CK (the outline's ``TTPParser``).

MITRE publishes ATT&CK as STIX 2.x bundles in the
``mitre-attack/attack-stix-data`` GitHub repository — one bundle per domain
(enterprise / mobile / ICS). Unlike the NVD source, this is a single large JSON
document rather than a paginated REST API, so the only thing reused from
:class:`BaseExtractor` is the resilient HTTP fetch; everything below it is
STIX-specific parsing.

The parser turns the raw STIX object graph into the structured entities the ML
pipeline cares about:

* **techniques**  — ATT&CK techniques/sub-techniques (the ``T####`` labels),
* **tactics**     — the kill-chain phases (``attack_phase`` feature),
* **groups**      — named threat actors (for the attribution extension), and
* **group → technique** mappings, derived from STIX ``uses`` relationships.

Data: https://github.com/mitre-attack/attack-stix-data
Tooling reference: https://github.com/mitre-attack/mitreattack-python
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from stix2 import Filter, MemoryStore

from .base import BaseExtractor

logger = logging.getLogger(__name__)

ATTACK_STIX_BASE = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master"

# ATT&CK domain -> directory/filename stem in the attack-stix-data repo.
DOMAIN_DIRS = {
    "enterprise": "enterprise-attack",
    "mobile": "mobile-attack",
    "ics": "ics-attack",
}

# Source name that marks an object's canonical ATT&CK ID in external_references.
_ATTACK_SOURCE = "mitre-attack"
_KILL_CHAIN = "mitre-attack"


# --------------------------------------------------------------------------- #
# Structured outputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Technique:
    technique_id: str                       # e.g. "T1059" or "T1059.001"
    name: str
    description: str
    tactics: tuple                          # tactic shortnames (kill-chain phases)
    platforms: tuple
    is_subtechnique: bool
    parent_technique_id: Optional[str]      # "T1059" for "T1059.001", else None
    deprecated: bool
    stix_id: str


@dataclass(frozen=True)
class Tactic:
    tactic_id: str                          # e.g. "TA0002"
    name: str                               # e.g. "Execution"
    shortname: str                          # e.g. "execution" (matches kill-chain phase)
    description: str
    stix_id: str


@dataclass(frozen=True)
class Group:
    group_id: str                           # e.g. "G0016"
    name: str
    aliases: tuple
    description: str
    deprecated: bool
    stix_id: str


@dataclass(frozen=True)
class GroupTechniqueMapping:
    group_id: str
    group_name: str
    technique_id: str
    technique_name: str


# --------------------------------------------------------------------------- #
# STIX field helpers (work on either stix2 objects or plain dicts — both are
# Mappings, so attribute access is uniform via .get / __getitem__)
# --------------------------------------------------------------------------- #
def _attack_id(obj: Mapping[str, Any]) -> Optional[str]:
    """Return the canonical ATT&CK external ID (T####, TA####, G####, ...)."""
    for ref in obj.get("external_references", []) or []:
        if ref.get("source_name") == _ATTACK_SOURCE and ref.get("external_id"):
            return ref["external_id"]
    return None


def _kill_chain_phases(obj: Mapping[str, Any]) -> List[str]:
    """Tactic shortnames from an object's ATT&CK kill-chain phases."""
    return [
        phase["phase_name"]
        for phase in obj.get("kill_chain_phases", []) or []
        if phase.get("kill_chain_name") == _KILL_CHAIN and phase.get("phase_name")
    ]


def _is_active(obj: Mapping[str, Any]) -> bool:
    """True unless the object is revoked or deprecated by ATT&CK."""
    return not (obj.get("revoked") or obj.get("x_mitre_deprecated"))


def _bundle_path(domain_dir: str, version: Optional[str]) -> str:
    """Relative path of a domain bundle; pinned to ``version`` if given."""
    stem = domain_dir if version is None else f"{domain_dir}-{version}"
    return f"/{domain_dir}/{stem}.json"


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class MITREExtractor(BaseExtractor):
    """Fetches and parses a MITRE ATT&CK STIX bundle into structured TTPs.

    The bundle is loaded lazily on first query, so typical use is a one-liner::

        with MITREExtractor(domain="enterprise") as mitre:
            techniques = mitre.techniques()
            mappings = mitre.group_technique_mappings()

    Pin a release with ``version`` (e.g. ``"14.1"``) for reproducible runs, or
    load a previously downloaded bundle offline via :meth:`from_file`.
    """

    base_url = ATTACK_STIX_BASE
    default_headers = {"Accept": "application/json"}

    def __init__(
        self,
        domain: str = "enterprise",
        *,
        version: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        if domain not in DOMAIN_DIRS:
            raise ValueError(
                f"unknown domain {domain!r}; expected one of {sorted(DOMAIN_DIRS)}"
            )
        # GitHub raw needs no client-side rate limiting for a single fetch.
        super().__init__(
            rate_limiter=None,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        self.domain = domain
        self.version = version
        self._store: Optional[MemoryStore] = None

    # -- loading ----------------------------------------------------------- #
    def load(self, objects: Optional[Sequence[Mapping[str, Any]]] = None) -> "MITREExtractor":
        """Build the in-memory STIX store, downloading the bundle if needed."""
        if objects is None:
            objects = self._download_objects()
        # allow_custom: ATT&CK defines custom object types (x-mitre-tactic, ...)
        # and properties (x_mitre_*) that vanilla STIX would reject.
        self._store = MemoryStore(stix_data=list(objects), allow_custom=True)
        logger.info("Loaded %d MITRE ATT&CK STIX objects (%s)", len(objects), self.domain)
        return self

    @classmethod
    def from_file(cls, path: str, domain: str = "enterprise") -> "MITREExtractor":
        """Construct a parser from a STIX bundle already on disk."""
        with open(path, "r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        return cls(domain=domain).load(bundle.get("objects", []))

    def _download_objects(self) -> List[Dict[str, Any]]:
        path = _bundle_path(DOMAIN_DIRS[self.domain], self.version)
        logger.info("Downloading ATT&CK bundle %s (version=%s)", path, self.version or "latest")
        bundle = self.get(path)
        return bundle.get("objects", [])

    def _query(self, *filters: Filter) -> List[Any]:
        if self._store is None:
            self.load()
        return self._store.query(list(filters))

    # -- structured accessors ---------------------------------------------- #
    def techniques(self, *, include_deprecated: bool = False) -> List[Technique]:
        """All ATT&CK techniques and sub-techniques."""
        results: List[Technique] = []
        for obj in self._query(Filter("type", "=", "attack-pattern")):
            if not include_deprecated and not _is_active(obj):
                continue
            attack_id = _attack_id(obj)
            if attack_id is None:
                continue
            is_sub = bool(obj.get("x_mitre_is_subtechnique", False))
            parent = attack_id.split(".")[0] if is_sub and "." in attack_id else None
            results.append(
                Technique(
                    technique_id=attack_id,
                    name=obj.get("name", ""),
                    description=obj.get("description", "") or "",
                    tactics=tuple(_kill_chain_phases(obj)),
                    platforms=tuple(obj.get("x_mitre_platforms", []) or []),
                    is_subtechnique=is_sub,
                    parent_technique_id=parent,
                    deprecated=not _is_active(obj),
                    stix_id=obj["id"],
                )
            )
        return results

    def tactics(self) -> List[Tactic]:
        """The ATT&CK tactics (kill-chain phases) for this domain."""
        results: List[Tactic] = []
        for obj in self._query(Filter("type", "=", "x-mitre-tactic")):
            attack_id = _attack_id(obj)
            if attack_id is None:
                continue
            results.append(
                Tactic(
                    tactic_id=attack_id,
                    name=obj.get("name", ""),
                    shortname=obj.get("x_mitre_shortname", ""),
                    description=obj.get("description", "") or "",
                    stix_id=obj["id"],
                )
            )
        return results

    def groups(self, *, include_deprecated: bool = False) -> List[Group]:
        """Named threat-actor groups (STIX intrusion-sets)."""
        results: List[Group] = []
        for obj in self._query(Filter("type", "=", "intrusion-set")):
            if not include_deprecated and not _is_active(obj):
                continue
            attack_id = _attack_id(obj)
            if attack_id is None:
                continue
            results.append(
                Group(
                    group_id=attack_id,
                    name=obj.get("name", ""),
                    aliases=tuple(obj.get("aliases", []) or []),
                    description=obj.get("description", "") or "",
                    deprecated=not _is_active(obj),
                    stix_id=obj["id"],
                )
            )
        return results

    def group_technique_mappings(self) -> List[GroupTechniqueMapping]:
        """Group → technique links, derived from STIX ``uses`` relationships."""
        # Index by STIX id so relationship endpoints resolve in O(1). Include
        # deprecated entities here so a relationship is never silently dropped
        # because one endpoint happens to be deprecated.
        techniques = {t.stix_id: t for t in self.techniques(include_deprecated=True)}
        groups = {g.stix_id: g for g in self.groups(include_deprecated=True)}

        mappings: List[GroupTechniqueMapping] = []
        for rel in self._query(
            Filter("type", "=", "relationship"),
            Filter("relationship_type", "=", "uses"),
        ):
            group = groups.get(rel.get("source_ref"))
            technique = techniques.get(rel.get("target_ref"))
            if group is not None and technique is not None:
                mappings.append(
                    GroupTechniqueMapping(
                        group_id=group.group_id,
                        group_name=group.name,
                        technique_id=technique.technique_id,
                        technique_name=technique.name,
                    )
                )
        return mappings

    def extract(self) -> Dict[str, List[Any]]:
        """Implements :meth:`BaseExtractor.extract`; returns every entity set."""
        return {
            "techniques": self.techniques(),
            "tactics": self.tactics(),
            "groups": self.groups(),
            "group_techniques": self.group_technique_mappings(),
        }