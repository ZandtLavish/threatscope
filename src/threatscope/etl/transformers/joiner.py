"""Joining: enrich :class:`ThreatEvent` records with MITRE ATT&CK context.

The normalizer seeds ``technique_ids`` where the source provides them (OTX
``attack_ids``); this stage turns those — plus any externally supplied CVE →
technique links — into the derived features the model consumes:

* resolve each technique against the ATT&CK catalog (sub-techniques fall back
  to their parent), and from the resolved techniques fill in
* ``tactics``   — the ATT&CK kill-chain phases (the ``attack_phase`` feature), and
* ``platforms`` — the technique's affected platforms.

It can optionally graft an actor's known techniques onto a pulse via the
group → technique mappings. The reference data (techniques, mappings) is
*injected* rather than fetched, so the joiner is decoupled from the extract
layer and easy to test with stand-in objects.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from .base import BaseTransformer
from .schema import SourceType, ThreatEvent

if TYPE_CHECKING:  # avoid a runtime transformers -> extractors dependency
    from ..extractors.mitre import GroupTechniqueMapping, Technique

logger = logging.getLogger(__name__)


def _ordered_unique(*iterables: Iterable[str]) -> Tuple[str, ...]:
    """Concatenate iterables, dropping falsy/duplicate items, preserving order."""
    seen: Dict[str, None] = {}
    for iterable in iterables:
        for item in iterable:
            if item:
                seen[item] = None
    return tuple(seen)


class MITREJoiner(BaseTransformer[ThreatEvent, ThreatEvent]):
    """Enriches each event with ATT&CK tactics/platforms from its techniques.

    Args:
        techniques: the ATT&CK technique catalog (e.g. ``MITREExtractor.techniques()``).
        group_mappings: optional group -> technique links, used only when
            ``expand_actor_techniques`` is set.
        cve_technique_map: optional ``{cve_id: [technique_id, ...]}`` to attach
            known CVE -> ATT&CK links (NVD itself carries none).
        expand_actor_techniques: if True, add an event actor's known techniques.
        drop_unknown: if True, discard technique IDs absent from the catalog
            instead of keeping them as opaque labels.
    """

    def __init__(
        self,
        techniques: "Iterable[Technique]",
        *,
        group_mappings: "Optional[Iterable[GroupTechniqueMapping]]" = None,
        cve_technique_map: Optional[Mapping[str, Iterable[str]]] = None,
        expand_actor_techniques: bool = False,
        drop_unknown: bool = False,
    ) -> None:
        self.expand_actor_techniques = expand_actor_techniques
        self.drop_unknown = drop_unknown

        # Catalog indexed by ATT&CK ID for O(1) resolution.
        self._technique_by_id: Dict[str, "Technique"] = {
            t.technique_id: t for t in techniques
        }

        # actor key (group id and name, lower-cased) -> ordered technique IDs.
        self._group_techniques: Dict[str, List[str]] = {}
        for mapping in group_mappings or []:
            for key in (mapping.group_id, mapping.group_name):
                if not key:
                    continue
                bucket = self._group_techniques.setdefault(key.strip().lower(), [])
                if mapping.technique_id not in bucket:
                    bucket.append(mapping.technique_id)

        # CVE -> techniques, with keys/values normalized to upper-case.
        self._cve_map: Dict[str, Tuple[str, ...]] = {
            str(cve).upper(): _ordered_unique(str(t).upper() for t in techs)
            for cve, techs in (cve_technique_map or {}).items()
        }

    def transform(self, records):
        enriched = 0
        for event in records:
            self._enrich(event)
            enriched += 1
            yield event
        logger.info("MITREJoiner: enriched %d events", enriched)

    def _resolve(self, technique_id: str) -> "Optional[Technique]":
        """Look up a technique, falling back to the parent for a sub-technique."""
        technique = self._technique_by_id.get(technique_id)
        if technique is None and "." in technique_id:
            technique = self._technique_by_id.get(technique_id.split(".")[0])
        return technique

    def _enrich(self, event: ThreatEvent) -> ThreatEvent:
        candidate_ids: List[str] = list(event.technique_ids)

        # Attach known CVE -> technique links (NVD provides none on its own).
        if self._cve_map and event.source == SourceType.NVD:
            candidate_ids.extend(self._cve_map.get(event.event_id.upper(), ()))

        # Optionally graft the actor's known techniques onto the event.
        if self.expand_actor_techniques and event.actor:
            candidate_ids.extend(self._group_techniques.get(event.actor.strip().lower(), ()))

        resolved_ids: List[str] = []
        tactics: List[str] = []
        platforms: List[str] = []
        for technique_id in _ordered_unique(candidate_ids):
            technique = self._resolve(technique_id)
            if technique is None:
                if not self.drop_unknown:
                    resolved_ids.append(technique_id)  # keep as opaque label
                continue
            resolved_ids.append(technique_id)
            tactics.extend(technique.tactics)
            platforms.extend(technique.platforms)

        # Reassign (don't mutate) the tuple fields, merging with any existing
        # values the normalizer may already have set.
        event.technique_ids = _ordered_unique(resolved_ids)
        event.tactics = _ordered_unique(event.tactics, tactics)
        event.platforms = _ordered_unique(event.platforms, platforms)
        return event

