"""Full ELT orchestration: extract -> transform -> load.

Wires every stage built so far into one run that ends with a populated feature
store (Parquet) and system-of-record database (SQLAlchemy) — i.e. the
``--store-root`` that ``src.ml.train`` consumes.

Flow:

1. **MITRE reference** is loaded first — its technique catalog is both the join
   dictionary for the :class:`MITREJoiner` and the stable label space
   (``technique_vocab``) for the encoder.
2. **Extract + normalize** CVEs (NVD) and pulses (OTX) into one
   :class:`ThreatEvent` stream.
3. **Join** each event with ATT&CK to fill in tactics/platforms (and optional
   CVE->technique links / actor techniques).
4. **Encode** the enriched events into model-ready arrays.
5. **Load** events into the database (upsert) and both the event tables and
   encoded features into the Parquet store.

Run it as a module from the project root::

    python -m src.pipeline --nvd-days 30 --otx-max-pulses 500
    python -m src.pipeline --sources nvd --no-db        # CVEs only, features only

API keys come from ``--*-api-key`` or the ``NVD_API_KEY`` / ``OTX_API_KEY``
environment variables.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .etl.extractors.mitre import MITREExtractor
from .etl.extractors.nvd import NVDExtractor, MAX_DATE_RANGE_DAYS
from .etl.extractors.otx import OTXExtractor
from .etl.loaders.db import EventDatabase
from .etl.loaders.feature_store import FeatureStore
from .etl.transformers.encoder import FeatureEncoder
from .etl.transformers.joiner import MITREJoiner
from .etl.transformers.normalizer import NVDNormalizer, OTXNormalizer
from .etl.transformers.schema import ThreatEvent

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Everything the pipeline needs; populated from the CLI (see :func:`main`)."""

    # Outputs
    store_root: str = "data/feature_store"
    db_url: str = "sqlite:///data/tactclass.db"
    feature_name: str = "features"
    write_db: bool = True
    write_features: bool = True

    # Which event sources to extract (MITRE reference is always loaded).
    sources: Tuple[str, ...] = ("nvd", "otx")

    # NVD
    nvd_api_key: Optional[str] = None
    nvd_days: int = 30
    nvd_results_per_page: int = 2000

    # OTX
    otx_api_key: Optional[str] = None
    otx_days: int = 30
    otx_max_pulses: Optional[int] = 500

    # MITRE ATT&CK
    mitre_domain: str = "enterprise"
    mitre_version: Optional[str] = None

    # Join behavior
    cve_technique_map_path: Optional[str] = None
    expand_actor_techniques: bool = False


def _date_windows(
    start: datetime, end: datetime, max_days: int
) -> Iterator[Tuple[datetime, datetime]]:
    """Split ``[start, end]`` into consecutive windows no wider than ``max_days``
    (NVD rejects date ranges over 120 days)."""
    step = timedelta(days=max_days)
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


class Pipeline:
    """Stateful orchestrator for one ELT run."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # Single reference instant: the NVD window end and the encoder's
        # "now" for days-since-published, so the run is internally consistent.
        self.reference_time = datetime.now(timezone.utc)

    # -- extract / transform per source ------------------------------------ #
    def load_mitre(self) -> Tuple[list, list]:
        """Load the ATT&CK technique catalog and group->technique mappings."""
        logger.info("Loading MITRE ATT&CK (%s)", self.config.mitre_domain)
        with MITREExtractor(self.config.mitre_domain, version=self.config.mitre_version) as mitre:
            techniques = mitre.techniques()
            group_mappings = mitre.group_technique_mappings()
        logger.info("MITRE: %d techniques, %d group->technique mappings",
                    len(techniques), len(group_mappings))
        return techniques, group_mappings

    def extract_nvd(self) -> List[ThreatEvent]:
        """Pull recently-modified CVEs and normalize them to events."""
        end = self.reference_time
        start = end - timedelta(days=self.config.nvd_days)
        normalizer = NVDNormalizer()
        logger.info("Extracting NVD CVEs modified in the last %d days", self.config.nvd_days)
        with NVDExtractor(
            api_key=self.config.nvd_api_key,
            results_per_page=self.config.nvd_results_per_page,
        ) as nvd:
            cves = (
                cve
                for window_start, window_end in _date_windows(start, end, MAX_DATE_RANGE_DAYS)
                for cve in nvd.iter_cves(last_mod_start_date=window_start, last_mod_end_date=window_end)
            )
            events = list(normalizer.transform(cves))
        logger.info("NVD: %d events", len(events))
        return events

    def extract_otx(self) -> List[ThreatEvent]:
        """Pull recent OTX pulses and normalize them to events."""
        if not self.config.otx_api_key:
            raise ValueError("OTX source requested but no API key (set OTX_API_KEY or --otx-api-key)")
        start = self.reference_time - timedelta(days=self.config.otx_days)
        normalizer = OTXNormalizer()
        logger.info("Extracting OTX pulses modified in the last %d days (max %s)",
                    self.config.otx_days, self.config.otx_max_pulses)
        with OTXExtractor(api_key=self.config.otx_api_key) as otx:
            pulses = otx.iter_pulses(modified_since=start, max_pulses=self.config.otx_max_pulses)
            events = list(normalizer.transform(pulses))
        logger.info("OTX: %d events", len(events))
        return events

    # -- orchestration ----------------------------------------------------- #
    def run(self) -> Dict[str, Any]:
        """Execute the full ELT run and return a summary of what happened."""
        techniques, group_mappings = self.load_mitre()
        technique_vocab = [t.technique_id for t in techniques]

        events: List[ThreatEvent] = []
        if "nvd" in self.config.sources:
            events += self.extract_nvd()
        if "otx" in self.config.sources:
            events += self.extract_otx()
        logger.info("Extracted %d events from %s", len(events), ", ".join(self.config.sources))

        # TRANSFORM: join with ATT&CK, then encode to model-ready arrays.
        joiner = MITREJoiner(
            techniques,
            group_mappings=group_mappings,
            cve_technique_map=self._load_cve_technique_map(),
            expand_actor_techniques=self.config.expand_actor_techniques,
        )
        events = list(joiner.transform(events))

        encoder = FeatureEncoder(technique_vocab=technique_vocab, reference_time=self.reference_time)
        encoded = list(encoder.fit_transform(events)) if events else []

        # LOAD
        summary: Dict[str, Any] = {
            "events": len(events),
            "encoded": len(encoded),
            "techniques": len(technique_vocab),
            "sources": list(self.config.sources),
            "outputs": {},
        }
        if not events:
            logger.warning("No events extracted; nothing to load.")
            return summary

        if self.config.write_db:
            db = EventDatabase(self.config.db_url)
            db.upsert_events(events)
            summary["outputs"]["db"] = self.config.db_url

        if self.config.write_features:
            store = FeatureStore(self.config.store_root)
            store.write_events(events, partition_by_source=True)
            store.write_encoded(encoded, name=self.config.feature_name, metadata={
                "feature_spec": encoder.feature_spec(),
                "technique_vocab": technique_vocab,
                "structured_feature_names": encoder.structured_feature_names(),
                "generated_at": self.reference_time.isoformat(),
            })
            # Persist the fitted encoder so `src.ml.predict` can encode raw input
            # identically at inference time.
            encoder_path = Path(self.config.store_root) / "encoder.json"
            encoder.save(encoder_path)
            summary["outputs"]["feature_store"] = self.config.store_root
            summary["outputs"]["encoder"] = str(encoder_path)

        logger.info("Pipeline complete: %s", summary)
        return summary

    def _load_cve_technique_map(self) -> Optional[Dict[str, List[str]]]:
        path = self.config.cve_technique_map_path
        if not path:
            return None
        mapping = json.loads(Path(path).read_text())
        logger.info("Loaded %d CVE->technique links from %s", len(mapping), path)
        return mapping


def _ensure_parent_dirs(config: PipelineConfig) -> None:
    """Create output directories so SQLite/Parquet writes don't fail on a fresh checkout."""
    Path(config.store_root).mkdir(parents=True, exist_ok=True)
    if config.db_url.startswith("sqlite:///"):
        db_path = Path(config.db_url[len("sqlite:///"):])
        if db_path.parent and not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)


def main(argv=None) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from .config import get_api_key, load_config

    # Pre-parse --config so the file's values can seed the other defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    config_path = pre.parse_known_args(argv)[0].config
    cfg = load_config(config_path)
    pipe = cfg["pipeline"]

    parser = argparse.ArgumentParser(
        description="Run the TactClass ELT pipeline (extract -> transform -> load)",
    )
    parser.add_argument("--config", default=config_path, help="path to a config.yaml (default: src/config.yaml)")
    parser.add_argument("--sources", nargs="+", choices=("nvd", "otx"), default=list(pipe["sources"]),
                        help="event sources to extract (otx is skipped if no API key is available)")
    parser.add_argument("--store-root", default=cfg["feature_store"]["root"])
    parser.add_argument("--db-url", default=cfg["database"]["url"])
    parser.add_argument("--name", default=cfg["feature_store"]["name"], help="encoded dataset name")
    parser.add_argument("--no-db", action="store_true", help="skip the SQLAlchemy load")
    parser.add_argument("--no-features", action="store_true", help="skip the Parquet feature store load")

    parser.add_argument("--nvd-api-key", default=get_api_key(cfg, "nvd"))
    parser.add_argument("--nvd-days", type=int, default=pipe["nvd_days"])
    parser.add_argument("--otx-api-key", default=get_api_key(cfg, "otx"))
    parser.add_argument("--otx-days", type=int, default=pipe["otx_days"])
    parser.add_argument("--otx-max-pulses", type=int, default=pipe["otx_max_pulses"])

    parser.add_argument("--mitre-domain", default=pipe["mitre_domain"],
                        choices=("enterprise", "mobile", "ics"))
    parser.add_argument("--mitre-version", default=pipe["mitre_version"])

    parser.add_argument("--cve-technique-map", default=pipe["cve_technique_map"],
                        help="JSON file of {cve_id: [technique_id, ...]} to attach CVE->ATT&CK links")
    parser.add_argument("--expand-actor-techniques", action="store_true",
                        default=pipe["expand_actor_techniques"],
                        help="add an actor's known techniques to its pulses")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Honor the requested sources, but drop OTX if no key is available.
    sources = tuple(args.sources)
    if "otx" in sources and not args.otx_api_key:
        logger.warning("OTX requested but no API key found; skipping it "
                       "(set api_keys.otx in config.yaml or OTX_API_KEY).")
        sources = tuple(s for s in sources if s != "otx")

    config = PipelineConfig(
        store_root=args.store_root,
        db_url=args.db_url,
        feature_name=args.name,
        write_db=pipe["write_db"] and not args.no_db,
        write_features=pipe["write_features"] and not args.no_features,
        sources=sources,
        nvd_api_key=args.nvd_api_key,
        nvd_days=args.nvd_days,
        nvd_results_per_page=pipe["nvd_results_per_page"],
        otx_api_key=args.otx_api_key,
        otx_days=args.otx_days,
        otx_max_pulses=args.otx_max_pulses,
        mitre_domain=args.mitre_domain,
        mitre_version=args.mitre_version,
        cve_technique_map_path=args.cve_technique_map,
        expand_actor_techniques=args.expand_actor_techniques,
    )
    _ensure_parent_dirs(config)

    summary = Pipeline(config).run()
    print(json.dumps(summary, indent=2))
