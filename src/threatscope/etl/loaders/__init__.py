"""Load stage: persist transformed records to durable stores.

Two complementary stores share the :class:`ThreatEvent` contract: the Parquet
:class:`FeatureStore` (append-only, ML-facing) and the SQLAlchemy
:class:`EventDatabase` (upsert-by-id system of record).
"""

from .db import Event, EventDatabase, EventTechnique
from .feature_store import EVENTS_SCHEMA, FeatureStore

__all__ = [
    "FeatureStore",
    "EVENTS_SCHEMA",
    "EventDatabase",
    "Event",
    "EventTechnique",
]
