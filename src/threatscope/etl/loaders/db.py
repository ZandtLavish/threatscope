"""Relational system-of-record — the SQLAlchemy half of the LOAD stage.

Where the Parquet :mod:`feature_store` is append-only and ML-facing, this is the
authoritative, queryable store of normalized events. It **upserts by event id**,
so re-ingesting the same CVE/pulse updates the row in place rather than
accumulating duplicates (the one gap the feature store leaves open).

Modeling choices, deliberately pragmatic:

* scalar fields map to typed columns; the descriptive multi-valued fields
  (``platforms``, ``vendors``, ``tags``, ``tactics``) are stored as portable
  JSON columns — they are metadata, rarely joined on; while
* ``technique_ids`` — the ATT&CK labels and the primary analytical join key —
  get a first-class ``event_techniques`` association so queries like "every
  event mapped to T1059" are indexed rather than JSON scans.

Runs on SQLite (dev) and PostgreSQL (prod) unchanged — JSON columns and the
get-then-write upsert are dialect-agnostic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, List, Optional

from sqlalchemy import ForeignKey, String, create_engine, func, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)
from sqlalchemy.types import JSON, DateTime, Float, Integer

if TYPE_CHECKING:  # type-only; the loader reads duck-typed ThreatEvent fields
    from ..transformers.schema import ThreatEvent

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Event(Base):
    """One normalized threat observation (mirrors :class:`ThreatEvent`)."""

    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[Optional[str]] = mapped_column(String, default=None)
    description: Mapped[Optional[str]] = mapped_column(String, default=None)
    published: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    modified: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)

    cvss_score: Mapped[Optional[float]] = mapped_column(Float, default=None)
    cvss_severity: Mapped[Optional[str]] = mapped_column(String, index=True, default=None)
    cvss_attack_vector: Mapped[Optional[str]] = mapped_column(String, default=None)
    cvss_attack_complexity: Mapped[Optional[str]] = mapped_column(String, default=None)
    cvss_privileges_required: Mapped[Optional[str]] = mapped_column(String, default=None)
    cvss_user_interaction: Mapped[Optional[str]] = mapped_column(String, default=None)
    cvss_scope: Mapped[Optional[str]] = mapped_column(String, default=None)

    ioc_count: Mapped[int] = mapped_column(Integer, default=0)
    actor: Mapped[Optional[str]] = mapped_column(String, index=True, default=None)

    # Descriptive multi-valued metadata — portable JSON, not joined on.
    platforms: Mapped[list] = mapped_column(JSON, default=list)
    vendors: Mapped[list] = mapped_column(JSON, default=list)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    tactics: Mapped[list] = mapped_column(JSON, default=list)

    # ATT&CK labels: first-class, indexed association.
    techniques: Mapped[List["EventTechnique"]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @property
    def technique_ids(self) -> List[str]:
        return [link.technique_id for link in self.techniques]


class EventTechnique(Base):
    """Association row linking an event to one ATT&CK technique id."""

    __tablename__ = "event_techniques"

    event_id: Mapped[str] = mapped_column(
        ForeignKey("events.event_id", ondelete="CASCADE"), primary_key=True
    )
    technique_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)

    event: Mapped["Event"] = relationship(back_populates="techniques")


class EventDatabase:
    """Connection + repository for persisting and querying :class:`Event` rows.

    Args:
        url: any SQLAlchemy URL (default a local SQLite file). Use a
            ``postgresql+psycopg://`` URL in production.
        echo: log emitted SQL.
        create: create tables on init if missing.
    """

    def __init__(self, url: str = "sqlite:///tactclass.db", *, echo: bool = False, create: bool = True) -> None:
        self.engine = create_engine(url, echo=echo)
        self._session_factory = sessionmaker(self.engine)
        if create:
            self.create_all()

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._session_factory()

    # -- writing ----------------------------------------------------------- #
    def upsert_events(self, events: "Iterable[ThreatEvent]") -> int:
        """Insert or update events by ``event_id`` in a single transaction.

        Returns the number processed. Uses a get-then-write upsert (portable
        across dialects); fine for incremental batches. For very large bulk
        loads, prefer a dialect-specific ``INSERT ... ON CONFLICT``.
        """
        processed = 0
        with self._session_factory.begin() as session:
            for event in events:
                self._upsert(session, event)
                processed += 1
        logger.info("Upserted %d events", processed)
        return processed

    @staticmethod
    def _upsert(session: Session, event: "ThreatEvent") -> None:
        row = session.get(Event, event.event_id)
        if row is None:
            row = Event(event_id=event.event_id)
            session.add(row)
        _apply(row, event)


    # -- reading ----------------------------------------------------------- #
    # Reads eager-load `techniques` so the returned rows stay usable after the
    # session closes (a detached instance can't fire a lazy load).
    def get_event(self, event_id: str) -> Optional[Event]:
        with self._session_factory() as session:
            return session.get(Event, event_id, options=[selectinload(Event.techniques)])

    def events_for_technique(self, technique_id: str) -> List[Event]:
        """All events mapped to a given ATT&CK technique."""
        stmt = (
            select(Event)
            .join(Event.techniques)
            .where(EventTechnique.technique_id == technique_id)
            .options(selectinload(Event.techniques))
        )
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def count(self) -> int:
        with self._session_factory() as session:
            return session.scalar(select(func.count()).select_from(Event)) or 0


def _apply(row: Event, event: "ThreatEvent") -> None:
    """Copy a :class:`ThreatEvent` onto an :class:`Event` row (scalars, JSON,
    and the technique association — reassigning the latter lets delete-orphan
    drop links removed since the last ingest)."""
    cvss = event.cvss
    row.source = event.source.value
    row.title = event.title
    row.description = event.description
    row.published = event.published
    row.modified = event.modified
    row.cvss_score = cvss.base_score if cvss else None
    row.cvss_severity = cvss.severity.value if cvss else None
    row.cvss_attack_vector = cvss.attack_vector if cvss else None
    row.cvss_attack_complexity = cvss.attack_complexity if cvss else None
    row.cvss_privileges_required = cvss.privileges_required if cvss else None
    row.cvss_user_interaction = cvss.user_interaction if cvss else None
    row.cvss_scope = cvss.scope if cvss else None
    row.ioc_count = event.ioc_count
    row.actor = event.actor
    row.platforms = list(event.platforms)
    row.vendors = list(event.vendors)
    row.tags = list(event.tags)
    row.tactics = list(event.tactics)
    row.techniques = [EventTechnique(technique_id=tid) for tid in event.technique_ids]
