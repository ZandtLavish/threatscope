"""Parquet feature store — the file-based half of the LOAD stage.

Persists the transform stage's two output shapes to Parquet:

* **event tables**   — :meth:`FeatureStore.write_events` writes the normalized,
  human-readable :meth:`ThreatEvent.to_row` records (one dataset per entity
  type, optionally partitioned by source); and
* **the feature store** — :meth:`FeatureStore.write_encoded` writes the
  :class:`FeatureEncoder` sample dicts (``description_embed`` / ``structured`` /
  ``labels``) that the ``ml`` package trains on.

Each named dataset is a directory of immutable ``part-NNNNN.parquet`` files, so
successive runs append rather than overwrite. Schemas are pinned explicitly so
part files written across runs always merge cleanly. The label semantics that
make the one-hot ``labels`` interpretable (the technique vocabulary, feature
spec, ...) ride along as Parquet schema metadata, keeping the label store and
its meaning together.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

if TYPE_CHECKING:  # type-only; the store relies on duck-typed .to_row() at runtime
    from ..transformers.schema import ThreatEvent

logger = logging.getLogger(__name__)

# Pinned schema for the normalized-event table (mirrors ThreatEvent.to_row()).
# Declaring it explicitly stops Arrow's per-batch type inference from drifting
# (e.g. an all-null cvss_score batch inferring `null` instead of `double`).
EVENTS_SCHEMA = pa.schema([
    ("event_id", pa.string()),
    ("source", pa.string()),
    ("title", pa.string()),
    ("description", pa.string()),
    ("published", pa.string()),
    ("modified", pa.string()),
    ("cvss_score", pa.float64()),
    ("cvss_severity", pa.string()),
    ("cvss_attack_vector", pa.string()),
    ("cvss_attack_complexity", pa.string()),
    ("cvss_privileges_required", pa.string()),
    ("cvss_user_interaction", pa.string()),
    ("cvss_scope", pa.string()),
    ("platforms", pa.list_(pa.string())),
    ("vendors", pa.list_(pa.string())),
    ("ioc_count", pa.int64()),
    ("tags", pa.list_(pa.string())),
    ("technique_ids", pa.list_(pa.string())),
    ("tactics", pa.list_(pa.string())),
    ("actor", pa.string()),
])

_METADATA_KEY = b"tactclass_feature_store"
_ARRAY_FIELDS = ("description_embed", "structured", "labels")


def _events_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=EVENTS_SCHEMA)


def _encoded_table(samples: Sequence[Mapping[str, Any]]) -> pa.Table:
    """Build a table with ``event_id`` plus one ``list<float32>`` per array field."""
    columns = {"event_id": pa.array([str(s["event_id"]) for s in samples], pa.string())}
    for field in _ARRAY_FIELDS:
        if field in samples[0]:
            data = [np.asarray(s[field], dtype=np.float32).tolist() for s in samples]
            columns[field] = pa.array(data, type=pa.list_(pa.float32()))
    return pa.table(columns)


class FeatureStore:
    """Reads and writes Parquet datasets under a single root directory."""

    def __init__(self, root: Union[str, Path], *, compression: str = "snappy") -> None:
        self.root = Path(root)
        self.compression = compression

    def path(self, name: str) -> Path:
        """Directory backing the dataset ``name``."""
        return self.root / name

    # -- writing ----------------------------------------------------------- #
    def write_events(
        self,
        events: "Iterable[ThreatEvent]",
        *,
        name: str = "events",
        partition_by_source: bool = False,
    ) -> List[Path]:
        """Persist normalized events. With ``partition_by_source`` each source's
        rows land under a ``source=<src>/`` subdirectory (Hive-style)."""
        rows = [event.to_row() for event in events]
        if not rows:
            logger.info("write_events: nothing to write for %r", name)
            return []
        if not partition_by_source:
            return [self._write_part(_events_table(rows), self.path(name))]

        paths: List[Path] = []
        by_source: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            by_source.setdefault(row["source"], []).append(row)
        for source, group in by_source.items():
            paths.append(self._write_part(_events_table(group), self.path(name) / f"source={source}"))
        return paths

    def write_encoded(
        self,
        samples: Iterable[Mapping[str, Any]],
        *,
        name: str = "features",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        """Persist encoded feature/label arrays. ``metadata`` (e.g. the encoder's
        ``feature_spec()`` and technique vocabulary) is stored as schema metadata
        so the one-hot ``labels`` stay interpretable downstream."""
        materialized = list(samples)
        if not materialized:
            logger.info("write_encoded: nothing to write for %r", name)
            return None
        table = _encoded_table(materialized)
        if metadata:
            table = table.replace_schema_metadata(
                {_METADATA_KEY: json.dumps(dict(metadata), default=str).encode("utf-8")}
            )
        return self._write_part(table, self.path(name))

    def _write_part(self, table: pa.Table, directory: Path) -> Path:
        """Append ``table`` as the next ``part-NNNNN.parquet`` in ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        index = len(list(directory.glob("*.parquet")))
        part_path = directory / f"part-{index:05d}.parquet"
        pq.write_table(table, part_path, compression=self.compression)
        logger.info("Wrote %d rows -> %s", table.num_rows, part_path)
        return part_path

    # -- reading ----------------------------------------------------------- #
    def read_table(self, name: str) -> pa.Table:
        """Read every part file of a dataset (Hive partitions auto-discovered)."""
        return ds.dataset(self.path(name), format="parquet").to_table()

    def read_pandas(self, name: str):
        return self.read_table(name).to_pandas()

    def read_arrays(self, name: str) -> Dict[str, Any]:
        """Load an encoded dataset back as stacked NumPy arrays, ready for ML.

        Returns ``event_id`` (list) plus a 2-D ``float32`` array per present
        array field, e.g. ``structured`` -> shape ``(n_samples, structured_dim)``.
        """
        frame = self.read_pandas(name)
        arrays: Dict[str, Any] = {"event_id": frame["event_id"].tolist()}
        for field in _ARRAY_FIELDS:
            if field in frame.columns:
                arrays[field] = np.asarray(frame[field].tolist(), dtype=np.float32)
        return arrays

    def read_metadata(self, name: str) -> Dict[str, Any]:
        """Return the schema metadata stored with an encoded dataset (or ``{}``)."""
        parts = sorted(self.path(name).rglob("*.parquet"))
        if not parts:
            return {}
        schema_metadata = pq.read_schema(parts[0]).metadata or {}
        raw = schema_metadata.get(_METADATA_KEY)
        return json.loads(raw) if raw else {}
