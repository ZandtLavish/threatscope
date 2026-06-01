"""Encoding: enriched :class:`ThreatEvent` -> model-ready feature arrays.

This is the last transform stage. It turns each event into the three arrays the
dual-input Keras model consumes:

* ``description_embed`` — a fixed-width text embedding (the model's text branch),
* ``structured``        — numeric + multi-hot categorical features (struct branch),
* ``labels``            — the multi-label ATT&CK technique target.

Unlike the upstream stages, the encoder is *stateful*: categorical features need
vocabularies, so it follows the familiar ``fit`` / ``transform`` lifecycle
(:meth:`fit` learns vocabularies from a sample of events; :meth:`transform`
emits arrays). Stable vocabularies — especially the technique label space —
should be supplied explicitly (e.g. from the MITRE catalog) so the feature
layout doesn't drift between training runs.

Notes on the feature set:

* ``tactics`` and ``platforms`` are **deliberately excluded** from the structured
  features. The joiner derives them from ``technique_ids`` (the labels), so
  encoding them would leak the target — a model could read the answer off the
  structured branch, score well offline, then collapse to ~0 at inference where
  those fields are absent. The structured branch therefore carries only signal
  available independently of the labels: CVSS, ``ioc_count``,
  ``days_since_pub``, and ``vendors`` (multi-hot — a CVE may list several).
* the CVSS vector metrics are mapped to the **official CVSS v3 numeric weights**,
  so they carry real signal instead of arbitrary label codes.

The text embedder is injected; the default :class:`HashingTextEmbedder` is a
deterministic, dependency-light stand-in so the pipeline runs end-to-end. Swap
in a semantic encoder (sentence-transformers, a trained Keras text model, ...)
for production via the ``text_embedder`` argument.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from .base import BaseTransformer
from .schema import ThreatEvent

logger = logging.getLogger(__name__)

# Official CVSS v3 metric weights (scope-unchanged canonical values). Keys are
# the normalized words the schema stores; unknown/missing metrics encode to 0.0.
_ATTACK_VECTOR_WEIGHT = {"NETWORK": 0.85, "ADJACENT_NETWORK": 0.62, "ADJACENT": 0.62, "LOCAL": 0.55, "PHYSICAL": 0.2}
_ATTACK_COMPLEXITY_WEIGHT = {"LOW": 0.77, "HIGH": 0.44}
_PRIVILEGES_REQUIRED_WEIGHT = {"NONE": 0.85, "LOW": 0.62, "HIGH": 0.27}
_USER_INTERACTION_WEIGHT = {"NONE": 0.85, "REQUIRED": 0.62}

# Fixed leading block of the structured vector (categorical blocks follow).
NUMERIC_FEATURE_NAMES = (
    "cvss_score",
    "has_cvss",
    "cvss_av",
    "cvss_ac",
    "cvss_pr",
    "cvss_ui",
    "ioc_count_log",
    "days_since_pub_log",
    "has_published",
)

_TOKEN_RE = re.compile(r"\w+")


def _weight(table: Dict[str, float], value: Optional[str]) -> float:
    return table.get((value or "").upper(), 0.0)


# --------------------------------------------------------------------------- #
# Reusable building blocks
# --------------------------------------------------------------------------- #
class CategoricalVocabulary:
    """Maps category values to indices and produces multi-hot vectors.

    Either constructed from an explicit ordered value list (already "fitted") or
    learned from data via :meth:`fit`, which keeps the most frequent values
    subject to ``min_freq`` / ``max_size`` — the latter taming high-cardinality
    fields like vendor. With ``use_oov`` a trailing bucket absorbs unseen values;
    otherwise they are ignored.
    """

    def __init__(
        self,
        values: Optional[Sequence[str]] = None,
        *,
        max_size: Optional[int] = None,
        min_freq: int = 1,
        use_oov: bool = False,
    ) -> None:
        self.max_size = max_size
        self.min_freq = min_freq
        self.use_oov = use_oov
        self._index: Dict[str, int] = {}
        self.fitted = False
        if values is not None:
            self._set_vocab(values)

    def _set_vocab(self, ordered_values: Iterable[str]) -> None:
        self._index = {value: i for i, value in enumerate(dict.fromkeys(ordered_values))}
        self.fitted = True

    def fit(self, values: Iterable[str]) -> "CategoricalVocabulary":
        counts = Counter(value for value in values if value)
        kept = [(value, count) for value, count in counts.items() if count >= self.min_freq]
        kept.sort(key=lambda item: (-item[1], item[0]))  # frequency desc, then name
        if self.max_size is not None:
            kept = kept[: self.max_size]
        self._set_vocab([value for value, _ in kept])
        return self

    @property
    def size(self) -> int:
        return len(self._index) + (1 if self.use_oov else 0)

    @property
    def values(self) -> List[str]:
        return list(self._index)

    def index(self, value: str) -> Optional[int]:
        position = self._index.get(value)
        if position is None and self.use_oov:
            return len(self._index)  # shared trailing OOV bucket
        return position

    def multi_hot(self, values: Iterable[str]) -> np.ndarray:
        vector = np.zeros(self.size, dtype=np.float32)
        for value in values:
            position = self.index(value)
            if position is not None:
                vector[position] = 1.0
        return vector

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the learned vocabulary and its config (order is preserved)."""
        return {
            "values": list(self._index),
            "max_size": self.max_size,
            "min_freq": self.min_freq,
            "use_oov": self.use_oov,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoricalVocabulary":
        return cls(
            values=data["values"],
            max_size=data.get("max_size"),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", False),
        )


class HashingTextEmbedder:
    """Deterministic feature-hashing text embedder (the default stand-in).

    Uses the signed hashing trick over word tokens into a fixed ``dim`` and
    L2-normalizes. It captures lexical overlap but no semantics — replace it with
    a real sentence encoder for production. Hashing uses :mod:`hashlib` (not the
    salted built-in ``hash``) so vectors are stable across processes.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN_RE.findall((text or "").lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0.0 else vector


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class FeatureEncoder(BaseTransformer[ThreatEvent, Dict[str, Any]]):
    """Encodes events into ``{description_embed, structured, labels}`` arrays.

    Args:
        text_embedder: callable ``str -> array[embedding_dim]``; defaults to
            :class:`HashingTextEmbedder`.
        embedding_dim: width of the text embedding.
        technique_vocab: the multi-label target space (strongly recommended —
            pass the MITRE technique catalog so the label set is stable). If
            omitted it is learned from the events seen during :meth:`fit`.
        max_vendors: cap on the (high-cardinality) vendor vocabulary; the rest
            fold into a shared OOV bucket.
        reference_time: "now" for the ``days_since_pub`` feature; pin it for
            reproducible training snapshots.
    """

    def __init__(
        self,
        *,
        text_embedder: Optional[Callable[[str], np.ndarray]] = None,
        embedding_dim: int = 128,
        technique_vocab: Optional[Sequence[str]] = None,
        max_vendors: int = 512,
        reference_time: Optional[datetime] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.text_embedder = text_embedder or HashingTextEmbedder(embedding_dim)
        self.reference_time = reference_time or datetime.now(timezone.utc)

        self._technique_vocab = CategoricalVocabulary(technique_vocab)
        self._vendor_vocab = CategoricalVocabulary(max_size=max_vendors, use_oov=True)

    # -- lifecycle --------------------------------------------------------- #
    def fit(self, events: Iterable[ThreatEvent]) -> "FeatureEncoder":
        """Learn any vocabularies not supplied up front. Pass a *materialized*
        sequence (it is consumed here and again in :meth:`transform`)."""
        vendors: List[str] = []
        techniques: List[str] = []
        for event in events:
            if not self._vendor_vocab.fitted:
                vendors.extend(event.vendors)
            if not self._technique_vocab.fitted:
                techniques.extend(event.technique_ids)

        for vocab, observed in (
            (self._vendor_vocab, vendors),
            (self._technique_vocab, techniques),
        ):
            if not vocab.fitted:
                vocab.fit(observed)

        logger.info(
            "FeatureEncoder fitted: structured_dim=%d techniques=%d (vendors=%d)",
            self.structured_dim, self.num_techniques, self._vendor_vocab.size,
        )
        return self

    def fit_transform(self, events: Sequence[ThreatEvent]):
        return self.fit(events).transform(events)

    @property
    def _fitted(self) -> bool:
        return all(v.fitted for v in (self._technique_vocab, self._vendor_vocab))

    # -- persistence ------------------------------------------------------- #
    # The fitted encoder IS the feature contract between training and serving:
    # inference must reproduce the exact vocabularies and embedding. Persist it
    # alongside the model so predictions are faithful.
    def save(self, path: Union[str, Path]) -> None:
        """Serialize the fitted encoder to JSON.

        Only the default :class:`HashingTextEmbedder` (stateless, reproducible
        from its dim) is serializable; a custom ``text_embedder`` must be
        re-supplied to :meth:`load`.
        """
        if not isinstance(self.text_embedder, HashingTextEmbedder):
            raise ValueError(
                "save() only persists the default HashingTextEmbedder; a custom "
                "text_embedder must be re-supplied at load() time"
            )
        config = {
            "embedding_dim": self.embedding_dim,
            "reference_time": self.reference_time.isoformat(),
            "text_embedder": {"type": "hashing", "dim": self.embedding_dim},
            "technique_vocab": self._technique_vocab.to_dict(),
            "vendor_vocab": self._vendor_vocab.to_dict(),
        }
        Path(path).write_text(json.dumps(config, indent=2))
        logger.info("Saved encoder -> %s", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        text_embedder: Optional[Callable[[str], np.ndarray]] = None,
        reference_time: Optional[datetime] = None,
    ) -> "FeatureEncoder":
        """Reconstruct a fitted encoder from :meth:`save` output.

        Pass ``text_embedder`` to override the default hashing embedder, or
        ``reference_time`` to recompute ``days_since_pub`` against a different
        "now" (otherwise the persisted training-time reference is used).
        """
        config = json.loads(Path(path).read_text())
        encoder = cls(
            text_embedder=text_embedder,
            embedding_dim=config["embedding_dim"],
            technique_vocab=config["technique_vocab"]["values"],
            reference_time=reference_time or datetime.fromisoformat(config["reference_time"]),
        )
        # Restore the vendor vocab in full (it carries max_size/use_oov + OOV bucket).
        encoder._vendor_vocab = CategoricalVocabulary.from_dict(config["vendor_vocab"])
        return encoder

    # -- dimensions -------------------------------------------------------- #
    @property
    def structured_dim(self) -> int:
        return len(NUMERIC_FEATURE_NAMES) + self._vendor_vocab.size

    @property
    def num_techniques(self) -> int:
        return self._technique_vocab.size

    def feature_spec(self) -> Dict[str, int]:
        """Input/output widths, for sizing the Keras model."""
        return {
            "description_embed": self.embedding_dim,
            "structured": self.structured_dim,
            "num_techniques": self.num_techniques,
        }

    def structured_feature_names(self) -> List[str]:
        """Human-readable name for every structured dimension (for debugging)."""
        names = list(NUMERIC_FEATURE_NAMES)
        names += [f"vendor={v}" for v in self._vendor_vocab.values]
        if self._vendor_vocab.use_oov:
            names.append("vendor=<OOV>")
        return names

    # -- encoding ---------------------------------------------------------- #
    def transform(self, events):
        if not self._fitted:
            raise RuntimeError("FeatureEncoder must be fit() before transform()")
        for event in events:
            yield self._encode(event)

    def _numeric_block(self, event: ThreatEvent) -> List[float]:
        cvss = event.cvss
        if cvss is not None:
            block = [
                cvss.base_score / 10.0,
                1.0,
                _weight(_ATTACK_VECTOR_WEIGHT, cvss.attack_vector),
                _weight(_ATTACK_COMPLEXITY_WEIGHT, cvss.attack_complexity),
                _weight(_PRIVILEGES_REQUIRED_WEIGHT, cvss.privileges_required),
                _weight(_USER_INTERACTION_WEIGHT, cvss.user_interaction),
            ]
        else:
            block = [0.0] * 6
        age = event.age_days(self.reference_time)
        block.append(math.log1p(max(event.ioc_count, 0)))
        block.append(math.log1p(age) if age and age > 0 else 0.0)
        block.append(1.0 if event.published is not None else 0.0)
        return block

    def _encode(self, event: ThreatEvent) -> Dict[str, Any]:
        structured = np.concatenate([
            np.array(self._numeric_block(event), dtype=np.float32),
            self._vendor_vocab.multi_hot(event.vendors),
        ])
        embedding = np.asarray(self.text_embedder(event.description), dtype=np.float32)
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"text_embedder returned shape {embedding.shape}, expected ({self.embedding_dim},)"
            )
        return {
            "event_id": event.event_id,
            "description_embed": embedding,
            "structured": structured,
            "labels": self._technique_vocab.multi_hot(event.technique_ids),
        }
