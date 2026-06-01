"""Dataset assembly — Parquet feature store -> ``tf.data`` for the dual-input model.

This bridges the LOAD stage and the model. It loads the encoded arrays the
:class:`~etl.loaders.feature_store.FeatureStore` persisted, splits them
reproducibly, and serves them as a ``tf.data.Dataset`` yielding the exact shape
the Keras model expects::

    ({INPUT_DESCRIPTION_EMBED: <embed>, INPUT_STRUCTURED: <struct>}, <labels>)

The input/output **names defined here are the contract** shared with
``model.py`` — both sides import these constants so the functional model's
named inputs line up with the dataset's feature dict.

Two layers, deliberately separated:

* a framework-agnostic NumPy layer (:class:`FeatureMatrices`, :func:`load_features`,
  :func:`split`) that needs only NumPy and is easy to test; and
* a thin ``tf.data`` layer (:func:`to_tf_dataset`, :func:`make_datasets`) that
  imports TensorFlow lazily, so importing this module never requires TF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # avoid hard deps at import time
    import tensorflow as tf

    from threatscope.etl.loaders.feature_store import FeatureStore

logger = logging.getLogger(__name__)

# --- the dataset <-> model contract: keep these in sync with model.py ---
INPUT_DESCRIPTION_EMBED = "description_embed"
INPUT_STRUCTURED = "structured"
OUTPUT_TECHNIQUES = "techniques"


@dataclass
class FeatureMatrices:
    """An aligned block of encoded samples (the in-memory training matrices).

    All arrays share the same first dimension (``num_samples``). ``feature_spec``
    and ``technique_vocab`` carry the metadata persisted with the feature store
    so the model can be sized and predictions mapped back to ATT&CK IDs.
    """

    event_ids: List[str]
    description_embed: np.ndarray           # (n, embedding_dim)
    structured: np.ndarray                  # (n, structured_dim)
    labels: np.ndarray                      # (n, num_techniques)
    feature_spec: Optional[Dict[str, int]] = None
    technique_vocab: Optional[List[str]] = None

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])

    def resolved_spec(self) -> Dict[str, int]:
        """Input/output widths, from stored metadata or inferred from the arrays."""
        if self.feature_spec:
            return dict(self.feature_spec)
        return {
            INPUT_DESCRIPTION_EMBED: int(self.description_embed.shape[1]),
            INPUT_STRUCTURED: int(self.structured.shape[1]),
            "num_techniques": int(self.labels.shape[1]),
        }

    def subset(self, indices: Sequence[int]) -> "FeatureMatrices":
        idx = np.asarray(indices, dtype=int)
        return FeatureMatrices(
            event_ids=[self.event_ids[i] for i in idx],
            description_embed=self.description_embed[idx],
            structured=self.structured[idx],
            labels=self.labels[idx],
            feature_spec=self.feature_spec,
            technique_vocab=self.technique_vocab,
        )


@dataclass
class DataSplits:
    train: FeatureMatrices
    val: FeatureMatrices
    test: FeatureMatrices


def load_features(store: "FeatureStore", name: str = "features") -> FeatureMatrices:
    """Load an encoded dataset (and its metadata) from the feature store."""
    arrays = store.read_arrays(name)
    metadata = store.read_metadata(name)
    matrices = FeatureMatrices(
        event_ids=arrays["event_id"],
        description_embed=arrays[INPUT_DESCRIPTION_EMBED],
        structured=arrays[INPUT_STRUCTURED],
        labels=arrays["labels"],
        feature_spec=metadata.get("feature_spec"),
        technique_vocab=metadata.get("technique_vocab"),
    )
    logger.info(
        "Loaded %d samples (spec=%s)", matrices.num_samples, matrices.resolved_spec()
    )
    return matrices


def split(
    matrices: FeatureMatrices,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> DataSplits:
    """Reproducibly partition samples into train/val/test.

    A plain seeded shuffle (no stratification — meaningful stratification over a
    multi-label target is ill-defined). The seed makes splits stable across runs.
    """
    if not 0.0 <= val_frac + test_frac < 1.0:
        raise ValueError("val_frac + test_frac must be in [0, 1)")
    n = matrices.num_samples
    order = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]
    logger.info("Split %d -> train=%d val=%d test=%d", n, len(train_idx), len(val_idx), len(test_idx))
    return DataSplits(
        train=matrices.subset(train_idx),
        val=matrices.subset(val_idx),
        test=matrices.subset(test_idx),
    )


def reduce_label_space(
    matrices: FeatureMatrices, *, min_label_support: int = 1
) -> "tuple[FeatureMatrices, np.ndarray]":
    """Restrict the target to techniques with support, and drop empty-label rows.

    The full label matrix (one column per MITRE technique, ~697) is mostly empty:
    techniques never seen can't be learned, and rows with no positive label (e.g.
    unmapped CVEs) only teach the model to predict nothing. This keeps columns
    with ``>= min_label_support`` positives over the whole dataset, then drops
    rows that are all-zero after that reduction.

    Returns the reduced matrices and ``kept_columns`` — the indices (into the
    original 697-wide vocab) that define the model's output space. That array is
    persisted with the model so evaluate/predict map output index -> technique id
    identically.
    """
    labels = matrices.labels
    full_vocab = matrices.technique_vocab or [str(i) for i in range(labels.shape[1])]

    support = labels.sum(axis=0)
    kept_columns = np.nonzero(support >= min_label_support)[0]
    if kept_columns.size == 0:
        raise ValueError(
            f"No techniques have >= {min_label_support} positive example(s). "
            "The dataset has no usable labels — supply OTX pulses (attack_ids) or a "
            "--cve-technique-map so events carry technique_ids."
        )

    reduced_labels = labels[:, kept_columns]
    row_mask = reduced_labels.sum(axis=1) > 0
    kept_rows = np.nonzero(row_mask)[0]

    spec = dict(matrices.feature_spec) if matrices.feature_spec else None
    if spec is not None:
        spec["num_techniques"] = int(kept_columns.size)

    reduced = FeatureMatrices(
        event_ids=[matrices.event_ids[i] for i in kept_rows],
        description_embed=matrices.description_embed[kept_rows],
        structured=matrices.structured[kept_rows],
        labels=reduced_labels[kept_rows],
        feature_spec=spec,
        technique_vocab=[full_vocab[i] for i in kept_columns],
    )
    logger.info(
        "Reduced labels: %d -> %d techniques (support>=%d); kept %d/%d rows with >=1 label",
        labels.shape[1], kept_columns.size, min_label_support, kept_rows.size, labels.shape[0],
    )
    return reduced, kept_columns


def compute_pos_weights(labels: np.ndarray, *, max_pos_weight: float = 100.0) -> np.ndarray:
    """Per-technique positive weight ``neg/pos`` (clipped) for weighted BCE.

    Counters the residual imbalance after reduction — each row still has only a
    couple of the kept techniques positive. Computed from the training split only.
    """
    n = labels.shape[0]
    pos = labels.sum(axis=0)
    neg = n - pos
    weights = np.where(pos > 0, neg / np.maximum(pos, 1.0), max_pos_weight)
    return np.minimum(weights, max_pos_weight).astype(np.float32)


def to_tf_dataset(
    matrices: FeatureMatrices,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: Optional[int] = None,
    repeat: bool = False,
    seed: int = 42,
) -> "tf.data.Dataset":
    """Wrap matrices as a ``tf.data.Dataset`` of ``(inputs_dict, labels)`` batches.

    Shuffle only the training split; leave val/test ordered. TensorFlow is
    imported here (not at module load) so the NumPy layer stays usable without it.
    """
    import tensorflow as tf

    inputs = {
        INPUT_DESCRIPTION_EMBED: matrices.description_embed,
        INPUT_STRUCTURED: matrices.structured,
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, matrices.labels))
    if shuffle:
        dataset = dataset.shuffle(
            shuffle_buffer or max(matrices.num_samples, 1),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_datasets(
    store: "FeatureStore",
    *,
    name: str = "features",
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    min_label_support: int = 1,
    max_pos_weight: float = 100.0,
) -> Dict[str, Any]:
    """End-to-end helper for ``train.py``: load -> reduce -> split -> ``tf.data``.

    Returns ``{"train", "val", "test"}`` datasets plus ``"spec"`` (reduced model
    input/output widths), ``"technique_vocab"`` (reduced output-index -> ATT&CK
    id), ``"kept_columns"`` (indices into the full vocab — persist with the
    model), and ``"pos_weights"`` (per-technique weights from the train split).
    """
    matrices = load_features(store, name)
    matrices, kept_columns = reduce_label_space(matrices, min_label_support=min_label_support)
    splits = split(matrices, val_frac=val_frac, test_frac=test_frac, seed=seed)
    pos_weights = compute_pos_weights(splits.train.labels, max_pos_weight=max_pos_weight)
    return {
        "train": to_tf_dataset(splits.train, batch_size=batch_size, shuffle=True, seed=seed),
        "val": to_tf_dataset(splits.val, batch_size=batch_size),
        "test": to_tf_dataset(splits.test, batch_size=batch_size),
        "spec": matrices.resolved_spec(),
        "technique_vocab": matrices.technique_vocab,
        "kept_columns": kept_columns.tolist(),
        "pos_weights": pos_weights.tolist(),
        "sizes": {
            "train": splits.train.num_samples,
            "val": splits.val.num_samples,
            "test": splits.test.num_samples,
        },
    }