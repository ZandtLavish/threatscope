"""The dual-input multi-label classifier (Keras).

One threat event carries two complementary signals: the free-text description
and the structured metadata (CVSS, platforms, tactics, ...). This model has a
branch for each and merges them, then predicts — as independent sigmoids — which
ATT&CK techniques apply. It is multi-label: an event can map to several techniques.

The model is built from the ``spec`` that :func:`ml.dataset.make_datasets`
returns (``{description_embed, structured, num_techniques}``) and its named
inputs/output reuse the constants from :mod:`ml.dataset`, so the dict the
``tf.data`` pipeline yields binds to the right tensors by name.

Keras is imported lazily inside :func:`build_model` so this module (and its
shared hyperparameter defaults) can be imported without TensorFlow installed.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

from .dataset import INPUT_DESCRIPTION_EMBED, INPUT_STRUCTURED, OUTPUT_TECHNIQUES

logger = logging.getLogger(__name__)

# Architecture / training defaults (override per call as needed).
TEXT_HIDDEN = 64
STRUCT_HIDDEN: Sequence[int] = (64, 32)
MERGE_HIDDEN = 128
DROPOUT_TEXT = 0.3
DROPOUT_MERGE = 0.4
LEARNING_RATE = 1e-3
DECISION_THRESHOLD = 0.5


def build_model(
    spec: Mapping[str, int],
    *,
    text_hidden: int = TEXT_HIDDEN,
    struct_hidden: Sequence[int] = STRUCT_HIDDEN,
    merge_hidden: int = MERGE_HIDDEN,
    dropout_text: float = DROPOUT_TEXT,
    dropout_merge: float = DROPOUT_MERGE,
    learning_rate: float = LEARNING_RATE,
    threshold: float = DECISION_THRESHOLD,
    pos_weights: Optional[Sequence[float]] = None,
):
    """Build and compile the dual-input model.

    Args:
        spec: input/output widths — ``{INPUT_DESCRIPTION_EMBED, INPUT_STRUCTURED,
            "num_techniques"}`` (i.e. ``FeatureMatrices.resolved_spec()``).
    Returns:
        A compiled ``keras.Model`` taking a ``{name: tensor}`` feature dict and
        emitting per-technique probabilities.
    """
    import keras
    from keras import layers, ops

    embedding_dim = int(spec[INPUT_DESCRIPTION_EMBED])
    structured_dim = int(spec[INPUT_STRUCTURED])
    num_techniques = int(spec["num_techniques"])

    # --- Text branch: pre-computed description embedding ---
    text_input = keras.Input(shape=(embedding_dim,), name=INPUT_DESCRIPTION_EMBED)
    text_x = layers.Dense(text_hidden, activation="relu")(text_input)
    text_x = layers.Dropout(dropout_text)(text_x)

    # --- Structured branch: numeric + multi-hot categorical features ---
    struct_input = keras.Input(shape=(structured_dim,), name=INPUT_STRUCTURED)
    struct_x = struct_input
    for i, units in enumerate(struct_hidden):
        struct_x = layers.Dense(units, activation="relu")(struct_x)
        if i == 0:  # normalize the heterogeneous feature scales early
            struct_x = layers.BatchNormalization()(struct_x)

    # --- Merge and classify ---
    merged = layers.Concatenate()([text_x, struct_x])
    merged = layers.Dense(merge_hidden, activation="relu")(merged)
    merged = layers.Dropout(dropout_merge)(merged)
    output = layers.Dense(num_techniques, activation="sigmoid", name=OUTPUT_TECHNIQUES)(merged)

    model = keras.Model(inputs=[text_input, struct_input], outputs=output, name="tactclass")

    # Weighted BCE counters class imbalance: each positive label is up-weighted
    # by pos_weights[j] (~neg/pos). Output stays sigmoid, so predict/evaluate keep
    # treating outputs as probabilities. Falls back to plain BCE when unweighted.
    if pos_weights is not None:
        weights = ops.convert_to_tensor(list(pos_weights), dtype="float32")

        def weighted_bce(y_true, y_pred):
            eps = 1e-7
            y_pred = ops.clip(y_pred, eps, 1.0 - eps)
            per_label = -(weights * y_true * ops.log(y_pred)
                          + (1.0 - y_true) * ops.log(1.0 - y_pred))
            return ops.mean(per_label)

        loss = weighted_bce
    else:
        loss = "binary_crossentropy"  # multi-label: independent per-technique

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
        metrics=[
            keras.metrics.AUC(name="auc", multi_label=True),
            keras.metrics.F1Score(name="f1", average="micro", threshold=threshold),
        ],
    )
    logger.info(
        "Built model: text=%d struct=%d -> %d techniques (%d params)",
        embedding_dim, structured_dim, num_techniques, model.count_params(),
    )
    return model
