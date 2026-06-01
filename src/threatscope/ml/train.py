"""Training entry point: feature store -> trained, persisted model.

Ties the ML layer together: :func:`ml.dataset.make_datasets` builds the
``tf.data`` splits, :func:`ml.model.build_model` sizes the network from the
dataset ``spec``, and this module runs ``fit`` with sensible callbacks, scores
the held-out test split, and saves the model alongside the metadata a serving
step needs (the ``spec`` and the ``technique_vocab`` that maps output indices
back to ATT&CK IDs).

Usage::

    python -m ml.train --store-root data/feature_store --model-dir artifacts/model
    python -m ml.train --demo --epochs 3        # synthetic self-contained smoke test

Keras is imported lazily inside :func:`train` so the module imports without TF.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .dataset import make_datasets
from .model import build_model

if TYPE_CHECKING:
    from ..etl.loaders.feature_store import FeatureStore

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "artifacts/model"


def train(
    store: "FeatureStore",
    *,
    name: str = "features",
    model_dir: str = DEFAULT_MODEL_DIR,
    epochs: int = 20,
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    learning_rate: float = 1e-3,
    patience: int = 4,
    verbose: int = 1,
    min_label_support: int = 1,
    max_pos_weight: float = 100.0,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Train on the feature store and persist the best model + metadata.

    Returns the model and the test-split metrics.
    """
    import keras

    keras.utils.set_random_seed(seed)

    data = make_datasets(
        store, name=name, batch_size=batch_size,
        val_frac=val_frac, test_frac=test_frac, seed=seed,
        min_label_support=min_label_support, max_pos_weight=max_pos_weight,
    )
    sizes = data["sizes"]
    if sizes["train"] == 0:
        raise ValueError(
            f"No training samples (sizes={sizes}). Check the feature store name/path "
            f"and that val_frac+test_frac leave room for training data."
        )
    logger.info(
        "Training on %d samples (val=%d, test=%d) for up to %d epochs",
        sizes["train"], sizes["val"], sizes["test"], epochs,
    )

    # Model hyperparameters come from config (model_params); learning_rate is
    # kept as an explicit arg/fallback for convenience.
    build_kwargs = dict(model_params or {})
    build_kwargs.setdefault("learning_rate", learning_rate)
    build_kwargs["pos_weights"] = data["pos_weights"]
    model = build_model(data["spec"], **build_kwargs)

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"

    # Monitor val_loss when there's a validation split, else fall back to the
    # training loss so the callbacks (and the run) don't stall on a missing metric.
    has_val = sizes["val"] > 0
    monitor = "val_loss" if has_val else "loss"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, monitor=monitor, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=max(1, patience // 2)),
    ]

    history = model.fit(
        data["train"],
        validation_data=data["val"] if has_val else None,
        epochs=epochs, callbacks=callbacks, verbose=verbose,
    )

    test_metrics = (
        {
            metric: float(value)
            for metric, value in model.evaluate(data["test"], return_dict=True, verbose=0).items()
        }
        if sizes["test"] > 0
        else {}
    )

    # EarlyStopping restored the best weights; persist that final state + metadata.
    model.save(model_path)
    metadata = {
        "spec": data["spec"],
        "technique_vocab": data["technique_vocab"],
        "kept_columns": data["kept_columns"],
        "min_label_support": min_label_support,
        "test_metrics": test_metrics,
        "epochs_run": len(history.history["loss"]),
        "seed": seed,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model -> %s | test metrics: %s", model_path, test_metrics)
    return model, test_metrics


def _build_demo_store(root: str, num_samples: int) -> "FeatureStore":
    """Create a small synthetic feature store for the ``--demo`` smoke test."""
    from ..etl.loaders.feature_store import FeatureStore
    from ..etl.transformers.encoder import FeatureEncoder
    from ..etl.transformers.schema import SourceType, ThreatEvent

    vocab = ["T1059", "T1190", "T1566", "T1486", "T1003"]
    events = []
    for i in range(num_samples):
        techniques = [vocab[i % len(vocab)]]
        if i % 3 == 0:  # some genuinely multi-label samples
            techniques.append(vocab[(i + 1) % len(vocab)])
        events.append(ThreatEvent(
            event_id=f"evt-{i}",
            source=SourceType.OTX,
            description=f"threat report {vocab[i % len(vocab)]} exploitation activity sample {i}",
            ioc_count=i % 20,
            tactics=("execution",) if i % 2 else ("initial-access",),
            technique_ids=tuple(techniques),
        ))

    encoder = FeatureEncoder(technique_vocab=vocab)
    encoded = list(encoder.fit_transform(events))
    store = FeatureStore(root)
    store.write_encoded(encoded, metadata={
        "feature_spec": encoder.feature_spec(), "technique_vocab": vocab,
    })
    return store


def main(argv=None) -> None:
    from ..config import load_config

    # Pre-parse --config so the file's values can seed the other defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    config_path = pre.parse_known_args(argv)[0].config
    cfg = load_config(config_path)
    tr, fs, mdl = cfg["train"], cfg["feature_store"], cfg["model"]

    parser = argparse.ArgumentParser(description="Train the TactClass multi-label model")
    parser.add_argument("--config", default=config_path, help="path to a config.yaml (default: src/config.yaml)")
    parser.add_argument("--store-root", default=fs["root"], help="root directory of the Parquet feature store")
    parser.add_argument("--name", default=fs["name"], help="encoded dataset name in the store")
    parser.add_argument("--model-dir", default=tr["model_dir"])
    parser.add_argument("--epochs", type=int, default=tr["epochs"])
    parser.add_argument("--batch-size", type=int, default=tr["batch_size"])
    parser.add_argument("--val-frac", type=float, default=tr["val_frac"])
    parser.add_argument("--test-frac", type=float, default=tr["test_frac"])
    parser.add_argument("--seed", type=int, default=tr["seed"])
    parser.add_argument("--learning-rate", type=float, default=mdl["learning_rate"])
    parser.add_argument("--min-label-support", type=int, default=tr["min_label_support"],
                        help="keep only techniques with at least this many positive examples")
    parser.add_argument("--max-pos-weight", type=float, default=tr["max_pos_weight"],
                        help="cap on per-technique positive weight in the weighted loss")
    parser.add_argument("--demo", action="store_true", help="train on a synthetic in-memory dataset")
    parser.add_argument("--demo-size", type=int, default=120)
    parser.add_argument(
        "--verbose", type=int, default=1, choices=(0, 1, 2),
        help="Keras fit verbosity: 1=per-step progress bar (default), 2=one line/epoch, 0=silent",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from ..etl.loaders.feature_store import FeatureStore

    if args.demo:
        import tempfile
        store = _build_demo_store(tempfile.mkdtemp(prefix="tactclass-demo-"), args.demo_size)
    elif args.store_root:
        store = FeatureStore(args.store_root)
    else:
        parser.error("either --store-root or --demo is required")

    # Architecture hyperparameters come from config.yaml's `model:` section.
    model_params = {
        "text_hidden": mdl["text_hidden"],
        "struct_hidden": mdl["struct_hidden"],
        "merge_hidden": mdl["merge_hidden"],
        "dropout_text": mdl["dropout_text"],
        "dropout_merge": mdl["dropout_merge"],
        "threshold": mdl["decision_threshold"],
        "learning_rate": args.learning_rate,
    }
    train(
        store, name=args.name, model_dir=args.model_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed, patience=tr["patience"], verbose=args.verbose,
        min_label_support=args.min_label_support, max_pos_weight=args.max_pos_weight,
        model_params=model_params,
    )
