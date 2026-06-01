"""Evaluation: score a trained model and break results down per technique.

Loads the model and metadata that :mod:`ml.train` persisted, reconstructs the
held-out split (same seed/fractions, so it matches what training set aside),
and reports:

* aggregate multi-label metrics — micro and macro precision/recall/F1, plus the
  compiled Keras metrics (loss, AUC, F1); and
* a **per-technique** breakdown (precision/recall/F1/support), mapped back to
  ATT&CK IDs via the stored ``technique_vocab`` — this is what tells an analyst
  which techniques the model actually predicts well versus which are starved of
  training examples.

Per-class metrics are computed in NumPy (no scikit-learn dependency). Usage::

    python -m src.ml.evaluate --model-dir artifacts/model --store-root data/feature_store
    python -m src.ml.evaluate --demo        # train a quick model, then evaluate it
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .dataset import (
    INPUT_DESCRIPTION_EMBED,
    INPUT_STRUCTURED,
    load_features,
    to_tf_dataset,
)
from .dataset import split as split_features
from .model import DECISION_THRESHOLD

if TYPE_CHECKING:
    from ..etl.loaders.feature_store import FeatureStore

logger = logging.getLogger(__name__)


def load_artifacts(model_dir: str):
    """Load the persisted Keras model and its training metadata.

    ``compile=False`` because training may use a custom weighted-BCE loss that
    isn't needed for inference and would otherwise require deserialization.
    """
    import keras

    path = Path(model_dir)
    model = keras.models.load_model(path / "model.keras", compile=False)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return model, metadata


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Micro ROC-AUC over flattened labels (rank/Mann-Whitney; ties ignored)."""
    y_true = y_true.ravel().astype(bool)
    y_score = y_score.ravel()
    n_pos = int(y_true.sum())
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=float)
    ranks[order] = np.arange(1, y_score.size + 1)
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise divide, yielding 0 where the denominator is 0."""
    return np.divide(
        numerator, denominator,
        out=np.zeros(np.broadcast(numerator, denominator).shape, dtype=float),
        where=denominator != 0,
    )


def multilabel_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    technique_vocab: Optional[List[str]] = None,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> Dict[str, Any]:
    """Compute per-technique and aggregate multi-label metrics.

    ``y_true``/``y_prob`` are ``(n_samples, n_techniques)``. Returns micro/macro
    aggregates and a per-technique list sorted by support (descending).
    """
    y_true_bool = y_true.astype(bool)
    y_pred = y_prob >= threshold

    tp = (y_true_bool & y_pred).sum(axis=0).astype(float)
    fp = (~y_true_bool & y_pred).sum(axis=0).astype(float)
    fn = (y_true_bool & ~y_pred).sum(axis=0).astype(float)
    support = y_true_bool.sum(axis=0).astype(int)
    predicted = y_pred.sum(axis=0).astype(int)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # Micro: pool counts across all techniques (sample-frequency weighted).
    tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
    micro_p = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) else 0.0
    micro_r = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) else 0.0
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # Macro: unweighted mean over the techniques actually present in y_true, so
    # the score isn't dominated by never-observed classes.
    present = support > 0
    macro = {
        "precision": float(precision[present].mean()) if present.any() else 0.0,
        "recall": float(recall[present].mean()) if present.any() else 0.0,
        "f1": float(f1[present].mean()) if present.any() else 0.0,
    }

    labels = technique_vocab or [str(i) for i in range(y_true.shape[1])]
    per_technique = [
        {
            "technique_id": labels[i],
            "support": int(support[i]),
            "predicted": int(predicted[i]),
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
        }
        for i in range(y_true.shape[1])
    ]
    per_technique.sort(key=lambda row: row["support"], reverse=True)

    return {
        "threshold": threshold,
        "num_samples": int(y_true.shape[0]),
        "num_techniques": int(y_true.shape[1]),
        "techniques_with_support": int(present.sum()),
        "micro_auc": _roc_auc(y_true, y_prob),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": macro,
        "per_technique": per_technique,
    }


def evaluate_model(
    model_dir: str,
    store: "FeatureStore",
    *,
    name: str = "features",
    which_split: str = "test",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    threshold: Optional[float] = None,
    batch_size: int = 256,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a saved model on one split of the feature store."""
    model, metadata = load_artifacts(model_dir)

    matrices = load_features(store, name)
    splits = split_features(matrices, val_frac=val_frac, test_frac=test_frac, seed=seed)
    subset = getattr(splits, which_split)
    if subset.num_samples == 0:
        raise ValueError(f"split {which_split!r} is empty; nothing to evaluate")

    # The model outputs the reduced label space; align the stored full-width
    # labels and the vocab to the kept columns persisted at training time.
    kept = metadata.get("kept_columns")
    y_true = subset.labels[:, kept] if kept is not None else subset.labels
    vocab = metadata.get("technique_vocab") or matrices.technique_vocab
    threshold = threshold if threshold is not None else DECISION_THRESHOLD

    y_prob = model.predict(
        {INPUT_DESCRIPTION_EMBED: subset.description_embed, INPUT_STRUCTURED: subset.structured},
        batch_size=batch_size, verbose=0,
    )
    report = multilabel_report(y_true, y_prob, vocab, threshold=threshold)
    report["split"] = which_split

    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
        logger.info("Wrote evaluation report -> %s", report_path)
    return report


def print_report(report: Dict[str, Any], *, top: int = 20) -> None:
    """Human-readable summary: aggregates, then the top techniques by support."""
    print(f"\n== Evaluation ({report.get('split', '?')} split) ==")
    print(f"samples={report['num_samples']}  techniques={report['num_techniques']} "
          f"(with support: {report['techniques_with_support']})  threshold={report['threshold']}")
    micro, macro = report["micro"], report["macro"]
    print(f"micro-AUC={report.get('micro_auc', float('nan')):.3f}")
    print(f"micro  P={micro['precision']:.3f} R={micro['recall']:.3f} F1={micro['f1']:.3f}")
    print(f"macro  P={macro['precision']:.3f} R={macro['recall']:.3f} F1={macro['f1']:.3f}")

    rows = [r for r in report["per_technique"] if r["support"] > 0][:top]
    if rows:
        print(f"\n{'technique':<14}{'support':>8}{'pred':>6}{'P':>8}{'R':>8}{'F1':>8}")
        for r in rows:
            print(f"{r['technique_id']:<14}{r['support']:>8}{r['predicted']:>6}"
                  f"{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}")


def main(argv=None) -> None:
    from ..config import load_config

    # Pre-parse --config so the file's values can seed the other defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    config_path = pre.parse_known_args(argv)[0].config
    cfg = load_config(config_path)
    ev, tr, fs = cfg["evaluate"], cfg["train"], cfg["feature_store"]

    parser = argparse.ArgumentParser(description="Evaluate a trained TactClass model")
    parser.add_argument("--config", default=config_path, help="path to a config.yaml (default: src/config.yaml)")
    parser.add_argument("--model-dir", default=tr["model_dir"], help="directory containing model.keras + metadata.json")
    parser.add_argument("--store-root", default=fs["root"], help="root of the Parquet feature store")
    parser.add_argument("--name", default=fs["name"])
    parser.add_argument("--split", default=ev["split"], choices=("train", "val", "test"))
    # val/test/seed default to the train: section so the split matches training.
    parser.add_argument("--val-frac", type=float, default=tr["val_frac"])
    parser.add_argument("--test-frac", type=float, default=tr["test_frac"])
    parser.add_argument("--seed", type=int, default=tr["seed"])
    parser.add_argument("--threshold", type=float, default=ev["threshold"])
    parser.add_argument("--batch-size", type=int, default=ev["batch_size"])
    parser.add_argument("--report", help="optional path to write the JSON report")
    parser.add_argument("--top", type=int, default=ev["top"], help="techniques to show in the table")
    parser.add_argument("--demo", action="store_true", help="train a quick model, then evaluate it")
    parser.add_argument("--demo-size", type=int, default=120)
    parser.add_argument("--demo-epochs", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from ..etl.loaders.feature_store import FeatureStore

    if args.demo:
        import tempfile

        from .train import _build_demo_store, train

        tmp = tempfile.mkdtemp(prefix="tactclass-eval-")
        store = _build_demo_store(f"{tmp}/store", args.demo_size)
        model_dir = f"{tmp}/model"   # isolate the throwaway demo model
        train(store, model_dir=model_dir, epochs=args.demo_epochs,
              batch_size=16, seed=args.seed, verbose=0)
    elif args.model_dir and args.store_root:
        store = FeatureStore(args.store_root)
        model_dir = args.model_dir
    else:
        parser.error("provide --model-dir and --store-root, or use --demo")

    report = evaluate_model(
        model_dir, store, name=args.name, which_split=args.split,
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
        threshold=args.threshold, batch_size=args.batch_size, report_path=args.report,
    )
    print_report(report, top=args.top)
