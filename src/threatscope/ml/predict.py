"""Inference: raw event -> predicted ATT&CK techniques.

The project's headline capability — hand it a CVE description or incident report
and get back the likely ATT&CK techniques. Faithful inference needs two
artifacts the training run produced:

* the trained model (``model.keras`` + ``metadata.json`` for the label vocab), and
* the **fitted encoder** (``encoder.json`` the pipeline saved next to the feature
  store) — so raw input is vectorized identically to training.

Three input modes:

* ``--description "<text>"`` (+ optional ``--cvss-score`` / ``--platforms`` / ...)
* ``--json '<obj-or-array>'`` or a path to a JSON file of event dict(s)
* ``--event-id <id>`` to re-score an event already encoded in the feature store
  (uses the stored vectors directly; no encoder needed)

Usage::

    python -m src.ml.predict --model-dir artifacts/model --store-root data/feature_store \\
        --description "Apache Log4j2 JNDI lookup enables remote code execution"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .dataset import INPUT_DESCRIPTION_EMBED, INPUT_STRUCTURED, load_features
from .evaluate import load_artifacts
from .model import DECISION_THRESHOLD

if TYPE_CHECKING:
    from ..etl.transformers.schema import ThreatEvent

logger = logging.getLogger(__name__)


def _event_from_dict(data: Dict[str, Any]) -> "ThreatEvent":
    """Build a :class:`ThreatEvent` from a loosely-typed input dict."""
    from ..etl.transformers.schema import CVSS, Severity, SourceType, ThreatEvent

    cvss = None
    if data.get("cvss_score") is not None:
        score = float(data["cvss_score"])
        cvss = CVSS(
            version=str(data.get("cvss_version", "3.1")),
            base_score=score,
            severity=Severity.from_score(score),
            vector_string=data.get("cvss_vector", ""),
            attack_vector=data.get("attack_vector"),
            attack_complexity=data.get("attack_complexity"),
            privileges_required=data.get("privileges_required"),
            user_interaction=data.get("user_interaction"),
            scope=data.get("scope"),
        )
    return ThreatEvent(
        event_id=str(data.get("event_id", "query")),
        source=SourceType(data["source"]) if data.get("source") else SourceType.OTX,
        title=data.get("title", ""),
        description=data.get("description", ""),
        cvss=cvss,
        platforms=tuple(data.get("platforms") or ()),
        vendors=tuple(data.get("vendors") or ()),
        ioc_count=int(data.get("ioc_count", 0) or 0),
        tags=tuple(data.get("tags") or ()),
        technique_ids=tuple(data.get("technique_ids") or ()),
        tactics=tuple(data.get("tactics") or ()),
        actor=data.get("actor"),
    )


def _rank(prob_row: np.ndarray, vocab: List[str], threshold: float, top_k: int) -> List[Dict[str, Any]]:
    """Top-``k`` techniques by probability, flagging those above ``threshold``."""
    order = np.argsort(prob_row)[::-1][:top_k]
    return [
        {
            "technique_id": vocab[j] if vocab and j < len(vocab) else str(j),
            "probability": round(float(prob_row[j]), 4),
            "predicted": bool(prob_row[j] >= threshold),
        }
        for j in order
    ]


def _resolve_vocab(metadata: Dict[str, Any], num_classes: int) -> List[str]:
    vocab = metadata.get("technique_vocab")
    return vocab if vocab else [str(i) for i in range(num_classes)]


def predict_events(
    model_dir: str,
    events: "List[ThreatEvent]",
    *,
    encoder_path: str,
    threshold: float = DECISION_THRESHOLD,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Encode raw events with the persisted encoder and predict techniques."""
    from ..etl.transformers.encoder import FeatureEncoder

    model, metadata = load_artifacts(model_dir)
    encoder = FeatureEncoder.load(encoder_path)

    samples = list(encoder.transform(events))
    inputs = {
        INPUT_DESCRIPTION_EMBED: np.stack([s[INPUT_DESCRIPTION_EMBED] for s in samples]),
        INPUT_STRUCTURED: np.stack([s[INPUT_STRUCTURED] for s in samples]),
    }
    probs = model.predict(inputs, verbose=0)
    vocab = _resolve_vocab(metadata, probs.shape[1])
    return [
        {"event_id": event.event_id, "title": event.title,
         "predictions": _rank(probs[i], vocab, threshold, top_k)}
        for i, event in enumerate(events)
    ]


def predict_stored(
    model_dir: str,
    store: Any,
    event_id: str,
    *,
    name: str = "features",
    threshold: float = DECISION_THRESHOLD,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Re-score an event already encoded in the feature store (no encoder needed)."""
    model, metadata = load_artifacts(model_dir)
    matrices = load_features(store, name)
    try:
        idx = matrices.event_ids.index(event_id)
    except ValueError:
        raise SystemExit(f"event_id {event_id!r} not found in dataset {name!r}")

    inputs = {
        INPUT_DESCRIPTION_EMBED: matrices.description_embed[idx:idx + 1],
        INPUT_STRUCTURED: matrices.structured[idx:idx + 1],
    }
    probs = model.predict(inputs, verbose=0)[0]
    # Use the model's reduced vocab from metadata (the store's technique_vocab is
    # the full catalog and would mismap to the model's narrower output).
    vocab = _resolve_vocab(metadata, probs.shape[0])
    return {"event_id": event_id, "predictions": _rank(probs, vocab, threshold, top_k)}


def _print_predictions(results: List[Dict[str, Any]]) -> None:
    for result in results:
        header = result["event_id"]
        if result.get("title"):
            header += f"  —  {result['title']}"
        print(f"\n# {header}")
        for pred in result["predictions"]:
            mark = "*" if pred["predicted"] else " "
            print(f"  {mark} {pred['technique_id']:<12} {pred['probability']:.3f}")
    print("\n(* = at or above the decision threshold)")


def main(argv=None) -> None:
    from ..config import load_config

    # Pre-parse --config so the file's values can seed the other defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    config_path = pre.parse_known_args(argv)[0].config
    cfg = load_config(config_path)
    pc, fs, tr = cfg["predict"], cfg["feature_store"], cfg["train"]
    default_threshold = pc["threshold"] if pc["threshold"] is not None else cfg["model"]["decision_threshold"]

    parser = argparse.ArgumentParser(description="Predict ATT&CK techniques for an event")
    parser.add_argument("--config", default=config_path, help="path to a config.yaml (default: src/config.yaml)")
    parser.add_argument("--model-dir", default=tr["model_dir"], help="dir with model.keras + metadata.json")
    parser.add_argument("--store-root", default=fs["root"], help="feature store root (locates encoder.json)")
    parser.add_argument("--encoder", help="path to encoder.json (default: <store-root>/encoder.json)")
    parser.add_argument("--name", default=fs["name"], help="encoded dataset name (for --event-id)")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--description", help="raw event/CVE description text")
    source.add_argument("--json", help="JSON event object/array, or path to a JSON file")
    source.add_argument("--event-id", help="re-score an event already in the feature store")

    # Optional structured fields for the --description mode. (Label-derived
    # tactics/platforms are intentionally not encoded, so they aren't accepted.)
    parser.add_argument("--cvss-score", type=float)
    parser.add_argument("--vendors", nargs="+", default=None)
    parser.add_argument("--ioc-count", type=int, default=0)
    parser.add_argument("--source", default="otx", choices=("nvd", "otx", "mitre"))

    parser.add_argument("--threshold", type=float, default=default_threshold)
    parser.add_argument("--top-k", type=int, default=pc["top_k"])
    parser.add_argument("--output", help="optional path to write predictions as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.event_id:
        if not args.store_root:
            parser.error("--event-id requires --store-root")
        from ..etl.loaders.feature_store import FeatureStore
        store = FeatureStore(args.store_root)
        results = [predict_stored(args.model_dir, store, args.event_id,
                                  name=args.name, threshold=args.threshold, top_k=args.top_k)]
    else:
        # Raw text or JSON -> needs the persisted encoder.
        encoder_path = args.encoder or (
            str(Path(args.store_root) / "encoder.json") if args.store_root
            else str(Path(args.model_dir) / "encoder.json")
        )
        if not Path(encoder_path).exists():
            parser.error(f"encoder not found at {encoder_path}; pass --encoder or --store-root")

        if args.json:
            payload = Path(args.json).read_text() if Path(args.json).exists() else args.json
            parsed = json.loads(payload)
            dicts = parsed if isinstance(parsed, list) else [parsed]
        else:
            dicts = [{
                "description": args.description,
                "source": args.source,
                "cvss_score": args.cvss_score,
                "vendors": args.vendors,
                "ioc_count": args.ioc_count,
            }]
        events = [_event_from_dict(d) for d in dicts]
        results = predict_events(args.model_dir, events, encoder_path=encoder_path,
                                 threshold=args.threshold, top_k=args.top_k)

    _print_predictions(results)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info("Wrote predictions -> %s", args.output)
