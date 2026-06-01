"""Central configuration: defaults, ``config.yaml``, and resolution helpers.

End users tune the project from ``config.yaml`` (API keys, DB URL, feature-store
paths, pipeline windows, model hyperparameters, ...). The CLIs load it and use
its values as their argument defaults, so the precedence end-to-end is:

    explicit CLI flag  >  config.yaml  >  environment variable (secrets)  >  built-in default

:data:`DEFAULTS` is the source of truth for the shape and the built-in values;
``config.yaml`` only needs to override what you want to change (missing keys
fall back to the defaults via a deep merge).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:  # PyYAML is required only to *read* a config file; defaults work without it.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# Built-in defaults — the canonical structure mirrored by config.yaml.
DEFAULTS: Dict[str, Any] = {
    "api_keys": {
        "nvd": None,   # else env NVD_API_KEY
        "otx": None,   # else env OTX_API_KEY
    },
    "database": {
        "url": "sqlite:///data/tactclass.db",
    },
    "feature_store": {
        "root": "data/feature_store",
        "name": "features",
    },
    "pipeline": {
        "sources": ["nvd", "otx"],
        "nvd_days": 30,
        "nvd_results_per_page": 2000,
        "otx_days": 30,
        "otx_max_pulses": 500,
        "mitre_domain": "enterprise",
        "mitre_version": None,
        "cve_technique_map": None,
        "expand_actor_techniques": False,
        "write_db": True,
        "write_features": True,
    },
    "model": {
        "embedding_dim": 128,
        "text_embedder": "hashing",          # hashing | sentence-transformer
        "text_embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "text_hidden": 64,
        "struct_hidden": [64, 32],
        "merge_hidden": 128,
        "dropout_text": 0.3,
        "dropout_merge": 0.4,
        "learning_rate": 0.001,
        "decision_threshold": 0.5,
    },
    "train": {
        "model_dir": "artifacts/model",
        "epochs": 20,
        "batch_size": 32,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "seed": 42,
        "patience": 4,
        "min_label_support": 1,              # drop techniques with fewer positives
        "max_pos_weight": 100.0,             # cap on per-technique positive weight
    },
    "evaluate": {
        "split": "test",
        "threshold": None,   # None -> model.decision_threshold
        "batch_size": 256,
        "top": 20,
    },
    "predict": {
        "threshold": None,   # None -> model.decision_threshold
        "top_k": 10,
    },
}


def default_config_path() -> Path:
    """Where config.yaml is looked up: ``$TACTCLASS_CONFIG`` else next to this file."""
    env = os.getenv("TACTCLASS_CONFIG")
    return Path(env) if env else Path(__file__).with_name("config.yaml")


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (nested dicts merged, not replaced)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Return the merged configuration (``DEFAULTS`` overlaid with the YAML file).

    A missing file is fine — defaults are used. A present file requires PyYAML.
    """
    config = copy.deepcopy(DEFAULTS)
    config_path = Path(path) if path else default_config_path()
    if config_path.exists():
        if yaml is None:
            raise RuntimeError(
                f"Reading {config_path} requires PyYAML — install it (pip install pyyaml) "
                "or remove the file to fall back to built-in defaults."
            )
        loaded = yaml.safe_load(config_path.read_text()) or {}
        _deep_update(config, loaded)
    return config


def get_api_key(config: Dict[str, Any], source: str) -> Optional[str]:
    """Resolve an API key: config.yaml value, else ``<SOURCE>_API_KEY`` env var."""
    value = (config.get("api_keys") or {}).get(source)
    return value or os.getenv(f"{source.upper()}_API_KEY") or None
