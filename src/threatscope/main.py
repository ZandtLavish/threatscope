"""TactClass unified CLI — a single entry point for the whole project.

Dispatches to the four stage commands, each of which owns its own flags and
``--help`` (run ``<command> --help`` for details):

    pipeline   run the ELT pipeline: extract -> transform -> load
    train      train the dual-input model on the feature store
    evaluate   score a trained model and break results down per technique
    predict    predict ATT&CK techniques for an event (raw text / JSON / stored id)

Run from the project root::

    python -m src.main pipeline --nvd-days 30
    python -m src.main train --store-root data/feature_store --model-dir artifacts/model
    python -m src.main evaluate --model-dir artifacts/model --store-root data/feature_store
    python -m src.main predict --model-dir artifacts/model --store-root data/feature_store \\
        --description "Apache Log4j2 JNDI lookup enables remote code execution"

Each handler imports its module lazily, so a command only pays for the
dependencies it actually uses (e.g. ``predict`` never imports the ELT stack).
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional

# command -> dotted module path exposing a ``main(argv)`` function.
_COMMANDS: Dict[str, str] = {
    "pipeline": "pipeline",
    "train": "ml.train",
    "evaluate": "ml.evaluate",
    "predict": "ml.predict",
}


def _delegate(module_suffix: str) -> Callable[[List[str]], None]:
    """Import ``src.<module_suffix>`` and return its ``main`` (lazily)."""
    import importlib

    module = importlib.import_module(f"{__package__}.{module_suffix}")
    return module.main


def _usage() -> str:
    commands = "\n".join(f"  {name:<10} {path}" for name, path in _COMMANDS.items())
    return (
        "usage: python -m src.main <command> [options]\n\n"
        "commands:\n" + commands +
        "\n\nRun 'python -m src.main <command> --help' for command-specific options."
    )


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return

    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        sys.exit(f"unknown command {command!r}\n\n{_usage()}")

    _delegate(_COMMANDS[command])(rest)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    main()
