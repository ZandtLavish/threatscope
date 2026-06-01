"""Transform stage: normalize, join, and encode extracted records.

Foundation (defined here): the :class:`BaseTransformer` abstraction and the
canonical :class:`ThreatEvent` schema that the concrete transformers
(normalizer, joiner, encoder) build on.
"""

from .base import BaseTransformer, Chain
from .encoder import CategoricalVocabulary, FeatureEncoder, HashingTextEmbedder
from .joiner import MITREJoiner
from .normalizer import BaseNormalizer, NVDNormalizer, OTXNormalizer
from .schema import CVSS, Severity, SourceType, ThreatEvent, to_utc

__all__ = [
    "BaseTransformer",
    "Chain",
    "ThreatEvent",
    "CVSS",
    "Severity",
    "SourceType",
    "to_utc",
    "BaseNormalizer",
    "NVDNormalizer",
    "OTXNormalizer",
    "MITREJoiner",
    "FeatureEncoder",
    "CategoricalVocabulary",
    "HashingTextEmbedder",
]
