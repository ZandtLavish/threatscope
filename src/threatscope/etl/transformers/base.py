"""Reusable abstraction for the transform stage.

A transformer maps a stream of input records to a stream of output records.
That is the whole contract — :class:`BaseTransformer` — and it mirrors the
generator-based, streaming style the extractors already use, so transformers
compose without materializing the full dataset in memory.

The concrete transformers fit this shape as follows:

* ``Normalizer``  : Iterable[raw source record] -> Iterator[ThreatEvent]
* ``Joiner``      : Iterable[ThreatEvent]        -> Iterator[ThreatEvent]
* ``Encoder``     : Iterable[ThreatEvent]        -> Iterator[feature row]

:class:`Chain` wires them together left-to-right so the pipeline reads as one
composed transformer: ``Chain(Normalizer(), Joiner(...), Encoder(...))``.
"""

from __future__ import annotations

import abc
from typing import Generic, Iterable, Iterator, TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class BaseTransformer(abc.ABC, Generic[InT, OutT]):
    """Maps an iterable of input records to an iterator of output records.

    Subclasses implement :meth:`transform` as a generator. Instances are
    callable for convenience, so ``transformer(records)`` is equivalent to
    ``transformer.transform(records)``.
    """

    @abc.abstractmethod
    def transform(self, records: Iterable[InT]) -> Iterator[OutT]:
        """Lazily consume ``records`` and yield transformed output."""
        raise NotImplementedError

    def __call__(self, records: Iterable[InT]) -> Iterator[OutT]:
        return self.transform(records)


class Chain(BaseTransformer[InT, OutT]):
    """Compose transformers left-to-right: ``Chain(a, b)(x) == b(a(x))``.

    Composition stays lazy — each stage pulls from the previous one's
    generator — so a long pipeline still streams one record at a time. Type
    parameters describe the chain's outer ends; intermediate stage types are
    not statically checked.
    """

    def __init__(self, *stages: BaseTransformer) -> None:
        if not stages:
            raise ValueError("Chain requires at least one transformer")
        self.stages = stages

    def transform(self, records: Iterable[InT]) -> Iterator[OutT]:
        stream: Iterable = records
        for stage in self.stages:
            stream = stage.transform(stream)
        return iter(stream)
