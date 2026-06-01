"""Extractor for AlienVault OTX (the outline's ``IOCFetcher``).

OTX exposes threat intelligence as *pulses* — curated reports that each bundle
a set of indicators of compromise (IOCs: hashes, domains, IPs, URLs, ...) along
with metadata that is directly useful as ML labels: ``tags``, ``adversary``,
``malware_families``, and ``attack_ids`` (MITRE ATT&CK technique IDs).

Like NVD this is a key-authenticated, page-based REST API, so the HTTP session,
retries, and rate limiting are inherited from :class:`BaseExtractor`. What is
OTX-specific lives here: the ``X-OTX-API-KEY`` auth header, the
``{"results": [...], "next": <url>}`` pagination style, and the pulse/indicator
endpoints.

API reference: https://otx.alienvault.com/api
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

from .base import BaseExtractor, RateLimiter

logger = logging.getLogger(__name__)

OTX_API_URL = "https://otx.alienvault.com/api/v1"

# OTX caps page size at 50 for the pulse/indicator list endpoints.
MAX_PAGE_LIMIT = 50
DEFAULT_PAGE_LIMIT = 50

# Polite default client-side budget (OTX allows far more for authenticated
# keys, but we average ~1 request/second unless the caller raises it).
DEFAULT_REQUESTS_PER_MINUTE = 60

DateLike = Union[str, datetime]


def _to_iso(value: DateLike) -> str:
    """Render a datetime as an ISO-8601 UTC string (strings pass through)."""
    if isinstance(value, str):
        return value
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.isoformat()


class OTXExtractor(BaseExtractor):
    """Pulls pulses and their IOCs from the AlienVault OTX DirectConnect API.

    Example::

        with OTXExtractor(api_key="...") as otx:
            for pulse in otx.iter_pulses(max_pulses=100):
                print(pulse["name"], pulse.get("attack_ids"))
    """

    base_url = OTX_API_URL

    def __init__(
        self,
        api_key: str,
        *,
        requests_per_minute: Optional[int] = DEFAULT_REQUESTS_PER_MINUTE,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError("OTX requires an API key")
        rate_limiter = (
            RateLimiter(requests_per_minute, 60.0) if requests_per_minute else None
        )
        super().__init__(
            rate_limiter=rate_limiter,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        self.api_key = api_key
        self._session.headers["X-OTX-API-KEY"] = api_key

    def iter_pulses(
        self,
        *,
        subscribed: bool = True,
        modified_since: Optional[DateLike] = None,
        query: Optional[str] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        max_pulses: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield raw pulse objects, transparently following pagination.

        With ``query`` set, searches all public pulses; otherwise reads the
        account's ``subscribed`` feed (or the global ``activity`` feed when
        ``subscribed=False``). ``modified_since`` enables incremental pulls and
        ``max_pulses`` caps how many are yielded.
        """
        if query is not None:
            path = "/search/pulses"
            params: Dict[str, Any] = {"q": query}
        else:
            path = "/pulses/subscribed" if subscribed else "/pulses/activity"
            params = {}
        params["limit"] = min(limit, MAX_PAGE_LIMIT)
        if modified_since is not None:
            params["modified_since"] = _to_iso(modified_since)

        yield from self._paginate(path, params, max_items=max_pulses)

    def get_pulse(self, pulse_id: str) -> Dict[str, Any]:
        """Fetch a single pulse's full detail by its ID."""
        return self.get(f"/pulses/{pulse_id}")

    def get_pulse_indicators(
        self,
        pulse_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        max_indicators: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return all IOCs attached to a pulse (this endpoint paginates too)."""
        return list(
            self._paginate(
                f"/pulses/{pulse_id}/indicators",
                {"limit": min(limit, MAX_PAGE_LIMIT)},
                max_items=max_indicators,
            )
        )

    def get_indicator_details(
        self,
        indicator_type: str,
        indicator: str,
        section: str = "general",
    ) -> Dict[str, Any]:
        """Look up enrichment for a single indicator (e.g. ``"IPv4"``, ``8.8.8.8``)."""
        return self.get(f"/indicators/{indicator_type}/{indicator}/{section}")

    def extract(self, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """Implements :meth:`BaseExtractor.extract`; delegates to :meth:`iter_pulses`."""
        return self.iter_pulses(**kwargs)

    def _paginate(
        self,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        max_items: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Walk an OTX list endpoint, following ``next`` until exhausted.

        The first request uses ``path`` + ``params``; OTX then returns an
        absolute ``next`` URL with the cursor already baked in, so subsequent
        requests just follow it verbatim.
        """
        yielded = 0
        next_url: Optional[str] = None
        first = True
        while first or next_url:
            page = self.get(path, params=params) if first else self.get(next_url)
            first = False
            results = page.get("results", [])
            if not results:
                break
            for item in results:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            next_url = page.get("next")
            logger.debug("OTX: %d items yielded, more=%s", yielded, bool(next_url))

