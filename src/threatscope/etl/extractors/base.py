"""Reusable building blocks shared by the OSINT data-source extractors.

Every extractor in this package pulls raw records from a public threat-
intelligence REST API. They all share the same plumbing concerns:

* a configured, connection-pooled HTTP session,
* polite rate limiting so we stay within each provider's quota, and
* resilient retries with exponential backoff on transient failures.

Those concerns live here in :class:`BaseExtractor` and :class:`RateLimiter`
so the concrete extractors (NVD, OTX, MITRE, ...) only have to describe what
is unique about their API: endpoints, parameters, and response shapes.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from collections import deque
from typing import Any, Mapping, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying: rate-limit + transient server errors.
RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class RateLimiter:
    """Sliding-window rate limiter: at most ``max_calls`` per ``period`` seconds.

    Unlike a fixed-window counter, this tracks the timestamp of every call and
    only blocks when the oldest call still inside the window would push us over
    the limit. It is thread-safe so a single limiter can guard a session that is
    shared across worker threads.
    """

    def __init__(self, max_calls: int, period: float) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if period <= 0:
            raise ValueError("period must be > 0")
        self.max_calls = max_calls
        self.period = float(period)
        self._calls: "deque[float]" = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is free, then reserve it."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps aged out of window
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                # Wait just long enough for the oldest call to expire
                sleep_for = self.period - (now - self._calls[0])
            logger.debug("Rate limit reached; sleeping %.2fs", sleep_for)
            time.sleep(max(sleep_for, 0.0))


class BaseExtractor(abc.ABC):
    """Abstract base for HTTP-backed extractors.

    Subclasses set :attr:`base_url` (and optionally :attr:`default_headers`),
    implement :meth:`extract`, and use :meth:`get` for all network access so
    that rate limiting and retries are applied uniformly.
    """

    base_url: str = ""
    default_headers: Mapping[str, str] = {}

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.rate_limiter = rate_limiter
        self._session = session or self._build_session(max_retries, backoff_factor)

    def _build_session(self, max_retries: int, backoff_factor: float) -> requests.Session:
        """Create a session with urllib3-level retry/backoff on idempotent GETs."""
        session = requests.Session()
        session.headers.update(self.default_headers)
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=RETRYABLE_STATUS,
            allowed_methods=("GET",),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def get(self, path: str = "", params: Optional[Mapping[str, Any]] = None) -> Any:
        """Perform a rate-limited GET and return the decoded JSON body.

        ``path`` may be a full URL or a fragment appended to :attr:`base_url`.
        Raises :class:`requests.HTTPError` on a non-2xx response.
        """
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        logger.debug("GET %s params=%s", url, params)
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @abc.abstractmethod
    def extract(self, *args: Any, **kwargs: Any) -> Any:
        """Pull raw records from the source. Defined by each concrete extractor."""
        raise NotImplementedError

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "BaseExtractor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
