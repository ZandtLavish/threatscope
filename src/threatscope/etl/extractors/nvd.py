"""Extractor for the NVD (National Vulnerability Database) CVE API 2.0.

This is the *Extract* stage for the NVD source: it pulls raw CVE records and
hands them downstream untouched (normalization/encoding happen in the
``transformers`` package). It handles the parts of the NVD API that are easy to
get wrong:

* pagination via ``startIndex`` / ``resultsPerPage`` (2000 records max/page),
* the rolling 30-second request budget (10x larger with an API key), and
* the 120-day cap and ISO-8601 formatting on the date-range filters.

API reference: https://nvd.nist.gov/developers/vulnerabilities
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from .base import BaseExtractor, RateLimiter

logger = logging.getLogger(__name__)

NVD_CVE_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Hard limits by NVD API
MAX_RESULTS_PER_PAGE = 2000
MAX_DATE_RANGE_DAYS = 120

# Rolling-window request budgets (calls, seconds). An API key raises the public allowance
# Stay a touch under the documented ceilings to avoid edge-of-quota 403/429 responses.
PUBLIC_RATE = (5, 30.0)
API_KEY_RATE = (50, 30.0)

_NVD_DT_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

DateLike = Union[str, datetime]


def _format_datetime(value: DateLike) -> str:
    """Render a datetime (or pass through a string) in NVD's ISO-8601 format."""
    if isinstance(value, str):
        return value
    # NVD treats naive timestamps as UTC; make explicit for tz-aware ones
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    # Trim microseconds to the milliseconds NVD documents
    return value.strftime(_NVD_DT_FORMAT)[:-3]


def _validate_date_window(
    start: Optional[DateLike],
    end: Optional[DateLike],
    label: str,
) -> None:
    """NVD requires date filters as a start/end pair spanning <= 120 days."""
    if (start is None) != (end is None):
        raise ValueError(f"{label} requires both a start and an end date, or neither")
    if isinstance(start, datetime) and isinstance(end, datetime):
        if end < start:
            raise ValueError(f"{label}: end date precedes start date")
        if end - start > timedelta(days=MAX_DATE_RANGE_DAYS):
            raise ValueError(
                f"{label}: range exceeds NVD's {MAX_DATE_RANGE_DAYS}-day limit"
            )


class NVDExtractor(BaseExtractor):
    """Pulls raw CVE records from the NVD CVE API 2.0.

    Example::

        with NVDExtractor(api_key="...") as nvd:
            for cve in nvd.iter_cves(
                pub_start_date=datetime(2024, 1, 1),
                pub_end_date=datetime(2024, 1, 31),
            ):
                print(cve["id"])
    """

    base_url = NVD_CVE_API_URL
    default_headers = {"Accept": "application/json"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        results_per_page: int = MAX_RESULTS_PER_PAGE,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        max_calls, period = API_KEY_RATE if api_key else PUBLIC_RATE
        super().__init__(
            rate_limiter=RateLimiter(max_calls, period),
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        self.api_key = api_key
        if api_key:
            # NVD authenticates via a custom header, not bearer auth.
            self._session.headers["apiKey"] = api_key
        self.results_per_page = max(1, min(results_per_page, MAX_RESULTS_PER_PAGE))

    def iter_cves(
        self,
        *,
        pub_start_date: Optional[DateLike] = None,
        pub_end_date: Optional[DateLike] = None,
        last_mod_start_date: Optional[DateLike] = None,
        last_mod_end_date: Optional[DateLike] = None,
        keyword: Optional[str] = None,
        cvss_v3_severity: Optional[str] = None,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield raw ``cve`` objects matching the given filters, page by page.

        Filtering by publication date (``pub_*``) or last-modified date
        (``last_mod_*``) is the basis for incremental pulls; each window must
        span no more than 120 days. ``extra_params`` is merged in last for any
        NVD parameter not surfaced here (e.g. ``cpeName``, ``cveTag``).
        """
        params = self._build_params(
            pub_start_date=pub_start_date,
            pub_end_date=pub_end_date,
            last_mod_start_date=last_mod_start_date,
            last_mod_end_date=last_mod_end_date,
            keyword=keyword,
            cvss_v3_severity=cvss_v3_severity,
            extra_params=extra_params,
        )

        start_index = 0
        total_results: Optional[int] = None
        while total_results is None or start_index < total_results:
            page = self.get(
                params={
                    **params,
                    "startIndex": start_index,
                    "resultsPerPage": self.results_per_page,
                }
            )
            total_results = page.get("totalResults", 0)
            vulnerabilities = page.get("vulnerabilities", [])
            if not vulnerabilities:
                break

            for entry in vulnerabilities:
                cve = entry.get("cve")
                if cve is not None:
                    yield cve

            start_index += len(vulnerabilities)
            logger.info(
                "NVD: fetched %d/%d CVEs", min(start_index, total_results), total_results
            )

    def fetch_cve(self, cve_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single CVE by its ID (e.g. ``"CVE-2021-44228"``)."""
        page = self.get(params={"cveId": cve_id})
        vulnerabilities = page.get("vulnerabilities", [])
        return vulnerabilities[0]["cve"] if vulnerabilities else None

    def extract(self, **filters: Any) -> Iterator[Dict[str, Any]]:
        """Implements :meth:`BaseExtractor.extract`; delegates to :meth:`iter_cves`."""
        return self.iter_cves(**filters)

    def _build_params(
        self,
        *,
        pub_start_date: Optional[DateLike],
        pub_end_date: Optional[DateLike],
        last_mod_start_date: Optional[DateLike],
        last_mod_end_date: Optional[DateLike],
        keyword: Optional[str],
        cvss_v3_severity: Optional[str],
        extra_params: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Validate filters and assemble the static query parameters."""
        _validate_date_window(pub_start_date, pub_end_date, "publication date")
        _validate_date_window(last_mod_start_date, last_mod_end_date, "last-modified date")

        params: Dict[str, Any] = {}
        if pub_start_date is not None:
            params["pubStartDate"] = _format_datetime(pub_start_date)
            params["pubEndDate"] = _format_datetime(pub_end_date)
        if last_mod_start_date is not None:
            params["lastModStartDate"] = _format_datetime(last_mod_start_date)
            params["lastModEndDate"] = _format_datetime(last_mod_end_date)
        if keyword:
            params["keywordSearch"] = keyword
        if cvss_v3_severity:
            params["cvssV3Severity"] = cvss_v3_severity.upper()
        if extra_params:
            params.update(extra_params)
        return params

