"""Mockable web-scraping utility for supplementary job-posting sources.

Currently supports two back-ends:

* ``"mock"``   – returns a synthetic DataFrame; no network calls made.
* ``"eluta"``  – scrapes Eluta.ca search results (best-effort; subject to
  site changes).

Usage::

    from job_trends_canada.data_ingestion.scraper import fetch_supplementary

    df = fetch_supplementary(source="mock", query="software engineer", pages=1)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

SourceType = Literal["mock", "eluta"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_supplementary(
    source: SourceType = "mock",
    *,
    query: str = "software engineer",
    province: str = "ON",
    pages: int = 1,
    delay: float = 1.0,
) -> pd.DataFrame:
    """Return supplementary job postings from *source*.

    Parameters
    ----------
    source:
        Back-end to use.  ``"mock"`` returns synthetic data with no network
        activity; ``"eluta"`` scrapes Eluta.ca.
    query:
        Job-title search query.
    province:
        Two-letter Canadian province code used to filter results.
    pages:
        Number of result pages to fetch (only relevant for ``"eluta"``).
    delay:
        Seconds to wait between page requests (polite scraping).

    Returns
    -------
    pandas.DataFrame
        Columns: ``source``, ``title``, ``province``, ``posted_date``,
        ``description``.
    """
    if source == "mock":
        return _mock_data(query=query, province=province)
    if source == "eluta":
        return _scrape_eluta(query=query, province=province, pages=pages, delay=delay)
    raise ValueError(f"Unknown source: {source!r}. Choose 'mock' or 'eluta'.")


# ---------------------------------------------------------------------------
# Mock back-end
# ---------------------------------------------------------------------------

_MOCK_ROWS = [
    {
        "source": "mock",
        "title": "Senior Product Manager",
        "province": "ON",
        "posted_date": "2024-03-01",
        "description": "Lead product strategy. Direct reports: 3. Hiring responsibility.",
    },
    {
        "source": "mock",
        "title": "Software Architect",
        "province": "BC",
        "posted_date": "2024-03-05",
        "description": "Design large-scale systems. Mentoring junior engineers.",
    },
    {
        "source": "mock",
        "title": "Engineering Manager",
        "province": "ON",
        "posted_date": "2024-03-10",
        "description": "Manage a team of 8 engineers. Budgetary responsibility of $500K.",
    },
    {
        "source": "mock",
        "title": "Marketing Analyst",
        "province": "QC",
        "posted_date": "2024-03-12",
        "description": "Analyse campaign performance and report to Marketing Director.",
    },
    {
        "source": "mock",
        "title": "Director of Sales",
        "province": "AB",
        "posted_date": "2024-03-15",
        "description": "P&L responsibility for western region. Report to VP Sales.",
    },
]


def _mock_data(query: str, province: str) -> pd.DataFrame:
    rows = _MOCK_ROWS.copy()
    df = pd.DataFrame(rows)
    df["posted_date"] = pd.to_datetime(df["posted_date"])
    return df


# ---------------------------------------------------------------------------
# Eluta.ca back-end (best-effort scraper)
# ---------------------------------------------------------------------------

_ELUTA_BASE = "https://www.eluta.ca/search"


def _scrape_eluta(
    query: str,
    province: str,
    pages: int,
    delay: float,
) -> pd.DataFrame:
    """Scrape Eluta.ca for job listings.

    Returns an empty DataFrame gracefully if scraping fails (e.g. network
    unavailable, site structure changed, or the request is blocked).
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        logger.error("Missing dependency for Eluta scraper: %s", exc)
        return pd.DataFrame(columns=["source", "title", "province", "posted_date", "description"])

    rows: list[dict] = []
    for page in range(1, pages + 1):
        params = {
            "q": query,
            "l": province,
            "pg": page,
        }
        try:
            resp = requests.get(_ELUTA_BASE, params=params, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Eluta request failed (page %d): %s", page, exc)
            break

        soup = BeautifulSoup(resp.text, "lxml")
        listings = soup.select("div.result") or soup.select("li.result")
        if not listings:
            logger.info("No listings found on page %d – stopping.", page)
            break

        for item in listings:
            title_el = item.select_one("a.jobtitle, h2.job-title, a[data-job-title]")
            desc_el = item.select_one("p.description, span.summary")
            date_el = item.select_one("span.date, time")

            rows.append(
                {
                    "source": "eluta",
                    "title": title_el.get_text(strip=True) if title_el else "",
                    "province": province,
                    "posted_date": date_el.get_text(strip=True) if date_el else "",
                    "description": desc_el.get_text(strip=True) if desc_el else "",
                }
            )
        if page < pages:
            time.sleep(delay)

    if not rows:
        return pd.DataFrame(columns=["source", "title", "province", "posted_date", "description"])

    df = pd.DataFrame(rows)
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    return df
