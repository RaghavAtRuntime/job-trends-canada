"""Canada Job Bank data ingestion module.

Downloads and parses the Government of Canada's Open Government Portal job
posting archives (CSV format) and exposes a clean :class:`pandas.DataFrame`.

The Job Bank publishes monthly CSV archives at a predictable URL pattern:
    https://open.canada.ca/data/en/dataset/...

Because the exact archive URL changes over time, this module attempts to
resolve the latest dataset resource URL from the CKAN API, then downloads and
parses the CSV.  When running in test/offline mode, a synthetic dataset is
returned instead.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CKAN package identifier for the Job Bank postings dataset
_CKAN_API = "https://open.canada.ca/api/3/action/package_show"
_DATASET_ID = "3f4a4b5a-13ac-4c84-95f5-7b2e0e11f98f"  # Job Bank – Job Postings

# Column normalisation map  (raw → internal name)
_COLUMN_MAP: dict[str, str] = {
    "job_id": "job_id",
    "title": "title",
    "noc": "noc_code",
    "noc_code": "noc_code",
    "province": "province",
    "naics": "naics_code",
    "naics_code": "naics_code",
    "industry": "naics_code",
    "posted_date": "posted_date",
    "date_posted": "posted_date",
    "description": "description",
    "job_description": "description",
    "salary": "salary",
    "employer": "employer",
    "city": "city",
    "location": "city",
}

# Minimum columns expected in a valid dataset
_REQUIRED_COLUMNS = {"title"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_job_bank_data(
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    use_sample: bool = False,
    csv_url: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return a normalised DataFrame of Job Bank postings.

    Parameters
    ----------
    start_date:
        If provided, keep only rows with ``posted_date >= start_date``.
    end_date:
        If provided, keep only rows with ``posted_date <= end_date``.
    use_sample:
        When ``True``, skip network calls and return the built-in synthetic
        sample dataset.  Useful for unit tests and offline development.
    csv_url:
        Override the auto-resolved CSV URL.  Ignored when *use_sample* is
        ``True``.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Normalised postings with at least the columns: ``job_id``, ``title``,
        ``noc_code``, ``province``, ``naics_code``, ``posted_date``,
        ``description``.
    """
    if use_sample:
        logger.info("Using built-in sample dataset (offline mode).")
        df = _build_sample_dataset()
    else:
        url = csv_url or _resolve_csv_url(timeout=timeout)
        logger.info("Downloading Job Bank data from %s", url)
        df = _download_and_parse(url, timeout=timeout)

    df = _normalise_columns(df)
    df = _parse_dates(df)
    df = _filter_date_range(df, start_date=start_date, end_date=end_date)
    df = _clean_titles(df)
    return df


def clean_titles(titles: pd.Series) -> pd.Series:
    """Return *titles* with whitespace normalised and casing title-cased.

    This function is also called internally by :func:`fetch_job_bank_data`.
    It is exposed publicly so that callers can clean titles from other sources
    using the same logic.
    """
    cleaned = titles.astype(str).str.strip()
    # Collapse multiple internal spaces
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    # Remove non-printable characters
    cleaned = cleaned.str.replace(r"[^\x20-\x7E]", "", regex=True)
    # Title-case
    cleaned = cleaned.str.title()
    return cleaned


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_csv_url(timeout: int = 30) -> str:
    """Query the CKAN API to find the latest CSV resource URL."""
    try:
        resp = requests.get(
            _CKAN_API,
            params={"id": _DATASET_ID},
            timeout=timeout,
        )
        resp.raise_for_status()
        resources = resp.json()["result"]["resources"]
        csv_resources = [
            r for r in resources if r.get("format", "").upper() == "CSV"
        ]
        if not csv_resources:
            raise ValueError("No CSV resource found in Job Bank dataset.")
        # Prefer the most recently created resource
        csv_resources.sort(key=lambda r: r.get("created", ""), reverse=True)
        return csv_resources[0]["url"]
    except Exception as exc:
        logger.warning(
            "Could not resolve CSV URL from CKAN API (%s). "
            "Falling back to sample dataset.",
            exc,
        )
        raise


def _download_and_parse(url: str, timeout: int = 30) -> pd.DataFrame:
    """Download *url* and return a parsed DataFrame."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), low_memory=False, dtype=str)


def _pad_noc(value: object) -> object:
    """Zero-pad a NOC code string to 5 characters; return as-is if not numeric."""
    if value is None or (isinstance(value, float)):
        return pd.NA
    s = str(value).strip()
    if not s or s.lower() == "nan" or s.lower() == "<na>":
        return pd.NA
    if s.isdigit():
        return s.zfill(5)
    return s


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to internal names using :data:`_COLUMN_MAP`."""
    # Lower-case and strip column names
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {c: _COLUMN_MAP[c] for c in df.columns if c in _COLUMN_MAP}
    df = df.rename(columns=rename)

    # Ensure required columns exist
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Add optional columns as NaN when absent
    for col in ["job_id", "noc_code", "province", "naics_code", "posted_date", "description"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce job_id: if missing, generate a sequential id
    if df["job_id"].isna().all():
        df["job_id"] = range(1, len(df) + 1)

    # NOC codes must be stored as strings to preserve leading zeros (e.g. "00014").
    # Pandas reads them as integers from CSV; pad back to at least 5 digits.
    if "noc_code" in df.columns:
        df["noc_code"] = (
            df["noc_code"]
            .dropna()
            .astype(str)
            .where(lambda s: s.str.strip().ne("nan"), other=pd.NA)
        )
        df["noc_code"] = df["noc_code"].apply(_pad_noc)

    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "posted_date" in df.columns:
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    return df


def _filter_date_range(
    df: pd.DataFrame,
    *,
    start_date: Optional[date],
    end_date: Optional[date],
) -> pd.DataFrame:
    if start_date and "posted_date" in df.columns:
        df = df[df["posted_date"].dt.date >= start_date]
    if end_date and "posted_date" in df.columns:
        df = df[df["posted_date"].dt.date <= end_date]
    return df.reset_index(drop=True)


def _clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    df["title"] = clean_titles(df["title"])
    return df


# ---------------------------------------------------------------------------
# Synthetic sample dataset (used in offline / test mode)
# ---------------------------------------------------------------------------

_SAMPLE_DATA = """job_id,title,noc_code,province,naics_code,posted_date,description
1,Software Developer,21232,ON,511210,2024-01-15,Develops and maintains web applications.
2,Project Manager,70010,BC,541511,2024-01-20,Leads cross-functional teams. Direct reports: 6.
3,Data Analyst,21223,AB,522110,2024-02-01,Analyzes datasets using Python and SQL.
4,Operations Manager,00014,QC,493110,2024-02-10,Manages warehouse operations. Budgetary responsibility of $2M.
5,Senior Software Engineer,21232,ON,511210,2024-02-15,Mentors junior developers and contributes to architecture strategy.
6,Marketing Coordinator,11202,MB,541810,2024-03-01,Coordinates marketing campaigns.
7,VP Engineering,00011,ON,511210,2024-03-10,P&L responsibility. Reports to CTO. Manages a team of 25.
8,Junior Developer,21232,SK,511210,2024-03-15,Entry-level position.
9,HR Manager,10011,BC,561310,2024-04-01,Manages HR team. Direct reports: 4. Hiring responsibility.
10,Business Analyst,11200,ON,522110,2024-04-05,Analyzes business processes.
11,Team Lead,21232,AB,511210,2024-04-10,Technical lead responsible for a team of 3.
12,Sales Manager,60010,QC,441110,2024-04-15,Manages regional sales team. Budgetary responsibility.
13,DevOps Engineer,21232,ON,511210,2024-05-01,CI/CD pipelines and cloud infrastructure.
14,Finance Manager,10011,ON,522110,2024-05-10,Oversees financial reporting. Direct reports: 5.
15,Principal Engineer,21232,BC,511210,2024-05-15,Senior IC role with mentoring responsibilities and hiring involvement.
16,Store Manager,62020,AB,441110,2024-06-01,Manages retail store. Report to District Manager.
17,UX Designer,52120,ON,511210,2024-06-10,Designs user interfaces.
18,Director of Product,70020,ON,511210,2024-06-15,Manages product managers. P&L ownership.
19,Data Engineer,21223,BC,511210,2024-07-01,Builds data pipelines.
20,Regional Manager,00014,MB,561499,2024-07-10,Manages multiple locations. Direct reports: 8.
"""


def _build_sample_dataset() -> pd.DataFrame:
    """Return the hard-coded sample dataset as a DataFrame."""
    return pd.read_csv(io.StringIO(_SAMPLE_DATA), dtype=str)
