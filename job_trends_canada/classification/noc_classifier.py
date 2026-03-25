"""NOC (National Occupational Classification) 2021 based classifier.

The NOC 2021 broad-structure uses a leading digit to indicate the broad
occupational category:

* ``0`` – Management occupations  → **Middle Management / Executive**
* ``1`` – Business, finance, admin → IC (varies)
* ``2`` – Natural / applied sciences → IC
* ``3`` – Health occupations → IC
* ``4`` – Education, law, social → IC
* ``5`` – Arts, culture, recreation → IC
* ``6`` – Sales and service → IC
* ``7`` – Trades, transport → IC
* ``8`` – Natural resources, agriculture → IC
* ``9`` – Manufacturing / utilities → IC

Within ``0``, we further distinguish *Executive* vs *Middle Management* by
the second digit:

* ``00`` – Senior management (Executive)
* ``01``–``09`` – Middle management

This module is intentionally code-only (no ML model required) so the full
pipeline can run offline.
"""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd

# Tier labels used throughout the pipeline
Tier = Literal["Entry/Junior IC", "Senior/Principal IC", "Middle Management", "Executive"]

_NOC_PREFIX_RE = re.compile(r"^\s*(\d)", )
_NOC_SENIOR_MGT_RE = re.compile(r"^\s*00")
_NOC_MGT_RE = re.compile(r"^\s*0")

# Title keywords that hint at seniority within IC roles
_SENIOR_TITLE_RE = re.compile(
    r"\b(senior|principal|staff|lead|architect|distinguished|fellow|director)\b",
    re.IGNORECASE,
)
_JUNIOR_TITLE_RE = re.compile(
    r"\b(junior|associate|entry[\s\-]?level|intern|trainee|apprentice)\b",
    re.IGNORECASE,
)


def classify_by_noc(noc_code: str | float | None, title: str = "") -> Tier:
    """Return the tier for a posting given its NOC code and title.

    Parameters
    ----------
    noc_code:
        Raw NOC code string (e.g. ``"21232"``, ``"00014"``).  May be
        ``None`` / ``NaN`` when absent.
    title:
        Job title, used to refine the IC seniority tier when the NOC code
        is in the non-management range.

    Returns
    -------
    Tier
        One of ``"Entry/Junior IC"``, ``"Senior/Principal IC"``,
        ``"Middle Management"``, or ``"Executive"``.
    """
    noc_str = _to_str(noc_code)

    if noc_str:
        if _NOC_SENIOR_MGT_RE.match(noc_str):
            return "Executive"
        if _NOC_MGT_RE.match(noc_str):
            return "Middle Management"

    # Non-management (or unknown) NOC: determine IC seniority from title
    return _ic_tier_from_title(title)


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``tier`` column to *df* using NOC codes and titles.

    Expects columns ``noc_code`` and ``title`` (both optional; missing values
    are handled gracefully).

    Returns a copy of *df* with the ``tier`` column added/overwritten.
    """
    df = df.copy()
    noc_col = df.get("noc_code", pd.Series([""] * len(df), index=df.index))
    title_col = df.get("title", pd.Series([""] * len(df), index=df.index))
    df["tier"] = [
        classify_by_noc(noc, title)
        for noc, title in zip(noc_col, title_col)
    ]
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_str(value: object) -> str:
    """Convert *value* to a stripped string; return empty string for null."""
    if value is None:
        return ""
    try:
        import math
        if math.isnan(float(value)):  # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _ic_tier_from_title(title: str) -> Tier:
    if _SENIOR_TITLE_RE.search(title):
        return "Senior/Principal IC"
    if _JUNIOR_TITLE_RE.search(title):
        return "Entry/Junior IC"
    return "Entry/Junior IC"  # default for unknown / ambiguous titles
