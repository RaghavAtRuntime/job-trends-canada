"""NLP / regex-based classifier for postings without NOC codes.

When a NOC code is absent or ambiguous, this module analyses the free-text
description and title to determine whether the role is a management role and,
if so, at what level.

Strategy
--------
1. **Regex keyword heuristics** – fast, no model required.  Looks for
   phrases like "Direct reports", "Budgetary responsibility", "P&L",
   "Report to", "manages a team".  A configurable *threshold* controls how
   many signals must fire before a posting is labelled Management.

2. **Optional sklearn pipeline** – when ``use_sklearn=True``, a lightweight
   ``TfidfVectorizer + LogisticRegression`` model is trained on the labelled
   rows produced by the regex heuristic and then used to score unlabelled
   rows.  Useful when you have a large corpus and want better generalisation.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from job_trends_canada.classification.noc_classifier import (
    Tier,
    _ic_tier_from_title,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword signal definitions
# ---------------------------------------------------------------------------

# Each entry is (pattern, weight).  Weights are summed; if total >= threshold
# the posting is classified as Management.
_MANAGEMENT_SIGNALS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bdirect\s+reports?\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bbudgetary\s+responsibilit", re.IGNORECASE), 1.0),
    (re.compile(r"\bP\s*&\s*L\b", re.IGNORECASE), 1.0),
    (re.compile(r"\breport(?:s)?\s+to\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bmanag(?:e|es|ing)\s+a\s+team\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bteam\s+of\s+\d+\b", re.IGNORECASE), 0.75),
    (re.compile(r"\bhiring\s+(?:responsibilit|authority|decisions?)\b", re.IGNORECASE), 0.75),
    (re.compile(r"\bperformance\s+review", re.IGNORECASE), 0.5),
    (re.compile(r"\bstaff(?:ing)?\s+decisions?\b", re.IGNORECASE), 0.5),
]

_EXECUTIVE_SIGNALS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bchief\s+\w+\s+officer\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bvice[\s\-]president\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bC[EFO]O\b"), 2.0),
    (re.compile(r"\bP\s*&\s*L\s+responsibilit", re.IGNORECASE), 1.5),
    (re.compile(r"\bboard\s+of\s+directors?\b", re.IGNORECASE), 1.0),
]

_MANAGEMENT_THRESHOLD = 1.0
_EXECUTIVE_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_by_text(
    title: str,
    description: str,
    *,
    management_threshold: float = _MANAGEMENT_THRESHOLD,
    executive_threshold: float = _EXECUTIVE_THRESHOLD,
) -> Tier:
    """Return the tier inferred from free-text *title* and *description*.

    Parameters
    ----------
    title:
        Job title string.
    description:
        Full job description text.
    management_threshold:
        Minimum summed signal weight to label a posting Management.
    executive_threshold:
        Minimum summed signal weight to label a posting Executive.

    Returns
    -------
    Tier
    """
    text = f"{title} {description}"

    exec_score = _score(text, _EXECUTIVE_SIGNALS)
    if exec_score >= executive_threshold:
        return "Executive"

    mgmt_score = _score(text, _MANAGEMENT_SIGNALS)
    if mgmt_score >= management_threshold:
        return "Middle Management"

    return _ic_tier_from_title(title)


def classify_dataframe_by_text(
    df: pd.DataFrame,
    *,
    management_threshold: float = _MANAGEMENT_THRESHOLD,
    executive_threshold: float = _EXECUTIVE_THRESHOLD,
) -> pd.DataFrame:
    """Add / overwrite the ``tier`` column using text classification.

    Only rows where ``tier`` is missing or the ``noc_code`` column is null are
    (re-)classified.  Rows already classified via NOC are left untouched
    unless ``force=True``.

    Parameters
    ----------
    df:
        DataFrame with at minimum a ``title`` column.  ``description`` is
        optional.
    """
    df = df.copy()
    title_col = df.get("title", pd.Series([""] * len(df), index=df.index))
    desc_col = df.get("description", pd.Series([""] * len(df), index=df.index))
    noc_col = df.get("noc_code", pd.Series([None] * len(df), index=df.index))

    existing_tiers = df.get("tier", pd.Series([None] * len(df), index=df.index))

    tiers: list[Tier] = []
    for title, desc, noc, existing_tier in zip(title_col, desc_col, noc_col, existing_tiers):
        has_noc = bool(noc and str(noc).strip() and str(noc).strip().lower() not in ("nan", "<na>"))
        if has_noc and existing_tier is not None:
            # NOC code present and already classified – preserve the existing tier
            tiers.append(existing_tier)
        else:
            tiers.append(
                classify_by_text(
                    str(title) if title else "",
                    str(desc) if desc else "",
                    management_threshold=management_threshold,
                    executive_threshold=executive_threshold,
                )
            )
    df["tier"] = tiers
    return df


def build_sklearn_classifier(df: pd.DataFrame):
    """Train and return a sklearn text classifier on *df*.

    Uses ``TfidfVectorizer + LogisticRegression``.  The training labels are
    the ``tier`` column (must already be present).

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Fitted ``Pipeline(tfidf, lr)`` ready for ``predict()``.
    label_encoder : sklearn.preprocessing.LabelEncoder
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError("scikit-learn is required for build_sklearn_classifier.") from exc

    if "tier" not in df.columns:
        raise ValueError("DataFrame must have a 'tier' column to train classifier.")

    texts = (
        df.get("title", pd.Series([""] * len(df))).astype(str)
        + " "
        + df.get("description", pd.Series([""] * len(df))).astype(str)
    ).tolist()

    le = LabelEncoder()
    y = le.fit_transform(df["tier"].astype(str))

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(texts, y)
    logger.info("Trained sklearn classifier on %d samples.", len(df))
    return pipeline, le


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score(text: str, signals: list[tuple[re.Pattern, float]]) -> float:
    total = 0.0
    for pattern, weight in signals:
        if pattern.search(text):
            total += weight
    return total
