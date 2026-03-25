"""Feature engineering – span-of-control and hybridisation signals.

Two complementary signals are extracted from raw job descriptions:

1. **Span of Control proxy** – parses numeric team sizes mentioned in the
   description (e.g. "managing a team of 5", "leads a group of 12 people").

2. **Hybridisation signal** – detects whether a *Senior IC* posting lists
   management-style responsibilities (mentoring, strategy, hiring) without
   carrying a formal "Manager" title.  A boolean ``is_hybrid_ic`` flag and a
   numeric ``hybrid_score`` are produced.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Span of control patterns
# ---------------------------------------------------------------------------

_TEAM_SIZE_RE = re.compile(
    r"""
    (?:
        (?:managing|leading|supervising|overseeing|responsible\s+for)
        \s+(?:a\s+)?(?:team|group|staff)\s+of\s+(\d+)
    )
    |
    (?:
        (?:team|group|staff)\s+of\s+(\d+)
    )
    |
    (?:
        (\d+)\s+(?:direct\s+)?reports?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Hybridisation patterns (IC with management responsibilities)
# ---------------------------------------------------------------------------

_HYBRID_SIGNALS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bmentor(?:ing|s)?\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bcoach(?:ing)?\b", re.IGNORECASE), 0.75),
    (re.compile(r"\bhiring\b", re.IGNORECASE), 1.0),
    (re.compile(r"\brecruit(?:ing|ment)?\b", re.IGNORECASE), 0.75),
    (re.compile(r"\bstrateg(?:y|ic)\b", re.IGNORECASE), 0.5),
    (re.compile(r"\broad[\s\-]?map\b", re.IGNORECASE), 0.5),
    (re.compile(r"\binfluence\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bcross[\s\-]?functional\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bonboarding\b", re.IGNORECASE), 0.75),
    (re.compile(r"\bknowledge[\s\-]?shar(?:e|ing)\b", re.IGNORECASE), 0.5),
]

_MANAGER_TITLE_RE = re.compile(r"\bmanager\b", re.IGNORECASE)
_HYBRID_THRESHOLD = 1.5

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_span_of_control(description: str) -> Optional[int]:
    """Return the largest team size mentioned in *description*, or ``None``.

    Searches for patterns like "team of 5", "8 direct reports", etc.
    When multiple mentions exist the maximum is returned (conservative proxy
    for the actual span of control).
    """
    if not isinstance(description, str):
        return None
    sizes: list[int] = []
    for match in _TEAM_SIZE_RE.finditer(description):
        for group in match.groups():
            if group is not None:
                sizes.append(int(group))
    return max(sizes) if sizes else None


def extract_hybrid_score(title: str, description: str) -> float:
    """Return a hybridisation score for a *Senior IC* posting.

    The score is the sum of weights for all detected management-style
    responsibility signals.  A higher score means the role lists more
    management activities despite (potentially) not having "Manager" in the
    title.
    """
    text = f"{title} {description}"
    return sum(
        weight for pattern, weight in _HYBRID_SIGNALS if pattern.search(text)
    )


def is_hybrid_ic(title: str, description: str, threshold: float = _HYBRID_THRESHOLD) -> bool:
    """Return ``True`` if a posting appears to be a hybrid IC/management role.

    A "hybrid IC" is a non-manager-titled role that lists a significant
    number of management-style responsibilities.
    """
    if _MANAGER_TITLE_RE.search(title):
        return False  # explicit manager title → not hybrid
    return extract_hybrid_score(title, description) >= threshold


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature-engineering columns to *df*.

    Columns added:

    ``span_of_control``
        Integer team size or ``NaN``.
    ``hybrid_score``
        Float hybridisation score.
    ``is_hybrid_ic``
        Boolean – ``True`` when role exhibits hybrid IC/management signals.

    Parameters
    ----------
    df:
        DataFrame with at minimum a ``title`` and ``description`` column.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with the three new columns added.
    """
    df = df.copy()

    title_col = df.get("title", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    desc_col = df.get("description", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)

    df["span_of_control"] = [
        extract_span_of_control(desc) for desc in desc_col
    ]
    df["hybrid_score"] = [
        extract_hybrid_score(title, desc)
        for title, desc in zip(title_col, desc_col)
    ]
    df["is_hybrid_ic"] = [
        is_hybrid_ic(title, desc)
        for title, desc in zip(title_col, desc_col)
    ]

    return df
