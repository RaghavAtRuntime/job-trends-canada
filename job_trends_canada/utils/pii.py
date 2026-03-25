"""PII (Personally Identifiable Information) removal utilities.

The ``drop_pii`` decorator strips recruiter names, e-mail addresses, phone
numbers, and other personal identifiers from any :class:`pandas.DataFrame`
returned or accepted by a function before it is stored or passed downstream.
"""

from __future__ import annotations

import functools
import re
from typing import Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Regex patterns for common PII tokens
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4})",
    re.IGNORECASE,
)
# Columns whose names suggest PII content
_PII_COLUMN_KEYWORDS = re.compile(
    r"(recruiter|contact|email|phone|mobile|name|posted_by|employer_contact)",
    re.IGNORECASE,
)


def _redact_series(series: pd.Series) -> pd.Series:
    """Return a copy of *series* with PII tokens replaced by ``[REDACTED]``."""
    # Accept both legacy object dtype and the newer StringDtype
    if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
        return series
    result = series.astype(str)
    result = result.str.replace(_EMAIL_RE.pattern, "[REDACTED_EMAIL]", regex=True)
    result = result.str.replace(_PHONE_RE.pattern, "[REDACTED_PHONE]", regex=True)
    return result


def sanitise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with PII columns dropped and tokens redacted.

    * Columns whose names match :data:`_PII_COLUMN_KEYWORDS` are dropped.
    * E-mail and phone patterns are redacted in every remaining text column.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    # Drop columns that are clearly PII by name
    pii_cols = [c for c in df.columns if _PII_COLUMN_KEYWORDS.search(str(c))]
    df = df.drop(columns=pii_cols, errors="ignore")

    # Redact patterns in all remaining object/string columns
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = _redact_series(df[col])

    return df


def drop_pii(func: Callable) -> Callable:
    """Decorator that sanitises :class:`~pandas.DataFrame` inputs and outputs.

    Wraps *func* so that:

    * Any positional or keyword argument that is a ``DataFrame`` is sanitised
      before being passed to *func*.
    * The return value, if it is a ``DataFrame``, is sanitised before being
      returned to the caller.

    Usage::

        @drop_pii
        def store_postings(df: pd.DataFrame) -> pd.DataFrame:
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        clean_args = tuple(
            sanitise_dataframe(a) if isinstance(a, pd.DataFrame) else a
            for a in args
        )
        clean_kwargs = {
            k: sanitise_dataframe(v) if isinstance(v, pd.DataFrame) else v
            for k, v in kwargs.items()
        }
        result = func(*clean_args, **clean_kwargs)
        if isinstance(result, pd.DataFrame):
            result = sanitise_dataframe(result)
        return result

    return wrapper
