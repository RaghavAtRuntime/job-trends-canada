"""Tests for the PII utility module."""

import pandas as pd
import pytest

from job_trends_canada.utils.pii import drop_pii, sanitise_dataframe


def test_email_redaction():
    df = pd.DataFrame({"description": ["Contact john.doe@example.com for info"]})
    clean = sanitise_dataframe(df)
    assert "[REDACTED_EMAIL]" in clean["description"].iloc[0]
    assert "john.doe@example.com" not in clean["description"].iloc[0]


def test_phone_redaction():
    df = pd.DataFrame({"description": ["Call us at 416-555-1234 anytime"]})
    clean = sanitise_dataframe(df)
    assert "[REDACTED_PHONE]" in clean["description"].iloc[0]
    assert "416-555-1234" not in clean["description"].iloc[0]


def test_pii_column_dropped():
    df = pd.DataFrame({"title": ["Engineer"], "recruiter": ["Jane Smith"], "email": ["jane@corp.com"]})
    clean = sanitise_dataframe(df)
    assert "recruiter" not in clean.columns
    assert "email" not in clean.columns
    assert "title" in clean.columns


def test_non_pii_column_preserved():
    df = pd.DataFrame({"title": ["Engineer"], "province": ["ON"]})
    clean = sanitise_dataframe(df)
    assert set(clean.columns) == {"title", "province"}


def test_drop_pii_decorator_sanitises_return_value():
    @drop_pii
    def get_data():
        return pd.DataFrame({"email": ["a@b.com"], "title": ["Dev"]})

    result = get_data()
    assert "email" not in result.columns


def test_drop_pii_decorator_sanitises_input():
    @drop_pii
    def passthrough(df: pd.DataFrame) -> pd.DataFrame:
        return df

    df = pd.DataFrame({"recruiter": ["Bob"], "title": ["Manager"]})
    result = passthrough(df)
    assert "recruiter" not in result.columns


def test_non_dataframe_passthrough():
    """Non-DataFrame arguments should pass through unmodified."""
    @drop_pii
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"
