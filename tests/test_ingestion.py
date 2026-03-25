"""Tests for the data ingestion modules."""

import pandas as pd
import pytest

from job_trends_canada.data_ingestion.job_bank import (
    clean_titles,
    fetch_job_bank_data,
)
from job_trends_canada.data_ingestion.scraper import fetch_supplementary


class TestJobBankIngestion:
    def test_sample_returns_dataframe(self):
        df = fetch_job_bank_data(use_sample=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns_present(self):
        df = fetch_job_bank_data(use_sample=True)
        for col in ["job_id", "title", "noc_code", "province", "posted_date", "description"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_titles_are_clean(self):
        df = fetch_job_bank_data(use_sample=True)
        # No leading/trailing whitespace
        assert (df["title"].str.strip() == df["title"]).all()

    def test_date_filter_start(self):
        from datetime import date
        df = fetch_job_bank_data(use_sample=True, start_date=date(2024, 4, 1))
        assert (df["posted_date"] >= pd.Timestamp("2024-04-01")).all()

    def test_date_filter_end(self):
        from datetime import date
        df = fetch_job_bank_data(use_sample=True, end_date=date(2024, 3, 31))
        assert (df["posted_date"] <= pd.Timestamp("2024-03-31")).all()


class TestCleanTitles:
    def test_strips_whitespace(self):
        result = clean_titles(pd.Series(["  Software Engineer  "]))
        assert result.iloc[0] == "Software Engineer"

    def test_title_case(self):
        result = clean_titles(pd.Series(["SENIOR DATA ANALYST"]))
        assert result.iloc[0] == "Senior Data Analyst"

    def test_collapses_spaces(self):
        result = clean_titles(pd.Series(["Data   Scientist"]))
        assert result.iloc[0] == "Data Scientist"


class TestScraper:
    def test_mock_returns_dataframe(self):
        df = fetch_supplementary(source="mock")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_mock_has_required_columns(self):
        df = fetch_supplementary(source="mock")
        for col in ["source", "title", "province", "posted_date", "description"]:
            assert col in df.columns

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            fetch_supplementary(source="invalid")  # type: ignore
