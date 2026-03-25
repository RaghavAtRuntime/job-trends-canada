"""Tests for the analysis / trend module."""

import pandas as pd
import pytest

from job_trends_canada.analysis.trends import (
    build_time_series,
    province_summary,
    steepest_mcr_decline,
)


@pytest.fixture()
def sample_postings():
    """Minimal postings DataFrame with required columns."""
    data = {
        "posted_date": pd.to_datetime(
            [
                "2024-01-10",
                "2024-01-15",
                "2024-02-01",
                "2024-02-20",
                "2024-03-05",
                "2024-03-18",
            ]
        ),
        "naics_code": ["511210", "511210", "522110", "522110", "511210", "522110"],
        "province": ["ON", "ON", "BC", "BC", "AB", "AB"],
        "tier": [
            "Middle Management",
            "Entry/Junior IC",
            "Executive",
            "Senior/Principal IC",
            "Middle Management",
            "Entry/Junior IC",
        ],
    }
    return pd.DataFrame(data)


class TestBuildTimeSeries:
    def test_output_columns(self, sample_postings):
        ts = build_time_series(sample_postings)
        for col in ["month", "naics_code", "total_postings", "management_postings", "mcr"]:
            assert col in ts.columns

    def test_mcr_between_zero_and_one(self, sample_postings):
        ts = build_time_series(sample_postings)
        assert (ts["mcr"] >= 0).all()
        assert (ts["mcr"] <= 1).all()

    def test_management_count_correct(self, sample_postings):
        ts = build_time_series(sample_postings)
        # Jan 2024, naics 511210: 2 postings (1 Management, 1 IC) → MCR 0.5
        jan_511 = ts[(ts["month"].astype(str) == "2024-01") & (ts["naics_code"] == "511210")]
        assert jan_511["management_postings"].iloc[0] == 1
        assert jan_511["total_postings"].iloc[0] == 2

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            build_time_series(pd.DataFrame({"title": ["Dev"]}))


class TestProvinceSummary:
    def test_output_columns(self, sample_postings):
        summary = province_summary(sample_postings)
        for col in ["province", "management", "non_management", "total", "management_ratio"]:
            assert col in summary.columns

    def test_ratio_correct(self, sample_postings):
        summary = province_summary(sample_postings)
        # ON: 2 postings, 1 management → ratio 0.5
        on_row = summary[summary["province"] == "ON"]
        assert abs(on_row["management_ratio"].iloc[0] - 0.5) < 1e-9

    def test_missing_columns_raises(self, sample_postings):
        with pytest.raises(ValueError, match="missing required columns"):
            province_summary(sample_postings.drop(columns=["tier"]))


class TestSteepestMcrDecline:
    def test_returns_dataframe(self):
        """Only checks that the function runs and returns a DF when years are present."""
        data = []
        for year in [2023, 2026]:
            for month in range(1, 3):
                period = pd.Period(f"{year}-0{month}", freq="M")
                data.append(
                    {
                        "month": period,
                        "naics_code": "511210",
                        "total_postings": 10,
                        "management_postings": 3 if year == 2023 else 1,
                        "mcr": 0.3 if year == 2023 else 0.1,
                    }
                )
        ts = pd.DataFrame(data)
        result = steepest_mcr_decline(ts, start_year=2023, end_year=2026)
        assert isinstance(result, pd.DataFrame)
        assert "mcr_change" in result.columns
        # A decline should be negative
        assert result["mcr_change"].iloc[0] < 0
