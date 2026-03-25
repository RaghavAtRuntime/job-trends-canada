"""Tests for the classification modules."""

import pandas as pd
import pytest

from job_trends_canada.classification.noc_classifier import (
    classify_by_noc,
    classify_dataframe,
)
from job_trends_canada.classification.nlp_classifier import (
    classify_by_text,
    classify_dataframe_by_text,
)


# ---------------------------------------------------------------------------
# NOC classifier
# ---------------------------------------------------------------------------


class TestNocClassifier:
    def test_executive_noc(self):
        assert classify_by_noc("00014") == "Executive"

    def test_middle_management_noc(self):
        assert classify_by_noc("01012") == "Middle Management"

    def test_ic_noc_default(self):
        result = classify_by_noc("21232", title="Developer")
        assert result in ("Entry/Junior IC", "Senior/Principal IC")

    def test_senior_ic_from_title(self):
        assert classify_by_noc("21232", title="Senior Software Engineer") == "Senior/Principal IC"

    def test_junior_ic_from_title(self):
        assert classify_by_noc("21232", title="Junior Developer") == "Entry/Junior IC"

    def test_nan_noc_falls_back_to_title(self):
        import math
        result = classify_by_noc(float("nan"), title="Principal Architect")
        assert result == "Senior/Principal IC"

    def test_none_noc(self):
        result = classify_by_noc(None, title="Intern")
        assert result == "Entry/Junior IC"

    def test_classify_dataframe(self):
        df = pd.DataFrame(
            {
                "noc_code": ["00014", "21232", None],
                "title": ["VP Sales", "Developer", "Senior Data Scientist"],
            }
        )
        result = classify_dataframe(df)
        assert "tier" in result.columns
        assert result.loc[0, "tier"] == "Executive"
        assert result.loc[2, "tier"] == "Senior/Principal IC"


# ---------------------------------------------------------------------------
# NLP classifier
# ---------------------------------------------------------------------------


class TestNlpClassifier:
    def test_direct_reports_is_management(self):
        tier = classify_by_text("Team Lead", "You will have 4 direct reports.")
        assert tier == "Middle Management"

    def test_budgetary_responsibility_is_management(self):
        tier = classify_by_text(
            "Operations Lead",
            "Budgetary responsibility of $1M. Reports to the Director.",
        )
        assert tier == "Middle Management"

    def test_pl_is_management(self):
        tier = classify_by_text("Director", "Full P&L responsibility for the region.")
        assert tier in ("Middle Management", "Executive")

    def test_no_signals_is_ic(self):
        tier = classify_by_text("Software Engineer", "Develops backend APIs in Python.")
        assert tier in ("Entry/Junior IC", "Senior/Principal IC")

    def test_vp_title_is_executive(self):
        tier = classify_by_text(
            "Vice President Engineering",
            "P&L responsibility. Reports to the board of directors.",
        )
        assert tier == "Executive"

    def test_classify_dataframe_by_text_no_noc(self):
        df = pd.DataFrame(
            {
                "title": ["Engineering Manager", "Software Developer"],
                "description": [
                    "Direct reports: 5. Budgetary responsibility of $500K.",
                    "Builds microservices using Go.",
                ],
                "noc_code": [None, None],
            }
        )
        result = classify_dataframe_by_text(df)
        assert result.loc[0, "tier"] == "Middle Management"
        assert result.loc[1, "tier"] in ("Entry/Junior IC", "Senior/Principal IC")
