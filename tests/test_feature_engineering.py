"""Tests for feature engineering extractors."""

import pandas as pd
import pytest

from job_trends_canada.feature_engineering.extractors import (
    add_features,
    extract_hybrid_score,
    extract_span_of_control,
    is_hybrid_ic,
)


class TestSpanOfControl:
    def test_team_of_n(self):
        assert extract_span_of_control("You will be managing a team of 5 engineers.") == 5

    def test_direct_reports(self):
        assert extract_span_of_control("The role has 8 direct reports.") == 8

    def test_multiple_mentions_returns_max(self):
        desc = "Lead a team of 3 analysts. Also mentors a group of 10 interns."
        assert extract_span_of_control(desc) == 10

    def test_no_mention_returns_none(self):
        assert extract_span_of_control("Develops REST APIs and writes unit tests.") is None

    def test_none_input(self):
        assert extract_span_of_control(None) is None


class TestHybridScore:
    def test_mentoring_increments_score(self):
        score = extract_hybrid_score("Staff Engineer", "Responsible for mentoring junior devs.")
        assert score > 0

    def test_hiring_increments_score(self):
        score = extract_hybrid_score("Principal Engineer", "Involved in hiring decisions.")
        assert score > 0

    def test_no_signals_zero(self):
        score = extract_hybrid_score("Backend Developer", "Builds APIs with Django.")
        assert score == 0.0


class TestIsHybridIc:
    def test_hybrid_senior_without_manager_title(self):
        result = is_hybrid_ic(
            "Principal Engineer",
            "You will be mentoring junior developers, involved in hiring, and defining strategy.",
        )
        assert result is True

    def test_manager_title_not_hybrid(self):
        result = is_hybrid_ic(
            "Engineering Manager",
            "Mentoring, hiring, strategy, roadmap.",
        )
        assert result is False  # explicit manager title → not hybrid

    def test_low_signals_not_hybrid(self):
        result = is_hybrid_ic("Developer", "Writes code.")
        assert result is False


class TestAddFeatures:
    def test_columns_added(self):
        df = pd.DataFrame(
            {
                "title": ["Senior Engineer", "Manager"],
                "description": [
                    "Mentoring and hiring involvement. Managing a team of 3.",
                    "Direct reports: 5. Budget responsibility.",
                ],
            }
        )
        result = add_features(df)
        assert "span_of_control" in result.columns
        assert "hybrid_score" in result.columns
        assert "is_hybrid_ic" in result.columns

    def test_span_extracted_correctly(self):
        df = pd.DataFrame(
            {"title": ["Lead"], "description": ["Team of 7 engineers."]}
        )
        result = add_features(df)
        assert result["span_of_control"].iloc[0] == 7

    def test_missing_description_handled(self):
        df = pd.DataFrame({"title": ["Developer"], "description": [None]})
        result = add_features(df)
        assert result["span_of_control"].iloc[0] is None
        assert result["hybrid_score"].iloc[0] == 0.0
