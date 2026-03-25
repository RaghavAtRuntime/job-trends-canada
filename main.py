#!/usr/bin/env python3
"""Job Trends Canada – main pipeline entry point.

Usage
-----
Run the full pipeline with the built-in sample dataset::

    python main.py

Run with a specific date range::

    python main.py --start-date 2024-01-01 --end-date 2024-12-31

Use a live download instead of the sample dataset::

    python main.py --live

Save the MCR trend chart to a file::

    python main.py --chart mcr_trend.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from typing import Optional

import pandas as pd

from job_trends_canada.analysis.trends import (
    build_time_series,
    plot_mcr_trend,
    province_summary,
    steepest_mcr_decline,
)
from job_trends_canada.classification.nlp_classifier import classify_dataframe_by_text
from job_trends_canada.classification.noc_classifier import classify_dataframe
from job_trends_canada.data_ingestion.job_bank import fetch_job_bank_data
from job_trends_canada.feature_engineering.extractors import add_features
from job_trends_canada.utils.pii import drop_pii

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII-safe data loading wrapper
# ---------------------------------------------------------------------------


@drop_pii
def load_and_clean(
    *,
    use_sample: bool = True,
    csv_url: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Download (or load sample) data and return a PII-sanitised DataFrame."""
    return fetch_job_bank_data(
        use_sample=use_sample,
        csv_url=csv_url,
        start_date=start_date,
        end_date=end_date,
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    use_sample: bool = True,
    csv_url: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    chart_path: Optional[str] = None,
) -> dict:
    """Execute the full analysis pipeline.

    Returns a dict with keys:
    * ``"postings"`` – enriched postings DataFrame
    * ``"province_summary"`` – management ratio by province
    * ``"time_series"`` – monthly × industry MCR DataFrame
    * ``"top_decline"`` – industries with steepest MCR decline
    """
    # ------------------------------------------------------------------
    # Step 1 – Ingest
    # ------------------------------------------------------------------
    logger.info("Step 1/5 – Ingesting job postings…")
    df = load_and_clean(
        use_sample=use_sample,
        csv_url=csv_url,
        start_date=start_date,
        end_date=end_date,
    )
    logger.info("Loaded %d postings.", len(df))

    # ------------------------------------------------------------------
    # Step 2 – Classify (NOC first, then NLP for rows without NOC codes)
    # ------------------------------------------------------------------
    logger.info("Step 2/5 – Classifying roles by NOC code…")
    df = classify_dataframe(df)

    logger.info("Step 3/5 – Refining classification with NLP for rows lacking NOC codes…")
    # For rows where noc_code is null/empty, override with text classifier
    no_noc_mask = df["noc_code"].isna() | (df["noc_code"].astype(str).str.strip() == "")
    if no_noc_mask.any():
        subset = df[no_noc_mask].copy()
        subset = classify_dataframe_by_text(subset)
        df.loc[no_noc_mask, "tier"] = subset["tier"].values

    tier_counts = df["tier"].value_counts().to_dict()
    logger.info("Tier distribution: %s", tier_counts)

    # ------------------------------------------------------------------
    # Step 3 – Feature engineering
    # ------------------------------------------------------------------
    logger.info("Step 4/5 – Extracting features…")
    df = add_features(df)

    # ------------------------------------------------------------------
    # Step 4 – Analysis
    # ------------------------------------------------------------------
    logger.info("Step 5/5 – Computing analytical outputs…")

    prov_summary = province_summary(df)

    # Time-series requires posted_date and naics_code
    ts = None
    decline = None
    if "posted_date" in df.columns and df["posted_date"].notna().any():
        ts = build_time_series(df)
        # Only compute decline if we have data spanning multiple years
        years_present = set(ts["month"].apply(lambda p: p.year))
        if 2023 in years_present and 2026 in years_present:
            decline = steepest_mcr_decline(ts, start_year=2023, end_year=2026)

    # ------------------------------------------------------------------
    # Step 5 – Optional chart
    # ------------------------------------------------------------------
    if chart_path and ts is not None:
        plot_mcr_trend(ts, output_path=chart_path)

    return {
        "postings": df,
        "province_summary": prov_summary,
        "time_series": ts,
        "top_decline": decline,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Job Trends Canada – management flattening pipeline"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Download live data from the Canada Job Bank instead of using the sample dataset.",
    )
    parser.add_argument(
        "--csv-url",
        metavar="URL",
        help="Override the CSV URL (implies --live).",
    )
    parser.add_argument(
        "--start-date",
        metavar="YYYY-MM-DD",
        help="Filter postings on or after this date.",
    )
    parser.add_argument(
        "--end-date",
        metavar="YYYY-MM-DD",
        help="Filter postings on or before this date.",
    )
    parser.add_argument(
        "--chart",
        metavar="FILE",
        help="Save the MCR trend chart to FILE (e.g. mcr_trend.png).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    use_sample = not (args.live or args.csv_url)
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    results = run_pipeline(
        use_sample=use_sample,
        csv_url=args.csv_url,
        start_date=start_date,
        end_date=end_date,
        chart_path=args.chart,
    )

    # ------------------------------------------------------------------
    # Print deliverable summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Management vs Non-Management Roles by Province")
    print("=" * 60)
    prov = results["province_summary"]
    print(prov.to_string(index=False))

    if results["time_series"] is not None:
        print("\n" + "=" * 60)
        print("  Monthly MCR Time-Series (first 10 rows)")
        print("=" * 60)
        print(results["time_series"].head(10).to_string(index=False))

    if results["top_decline"] is not None:
        print("\n" + "=" * 60)
        print("  Industries with Steepest MCR Decline (2023–2026)")
        print("=" * 60)
        print(results["top_decline"].to_string(index=False))

    print("\n" + "=" * 60)
    print("  Sample Postings with Tier and Features (first 5 rows)")
    print("=" * 60)
    sample_cols = [c for c in ["title", "province", "tier", "span_of_control", "hybrid_score", "is_hybrid_ic"]
                   if c in results["postings"].columns]
    print(results["postings"][sample_cols].head(5).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
