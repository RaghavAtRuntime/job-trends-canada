"""Analytical layer – Management Concentration Ratio (MCR) and trend analysis.

Key deliverables
----------------
* :func:`build_time_series` – produces a monthly × industry DataFrame with MCR.
* :func:`steepest_mcr_decline` – ranks industries by MCR decline between two
  calendar years.
* :func:`province_summary` – summary table of management vs non-management
  counts and ratio by province (initial deliverable from the spec).
* :func:`plot_mcr_trend` – line chart of MCR over time per industry.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Tier labels that count as "Management"
_MANAGEMENT_TIERS = {"Middle Management", "Executive"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a monthly × industry DataFrame with MCR and posting counts.

    Parameters
    ----------
    df:
        Postings DataFrame.  Must have columns: ``posted_date`` (datetime),
        ``naics_code``, ``tier``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(month, naics_code)`` with columns:

        ``total_postings``
            Total job postings in that month/industry cell.
        ``management_postings``
            Number of Management or Executive postings.
        ``mcr``
            Management Concentration Ratio = management / total.
    """
    required = {"posted_date", "naics_code", "tier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    work = df.copy()
    work["month"] = pd.to_datetime(work["posted_date"]).dt.to_period("M")
    work["is_management"] = work["tier"].isin(_MANAGEMENT_TIERS)
    work["naics_code"] = work["naics_code"].fillna("Unknown")

    grouped = (
        work.groupby(["month", "naics_code"])
        .agg(
            total_postings=("is_management", "count"),
            management_postings=("is_management", "sum"),
        )
        .reset_index()
    )
    grouped["mcr"] = grouped["management_postings"] / grouped["total_postings"]
    grouped = grouped.sort_values(["month", "naics_code"]).reset_index(drop=True)
    return grouped


def steepest_mcr_decline(
    time_series: pd.DataFrame,
    *,
    start_year: int = 2023,
    end_year: int = 2026,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the industries with the steepest MCR decline between two years.

    Parameters
    ----------
    time_series:
        Output of :func:`build_time_series`.
    start_year:
        Reference year for the baseline MCR.
    end_year:
        Target year for the comparison MCR.
    top_n:
        Return only the *top_n* industries with the largest decline.

    Returns
    -------
    pandas.DataFrame
        Columns: ``naics_code``, ``mcr_start``, ``mcr_end``, ``mcr_change``
        (negative = decline), sorted by ``mcr_change`` ascending.
    """
    ts = time_series.copy()
    ts["year"] = ts["month"].dt.year if hasattr(ts["month"], "dt") else ts["month"].apply(lambda p: p.year)

    def _year_avg(year: int) -> pd.DataFrame:
        subset = ts[ts["year"] == year]
        return (
            subset.groupby("naics_code")["mcr"]
            .mean()
            .rename(f"mcr_{year}")
            .reset_index()
        )

    start_df = _year_avg(start_year).rename(columns={f"mcr_{start_year}": "mcr_start"})
    end_df = _year_avg(end_year).rename(columns={f"mcr_{end_year}": "mcr_end"})

    merged = start_df.merge(end_df, on="naics_code", how="inner")
    merged["mcr_change"] = merged["mcr_end"] - merged["mcr_start"]
    merged = merged.sort_values("mcr_change").head(top_n).reset_index(drop=True)
    return merged


def province_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of management vs non-management roles by province.

    This is the **initial deliverable** described in the problem statement:
    a clean summary showing the ratio of Management vs Non-Management roles
    per Canadian province.

    Parameters
    ----------
    df:
        Postings DataFrame.  Must have columns ``province`` and ``tier``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``province``, ``management``, ``non_management``,
        ``total``, ``management_ratio``.
    """
    required = {"province", "tier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    work = df.copy()
    work["is_management"] = work["tier"].isin(_MANAGEMENT_TIERS)
    work["province"] = work["province"].fillna("Unknown")

    grouped = (
        work.groupby("province")["is_management"]
        .agg(
            management="sum",
            total="count",
        )
        .reset_index()
    )
    grouped["non_management"] = grouped["total"] - grouped["management"]
    grouped["management_ratio"] = grouped["management"] / grouped["total"]
    grouped = grouped[["province", "management", "non_management", "total", "management_ratio"]]
    grouped = grouped.sort_values("management_ratio", ascending=False).reset_index(drop=True)
    return grouped


def plot_mcr_trend(
    time_series: pd.DataFrame,
    *,
    naics_codes: Optional[list[str]] = None,
    title: str = "Management Concentration Ratio (MCR) Over Time",
    output_path: Optional[str] = None,
) -> None:
    """Plot the MCR trend over time, optionally filtered to specific industries.

    Parameters
    ----------
    time_series:
        Output of :func:`build_time_series`.
    naics_codes:
        List of NAICS codes to include.  When ``None``, all codes are plotted.
    title:
        Chart title.
    output_path:
        If given, save the figure to this file path (PNG/SVG/PDF determined by
        extension).  Otherwise the plot is displayed interactively.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("matplotlib and seaborn are required for plotting.") from exc

    ts = time_series.copy()
    # Convert Period to timestamp for matplotlib
    if hasattr(ts["month"].iloc[0], "to_timestamp"):
        ts["month"] = ts["month"].apply(lambda p: p.to_timestamp())

    if naics_codes:
        ts = ts[ts["naics_code"].isin(naics_codes)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=ts,
        x="month",
        y="mcr",
        hue="naics_code",
        marker="o",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Management Concentration Ratio")
    ax.set_ylim(0, 1)
    ax.legend(title="NAICS Code", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        logger.info("Saved MCR trend chart to %s", output_path)
    else:
        plt.show()
    plt.close(fig)
