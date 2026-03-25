# job-trends-canada

A modular Python pipeline that analyses whether a **"middle management flattening"** trend exists in the Canadian job market by comparing the ratio of management roles to individual contributor (IC) roles over time.

---

## Features

| Module | Description |
|---|---|
| **Data Ingestion** | Downloads monthly job-posting archives from the [Canada Job Bank](https://www.jobbank.gc.ca/) (Open Government Portal) or returns a built-in synthetic sample for offline development |
| **Supplementary Scraper** | Mockable scraping utility for Eluta.ca (mock mode ships out of the box; no network needed) |
| **Classification Engine** | 3-tier hierarchy (**Entry/Junior IC · Senior/Principal IC · Middle Management · Executive**) using NOC 2021 codes first, NLP/regex heuristics for postings without codes |
| **Feature Engineering** | Extracts *span-of-control* proxies (team sizes) and *hybridisation* signals (IC roles with management duties) |
| **Analytical Output** | Monthly × industry MCR time-series, province-level summary, steepest-decline ranking, and Matplotlib/Seaborn trend chart |
| **PII Protection** | `@drop_pii` decorator strips recruiter names, e-mails, and phone numbers from every DataFrame before storage |

---

## Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Run the pipeline (offline sample data)

```bash
python main.py
```

### 3 · Run with a date range

```bash
python main.py --start-date 2024-01-01 --end-date 2024-06-30
```

### 4 · Download live data from Canada Job Bank

```bash
python main.py --live
```

### 5 · Save the MCR trend chart

```bash
python main.py --chart mcr_trend.png
```

---

## Project Structure

```
job_trends_canada/
├── data_ingestion/
│   ├── job_bank.py          # Canada Job Bank CSV fetcher & normaliser
│   └── scraper.py           # Mockable Eluta.ca / Indeed scraper
├── classification/
│   ├── noc_classifier.py    # NOC 2021 code → tier mapping
│   └── nlp_classifier.py    # Regex/NLP heuristics for free-text descriptions
├── feature_engineering/
│   └── extractors.py        # Span-of-control & hybridisation signal extraction
├── analysis/
│   └── trends.py            # MCR computation, province summary, trend charts
└── utils/
    └── pii.py               # @drop_pii decorator and sanitise_dataframe helper

tests/
├── test_pii.py
├── test_classification.py
├── test_feature_engineering.py
├── test_analysis.py
└── test_ingestion.py

main.py                      # CLI entry point
requirements.txt
```

---

## Classification Logic

### NOC 2021 (priority)

| NOC prefix | Tier |
|---|---|
| `00` | Executive |
| `01`–`09` | Middle Management |
| Any other | IC (seniority from title keywords) |

### NLP / regex fallback (no NOC code)

Management signals scored and summed:

- "Direct reports" → 1.0
- "Budgetary responsibility" → 1.0
- "P&L" → 1.0
- "manages a team of N" → 1.0
- "hiring responsibility" → 0.75
- "team of N" → 0.75
- "Report to" → 0.5

Executive signals override when score ≥ 2.0 (e.g. "Vice President", "CFO", "board of directors").

---

## Key Metrics

**Management Concentration Ratio (MCR)**

```
MCR = Management_Postings / Total_Postings
```

Computed monthly per NAICS industry code. A declining MCR signals management-layer flattening.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Requirements

- Python 3.10+
- pandas, requests, beautifulsoup4, lxml
- scikit-learn (optional — for `build_sklearn_classifier`)
- matplotlib, seaborn
