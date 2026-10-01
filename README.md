# Kestrel Home — Warranty Claim Fraud Scorer

> **A production-ready fraud scoring service for Kestrel's warranty investigation desk.**  
> Scores one claim at a time, explains every decision in plain English, and requires no API key, no internet connection, and no external model service.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Run the API Server](#2-run-the-api-server)
  - [3. Run the Streamlit UI](#3-run-the-streamlit-ui)
- [API Reference](#api-reference)
  - [GET /health](#get-health)
  - [POST /score](#post-score)
- [Reproduce Training & Evidence](#reproduce-training--evidence)
- [Model Design](#model-design)
- [Performance Summary](#performance-summary)
- [Known Limitations](#known-limitations)
- [Data Handling & Privacy](#data-handling--privacy)

---

## Overview

The Kestrel investigation desk reviews up to **40 warranty claims per month**. Manually triaging thousands of incoming claims is impractical, so this tool ranks each claim by its fraud probability and explains why — helping investigators focus effort where it is most likely to pay off.

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Logistic regression (7 inputs) | With only ~145 confirmed frauds, a small interpretable model outperformed boosted trees on every out-of-time check |
| Rupees stopped per claim checked (not accuracy) | Accuracy of "flag nothing" is already 98.7%; it cannot distinguish a useful model from a useless one |
| 60-day rolling partner history | The fraud pattern shifted sharply in May 2026; longer history misleads the model |
| Out-of-time validation (never random splits) | Random splits leak outlet identity and hide the V1 failure (AUC 0.09 after May 2026) |
| No LLM, no external API | The service needs no key and works fully offline |

---

## Architecture

```
┌─────────────────────────────────────────┐
│          Streamlit UI  (port 8501)       │
│          app/ui.py                       │
│  Analyst fills in claim fields           │
│  and sees: score, risk band, reasons     │
└──────────────────┬──────────────────────┘
                   │  HTTP POST /score
┌──────────────────▼──────────────────────┐
│          FastAPI Service  (port 8000)    │
│          app/api.py                      │
│  Validates input via Pydantic            │
│  Calls kestrel.scoring.Scorer            │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         kestrel/scoring.py               │
│  Loads artifacts/model.json (no pickle) │
│  Loads artifacts/reference.json         │
│  Computes logistic score                 │
│  Generates human-readable reasons       │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┴───────────┐
       │ artifacts/model.json  │  ← 7-input logistic regression weights
       │ artifacts/reference.json │  ← partner profiles, product catalogue
       └───────────────────────┘
```

---

## Project Structure

```
Kestrelhome_submission/
│
├── app/
│   ├── api.py              # FastAPI endpoint — uvicorn app.api:app --port 8000
│   └── ui.py               # Streamlit analyst screen
│
├── kestrel/
│   ├── features.py         # Data cleaning, feature engineering (shared by training & serving)
│   ├── scoring.py          # Loads artifacts, scores one claim, writes reasons
│   ├── train.py            # Out-of-time validation, baselines, final fit, predictions
│   └── metrics.py          # Rupee-based queue metrics (fraud Rs stopped, goodwill cost)
│
├── artifacts/
│   ├── model.json          # Trained model weights (plain JSON — no sklearn needed to serve)
│   └── reference.json      # Partner profiles, product catalogue, queue threshold
│
├── evidence/
│   ├── EVIDENCE.md         # Full validation narrative with tables and commentary
│   ├── metrics.json        # Machine-readable validation results
│   └── evidence.png        # Charts: fraud rate over time, AUC by fold, net Rs vs queue size
│
├── tests/
│   └── test_api.py         # API integration tests (pytest)
│
├── data/                   # Raw CSVs — git-ignored, not distributed
│   └── README.txt
│
├── predictions.csv         # Final test-set scores (claim_id, score)
├── requirements.txt        # All Python dependencies
├── submission-form.md      # Submission answers
├── memo.md                 # Business context and notes
├── EVIDENCE.md             # Top-level copy of evidence narrative
└── RECORDING_SCRIPT.md     # Script for the demo recording
```

---

## Prerequisites

- **Python 3.10 or later**
- No other system dependencies required

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/riteshpandey2024-cyber/Kestrelhome_submission.git
cd Kestrelhome_submission

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

> **Note:** The trained artifacts are already in `artifacts/`, so no training step is required to run the service.

---

### 2. Run the API Server

Open a terminal and run:

```bash
uvicorn app.api:app --port 8000
```

Verify it is working:

```bash
curl -s localhost:8000/health
```

Expected output:

```json
{
  "status": "ok",
  "model": "kestrel-fraud-lr-2026-10-01",
  "partner_history_as_of": "2026-06-30"
}
```

If you see `"status": "degraded"`, the model artifacts are missing — run `python -m kestrel.train` (see [Reproduce Training](#reproduce-training--evidence)).

---

### 3. Run the Streamlit UI

Open a **second** terminal (keep the API running in the first):

```bash
streamlit run app/ui.py
```

This opens the analyst dashboard at **http://localhost:8501**.

From there you can:
- Select a preset example claim
- Fill in claim fields manually
- Click **Check claim** to see the fraud score, risk band, and plain-English reasons

---

## API Reference

Interactive Swagger docs: **http://localhost:8000/docs**

### GET `/health`

Returns model status and the partner-history freeze date.

```bash
curl -s localhost:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "model": "kestrel-fraud-lr-2026-10-01",
  "partner_history_as_of": "2026-06-30"
}
```

---

### POST `/score`

Scores one warranty claim and returns a full explanation.

**Request body (JSON):**

| Field | Type | Required | Description |
|---|---|---|---|
| `submitted_at` | string | ✅ | IST timestamp — `"YYYY-MM-DD HH:MM"` |
| `partner_id` | string | ✅ | e.g. `"SP3160"` |
| `sku` | string | ✅ | e.g. `"KH-AF-02"` |
| `claim_amount_inr` | number | ✅ | Claim value in rupees (> 0) |
| `partner_inspected` | string | ✅ | `"Y"` or `"N"` |
| `days_since_purchase` | integer | | Days since product purchase (default 0) |
| `photo_attached` | string | | `"Y"` or `"N"` (default `"N"`) |
| `product_serial` | string | | Product serial number |
| `claim_description` | string | | Fault description text |
| `inspector_note` | string | | Inspector's note (if any) |
| `customer_prior_claims` | integer | | Number of prior claims by this customer (default 0) |
| `claim_id` | string | | Optional — returned as-is in the response |

**Example request:**

```bash
curl -s -X POST localhost:8000/score \
  -H 'content-type: application/json' \
  -d '{
    "submitted_at": "2026-09-12 14:05",
    "partner_id": "SP3160",
    "sku": "KH-AF-02",
    "product_serial": "kh-662573314",
    "days_since_purchase": 140,
    "claim_amount_inr": 1450,
    "photo_attached": "Y",
    "partner_inspected": "N",
    "claim_description": "motor not running",
    "customer_prior_claims": 1
  }'
```

**Response fields:**

| Field | Description |
|---|---|
| `fraud_score` | Probability of fraud (0–1). Use for **ranking**, not as a literal percentage |
| `risk_band` | `HIGH` (investigation queue), `WATCH` (monitor partner), or `LOW` (pay normally) |
| `expected_net_inr_if_held` | Expected rupees gained by holding this claim (fraud saved minus goodwill cost) |
| `recommendation` | Plain-English action for the desk |
| `reasons[]` | Ranked list of factors with direction (raises / lowers) and log-odds contribution |
| `data_warnings[]` | Unknown partner, non-standard serial format, embedded instructions in description |
| `model` | Version, type, partner history date, and queue threshold |
| `caveat` | Reminder that the score is a ranking aid, not a verdict |

---

## Reproduce Training & Evidence

> **Requires the raw data files.** Place the following CSVs in the `data/` directory:
> `train.csv`, `test_unlabelled.csv`, `partners.csv`, `products.csv`, `sample_submission.csv`

```bash
# Retrain the model and regenerate all artifacts, predictions, and evidence
python -m kestrel.train

# Run the API integration tests
python -m pytest -q tests
```

**Outputs generated:**

| File | Description |
|---|---|
| `predictions.csv` | Fraud score for every claim in `test_unlabelled.csv` |
| `artifacts/model.json` | Updated model weights |
| `artifacts/reference.json` | Updated partner profiles and queue threshold |
| `evidence/metrics.json` | Full validation results (data facts, V0/V1/V2 checks, test expectations) |
| `evidence/evidence.png` | Fraud-rate timeline, AUC by fold, net-rupees vs queue-size chart |

---

## Model Design

The model is a **logistic regression with 7 inputs**, stored as plain JSON. No scikit-learn is needed to run the service — only `numpy` and `pandas`.

### Features used

| Feature | What it captures |
|---|---|
| `small_uninspected` | Claim < Rs 2,000 and not partner-inspected (the auto-approval loophole) |
| `is_new_partner` | Partner onboarded on or after 1 Nov 2025 |
| `new_small_unins` | Interaction: new partner × small uninspected claim |
| `prior_claims` | Number of previous warranty claims by this customer |
| `p_rate` | Partner's fraud rate in the last 60 days (shrunk toward the 1.2% base rate) |
| `p_small_unins_share` | Share of partner's recent claims that are small and uninspected |
| `log_amt` | Log of the claim amount |

### Features considered but not used

- Serial number format, serial reuse, hour of submission, fault text embedding
- 180-day partner history (kept old-regime bad actors at the top after they went quiet)
- Boosted trees on 20 features (better AUC but fewer frauds caught at the top of the queue on the rupee metric)
- A rate × small-claim interaction (sign flipped out-of-time; AUC 0.25 on V1)

---

## Performance Summary

Validation is always **out-of-time** (model frozen at date D, scored on claims after D). Random splits are not used because they leak outlet identity and hide catastrophic drift.

| Check | Period scored | AUC | Frauds caught (queue) | Net Rs / claim checked |
|---|---|---|---|---|
| V0 — old regime | Feb–Apr 2026 | 0.84 (CI 0.73–0.93) | 14 / 23 in 117 | +Rs 597 |
| **V1 — after May rule** | May–Jun 2026 | **0.09** ❌ | 1 / 36 in 80 | **−Rs 152** |
| **V2 — retrained** | Jun 2026 | **0.94** ✅ | **19 / 22 in 39** | **+Rs 464** |

> **V1 is the critical lesson.** A model trained before the May 2026 auto-approval policy change was worse than a coin flip (AUC 0.09). **The model must be retrained monthly.** See [`evidence/EVIDENCE.md`](evidence/EVIDENCE.md) for the full analysis.

**Expected performance on the test set (Jul–Sep 2026, 120 investigations):**

- ~60 frauds caught out of ~78 (precision ~50%)
- ~Rs 750 of fraud stopped per claim checked; ~Rs 560 net after goodwill
- AUC ~0.90 (plausible range 0.80–0.97)

---

## Known Limitations

- **Partner history is frozen at 30 Jun 2026.** 81 test claims come from partners the model has never seen (scored on claim structure only). Retrain monthly.
- **Scores over-state probability by ~1.5x.** V2 predicted 4.7% vs 3.1% observed. Use the score for ranking; it is not a calibrated probability.
- **~49% of flagged claims are genuine** (false positives). Each costs Rs 380 in goodwill (Rs 7,600 total in the V2 window).
- **~14% of frauds are missed** (false negatives) based on the June 2026 validation.
- **Old-regime fraud (large claims through a few established franchises) would score low.** No such claims have appeared since May 2026, but nothing in the model actively monitors for a recurrence.
- **Legacy Zoho data (pre-Oct 2025)** has `0` for all blank fields; it is unknown whether those zeros represent "not fraud" or "undecided". This cannot be corrected without a re-export.

---

## Data Handling & Privacy

- `data/*.csv` is **git-ignored**. Raw claim data is never committed to version control.
- `artifacts/reference.json` is derived from partner and product data. **Keep this repository private** (policy s10).
- No claim data was sent to any external service during development or in production.
- No LLM or external API is used inside the product.

---

