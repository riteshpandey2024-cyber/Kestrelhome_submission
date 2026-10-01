# Kestrel Home - warranty claim fraud scorer

Scores **one** warranty claim and explains why, for the 40-claims-a-month investigation desk.
No paid API, no API key, no network calls. The model is a 7-input logistic regression stored as plain JSON.

> **Client data.** `artifacts/reference.json` is derived from Kestrel's partner/claim data and `data/` holds the raw CSVs.
> Keep this repo **private** (policy s10). `data/*.csv` is git-ignored.

## Run it (clean machine, Python 3.10+)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# terminal 1 - the endpoint
uvicorn app.api:app --port 8000
# terminal 2 - the screen
streamlit run app/ui.py                                   # opens http://localhost:8501
```

The trained artifacts are in `artifacts/`, so nothing else is needed. If the API is down the screen says so politely;
if artifacts are missing `/health` reports `degraded` and `/score` returns 503 with the fix.

Try the endpoint directly:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d '{
 "submitted_at":"2026-09-12 14:05","partner_id":"SP3160","sku":"KH-AF-02","product_serial":"kh-662573314",
 "days_since_purchase":140,"claim_amount_inr":1450,"photo_attached":"Y","partner_inspected":"N",
 "claim_description":"motor not running","customer_prior_claims":1}'
```

Returns `fraud_score`, `risk_band` (HIGH = would make this quarter's 40-a-month queue), `expected_net_inr_if_held`,
a plain-English `recommendation`, `reasons[]`, `data_warnings[]` (unknown partner, odd serial, junk text in the description) and a `caveat`.
Interactive docs: http://localhost:8000/docs

## Reproduce training, predictions and evidence

Put `train.csv, test_unlabelled.csv, partners.csv, products.csv, sample_submission.csv` in `data/`, then:

```bash
python -m kestrel.train        # -> predictions.csv, artifacts/, evidence/metrics.json, evidence/evidence.png
python -m pytest -q tests      # API tests
```

## Layout

```
kestrel/features.py   cleaning (dedupe, serial/description clean-up), partner profile, features - shared by training and serving
kestrel/train.py      out-of-time validation, baselines, bootstrap CIs, final fit, predictions, evidence
kestrel/scoring.py    loads artifacts, scores one claim, writes the reasons
kestrel/metrics.py    review-queue rupee metrics (fraud Rs stopped, goodwill Rs 380/held genuine claim)
app/api.py            FastAPI endpoint   app/ui.py   Streamlit screen
evidence/             EVIDENCE.md (read this), metrics.json, evidence.png
memo.md  submission-form.md  RECORDING_SCRIPT.md  predictions.csv
```

## Things you should know
* Partner history in the service is **frozen at 30 Jun 2026** (the last labelled day). Retrain monthly; see EVIDENCE.md for why this matters.
* Text in `claim_description` is reduced to the fault wording. A few training rows contain instructions aimed at automated reviewers - they are ignored and surfaced as a warning.
* No LLM is used in the product, so there is nothing to fail without a key.
