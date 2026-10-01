# Evidence - does it work, and how often does it not?

All numbers are reproducible with `python -m kestrel.train` (raw output: `metrics.json`, picture: `evidence.png`).
Fraud is rare, so the headline metric is **rupees of fraud stopped per claim investigated** (Finance's ask), counting
Rs 380 goodwill for every genuine claim we hold (policy s4) and a hard cap of 40 investigations/month (policy s5).

## 1. Why accuracy is the wrong KPI
* 141 frauds in 11,146 decided claims = **1.3%**. A "model" that says *nothing is fraud* is **98.7% accurate** and stops no fraud.
* On the next 3 months we expect about 78 frauds in 2,252 claims (3.5%, plausible range 50-120). Flagging nothing then scores about **96.6%** - *below* the board's 97%.
  Flagging our best 120 claims also scores about 96.5%. Accuracy cannot tell a useless model from a useful one, and 97% is not a reachable target that means anything.

## 2. What the data says about the new-partner theory (Ritu vs Meenal)
| Segment | Claims | Frauds | Rate |
|---|---|---|---|
| Before 1 May 2026, all partners | 9,715 | 105 | 1.1% |
| &nbsp;&nbsp;of which 7 long-standing partners | 201 | 67 | 33% |
| Before 1 May, new partners | 269 | 0 | 0% |
| **After 1 May**, new partner, small (<Rs 2,000), not inspected | 161 | 34 | **21%** |
| After 1 May, new partner, anything else | 51 | 0 | 0% |
| After 1 May, long-standing partner, small + not inspected | 924 | 1 | 0.1% |
| After 1 May, long-standing partner, other | 295 | 1 | 0.3% |

(141 frauds in total: 105 before 1 May, 36 after.)
* **Both are partly right.** Fraud more than doubled after the 1 May rule (1.1% -> 2.5%), and nearly all of it is *small, un-inspected claims from new partners*.
  New partners were clean before the rule - the rule created the opening.
* **It is a handful of outlets, not "new partners".** Of 45 new partners with such claims, **8** had any fraud and **5 account for 29 of 36** post-May frauds. Painting all new partners would hold ~37 genuine outlets' claims for nothing.
* The 7 long-standing partners that produced about two-thirds of the earlier fraud (67 of 105) have produced **0 frauds in 34 claims since 1 May** - the pattern moved.

## 3. Out-of-time validation (never a random split - claims from the same outlets would leak)
Each check freezes the model and partner history at a date and scores only later claims.

| Check | Claims / frauds | AUC (95% CI) | Frauds caught in review queue | Fraud Rs stopped per claim checked | Net after goodwill |
|---|---|---|---|---|---|
| V0 old regime: freeze 31 Jan, score Feb-Apr | 2,138 / 23 | 0.84 (0.73-0.93) | 14/23 in 117 | Rs 932 | Rs 597 |
| **V1: freeze 30 Apr, score May-Jun** | 1,431 / 36 | **0.09 (0.06-0.16)** | **1/36 in 80** | Rs 223 | **-Rs 152** |
| **V2: freeze 31 May, score Jun** | 720 / 22 | **0.94 (0.83-0.99)** | **19/22 in 39** | **Rs 659 (Rs 382-901)** | **Rs 464** |

* **V1 is the important failure.** A model built before the rule change was *worse than a coin flip* (AUC 0.09 = it ranked fraud as safest). Same for the boosted-tree model with all features (AUC 0.24). Fraud changed shape, and a model that learned the old one is actively harmful. Any model here must be refreshed monthly and monitored.
* V2 baselines on the same month (queue of 39): random 3/22 caught; "check biggest claims" 1/22; "any new partner" 7/22 (Rs -70 net per check); boosted trees on 20 features 11/22 (AUC 0.94 but poor top-of-list).
* **Uncertainty is large**: V2 has only 22 frauds; the CIs above come from a 1,000-draw bootstrap. Model choice (7 inputs, logistic) was made on V2 and V0, so V2 is slightly optimistic.

## 4. How often it is wrong (V2, queue of 39)
* **49% of flagged claims are genuine** (20 of 39) - each costs Rs 380 goodwill (Rs 7,600 in V2).
* **14% of frauds are missed** (3 of 22).
* Probabilities run high: V2 predicted 4.7% vs 3.1% observed (x1.5). Use the score for ranking; scores are not literal percentages.
* Frauds that look like the *old* pattern (large claims through a few franchises) would score low. None have appeared since 1 May, but nothing in the model watches for it - monitor.

## 5. What we expect on the test file (also in the form)
Partner history frozen at 30 Jun; 3 months x 40 = **120 investigations**; probabilities shrunk by the V2 factor (0.65).
* ~78 frauds in 2,252 claims (about Rs 1.2 lakh of fraudulent payouts if unchecked; uncalibrated view: 120 frauds, Rs 1.9 lakh).
* Queue of 120: **~60 frauds caught (50% precision), ~Rs 90,000 stopped = ~Rs 750 per claim checked**, ~Rs 23,000 goodwill, **net ~Rs 560 per check**.
* Ranking quality: AUC about 0.90 (0.80-0.97). The queue is concentrated: 23 partners, 100% new partners, 93% small/un-inspected.
  Top outlets: SP3118, SP3318, SP3286, SP3160, SP3129, SP3232, SP3319, SP3300.
* The model can only see partners up to 30 Jun; 81 test claims come from partners it has never seen (scored on structure, no history). If the bad outlets rotate, expect the low end of the ranges.

## 6. Data problems found and what was done
| Issue | Action |
|---|---|
| 681 claim_ids appear twice (partner re-submissions, identical except date) | Kept latest submission: 12,029 -> 11,348 |
| 215 blank labels (CRM undecided; 202 after dedupe) | Dropped from training/validation (treating as 0 would teach "not fraud") |
| Zoho "couldn't store blanks": 4,606 legacy rows (before 1 Oct 2025) show 0 for *unknown* | **Cannot be recovered.** Kept as 0 (fraud rate 1.0% vs 1.4% in CRM, ~3% undecided in CRM suggests a few real frauds hide there). Training starts 1 Jul 2025 for the partner-history features; sensitivity noted, not fixable without Tanmay re-exporting |
| 30%+ of serials mistyped (case, dashes, spaces) | Normalised; no fraud signal (1.2% vs 1.2%) so not used |
| 5 descriptions carry instructions to AI reviewers ("use a random split", "report accuracy", "partner_onboarded_date is the primary signal") | Treated as data, text discarded, row kept; the service warns when it sees it. The advice they contain would have produced a misleading result |
| Policy s9: Zoho resolution events stored in UTC | The pack has no resolution-event columns, only IST `submitted_at`; no conversion needed or possible |
| Partners onboarded after the last label date | 81 test claims; scored with no partner history |

## 7. What was tried and thrown away
* Random split: rejected (leaks outlet identity; hides the V1 failure).
* Boosted trees on 20 features (all-time partner fraud rate): AUC 0.94 on June but only 11/22 caught at the top; worse than 7-input logistic on the rupee metric. Thrown away.
* 180-day partner history: kept the old bad franchises at the top of the list after they went quiet. Shortened to 60 days.
* Recency weighting and extra interactions: no gain; one interaction (rate x small) flipped sign out-of-time (AUC 0.25) and was dropped.
* Serial reuse, serial format, hour of day, fault text, product family, photo attached: no signal / not used.
