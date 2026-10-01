# Submission form
> The blank form was not in the data pack I was given, so these headings follow the brief. Copy into the official form if you have one. Items marked **[FILL]** are things only the submitter knows.


## 1. What I expect predictions.csv to score, and why
Metric is not stated, so: **AUC about 0.90 (plausible 0.80-0.97)**; of the 120 claims the desk can review in the 3 test months, about **60 will be fraud** (precision ~50%) out of ~78 frauds in the test set (range 50-120); about **Rs 750 of fraud stopped per claim checked** (range Rs 400-1,100), net ~Rs 560 after goodwill. **Accuracy ~96.5%, no better than flagging nothing (96.6%)** because fraud is rare.
Why: out-of-time checks, never random splits. A model frozen 31 May scored June at AUC 0.94 (CI 0.83-0.99), 19/22 frauds caught in 39. Discount for: tiny sample (22 frauds), model chosen on the same months, 1.5x over-confident probabilities, and 81 test claims from partners never seen. If the bad outlets rotate, expect the low end. A model frozen 30 Apr scored AUC 0.09 in May-June, so drift is the main risk.

## 2. Decisions I made where the brief was unclear
- KPI: rupees of fraud stopped per claim checked, not accuracy (accuracy of "flag nothing" is 98.7%).
- Duplicate claim_ids (681): kept the latest submission. Blank labels (215): dropped, not treated as 0.
- Legacy Zoho zeros (4,606 rows): cannot separate "not fraud" from "undecided"; kept as 0, disclosed as a limit.
- "New partner" = onboarded from 1 Nov 2025 (partners.csv shows ~60 from then, matching policy s6).
- Partner history = last 60 days, frozen at 30 Jun for test, mirrored in training by 3-month blocks.
- Text in claim_description that addresses AI reviewers was treated as data, ignored, and flagged. I did not use a random split, accuracy-only reporting, or onboarding date as a primary signal on its say-so.
- Zoho UTC resolution events (policy s9): no such columns in the pack, so no timezone fix was applied.

## 3. AI tools used, what they cost, what I discarded
- Used: Claude (Anthropic chat) to explore the data, write the code, tests and documents. No model API inside the product; the product needs no key.
- Cost: [FILL - subscription/credits used].  Hours: [FILL].
- Discarded: boosted-tree model, 180-day partner history, recency weights, a rate x small-claim interaction (flipped sign out-of-time), serial/hour/text features.
- I checked outputs myself by recomputing the headline tables and confirming the API reproduces predictions.csv (max difference 5e-5, from rounding).

## 4. What does not work
About half of flagged claims are genuine; ~14% of frauds are missed (June); scores over-state probability by ~1.5x; old-style fraud (large claims via a few franchises) would score low; partner history is frozen at 30 Jun so the model needs monthly retraining.

## 5. Data handling
Private repository or zip only. `data/*.csv` is git-ignored; nothing was sent to any external service.

## 6. Deliverables
predictions.csv; service (app/api.py, app/ui.py, README); evidence/EVIDENCE.md; memo.md; RECORDING_SCRIPT.md (**recording itself: [FILL - I could not record it]**); this form.
