# Screen recording script (<= 3:00, no slides)  -  I could not record this myself; read it over your screen.

**0:00-0:25  The brief.** Show email-thread.txt and ops-policy.pdf s4/s5. "Board wants >97% accuracy. Policy: 40 reviews/month, Rs 380 per genuine claim held, 1 May rule auto-approves < Rs 2,000."
**0:25-0:55  First look at the data.** Terminal: 12,029 rows -> 11,348 after duplicates; 202 blank labels dropped; 1.3% fraud. "Flag nothing = 98.7% accurate." Show the 5 descriptions with instructions to AI reviewers; "I ignored them".
**0:55-1:35  What I found.** Open evidence/evidence.png. Fraud by month jumps after May; new-partner x small x uninspected = 21%; 5 outlets = 29 of 36; old franchises went quiet.
**1:35-2:10  What I tried and threw away.** Random split (leaks), boosted trees on 20 features, 180-day partner history, extra interactions. Keep: 7-input logistic regression. Show V1: model frozen in April = AUC 0.09 - worse than a coin flip.
**2:10-2:45  The product.** `uvicorn` + `streamlit`; run the three presets: flagged-outlet claim (HIGH, reasons), long-standing partner (LOW), large inspected (LOW). Paste a description with injected text -> warning shown.
**2:45-3:00  The number.** "~Rs 750 of fraud stopped per claim checked, ~half of flags are genuine; retrain monthly."
