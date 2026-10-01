"""Data cleaning + feature engineering for the Kestrel warranty-fraud model.

Everything the live service needs is here so training and serving cannot drift apart.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

POLICY_CHANGE = pd.Timestamp("2026-05-01")      # ops-policy s5: claims < Rs 2,000 auto-approved
SMALL_CLAIM = 2000.0
NEW_PARTNER_FROM = pd.Timestamp("2025-11-01")   # partners.csv: the ~60 partners of policy s6 start here
GOODWILL_COST = 380.0                           # ops-policy s4: cost of holding a genuine claim
REVIEW_PER_MONTH = 40                           # ops-policy s5

KNOWN_FAULTS = [
    "remote not working", "loud noise while running", "power button not working",
    "filter indicator stuck", "motor not running", "water leaking", "not charging",
    "blade jammed", "tripping mcb", "burning smell", "unit not heating", "display not working",
    "display blank",
]

PROFILE_COLS = ["p_n_decided", "p_n_fraud", "p_rate", "p_vol", "p_small_unins", "p_small_unins_share"]
FEATURES = [
    "log_amt", "amt_ratio", "is_small", "days_since", "prior_claims",
    "photo", "inspected", "note_present", "small_uninspected", "post_policy",
    "tenure_days", "is_new_partner", "type_franchise", "type_freelance", "hour",
] + PROFILE_COLS


# Deliberately small: with ~145 confirmed frauds, a short interpretable model beat boosted trees
# on every out-of-time check (see evidence/EVIDENCE.md).
MODEL_FEATURES = ["small_uninspected", "is_new_partner", "new_small_unins", "prior_claims",
                  "p_rate", "p_small_unins_share", "log_amt"]


def clean_description(text) -> tuple[str, bool]:
    """Return (clean fault text, had_embedded_instruction). A few rows carry free text addressed
    to automated reviewers; it is treated as data and discarded."""
    t = str(text or "").strip().lower()
    for k in KNOWN_FAULTS:
        if t.startswith(k):
            return k, len(t) > len(k) + 2
    return re.split(r"[.;]", t)[0].strip(), len(t) > 40


def normalise_serial(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def load_claims(path, partners, products) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["submitted_at"])
    df["serial_norm"] = df["product_serial"].map(normalise_serial)
    cd = df["claim_description"].map(clean_description)
    df["fault"] = [c[0] for c in cd]
    df["desc_flagged"] = [c[1] for c in cd]
    # partners re-submit bounced claims: same claim_id twice -> keep the latest submission
    df = df.sort_values("ts").drop_duplicates("claim_id", keep="last").reset_index(drop=True)
    return df.merge(partners, on="partner_id", how="left").merge(products, on="sku", how="left")


def partner_profile(hist: pd.DataFrame, cutoff: pd.Timestamp, label_days: int = 60, vol_days: int = 60) -> pd.DataFrame:
    """Per-partner behaviour as known at `cutoff`. Labels only from decided claims in the last `label_days`
    before cutoff; volume/mix from all claims in the last `vol_days`. Short windows on purpose: the
    fraud pattern changed in May 2026 and old partner history misleads."""
    h = hist[hist.ts <= cutoff]
    lab = h[(h.ts > cutoff - pd.Timedelta(days=label_days)) & h.is_fraud.notna()]
    g = lab.groupby("partner_id").is_fraud.agg(p_n_decided="size", p_n_fraud="sum")
    rec = h[h.ts > cutoff - pd.Timedelta(days=vol_days)].copy()
    rec["su"] = ((rec.claim_amount_inr < SMALL_CLAIM) & (rec.partner_inspected == "N")).astype(int)
    v = rec.groupby("partner_id").agg(p_vol=("claim_id", "size"), p_small_unins=("su", "sum"))
    out = g.join(v, how="outer").fillna(0)
    # shrunk to the base rate, then capped at 0.25 (the highest value seen when training) so one
    # outlet cannot push a score off the scale
    out["p_rate"] = ((out.p_n_fraud + 20 * 0.012) / (out.p_n_decided + 20)).clip(upper=0.25)
    out["p_small_unins_share"] = out.p_small_unins / out.p_vol.clip(lower=1)
    return out[PROFILE_COLS]


EMPTY_PROFILE = {"p_n_decided": 0.0, "p_n_fraud": 0.0, "p_rate": 0.012, "p_vol": 0.0,
                 "p_small_unins": 0.0, "p_small_unins_share": 0.0}


def make_X(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    p = prof.reindex(df.partner_id.values).copy()
    for c, v in EMPTY_PROFILE.items():
        p[c] = p[c].fillna(v)
    p.index = df.index
    onb = pd.to_datetime(df.onboarded_date)
    amt = df.claim_amount_inr.astype(float)
    X = pd.DataFrame(index=df.index)
    X["log_amt"] = np.log(amt)
    X["amt_ratio"] = amt / df.list_price_inr
    X["is_small"] = (amt < SMALL_CLAIM).astype(int)
    X["days_since"] = df.days_since_purchase
    X["prior_claims"] = df.customer_prior_claims
    X["photo"] = (df.photo_attached == "Y").astype(int)
    X["inspected"] = (df.partner_inspected == "Y").astype(int)
    X["note_present"] = df.inspector_note.notna().astype(int)
    X["small_uninspected"] = ((amt < SMALL_CLAIM) & (df.partner_inspected == "N")).astype(int)
    X["post_policy"] = (df.ts >= POLICY_CHANGE).astype(int)
    X["tenure_days"] = (df.ts - onb).dt.days.clip(lower=0)
    X["is_new_partner"] = (onb >= NEW_PARTNER_FROM).astype(int)
    X["type_franchise"] = (df.partner_type == "franchise").astype(int)
    X["type_freelance"] = (df.partner_type == "freelance_technician").astype(int)
    X["hour"] = df.ts.dt.hour
    X["new_small_unins"] = X["is_new_partner"] * X["small_uninspected"]
    return pd.concat([X, p[PROFILE_COLS]], axis=1)[FEATURES + ["new_small_unins"]]


def build_training(df: pd.DataFrame, first_block="2025-07-01", months=3, last_cutoff=None, **pkw):
    """Training matrix that mimics deployment: each claim only sees partner behaviour frozen at
    the start of its `months`-month block (the test set is the 3 months after the last label).
    Rows before `first_block` have no history and are not used."""
    rows, metas = [], []
    starts = list(pd.date_range(first_block, "2027-01-01", freq=f"{months}MS"))
    for a, b in zip(starts[:-1], starts[1:]):
        blk = df[(df.ts >= a) & (df.ts < b)]
        if last_cutoff is not None:
            blk = blk[blk.ts <= last_cutoff]
        if blk.empty:
            continue
        prof = partner_profile(df, a - pd.Timedelta(seconds=1), **pkw)
        rows.append(make_X(blk, prof)); metas.append(blk)
    return pd.concat(rows), pd.concat(metas)
