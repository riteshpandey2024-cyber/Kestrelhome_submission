"""Load the trained artifacts, score one claim, explain it in words a Kestrel employee can read."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import (MODEL_FEATURES, GOODWILL_COST, SMALL_CLAIM, POLICY_CHANGE, clean_description,
                       normalise_serial, make_X)

ART = Path(__file__).resolve().parent.parent / "artifacts"

REQUIRED = ["partner_id", "sku", "claim_amount_inr", "partner_inspected", "submitted_at"]


class Scorer:
    def __init__(self, art_dir: Path = ART):
        m = json.loads((art_dir / "model.json").read_text())   # plain JSON: no pickle, no sklearn needed to serve
        self.version = m["version"]
        self.mean, self.scale = np.array(m["mean"]), np.array(m["scale"])
        self.coef, self.intercept = np.array(m["coef"]), m["intercept"]
        ref = json.loads((art_dir / "reference.json").read_text())
        self.partners = ref["partners"]           # partner_id -> {city, onboarded_date, partner_type}
        self.products = ref["products"]           # sku -> {family, list_price_inr, warranty_months}
        self.profile = pd.DataFrame(ref["profile"]).T if ref["profile"] else pd.DataFrame()
        self.as_of = ref["as_of"]
        self.queue_threshold = ref["queue_threshold"]   # score of the 40/month-th claim in the test window
        self.meta = ref.get("meta", {})

    # ---- scoring -------------------------------------------------------------------------
    def _frame(self, rec: dict) -> tuple[pd.DataFrame, list[str]]:
        warns = []
        miss = [c for c in REQUIRED if rec.get(c) in (None, "")]
        if miss:
            raise ValueError(f"missing required field(s): {', '.join(miss)}")
        pid, sku = str(rec["partner_id"]), str(rec["sku"])
        part = self.partners.get(pid)
        if part is None:
            warns.append(f"Partner {pid} is not in partners.csv - treated as an unknown, brand-new partner.")
            part = {"city": "unknown", "onboarded_date": str(pd.Timestamp(rec["submitted_at"]).date()),
                    "partner_type": "authorised_service_centre"}
        prod = self.products.get(sku)
        if prod is None:
            warns.append(f"SKU {sku} is not in products.csv - price ratio uses the claim amount itself.")
            prod = {"family": "unknown", "list_price_inr": float(rec["claim_amount_inr"]) * 3, "warranty_months": 12}
        desc, flagged = clean_description(rec.get("claim_description", ""))
        if flagged:
            warns.append("The fault description contains extra text (possibly instructions to a reviewer/AI). "
                         "It was ignored; only the fault wording is used.")
        ser = normalise_serial(rec.get("product_serial", ""))
        if rec.get("product_serial") and (not ser.startswith("KH") or len(ser) != 11):
            warns.append(f"Serial '{rec['product_serial']}' is not in the standard KH+9 digits format (cleaned: {ser}).")
        row = {
            "claim_id": rec.get("claim_id", "NEW"), "ts": pd.Timestamp(rec["submitted_at"]), "partner_id": pid,
            "claim_amount_inr": float(rec["claim_amount_inr"]), "partner_inspected": str(rec["partner_inspected"]).upper()[:1],
            "photo_attached": str(rec.get("photo_attached", "N")).upper()[:1],
            "days_since_purchase": int(rec.get("days_since_purchase") or 0),
            "customer_prior_claims": int(rec.get("customer_prior_claims") or 0),
            "inspector_note": rec.get("inspector_note") or np.nan, "sku": sku,
            **{k: part[k] for k in ("city", "onboarded_date", "partner_type")}, **prod,
        }
        return pd.DataFrame([row]), warns

    def score(self, rec: dict) -> dict:
        df, warns = self._frame(rec)
        X = make_X(df, self.profile)
        z = (X[MODEL_FEATURES].values[0].astype(float) - self.mean) / self.scale
        p = float(1 / (1 + np.exp(-(self.intercept + self.coef @ z))))
        amt = float(df.claim_amount_inr.iloc[0])
        ev = p * amt - (1 - p) * GOODWILL_COST        # rupees gained by holding this claim, expected
        band = "HIGH" if p >= self.queue_threshold else ("WATCH" if p >= self.queue_threshold / 3 else "LOW")
        reasons = self.explain(X.iloc[0], df.iloc[0], p)
        return {
            "claim_id": str(rec.get("claim_id") or "NEW"), "fraud_score": round(p, 4), "risk_band": band,
            "expected_net_inr_if_held": round(ev, 0),
            "recommendation": {
                "HIGH": "Send to the investigation desk before payout.",
                "WATCH": "Pay normally; keep an eye on this partner.",
                "LOW": "Pay normally."}[band],
            "reasons": reasons, "data_warnings": warns,
            "model": {"version": self.version, "type": "logistic regression (7 inputs)",
                      "partner_history_as_of": self.as_of,
                      "queue_threshold": round(self.queue_threshold, 4)},
            "caveat": "A score is a ranking aid, not a verdict. Fraud in this data is rare (about 1 claim in 80) and "
                      "most flagged claims are still genuine. Capacity: 40 investigations a month.",
        }

    # ---- reasons ---------------------------------------------------------------------------
    def explain(self, x: pd.Series, row: pd.Series, p: float) -> list[dict]:
        z = (x[MODEL_FEATURES].values.astype(float) - self.mean) / self.scale
        contrib = dict(zip(MODEL_FEATURES, self.coef * z))        # log-odds vs an average claim
        out = []
        pid = row.partner_id
        small = row.claim_amount_inr < SMALL_CLAIM
        if x.small_uninspected:
            out.append(("small_uninspected", contrib["small_uninspected"] + contrib["new_small_unins"],
                        f"Claim is under Rs {SMALL_CLAIM:,.0f} and was not inspected. Since "
                        f"{POLICY_CHANGE:%d %b %Y} such claims are auto-approved, and this is where the recent fraud sits."))
        if x.is_new_partner and contrib["is_new_partner"] + contrib["new_small_unins"] > 0:
            out.append(("is_new_partner", contrib["is_new_partner"] + contrib["new_small_unins"],
                        f"Partner {pid} joined on {row.onboarded_date} (one of the ~60 new partners). "
                        "Being new alone is weak evidence - most new partners are fine."))
        if x.p_n_fraud > 0 or x.p_rate > 0.05:
            out.append(("p_rate", contrib["p_rate"],
                        f"Partner {pid} had {int(x.p_n_fraud)} confirmed fraud(s) among {int(x.p_n_decided)} decided claims "
                        f"in the 60 days to {self.as_of}."))
        if x.p_small_unins_share >= 0.5 and x.p_vol >= 5 and contrib["p_small_unins_share"] > 0:
            out.append(("p_small_unins_share", contrib["p_small_unins_share"],
                        f"{x.p_small_unins_share:.0%} of partner {pid}'s recent claims ({int(x.p_vol)}) are small and uninspected."))
        if row.customer_prior_claims >= 2:
            out.append(("prior_claims", contrib["prior_claims"],
                        f"Customer already has {int(row.customer_prior_claims)} earlier warranty claims."))
        if not out:
            out.append(("none", 0.0, "No risk pattern found: inspected or larger claim from a partner with no recent fraud."))
        out.sort(key=lambda t: -t[1])
        return [{"factor": f, "effect": ("raises" if c > 0.05 else "lowers" if c < -0.05 else "neutral"),
                 "log_odds": round(float(c), 2), "text": t} for f, c, t in out]


_scorer: Scorer | None = None


def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer()
    return _scorer
