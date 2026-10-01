"""Train, validate out-of-time, write predictions + artifacts + evidence.

    python -m kestrel.train            (expects the CSVs in ./data)
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from .features import *
from .metrics import summary, topk_report

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA, ART, EVD = ROOT / "data", ROOT / "artifacts", ROOT / "evidence"
VERSION = "kestrel-fraud-lr-2026-10-01"
RNG = np.random.default_rng(7)


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))


def fit_until(tr, cutoff):
    X, m = build_training(tr, last_cutoff=cutoff)
    ok = m.is_fraud.notna()
    return new_model().fit(X.loc[ok, MODEL_FEATURES], m.is_fraud[ok].astype(int)), X[ok], m[ok]


def validate(tr, name, cutoff, lo, hi, with_baselines=True):
    cutoff = pd.Timestamp(cutoff)
    v = tr[(tr.ts >= lo) & (tr.ts < hi) & tr.is_fraud.notna()]
    y, amt = v.is_fraud.astype(int).values, v.claim_amount_inr.values
    k = int(round(REVIEW_PER_MONTH * (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 30.4))
    Xv = make_X(v, partner_profile(tr, cutoff))
    mod, Xt, mt = fit_until(tr, cutoff)
    out = {"name": name, "train_rows": int(len(Xt)), "train_frauds": int(mt.is_fraud.sum()), "val_rows": int(len(v)), "K": k}
    s = mod.predict_proba(Xv[MODEL_FEATURES])[:, 1]
    out["model"] = summary(y, s, amt, k)
    # bootstrap (rows) for uncertainty
    aucs, spc = [], []
    for _ in range(1000):
        i = RNG.integers(0, len(y), len(y))
        if y[i].sum() == 0: continue
        aucs.append(roc_auc_score(y[i], s[i])); spc.append(topk_report(y[i], s[i], amt[i], k)["stopped_per_check"])
    out["auc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    out["stopped_per_check_ci95"] = [float(np.percentile(spc, 2.5)), float(np.percentile(spc, 97.5))]
    out["calibration"] = {"mean_predicted": float(s.mean()), "actual_rate": float(y.mean())}
    if with_baselines:
        rng = RNG.random(len(y))
        out["baseline_random"] = summary(y, rng, amt, k)
        out["baseline_amount_only"] = summary(y, amt, amt, k)      # "check the biggest claims"
        out["baseline_new_partner_only"] = summary(y, Xv.is_new_partner.values + 1e-6 * rng, amt, k)
        hg = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                            l2_regularization=5, random_state=0)
        Xa, ma = build_training(tr, last_cutoff=cutoff, label_days=180, vol_days=90)
        ok = ma.is_fraud.notna()
        hg.fit(Xa.loc[ok, FEATURES], ma.is_fraud[ok].astype(int))
        Xva = make_X(v, partner_profile(tr, cutoff, 180, 90))
        out["baseline_boosted_trees_all_features"] = summary(y, hg.predict_proba(Xva[FEATURES])[:, 1], amt, k)
    return out, (y, s, amt)


def main():
    pa, pr = pd.read_csv(DATA / "partners.csv"), pd.read_csv(DATA / "products.csv")
    tr = load_claims(DATA / "train.csv", pa, pr)
    te = load_claims(DATA / "test_unlabelled.csv", pa, pr)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    raw = pd.read_csv(DATA / "train.csv")

    # ---------------- data facts -----------------
    lab = tr[tr.is_fraud.notna()]
    facts = {
        "train_rows_raw": int(len(raw)), "train_rows_after_dedupe": int(len(tr)),
        "duplicate_resubmissions_removed": int(len(raw) - len(tr)),
        "undecided_labels_dropped": int(tr.is_fraud.isna().sum()),
        "labelled_rows": int(len(lab)), "frauds": int(lab.is_fraud.sum()), "base_rate": float(lab.is_fraud.mean()),
        "accuracy_of_flagging_nothing": float(1 - lab.is_fraud.mean()),
        "rows_with_embedded_instructions_in_description": int(tr.desc_flagged.sum()),
        "legacy_zoho_rows_zero_may_hide_undecided": int((tr.source == "legacy_zoho").sum()),
        "fraud_rate_pre_policy_change": float(lab[lab.ts < POLICY_CHANGE].is_fraud.mean()),
        "fraud_rate_post_policy_change": float(lab[lab.ts >= POLICY_CHANGE].is_fraud.mean()),
    }
    post = lab[lab.ts >= POLICY_CHANGE]
    onb = pd.to_datetime(post.onboarded_date) >= NEW_PARTNER_FROM
    su = (post.claim_amount_inr < SMALL_CLAIM) & (post.partner_inspected == "N")
    facts["post_policy_segments"] = {
        "new_partner_small_uninspected": [int((onb & su).sum()), int(post[onb & su].is_fraud.sum())],
        "new_partner_other": [int((onb & ~su).sum()), int(post[onb & ~su].is_fraud.sum())],
        "old_partner_small_uninspected": [int((~onb & su).sum()), int(post[~onb & su].is_fraud.sum())],
        "old_partner_other": [int((~onb & ~su).sum()), int(post[~onb & ~su].is_fraud.sum())],
    }
    pf = post[onb & su].groupby("partner_id").is_fraud.agg(["size", "sum"]).sort_values("sum", ascending=False)
    facts["new_partners_with_small_uninspected_post_policy"] = int(len(pf))
    facts["new_partners_with_any_fraud_post_policy"] = int((pf["sum"] > 0).sum())
    facts["frauds_in_top_5_new_partners_post_policy"] = int(pf["sum"].head(5).sum())
    facts["post_policy_frauds_total"] = int(post.is_fraud.sum())

    # ---------------- out-of-time validation -----------------
    v1, _ = validate(tr, "V1: model frozen 30 Apr 2026, scored on May-Jun 2026", "2026-04-30 23:59:59", "2026-05-01", "2026-07-01")
    v2, (y2, s2, a2) = validate(tr, "V2: model frozen 31 May 2026, scored on Jun 2026", "2026-05-31 23:59:59", "2026-06-01", "2026-07-01")
    v0, _ = validate(tr, "V0: model frozen 31 Jan 2026, scored on Feb-Apr 2026 (old regime)", "2026-01-31 23:59:59", "2026-02-01", "2026-05-01", with_baselines=False)

    # ---------------- final fit -----------------
    last = tr.ts.max()
    X, m = build_training(tr)
    ok = m.is_fraud.notna()
    model = new_model().fit(X.loc[ok, MODEL_FEATURES], m.is_fraud[ok].astype(int))
    cutoff = pd.Timestamp("2026-06-30 23:59:59")
    prof = partner_profile(tr, cutoff)
    Xte = make_X(te, prof)
    te["score"] = model.predict_proba(Xte[MODEL_FEATURES])[:, 1]

    out = sub[["claim_id"]].merge(te[["claim_id", "score"]], on="claim_id", how="left")
    assert out.score.notna().all() and len(out) == len(sub) == te.claim_id.nunique()
    out["score"] = out.score.round(6)
    out.to_csv(ROOT / "predictions.csv", index=False)

    # ---------------- what to expect on test -----------------
    p = te.score.values
    K = REVIEW_PER_MONTH * 3
    order = np.argsort(-p)
    q = order[:K]
    thr = float(p[order[K - 1]])
    exp = {
        "test_rows": int(len(te)), "expected_fraud_count": float(p.sum()), "expected_fraud_rate": float(p.mean()),
        "expected_accuracy_flag_nothing": float(1 - p.mean()),
        "queue_size_K": K, "queue_score_threshold": thr,
        "expected_frauds_in_queue": float(p[q].sum()),
        "expected_fraud_inr_stopped": float((p[q] * te.claim_amount_inr.values[q]).sum()),
        "expected_goodwill_inr": float(((1 - p[q]) * GOODWILL_COST).sum()),
        "total_expected_fraud_inr_in_test": float((p * te.claim_amount_inr.values).sum()),
        "queue_new_partner_share": float(Xte.is_new_partner.values[q].mean()),
        "queue_distinct_partners": int(te.partner_id.values[q].shape[0] and len(set(te.partner_id.values[q]))),
        "queue_small_uninspected_share": float(Xte.small_uninspected.values[q].mean()),
        "test_new_partner_small_uninspected_rows": int((Xte.new_small_unins == 1).sum()),
    }
    # V2 showed the model over-predicts by ~1.5x (4.7% predicted vs 3.1% seen) -> shrink for the headline
    f = v2["calibration"]["actual_rate"] / v2["calibration"]["mean_predicted"]
    pc = np.minimum(1.0, p * f)
    exp["calibration_factor_from_V2"] = float(f)
    exp["calibrated_expected_fraud_count"] = float(pc.sum())
    exp["calibrated_expected_frauds_in_queue"] = float(pc[q].sum())
    exp["calibrated_stopped_inr"] = float((pc[q] * te.claim_amount_inr.values[q]).sum())
    exp["calibrated_goodwill_inr"] = float(((1 - pc[q]) * GOODWILL_COST).sum())
    exp["calibrated_stopped_per_check"] = exp["calibrated_stopped_inr"] / K
    exp["calibrated_net_per_check"] = (exp["calibrated_stopped_inr"] - exp["calibrated_goodwill_inr"]) / K
    exp["calibrated_accuracy_flag_nothing"] = float(1 - pc.mean())
    exp["expected_stopped_per_check"] = exp["expected_fraud_inr_stopped"] / K
    exp["expected_net_per_check"] = (exp["expected_fraud_inr_stopped"] - exp["expected_goodwill_inr"]) / K
    # expected accuracy if the 120 queue claims were all labelled "fraud"
    exp["expected_accuracy_queue_flagged"] = float(1 - (p[q].mean() * 0 + ((1 - p[q]).sum() + p[order[K:]].sum()) / len(p)))
    exp["top_queue_partners"] = (te.iloc[q].groupby("partner_id").size().sort_values(ascending=False).head(10).to_dict())

    # ---------------- artifacts -----------------
    ART.mkdir(exist_ok=True)
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    (ART / "model.json").write_text(json.dumps({
        "version": VERSION, "features": MODEL_FEATURES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}, indent=1))
    ref = {
        "as_of": str(cutoff.date()), "queue_threshold": thr,
        "partners": pa.set_index("partner_id").to_dict("index"),
        "products": pr.set_index("sku").to_dict("index"),
        "profile": prof.round(6).to_dict("index"),
        "meta": {"trained_rows": int(ok.sum()), "trained_frauds": int(m.is_fraud[ok].sum())},
    }
    (ART / "reference.json").write_text(json.dumps(ref))
    coefs = dict(zip(MODEL_FEATURES, model.named_steps["logisticregression"].coef_[0].round(3).tolist()))

    EVD.mkdir(exist_ok=True)
    (EVD / "metrics.json").write_text(json.dumps({"data_facts": facts, "validation": [v0, v1, v2],
                                                  "test_expectation": exp, "coefficients_standardised": coefs}, indent=2, default=float))

    # ---------------- charts -----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    mo = lab.groupby(lab.ts.dt.to_period("M")).is_fraud.mean() * 100
    ax[0].bar(mo.index.astype(str), mo.values, color=["#c0392b" if i >= "2026-05" else "#7f8c8d" for i in mo.index.astype(str)])
    ax[0].set_title("Fraud rate by month (%)  - red = after May 2026"); ax[0].tick_params(axis="x", rotation=70)
    names = ["pre-May model\n(V1)", "V2"]; vals = [v1["model"]["auc"], v2["model"]["auc"]]
    ax[1].bar(["V0\nFeb-Apr\n(old regime)", "V1\nMay-Jun, model\nfrozen 30 Apr", "V2\nJun, model\nfrozen 31 May"],
              [v0["model"]["auc"], v1["model"]["auc"], v2["model"]["auc"]], color=["#7f8c8d", "#c0392b", "#27ae60"])
    ax[1].axhline(0.5, ls="--", c="k", lw=0.8); ax[1].set_ylim(0, 1); ax[1].set_title("Out-of-time AUC (0.5 = coin flip)")
    ks = np.arange(1, 121); o2 = np.argsort(-s2)
    net = [topk_report(y2, s2, a2, k)["net_inr"] for k in ks[:80]]
    ax[2].plot(ks[:80], net); ax[2].axhline(0, c="k", lw=0.8); ax[2].set_xlabel("claims investigated (June)")
    ax[2].set_ylabel("Rs fraud stopped - goodwill"); ax[2].set_title("V2: net rupees vs claims checked")
    plt.tight_layout(); plt.savefig(EVD / "evidence.png", dpi=110)

    print(json.dumps({"facts": facts, "V1": v1["model"], "V2": v2["model"], "V0": v0["model"], "expect": exp, "coefs": coefs}, indent=1, default=float))


if __name__ == "__main__":
    main()
