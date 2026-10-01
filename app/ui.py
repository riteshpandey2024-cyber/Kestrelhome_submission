"""Streamlit screen.  streamlit run app/ui.py   (needs the API running: uvicorn app.api:app)"""
import os
import requests
import streamlit as st

API = os.environ.get("KESTREL_API", "http://127.0.0.1:8000")
st.set_page_config(page_title="Kestrel claim check", page_icon="🔎", layout="wide")
st.title("🔎 Warranty claim check")
st.caption("Scores one claim before payout and says why. A ranking aid for the 40-a-month investigation desk - not a verdict.")

PRESETS = {
    "Small, uninspected, recently-flagged new partner": dict(partner_id="SP3160", sku="KH-AF-02", amt=1450.0, insp="N", photo="Y", prior=1, days=140, d="2026-09-12"),
    "Small, uninspected, long-standing partner": dict(partner_id="SP3033", sku="KH-AF-02", amt=1450.0, insp="N", photo="Y", prior=0, days=140, d="2026-09-12"),
    "Large, inspected claim": dict(partner_id="SP3033", sku="KH-RV-03", amt=9800.0, insp="Y", photo="Y", prior=0, days=60, d="2026-09-12"),
}
choice = st.selectbox("Start from an example", list(PRESETS))
p = PRESETS[choice]

with st.form("claim"):
    c1, c2, c3 = st.columns(3)
    partner_id = c1.text_input("Partner ID", p["partner_id"])
    sku = c1.text_input("SKU", p["sku"])
    serial = c1.text_input("Serial (as typed)", "kh-662573314")
    amount = c2.number_input("Claim amount (Rs)", 1.0, 100000.0, p["amt"], 50.0)
    days = c2.number_input("Days since purchase", 0, 2000, p["days"])
    prior = c2.number_input("Customer's earlier claims", 0, 20, p["prior"])
    inspected = c3.selectbox("Partner inspected?", ["N", "Y"], index=["N", "Y"].index(p["insp"]))
    photo = c3.selectbox("Photo attached?", ["Y", "N"], index=["Y", "N"].index(p["photo"]))
    when = c3.text_input("Submitted at (IST)", p["d"] + " 14:05")
    desc = st.text_input("Fault description", "motor not running")
    note = st.text_input("Inspector note (blank if none)", "")
    go = st.form_submit_button("Check claim", type="primary")

if go:
    payload = dict(submitted_at=when, partner_id=partner_id.strip(), sku=sku.strip(), product_serial=serial,
                   days_since_purchase=int(days), claim_amount_inr=float(amount), photo_attached=photo,
                   partner_inspected=inspected, claim_description=desc, inspector_note=note or None,
                   customer_prior_claims=int(prior))
    try:
        r = requests.post(f"{API}/score", json=payload, timeout=10)
    except requests.RequestException:
        st.error(f"The scoring service is not reachable at {API}. Start it with: uvicorn app.api:app --port 8000")
        st.stop()
    if r.status_code != 200:
        st.warning(f"The service could not score this claim: {r.json().get('detail', r.text)}")
        st.stop()
    out = r.json()
    colour = {"HIGH": "🔴", "WATCH": "🟠", "LOW": "🟢"}[out["risk_band"]]
    a, b, c = st.columns(3)
    a.metric("Fraud score", f"{out['fraud_score']:.1%}")
    b.metric("Risk band", f"{colour} {out['risk_band']}")
    c.metric("Expected Rs gained by holding", f"{out['expected_net_inr_if_held']:,.0f}")
    st.subheader(out["recommendation"])
    st.markdown("**Why**")
    for rs in out["reasons"]:
        icon = {"raises": "⬆️", "lowers": "⬇️"}.get(rs["effect"], "•")
        st.write(f"{icon} {rs['text']}")
    for w in out["data_warnings"]:
        st.warning(w)
    st.caption(out["caveat"])
    with st.expander("Raw model output"):
        st.json(out)
