"""FastAPI service.  uvicorn app.api:app --port 8000"""
from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from kestrel.scoring import get_scorer

app = FastAPI(title="Kestrel warranty-claim fraud scorer", version="1.0",
              description="Scores ONE warranty claim and explains why. No external API or key is used.")


class Claim(BaseModel):
    claim_id: Optional[str] = Field(None, examples=["WC711350"])
    submitted_at: str = Field(..., examples=["2026-09-12 14:05"], description="IST, 'YYYY-MM-DD HH:MM'")
    partner_id: str = Field(..., examples=["SP3160"])
    sku: str = Field(..., examples=["KH-AF-02"])
    product_serial: Optional[str] = Field(None, examples=["kh-662573314"])
    days_since_purchase: int = Field(0, ge=0)
    claim_amount_inr: float = Field(..., gt=0)
    photo_attached: str = Field("N", pattern="^[YyNn]$")
    partner_inspected: str = Field(..., pattern="^[YyNn]$")
    claim_description: Optional[str] = None
    inspector_note: Optional[str] = None
    customer_prior_claims: int = Field(0, ge=0)
    source: Optional[str] = "crm"


@app.get("/health")
def health():
    try:
        s = get_scorer()
        return {"status": "ok", "model": s.version, "partner_history_as_of": s.as_of}
    except Exception as e:  # model artifacts missing -> say so politely
        return {"status": "degraded", "detail": f"model not loaded: {e}. Run `python -m kestrel.train` first."}


@app.post("/score")
def score(claim: Claim):
    try:
        return get_scorer().score(claim.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run `python -m kestrel.train` first.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not score this claim: {e}")
