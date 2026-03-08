"""
api/main.py
InsureAgent FastAPI inference server.

Endpoints:
  POST /process_claim  — run full agent loop, return verdict + trace
  GET  /health         — health check
"""

from fastapi import FastAPI, HTTPException
from api.schemas import ClaimRequest, ClaimResponse
from api.inference import process_claim
from utils.logger import get_logger, Events

log = get_logger(__name__)

app = FastAPI(title="InsureAgent API")


@app.post("/process_claim", response_model=ClaimResponse)
async def handle_claim(request: ClaimRequest):
    log.info(Events.CLAIM_RECEIVED,
        user_id=request.user_id,
        claimed_amount=request.claimed_amount,
        model=request.model,
    )
    try:
        result = process_claim(
            claim_text=request.claim_text,
            user_id=request.user_id,
            claimed_amount=request.claimed_amount,
            model=request.model,
        )
        log.info(Events.CLAIM_PROCESSED,
            user_id=request.user_id,
            verdict=result["verdict"],
            payout=result["payout"],
            latency_ms=result["latency_ms"],
            model=result["model_used"],
        )
        return ClaimResponse(**result)
    except Exception as e:
        log.error("claim_error",
            user_id=request.user_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}