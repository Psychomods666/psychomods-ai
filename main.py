import hashlib
import hmac
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("psychomods_ai")

app = FastAPI(
    title="PSYCHOMODS AI",
    description="Instagram webhook starter for Meta's Instagram API.",
    version="0.1.0",
)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")


@app.get("/")
async def home() -> dict[str, str]:
    return {
        "project": "PSYCHOMODS AI",
        "status": "online",
        "service": "Instagram Webhook",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request) -> PlainTextResponse:
    """Handle Meta's webhook subscription verification challenge."""
    if not VERIFY_TOKEN:
        logger.error("VERIFY_TOKEN is not configured")
        raise HTTPException(status_code=503, detail="Webhook is not configured")

    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token and hmac.compare_digest(token, VERIFY_TOKEN):
        logger.info("Meta webhook verification successful")
        return PlainTextResponse(content=challenge or "")

    logger.warning("Webhook verification rejected")
    raise HTTPException(status_code=403, detail="Verification failed")


def verify_meta_signature(body: bytes, signature_header: str) -> bool:
    """Validate X-Hub-Signature-256 using the Meta app secret."""
    if not META_APP_SECRET or not signature_header.startswith("sha256="):
        return False

    expected = "sha256=" + hmac.new(
        META_APP_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature_header, expected)


@app.post("/webhook")
async def receive_webhook(request: Request) -> dict[str, str]:
    """Receive Instagram webhook events after validating Meta's signature."""
    if not META_APP_SECRET:
        logger.error("META_APP_SECRET is not configured")
        raise HTTPException(status_code=503, detail="Signature validation is not configured")

    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not verify_meta_signature(body, signature):
        logger.warning("Rejected webhook with missing or invalid signature")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload: Any = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    # Avoid logging private message/comment contents or personal information.
    entries = payload.get("entry", []) if isinstance(payload, dict) else []
    logger.info(
        "Verified webhook received: object=%s, entries=%d",
        payload.get("object", "unknown") if isinstance(payload, dict) else "unknown",
        len(entries) if isinstance(entries, list) else 0,
    )

    # TODO: Add event parsing, deduplication, and Instagram reply logic.
    # Do not send replies until permissions, policy limits, and event types
    # have been tested with your own account.
    return {"status": "received"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
