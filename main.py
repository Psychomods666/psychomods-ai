
import hashlib
import hmac
import logging
import os
from collections import deque
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from groq import AsyncGroq

# Configuration
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("psychomods_ai")

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v27.0")

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(
    title="PSYCHOMODS AI",
    description="AI-powered Instagram webhook assistant.",
    version="0.2.0",
)

# Basic in-memory duplicate protection.
# This resets when the service restarts.
seen_message_ids: set[str] = set()
recent_message_ids: deque[str] = deque()
MAX_SEEN_IDS = 5000


def remember_message(message_id: str) -> bool:
    if message_id in seen_message_ids:
        return False

    seen_message_ids.add(message_id)
    recent_message_ids.append(message_id)

    while len(recent_message_ids) > MAX_SEEN_IDS:
        old_id = recent_message_ids.popleft()
        seen_message_ids.discard(old_id)

    return True


def verify_meta_signature(body: bytes, signature: str) -> bool:
    if not META_APP_SECRET or not signature.startswith("sha256="):
        return False

    expected = "sha256=" + hmac.new(
        META_APP_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature, expected)


async def generate_ai_reply(user_message: str) -> str:
    if not groq_client:
        raise RuntimeError("Groq is not configured")

    completion = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the official Instagram assistant for PSYCHO MOD'S, "
                    "a technology and cybersecurity brand. "
                    "Write friendly, natural, professional and concise replies. "
                    "Understand the user's question before answering. "
                    "For cybersecurity topics, support ethical learning, "
                    "defensive security and authorized labs only. "
                    "Never help with credential theft, malware deployment, "
                    "unauthorized access or harming others. "
                    "Never invent prices, products, services, links or company facts. "
                    "If you do not know something, say so politely. "
                    "Keep replies suitable for Instagram DMs."
                ),
            },
            {
                "role": "user",
                "content": user_message[:3000],
            },
        ],
        temperature=0.5,
        max_completion_tokens=220,
    )

    reply = completion.choices[0].message.content
    if not reply:
        raise RuntimeError("Groq returned an empty reply")

    return reply.strip()[:3000]


async def send_instagram_dm(
    instagram_user_id: str,
    recipient_id: str,
    reply_text: str,
) -> None:
    if not INSTAGRAM_ACCESS_TOKEN:
        logger.error("Instagram access token is not configured")
        return

    url = (
        f"https://graph.facebook.com/{GRAPH_API_VERSION}/"
        f"{instagram_user_id}/messages"
    )

    headers = {
        "Authorization": f"Bearer {INSTAGRAM_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": reply_text},
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        logger.info("Instagram DM reply sent successfully")

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Instagram send API returned HTTP %s",
            exc.response.status_code,
        )
    except httpx.RequestError:
        logger.error("Instagram send API network request failed")


async def process_dm(
    instagram_user_id: str,
    sender_id: str,
    message_text: str,
) -> None:
    try:
        reply = await generate_ai_reply(message_text)
        await send_instagram_dm(
            instagram_user_id,
            sender_id,
            reply,
        )
    except Exception as exc:
        logger.error(
            "DM processing failed (%s)",
            type(exc).__name__,
        )


@app.get("/")
async def home() -> dict[str, str]:
    return {
        "project": "PSYCHOMODS AI",
        "status": "online",
        "service": "Instagram AI Assistant",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request) -> PlainTextResponse:
    if not VERIFY_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Webhook verification is not configured",
        )

    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if (
        mode == "subscribe"
        and token
        and hmac.compare_digest(token, VERIFY_TOKEN)
    ):
        logger.info("Meta webhook verification successful")
        return PlainTextResponse(content=challenge or "")

    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    if not META_APP_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Webhook signature validation is not configured",
        )

    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not verify_meta_signature(body, signature):
        logger.warning("Rejected webhook with invalid signature")
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature",
        )

    try:
        payload: Any = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    entries = payload.get("entry", [])
    scheduled = 0

    if payload.get("object") == "instagram" and isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            instagram_user_id = str(entry.get("id", ""))

            for event in entry.get("messaging", []):
                if not isinstance(event, dict):
                    continue

                message = event.get("message", {})
                if not isinstance(message, dict):
                    continue

                # Ignore messages sent by our own account.
                if message.get("is_echo"):
                    continue

                sender_id = str(event.get("sender", {}).get("id", ""))
                recipient_id = str(event.get("recipient", {}).get("id", ""))
                message_text = message.get("text", "")
                message_id = str(message.get("mid", ""))

                account_id = instagram_user_id or recipient_id

                if not (
                    account_id
                    and sender_id
                    and message_text
                    and message_id
                ):
                    continue

                if sender_id == account_id:
                    continue

                if not remember_message(message_id):
                    logger.info("Duplicate DM event ignored")
                    continue

                background_tasks.add_task(
                    process_dm,
                    account_id,
                    sender_id,
                    message_text,
                )
                scheduled += 1

    # Never log private message contents or access tokens.
    logger.info(
        "Verified Instagram webhook received; entries=%d, DMs scheduled=%d",
        len(entries) if isinstance(entries, list) else 0,
        scheduled,
    )

    return {"status": "received"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
