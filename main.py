
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

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger("psychomods_ai")

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v27.0")

# Select the host that matches your Meta login configuration.
# Instagram Login: https://graph.instagram.com
# Facebook Login:  https://graph.facebook.com
META_GRAPH_BASE_URL = os.getenv(
    "META_GRAPH_BASE_URL",
    "https://graph.facebook.com",
).rstrip("/")

# Usually the Instagram professional account ID.
# Leave empty to use entry.id from the webhook.
INSTAGRAM_SEND_ID = os.getenv("INSTAGRAM_SEND_ID", "").strip()

groq_client = (
    AsyncGroq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)

app = FastAPI(
    title="PSYCHOMODS AI",
    description="AI-powered Instagram DM assistant",
    version="1.0.0",
)

# --------------------------------------------------
# DUPLICATE MESSAGE PROTECTION
# --------------------------------------------------

seen_message_ids: set[str] = set()
recent_message_ids: deque[str] = deque()

MAX_SEEN_IDS = 5000


def remember_message(message_id: str) -> bool:
    if not message_id:
        return False

    if message_id in seen_message_ids:
        return False

    seen_message_ids.add(message_id)
    recent_message_ids.append(message_id)

    while len(recent_message_ids) > MAX_SEEN_IDS:
        old_id = recent_message_ids.popleft()
        seen_message_ids.discard(old_id)

    return True


# --------------------------------------------------
# META SIGNATURE VALIDATION
# --------------------------------------------------

def verify_meta_signature(body: bytes, signature: str) -> bool:
    if not META_APP_SECRET:
        return False

    if not signature.startswith("sha256="):
        return False

    expected = "sha256=" + hmac.new(
        META_APP_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature, expected)


# --------------------------------------------------
# GROQ AI RESPONSE
# --------------------------------------------------

async def generate_ai_reply(user_message: str) -> str:
    if not groq_client:
        raise RuntimeError("Groq API key is not configured")

    completion = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the official Instagram AI assistant "
                    "for PSYCHO MOD'S, a technology and cybersecurity brand. "
                    "Be friendly, professional, helpful and concise. "
                    "Use natural conversational language. "
                    "Understand the question before answering. "
                    "For cybersecurity, support ethical learning, "
                    "defensive security and authorized labs only. "
                    "Never help with credential theft, malware deployment, "
                    "unauthorized access or harming others. "
                    "Never invent prices, services, links or company facts. "
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

    if not reply or not reply.strip():
        raise RuntimeError("Groq returned an empty response")

    return reply.strip()[:3000]


# --------------------------------------------------
# SEND INSTAGRAM DM
# --------------------------------------------------

async def send_instagram_dm(
    instagram_user_id: str,
    recipient_id: str,
    reply_text: str,
) -> bool:

    if not INSTAGRAM_ACCESS_TOKEN:
        logger.error("Instagram access token is not configured")
        return False

    account_id = INSTAGRAM_SEND_ID or instagram_user_id

    if not account_id or not recipient_id:
        logger.error("Missing Instagram account or recipient ID")
        return False

    url = (
        f"{META_GRAPH_BASE_URL}/"
        f"{GRAPH_API_VERSION}/"
        f"{account_id}/messages"
    )

    headers = {
        "Authorization": f"Bearer {INSTAGRAM_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "recipient": {
            "id": recipient_id,
        },
        "message": {
            "text": reply_text[:1000],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )

        if response.is_success:
            logger.info("Instagram DM reply sent successfully")
            return True

        # Log the API error response, but never the access token.
        logger.error(
            "Instagram API error: HTTP %s; response=%s",
            response.status_code,
            response.text[:1000],
        )

        return False

    except httpx.RequestError as exc:
        logger.error(
            "Instagram API network error: %s",
            type(exc).__name__,
        )
        return False


# --------------------------------------------------
# PROCESS ONE INCOMING MESSAGE
# --------------------------------------------------

async def process_dm(
    instagram_user_id: str,
    sender_id: str,
    message_text: str,
) -> None:

    try:
        logger.info("Starting AI response generation")

        reply = await generate_ai_reply(message_text)

        logger.info("AI response generated successfully")

        await send_instagram_dm(
            instagram_user_id,
            sender_id,
            reply,
        )

    except Exception as exc:
        logger.exception(
            "DM processing failed: %s",
            type(exc).__name__,
        )


# --------------------------------------------------
# NORMALIZE INCOMING WEBHOOK EVENTS
# --------------------------------------------------

def extract_messaging_events(entry: dict[str, Any]) -> list[dict]:
    """
    Instagram DM notifications normally use entry.messaging[].
    Other field-change notifications may use entry.changes[].
    We inspect both without assuming every change is a DM.
    """

    events: list[dict] = []

    messaging = entry.get("messaging", [])

    if isinstance(messaging, list):
        events.extend(
            item for item in messaging
            if isinstance(item, dict)
        )

    changes = entry.get("changes", [])

    if isinstance(changes, list):
        for change in changes:
            if not isinstance(change, dict):
                continue

            field = change.get("field")
            value = change.get("value")

            if field not in ("messages", "messaging"):
                continue

            if not isinstance(value, dict):
                continue

            # Only accept change events that contain a usable
            # messaging-style message structure.
            if isinstance(value.get("message"), dict):
                events.append(value)

    return events


def extract_incoming_message(
    event: dict[str, Any],
    account_id: str,
) -> tuple[str, str, str] | None:

    message = event.get("message")

    if not isinstance(message, dict):
        return None

    if message.get("is_echo"):
        return None

    sender = event.get("sender") or {}
    recipient = event.get("recipient") or {}

    if not isinstance(sender, dict):
        return None

    if not isinstance(recipient, dict):
        return None

    sender_id = str(sender.get("id", ""))
    recipient_id = str(recipient.get("id", ""))

    message_text = message.get("text", "")
    message_id = str(message.get("mid", ""))

    if not isinstance(message_text, str):
        return None

    message_text = message_text.strip()

    if not sender_id or not message_id or not message_text:
        return None

    if sender_id == account_id:
        return None

    # The sender is the person who messaged your business account.
    return sender_id, message_id, message_text


# --------------------------------------------------
# BASIC ROUTES
# --------------------------------------------------

@app.get("/")
async def home():
    return {
        "project": "PSYCHOMODS AI",
        "status": "online",
        "service": "Instagram AI Assistant",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "groq_configured": groq_client is not None,
        "instagram_token_configured": bool(INSTAGRAM_ACCESS_TOKEN),
        "meta_app_secret_configured": bool(META_APP_SECRET),
    }


# --------------------------------------------------
# META WEBHOOK VERIFICATION
# --------------------------------------------------

@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):

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

        return PlainTextResponse(
            content=challenge or "",
        )

    raise HTTPException(
        status_code=403,
        detail="Webhook verification failed",
    )


# --------------------------------------------------
# RECEIVE INSTAGRAM WEBHOOK
# --------------------------------------------------

@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):

    if not META_APP_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Meta App Secret is not configured",
        )

    body = await request.body()

    signature = request.headers.get(
        "X-Hub-Signature-256",
        "",
    )

    if not verify_meta_signature(body, signature):
        logger.warning("Rejected webhook: invalid signature")

        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature",
        )

    try:
        payload = await request.json()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload",
        )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid webhook payload",
        )

    if payload.get("object") != "instagram":
        logger.info(
            "Ignoring webhook object type: %s",
            payload.get("object"),
        )

        return {"status": "ignored"}

    entries = payload.get("entry", [])

    if not isinstance(entries, list):
        return {"status": "received"}

    scheduled = 0
    skipped = 0

    for entry in entries:

        if not isinstance(entry, dict):
            continue

        account_id = str(entry.get("id", ""))

        events = extract_messaging_events(entry)

        changes = entry.get("changes", [])

        change_fields = [
            item.get("field")
            for item in changes
            if isinstance(item, dict)
        ] if isinstance(changes, list) else []

        logger.info(
            "Webhook diagnostic: entry_keys=%s messaging_count=%s "
            "change_fields=%s",
            sorted(entry.keys()),
            len(entry.get("messaging", []))
            if isinstance(entry.get("messaging", []), list)
            else -1,
            change_fields,
        )

        logger.info(
            "Messaging events discovered: %d",
            len(events),
        )

        for event in events:

            message = event.get("message")

            if not isinstance(message, dict):
                skipped += 1
                logger.info("Skipped event: no message object")
                continue

            logger.info(
                "Message diagnostic: event_keys=%s message_keys=%s "
                "has_text=%s has_message_id=%s is_echo=%s",
                sorted(event.keys()),
                sorted(message.keys()),
                bool(message.get("text")),
                bool(message.get("mid")),
                bool(message.get("is_echo")),
            )

            result = extract_incoming_message(
                event,
                account_id,
            )

            if result is None:
                skipped += 1
                logger.info(
                    "Skipped event: not a processable incoming text DM"
                )
                continue

            sender_id, message_id, message_text = result

            if not account_id:
                skipped += 1
                logger.warning(
                    "Skipped DM: Instagram account ID is missing"
                )
                continue

            if not remember_message(message_id):
                skipped += 1
                logger.info("Duplicate DM event ignored")
                continue

            background_tasks.add_task(
                process_dm,
                account_id,
                sender_id,
                message_text,
            )

            scheduled += 1

    logger.info(
        "Webhook completed: entries=%d, scheduled=%d, skipped=%d",
        len(entries),
        scheduled,
        skipped,
    )

    return {
        "status": "received",
        "scheduled": str(scheduled),
    }


# --------------------------------------------------
# LOCAL STARTUP
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
