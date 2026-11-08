
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

# ==================================================
# CONFIGURATION
# ==================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger("psychomods_ai")

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

GRAPH_API_VERSION = os.getenv(
    "GRAPH_API_VERSION",
    "v27.0"
)

# Choose explicitly based on your Meta login configuration.
API_MODE = os.getenv(
    "INSTAGRAM_API_MODE",
    "instagram_login"
).lower().strip()

API_HOSTS = {
    "instagram_login": "https://graph.instagram.com",
    "facebook_login": "https://graph.facebook.com",
}

if API_MODE not in API_HOSTS:
    raise RuntimeError(
        "INSTAGRAM_API_MODE must be instagram_login "
        "or facebook_login"
    )

META_GRAPH_BASE_URL = os.getenv(
    "META_GRAPH_BASE_URL",
    API_HOSTS[API_MODE]
).rstrip("/")

# Accept both variable names.
INSTAGRAM_ACCOUNT_ID = (
    os.getenv("INSTAGRAM_SENDER_ID", "").strip()
    or os.getenv("INSTAGRAM_SEND_ID", "").strip()
)

groq_client = (
    AsyncGroq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY else None
)

app = FastAPI(
    title="PSYCHOMODS AI",
    description="AI-powered Instagram DM assistant",
    version="2.0.0",
)

# ==================================================
# DUPLICATE MESSAGE PROTECTION
# ==================================================

seen_message_ids = set()
recent_message_ids = deque()

MAX_SEEN_IDS = 5000


def remember_message(message_id: str) -> bool:

    if not message_id:
        return False

    if message_id in seen_message_ids:
        return False

    seen_message_ids.add(message_id)
    recent_message_ids.append(message_id)

    while len(recent_message_ids) > MAX_SEEN_IDS:
        old = recent_message_ids.popleft()
        seen_message_ids.discard(old)

    return True


# ==================================================
# META SIGNATURE VALIDATION
# ==================================================

def verify_meta_signature(body: bytes, signature: str) -> bool:

    if not META_APP_SECRET:
        return False

    if not signature.startswith("sha256="):
        return False

    expected = "sha256=" + hmac.new(
        META_APP_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


# ==================================================
# API CONFIGURATION DIAGNOSTICS
# ==================================================

@app.get("/diagnostics")
async def diagnostics():

    result = {
        "project": "PSYCHOMODS AI",
        "configured_mode": API_MODE,
        "configured_host": META_GRAPH_BASE_URL,
        "graph_version": GRAPH_API_VERSION,
        "access_token_configured": bool(INSTAGRAM_ACCESS_TOKEN),
        "groq_configured": groq_client is not None,
        "app_secret_configured": bool(META_APP_SECRET),
        "account_id_configured": bool(INSTAGRAM_ACCOUNT_ID),
        "account_id_is_zero": INSTAGRAM_ACCOUNT_ID == "0",
    }

    if not INSTAGRAM_ACCESS_TOKEN:
        result["error"] = "Instagram access token missing"
        return result

    # Test both hosts without exposing the token.
    async with httpx.AsyncClient(timeout=20.0) as client:

        for mode, host in API_HOSTS.items():

            if mode == "instagram_login":
                fields = "user_id,username"
            else:
                fields = "id,name"

            url = (
                f"{host}/{GRAPH_API_VERSION}/me"
                f"?fields={fields}"
            )

            try:
                response = await client.get(
                    url,
                    headers={
                        "Authorization":
                            f"Bearer {INSTAGRAM_ACCESS_TOKEN}"
                    },
                )

                try:
                    data = response.json()
                except ValueError:
                    data = {}

                result[mode] = {
                    "http_status": response.status_code,
                    "success": response.is_success,
                    "identity": data if response.is_success else None,
                    "error": (
                        data.get("error", {}).get("message")
                        if isinstance(data, dict)
                        and isinstance(data.get("error"), dict)
                        else None
                    ),
                }

            except httpx.RequestError as exc:
                result[mode] = {
                    "success": False,
                    "error": type(exc).__name__,
                }

    return result


# ==================================================
# CHECK ACCOUNT ID
# ==================================================

@app.get("/diagnostics/account")
async def check_account():

    if not INSTAGRAM_ACCESS_TOKEN:
        raise HTTPException(503, "Access token missing")

    if not INSTAGRAM_ACCOUNT_ID:
        raise HTTPException(400, "Account ID missing")

    if INSTAGRAM_ACCOUNT_ID == "0":
        raise HTTPException(400, "Account ID cannot be zero")

    url = (
        f"{META_GRAPH_BASE_URL}/"
        f"{GRAPH_API_VERSION}/"
        f"{INSTAGRAM_ACCOUNT_ID}"
    )

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:

            response = await client.get(
                url,
                params={"fields": "id,username"},
                headers={
                    "Authorization":
                        f"Bearer {INSTAGRAM_ACCESS_TOKEN}"
                },
            )

        try:
            data = response.json()
        except ValueError:
            data = {}

        return {
            "configured_account_id": INSTAGRAM_ACCOUNT_ID,
            "api_host": META_GRAPH_BASE_URL,
            "http_status": response.status_code,
            "success": response.is_success,
            "result": data,
        }

    except httpx.RequestError as exc:
        raise HTTPException(
            502,
            f"Meta connection error: {type(exc).__name__}"
        )


# ==================================================
# GROQ AI
# ==================================================

async def generate_ai_reply(message: str) -> str:

    if not groq_client:
        raise RuntimeError("Groq API key missing")

    completion = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the official PSYCHO MOD'S Instagram "
                    "AI assistant. Be friendly, professional, "
                    "helpful and concise. Support ethical "
                    "cybersecurity education and defensive security. "
                    "Never assist unauthorized access, credential "
                    "theft or malware deployment. Never invent "
                    "company facts, prices or links."
                ),
            },
            {
                "role": "user",
                "content": message[:3000],
            },
        ],
        temperature=0.5,
        max_completion_tokens=220,
    )

    reply = completion.choices[0].message.content

    if not reply or not reply.strip():
        raise RuntimeError("Empty Groq response")

    return reply.strip()[:3000]


# ==================================================
# SEND INSTAGRAM DM
# ==================================================

async def send_instagram_dm(
    account_id: str,
    recipient_id: str,
    reply_text: str,
) -> bool:

    if not INSTAGRAM_ACCESS_TOKEN:
        logger.error("Instagram token missing")
        return False

    if not account_id or account_id == "0":
        logger.error("Invalid Instagram account ID")
        return False

    if not recipient_id or recipient_id == "0":
        logger.error("Invalid recipient ID")
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
            "id": recipient_id
        },
        "message": {
            "text": reply_text[:1000]
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
            logger.info("Instagram DM sent successfully")
            return True

        logger.error(
            "Instagram API error: HTTP %s; response=%s",
            response.status_code,
            response.text[:1000],
        )

        return False

    except httpx.RequestError as exc:
        logger.error(
            "Instagram network error: %s",
            type(exc).__name__,
        )
        return False


# ==================================================
# PROCESS INCOMING DM
# ==================================================

async def process_dm(
    account_id: str,
    sender_id: str,
    message_text: str,
):

    try:
        logger.info("Starting AI response generation")

        reply = await generate_ai_reply(message_text)

        logger.info("AI response generated successfully")

        await send_instagram_dm(
            account_id,
            sender_id,
            reply,
        )

    except Exception as exc:
        logger.exception(
            "DM processing failed: %s",
            type(exc).__name__,
        )


# ==================================================
# EXTRACT WEBHOOK EVENTS
# ==================================================

def extract_messaging_events(entry: dict[str, Any]):

    events = []

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

            if change.get("field") not in (
                "messages",
                "messaging"
            ):
                continue

            value = change.get("value")

            if (
                isinstance(value, dict)
                and isinstance(value.get("message"), dict)
            ):
                events.append(value)

    return events


def extract_incoming_message(event: dict):

    message = event.get("message")

    if not isinstance(message, dict):
        return None

    if message.get("is_echo"):
        return None

    sender = event.get("sender") or {}

    if not isinstance(sender, dict):
        return None

    sender_id = str(sender.get("id", ""))
    message_id = str(message.get("mid", ""))
    message_text = message.get("text", "")

    if not isinstance(message_text, str):
        return None

    message_text = message_text.strip()

    if not sender_id or not message_id or not message_text:
        return None

    return sender_id, message_id, message_text


# ==================================================
# HOME AND HEALTH
# ==================================================

@app.get("/")
async def home():

    return {
        "project": "PSYCHOMODS AI",
        "status": "online",
        "version": "2.0.0",
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "groq_configured": groq_client is not None,
        "instagram_token_configured":
            bool(INSTAGRAM_ACCESS_TOKEN),
        "account_id_configured":
            bool(INSTAGRAM_ACCOUNT_ID),
        "api_mode": API_MODE,
    }


# ==================================================
# META WEBHOOK VERIFICATION
# ==================================================

@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):

    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if (
        mode == "subscribe"
        and VERIFY_TOKEN
        and token
        and hmac.compare_digest(token, VERIFY_TOKEN)
    ):
        logger.info("Webhook verification successful")
        return PlainTextResponse(challenge or "")

    raise HTTPException(403, "Webhook verification failed")


# ==================================================
# RECEIVE INSTAGRAM WEBHOOK
# ==================================================

@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):

    if not META_APP_SECRET:
        raise HTTPException(503, "Meta App Secret missing")

    body = await request.body()

    signature = request.headers.get(
        "X-Hub-Signature-256",
        "",
    )

    if not verify_meta_signature(body, signature):
        raise HTTPException(401, "Invalid webhook signature")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(400, "Invalid payload")

    if payload.get("object") != "instagram":
        return {"status": "ignored"}

    entries = payload.get("entry", [])

    if not isinstance(entries, list):
        return {"status": "received"}

    scheduled = 0
    skipped = 0

    for entry in entries:

        if not isinstance(entry, dict):
            continue

        # Actual Instagram professional account ID from webhook.
        webhook_account_id = str(entry.get("id", "")).strip()

        # Manual configured ID takes precedence if valid.
        account_id = (
            INSTAGRAM_ACCOUNT_ID
            if INSTAGRAM_ACCOUNT_ID
            else webhook_account_id
        )

        logger.info(
            "Webhook account ID present=%s; configured ID present=%s",
            bool(webhook_account_id),
            bool(INSTAGRAM_ACCOUNT_ID),
        )

        if not account_id or account_id == "0":
            logger.error("Invalid account ID; skipping entry")
            skipped += 1
            continue

        events = extract_messaging_events(entry)

        logger.info("Messaging events discovered: %d", len(events))

        for event in events:

            result = extract_incoming_message(event)

            if result is None:
                skipped += 1
                continue

            sender_id, message_id, message_text = result

            if sender_id == account_id:
                skipped += 1
                continue

            if not remember_message(message_id):
                skipped += 1
                continue

            background_tasks.add_task(
                process_dm,
                account_id,
                sender_id,
                message_text,
            )

            scheduled += 1

    logger.info(
        "Webhook completed: scheduled=%d skipped=%d",
        scheduled,
        skipped,
    )

    return {
        "status": "received",
        "scheduled": scheduled,
        "skipped": skipped,
    }


# ==================================================
# STARTUP
# ==================================================

@app.on_event("startup")
async def startup():

    logger.info("PSYCHOMODS AI starting")
    logger.info("API mode: %s", API_MODE)
    logger.info("API host: %s", META_GRAPH_BASE_URL)
    logger.info("Account ID configured: %s", bool(INSTAGRAM_ACCOUNT_ID))
    logger.info("Groq configured: %s", bool(groq_client))


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
