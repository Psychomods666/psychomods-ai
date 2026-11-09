import hashlib
import hmac
import json
import logging
import os
import re
from collections import deque
from contextlib import asynccontextmanager
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

VERSION = "2.4.0"


def clean_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if len(value) >= 2:
        if value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1].strip()

    return value


VERIFY_TOKEN = clean_env("VERIFY_TOKEN")
META_APP_SECRET = clean_env("META_APP_SECRET")
INSTAGRAM_APP_SECRET = clean_env("INSTAGRAM_APP_SECRET")
INSTAGRAM_ACCESS_TOKEN = clean_env("INSTAGRAM_ACCESS_TOKEN")

GROQ_API_KEY = clean_env("GROQ_API_KEY")
GROQ_MODEL = clean_env("GROQ_MODEL") or "openai/gpt-oss-20b"
GROQ_REASONING_EFFORT = clean_env("GROQ_REASONING_EFFORT") or "low"
GROQ_MAX_TOKENS = int(clean_env("GROQ_MAX_TOKENS") or "800")

GRAPH_API_VERSION = clean_env("GRAPH_API_VERSION") or "v27.0"

API_MODE = (
    clean_env("INSTAGRAM_API_MODE") or "instagram_login"
).lower()

API_HOSTS = {
    "instagram_login": "https://graph.instagram.com",
    "facebook_login": "https://graph.facebook.com",
}

if API_MODE not in API_HOSTS:
    raise RuntimeError(
        "INSTAGRAM_API_MODE must be instagram_login or facebook_login"
    )

META_GRAPH_BASE_URL = (
    clean_env("META_GRAPH_BASE_URL") or API_HOSTS[API_MODE]
).rstrip("/")

INSTAGRAM_ACCOUNT_ID = (
    clean_env("INSTAGRAM_SENDER_ID")
    or clean_env("INSTAGRAM_SEND_ID")
)

groq_client = (
    AsyncGroq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)


# ==================================================
# STARTUP (lifespan replaces deprecated on_event)
# ==================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("PSYCHOMODS AI starting")
    logger.info("Version: %s", VERSION)
    logger.info("API mode: %s", API_MODE)
    logger.info("API host: %s", META_GRAPH_BASE_URL)
    logger.info("Account ID configured: %s", bool(INSTAGRAM_ACCOUNT_ID))
    logger.info(
        "Instagram access token configured: %s",
        bool(INSTAGRAM_ACCESS_TOKEN),
    )
    logger.info("Meta App Secret configured: %s", bool(META_APP_SECRET))
    logger.info(
        "Instagram App Secret configured: %s",
        bool(INSTAGRAM_APP_SECRET),
    )
    logger.info("Groq configured: %s", bool(groq_client))
    logger.info(
        "Signature secret candidates: %d",
        len(get_signature_secrets()),
    )
    yield


app = FastAPI(
    title="PSYCHOMODS AI",
    description="AI-powered Instagram DM assistant",
    version=VERSION,
    lifespan=lifespan,
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
        old_id = recent_message_ids.popleft()
        seen_message_ids.discard(old_id)

    return True


def forget_message(message_id: str) -> None:
    """Allow a message to be processed again (used after a failed send)."""
    seen_message_ids.discard(message_id)

    try:
        recent_message_ids.remove(message_id)
    except ValueError:
        pass


# ==================================================
# META WEBHOOK SIGNATURE VALIDATION
# ==================================================

def get_signature_secrets() -> list[tuple[str, str]]:
    candidates = []

    if META_APP_SECRET:
        candidates.append(("META_APP_SECRET", META_APP_SECRET))

    if INSTAGRAM_APP_SECRET:
        candidates.append(("INSTAGRAM_APP_SECRET", INSTAGRAM_APP_SECRET))

    unique = []
    seen = set()

    for name, secret in candidates:
        if secret not in seen:
            seen.add(secret)
            unique.append((name, secret))

    return unique


def verify_meta_signature(
    body: bytes,
    signature: str,
) -> tuple[bool, str]:

    if not signature:
        return False, "missing_header"

    signature = signature.strip()

    if not signature.startswith("sha256="):
        return False, "invalid_header_format"

    received_digest = signature[len("sha256="):]

    if not re.fullmatch(r"[0-9a-fA-F]{64}", received_digest):
        return False, "invalid_digest_format"

    secrets = get_signature_secrets()

    if not secrets:
        return False, "no_secret_configured"

    for secret_name, secret_value in secrets:
        expected_digest = hmac.new(
            secret_value.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()

        if hmac.compare_digest(
            expected_digest,
            received_digest.lower(),
        ):
            return True, secret_name

    return False, "digest_mismatch"


# ==================================================
# DIAGNOSTICS
# ==================================================

@app.get("/")
async def home():
    return {
        "project": "PSYCHOMODS AI",
        "status": "online",
        "version": VERSION,
    }


@app.head("/")
async def home_head():
    return PlainTextResponse("", status_code=200)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": VERSION,
        "groq_configured": groq_client is not None,
        "instagram_token_configured": bool(INSTAGRAM_ACCESS_TOKEN),
        "account_id_configured": bool(INSTAGRAM_ACCOUNT_ID),
        "api_mode": API_MODE,
    }


@app.get("/diagnostics")
async def diagnostics():
    return {
        "project": "PSYCHOMODS AI",
        "version": VERSION,
        "configured_mode": API_MODE,
        "configured_host": META_GRAPH_BASE_URL,
        "graph_version": GRAPH_API_VERSION,
        "access_token_configured": bool(INSTAGRAM_ACCESS_TOKEN),
        "groq_configured": groq_client is not None,
        "groq_model": GROQ_MODEL,
        "meta_app_secret_configured": bool(META_APP_SECRET),
        "instagram_app_secret_configured": bool(INSTAGRAM_APP_SECRET),
        "signature_secret_count": len(get_signature_secrets()),
        "verify_token_configured": bool(VERIFY_TOKEN),
        "account_id_configured": bool(INSTAGRAM_ACCOUNT_ID),
        "account_id_is_zero": INSTAGRAM_ACCOUNT_ID == "0",
    }


@app.get("/diagnostics/account")
async def check_account():

    if not INSTAGRAM_ACCESS_TOKEN:
        raise HTTPException(503, "Instagram access token missing")

    if not INSTAGRAM_ACCOUNT_ID or INSTAGRAM_ACCOUNT_ID == "0":
        raise HTTPException(400, "Valid Instagram account ID missing")

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
                    "Authorization": f"Bearer {INSTAGRAM_ACCESS_TOKEN}"
                },
            )

        try:
            data = response.json()
        except ValueError:
            data = {"error": "Meta returned a non-JSON response"}

        return {
            "configured_account_id": INSTAGRAM_ACCOUNT_ID,
            "api_host": META_GRAPH_BASE_URL,
            "http_status": response.status_code,
            "success": response.is_success,
            "result": data,
        }

    except httpx.RequestError as exc:
        logger.exception("Account diagnostic connection failed")

        raise HTTPException(
            502,
            f"Meta connection error: {type(exc).__name__}",
        )


# ==================================================
# GROQ AI REPLY
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
                    "You are the official PSYCHO MOD'S Instagram AI "
                    "assistant. Be friendly, professional, helpful, "
                    "and concise. Support ethical cybersecurity "
                    "education and defensive security. Never assist "
                    "unauthorized access, credential theft, or malware "
                    "deployment. Never invent company facts, prices, "
                    "or links."
                ),
            },
            {
                "role": "user",
                "content": message[:3000],
            },
        ],
        temperature=0.5,
        # Reasoning tokens count toward this limit, so keep it generous
        max_completion_tokens=GROQ_MAX_TOKENS,
        reasoning_effort=GROQ_REASONING_EFFORT,
    )

    reply = completion.choices[0].message.content

    if not reply or not reply.strip():
        raise RuntimeError("Groq returned an empty response")

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
        logger.error("Instagram access token is missing")
        return False

    if not account_id or account_id == "0":
        logger.error("Invalid Instagram account ID")
        return False

    if not recipient_id or recipient_id == "0":
        logger.error("Invalid Instagram recipient ID")
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
        "recipient": {"id": recipient_id},
        "message": {"text": reply_text[:1000]},
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
            "Instagram send API failed: HTTP %s; response=%s",
            response.status_code,
            response.text[:1000],
        )

        return False

    except httpx.RequestError as exc:
        logger.exception(
            "Instagram send request failed: %s",
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
    message_id: str,
):

    try:
        logger.info("Generating AI reply")

        reply = await generate_ai_reply(message_text)

        logger.info("AI reply generated")

        sent = await send_instagram_dm(
            account_id,
            sender_id,
            reply,
        )

        if not sent:
            logger.error("AI reply could not be sent to Instagram")
            forget_message(message_id)

    except Exception as exc:
        logger.exception(
            "DM processing failed: %s",
            type(exc).__name__,
        )
        forget_message(message_id)


# ==================================================
# WEBHOOK EVENT PARSING
# ==================================================

def extract_messaging_events(
    entry: dict[str, Any],
) -> list[dict[str, Any]]:

    events = []

    messaging = entry.get("messaging", [])

    if isinstance(messaging, list):
        events.extend(
            item
            for item in messaging
            if isinstance(item, dict)
        )

    changes = entry.get("changes", [])

    if isinstance(changes, list):
        for change in changes:

            if not isinstance(change, dict):
                continue

            if change.get("field") not in ("messages", "messaging"):
                continue

            value = change.get("value")

            if not isinstance(value, dict):
                continue

            if isinstance(value.get("message"), dict):
                events.append(value)

    return events


def extract_incoming_message(event: dict[str, Any]):

    message = event.get("message")

    if not isinstance(message, dict):
        return None

    if message.get("is_echo"):
        return None

    sender = event.get("sender") or {}

    if not isinstance(sender, dict):
        return None

    sender_id = str(sender.get("id", "")).strip()
    message_id = str(message.get("mid", "")).strip()
    message_text = message.get("text", "")

    if not isinstance(message_text, str):
        return None

    message_text = message_text.strip()

    if not sender_id or not message_id or not message_text:
        return None

    return sender_id, message_id, message_text


# ==================================================
# META WEBHOOK VERIFICATION - GET
# ==================================================

@app.get(
    "/webhook",
    response_class=PlainTextResponse,
)
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
        logger.info("Webhook GET verification successful")
        return PlainTextResponse(challenge or "")

    logger.warning("Webhook GET verification rejected")

    raise HTTPException(403, "Webhook verification failed")


# ==================================================
# RECEIVE INSTAGRAM WEBHOOK - POST
# ==================================================

@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):

    body = await request.body()

    signature = request.headers.get(
        "X-Hub-Signature-256",
        "",
    )

    if not get_signature_secrets():
        logger.error("No webhook signature secret is configured")
        raise HTTPException(
            503,
            "Webhook signature secret missing",
        )

    if not signature:
        logger.warning("Webhook signature rejected: header missing")
        raise HTTPException(
            401,
            "Webhook signature header missing",
        )

    valid, result = verify_meta_signature(body, signature)

    if not valid:
        logger.warning(
            "WEBHOOK SIGNATURE DIAGNOSTICS | "
            "reason=%s | "
            "header_present=%s | "
            "header_length=%d | "
            "header_prefix=%s | "
            "content_type=%s | "
            "body_length=%d | "
            "body_sha256=%s",
            result,
            bool(signature),
            len(signature),
            signature.split("=", 1)[0] if "=" in signature else "invalid",
            request.headers.get("content-type", "missing"),
            len(body),
            hashlib.sha256(body).hexdigest()[:16],
        )

        raise HTTPException(
            401,
            "Invalid webhook signature",
        )

    logger.info(
        "Webhook signature validated using %s",
        result,
    )

    try:
        payload = json.loads(body)

    except (ValueError, UnicodeDecodeError):
        logger.warning("Webhook received invalid JSON")
        raise HTTPException(400, "Invalid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(400, "Invalid payload")

    if payload.get("object") != "instagram":
        logger.info(
            "Ignoring webhook object: %s",
            payload.get("object"),
        )
        return {"status": "ignored"}

    entries = payload.get("entry", [])

    if not isinstance(entries, list):
        return {
            "status": "received",
            "scheduled": 0,
        }

    scheduled = 0
    skipped = 0

    for entry in entries:

        if not isinstance(entry, dict):
            continue

        webhook_account_id = str(
            entry.get("id", "")
        ).strip()

        account_id = (
            INSTAGRAM_ACCOUNT_ID
            or webhook_account_id
        )

        if not account_id or account_id == "0":
            logger.error("Invalid Instagram account ID in webhook")
            skipped += 1
            continue

        events = extract_messaging_events(entry)

        logger.info(
            "Messaging events discovered: %d",
            len(events),
        )

        for event in events:

            message_result = extract_incoming_message(event)

            if message_result is None:
                skipped += 1
                continue

            sender_id, message_id, message_text = message_result

            # SAFE RECIPIENT ID DIAGNOSTICS
            logger.info(
                "DM RECIPIENT CHECK | "
                "sender_id_last4=%s | "
                "sender_id_length=%d | "
                "account_id_last4=%s | "
                "account_id_length=%d | "
                "same_id=%s",
                sender_id[-4:],
                len(sender_id),
                account_id[-4:],
                len(account_id),
                sender_id == account_id,
            )

            if sender_id == account_id:
                logger.info("Own account message ignored")
                skipped += 1
                continue

            if not remember_message(message_id):
                logger.info("Duplicate message ignored")
                skipped += 1
                continue

            logger.info("Incoming Instagram message received")

            background_tasks.add_task(
                process_dm,
                account_id,
                sender_id,
                message_text,
                message_id,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
