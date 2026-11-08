# PSYCHOMODS AI — Instagram Webhook Starter

A small FastAPI service for Meta's Instagram webhook verification and event reception.

**This is a webhook foundation, not a finished auto-reply bot.** It verifies Meta's GET challenge, validates POST webhook signatures, and acknowledges events. It does not yet generate or send Instagram replies.

## 1. Create your secrets

Create a long random `VERIFY_TOKEN` value. It can be any private string you choose.

Get your **Meta App Secret** from your Meta app's Settings → Basic page.

For local development:
1. Copy `.env.example` to `.env`.
2. Fill in `VERIFY_TOKEN` and `META_APP_SECRET`.
3. Keep `.env` private. Never commit it to GitHub or paste it into chat.

For Render, add both values under your service's Environment settings instead of uploading `.env`.

## 2. Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Set environment variables in your terminal or use a local environment loader of your choice. Then run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Check:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

Meta cannot verify a localhost URL. Use a public HTTPS deployment for Meta webhooks.

## 3. Deploy to Render

1. Create a **private** GitHub repository and upload these project files.
2. In Render, choose New → Web Service and connect that repository.
3. Use:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:
   - `VERIFY_TOKEN`
   - `META_APP_SECRET`
   - `LOG_LEVEL` = `INFO`
5. Deploy and confirm `/health` returns `{"status":"healthy"}`.

The included `render.yaml` can also be used as a Blueprint configuration. Free hosting may sleep when idle and is not suitable for guaranteed real-time production delivery.

## 4. Configure Meta

In the Instagram API product's webhook configuration, enter:

- Callback URL: `https://YOUR-RENDER-SERVICE.onrender.com/webhook`
- Verify token: the exact same value you configured as `VERIFY_TOKEN`

Click Verify and Save. Subscribe to the event fields you need, such as `messages` and `comments`, when available for your selected API/login configuration.

The callback must be publicly reachable over HTTPS. Meta's app mode, access level, permissions, account type, and review status determine which real events can be delivered.

## 5. Security and next steps

- Never put Instagram access tokens or the Meta App Secret in source code.
- Never commit `.env`.
- Do not share tokens in screenshots or chat.
- The POST endpoint rejects events unless `X-Hub-Signature-256` validates against the Meta App Secret.
- Do not log private message/comment contents.
- Before enabling replies, implement event parsing, duplicate-event protection, permission/error handling, user-visible safeguards, and the current Instagram messaging/comment policy limits.
- Test with your own authorized account before seeking the access required for other users.

## Project files

- `main.py` — FastAPI webhook endpoints
- `requirements.txt` — Python dependencies
- `.env.example` — safe environment-variable template
- `.gitignore` — excludes secrets and Python cache
- `render.yaml` — optional Render deployment blueprint
