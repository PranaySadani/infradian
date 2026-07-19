"""A stdlib-only OpenAI client.

Deliberately not the official SDK. The Vercel serverless function that serves this app is capped
at 500MB and ships stdlib plus numpy only, so every dependency has to earn its place. The OpenAI
REST API is plain HTTPS, so `urllib` is enough, and using it means the exact same module runs in
the tiny serverless function and in the full Docker backend with no divergence.

Everything here fails soft. If no key is configured, `available()` is False and callers fall back
to their deterministic paths. A network error, a timeout, or a malformed response raises
`OpenAIUnavailable`, which callers treat the same way as "no key". The app is never allowed to
break because a model call did.
"""

from __future__ import annotations

import contextlib
import json
import os
import ssl
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API_ROOT = "https://api.openai.com/v1"

# Model choices are overridable so a deployment can move without a code change.
CHAT_MODEL = os.environ.get("INFRADIAN_CHAT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.environ.get("INFRADIAN_VISION_MODEL", "gpt-4o-mini")
TRANSCRIBE_MODEL = os.environ.get("INFRADIAN_TRANSCRIBE_MODEL", "whisper-1")

DEFAULT_TIMEOUT = float(os.environ.get("INFRADIAN_LLM_TIMEOUT", "20"))


class OpenAIUnavailable(RuntimeError):
    """Raised when a call cannot be completed. Callers must degrade, never propagate to the user."""


def _load_dotenv_once() -> None:
    """Populate os.environ from a local .env if present.

    Vercel injects real environment variables, so this is a no-op there. It exists so that running
    the Docker backend or a local script picks up the key the developer put in .env.
    """
    if os.environ.get("_INFRADIAN_DOTENV_LOADED"):
        return
    os.environ["_INFRADIAN_DOTENV_LOADED"] = "1"
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            if k and v and not os.environ.get(k):
                os.environ[k] = v
        break


def api_key() -> str | None:
    _load_dotenv_once()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key or None


def available() -> bool:
    return api_key() is not None


def _request(path: str, data: bytes, content_type: str, timeout: float) -> dict:
    key = api_key()
    if not key:
        raise OpenAIUnavailable("no OPENAI_API_KEY configured")

    req = urllib.request.Request(
        f"{API_ROOT}{path}",
        data=data,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": content_type},
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        with contextlib.suppress(Exception):
            detail = json.loads(e.read().decode()).get("error", {}).get("message", "")
        raise OpenAIUnavailable(f"HTTP {e.code}: {detail or 'request rejected'}") from e
    except Exception as e:  # noqa: BLE001 — timeouts, DNS, TLS, malformed JSON
        raise OpenAIUnavailable(f"{type(e).__name__}") from e


def chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 700,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Chat completion constrained to a JSON object response. Returns the parsed object."""
    payload = {
        "model": model or CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body = _request("/chat/completions", json.dumps(payload).encode(), "application/json", timeout)
    try:
        return json.loads(body["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001
        raise OpenAIUnavailable("model returned unparseable JSON") from e


def vision_json(
    system: str,
    user: str,
    image_data_uri: str,
    *,
    model: str | None = None,
    max_tokens: int = 500,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Vision call over a data: URI image, constrained to a JSON object response."""
    payload = {
        "model": model or VISION_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": image_data_uri, "detail": "low"}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body = _request("/chat/completions", json.dumps(payload).encode(), "application/json", timeout)
    try:
        return json.loads(body["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001
        raise OpenAIUnavailable("vision model returned unparseable JSON") from e


def transcribe(audio_bytes: bytes, filename: str = "clip.webm", *, timeout: float = 30.0) -> str:
    """Whisper transcription. Hand-rolled multipart so no HTTP library is needed."""
    if not audio_bytes:
        raise OpenAIUnavailable("empty audio")

    boundary = f"----infradian{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )

    field("model", TRANSCRIBE_MODEL)
    field("response_format", "json")
    field("language", "en")

    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(audio_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())

    body = _request(
        "/audio/transcriptions",
        b"".join(parts),
        f"multipart/form-data; boundary={boundary}",
        timeout,
    )
    text = (body.get("text") or "").strip()
    if not text:
        raise OpenAIUnavailable("transcription returned no text")
    return text


def status() -> dict:
    """Non-secret description of the AI configuration, safe to expose to the frontend."""
    return {
        "enabled": available(),
        "chat_model": CHAT_MODEL,
        "vision_model": VISION_MODEL,
        "transcribe_model": TRANSCRIBE_MODEL,
        "note": (
            "OpenAI-backed paths are active."
            if available()
            else "No key configured. Deterministic fallbacks are in use and every feature still works."
        ),
    }
