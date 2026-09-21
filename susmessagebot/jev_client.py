from __future__ import annotations

import logging

import requests

from . import config
from .jev_policy import POLICY_CRITERIA, POLICY_INSTRUCTIONS
from .text_context import ContextTurn

_JEV_TIMEOUT_SECONDS = 10.0
_VALID_VERDICTS = {"BAN", "SAFE"}


def _evaluation_url() -> str:
    base = config.AI_GATEWAY_BASE_URL.rstrip("/")
    if base.endswith("/evaluate") or base.endswith("/evaluation-model"):
        return base
    if base.endswith("/v4/ai"):
        return f"{base}/evaluation-model"
    return f"{base}/evaluate"


def _questions() -> dict:
    return {
        "verdict": {
            "type": "choice",
            "instructions": POLICY_INSTRUCTIONS,
            "criteria": POLICY_CRITERIA,
        }
    }


def classify_with_jev(
    message: str,
    context: list[ContextTurn] | None = None,
) -> str | None:
    """
    Ask Jev for BAN/SAFE. Return None so callers can fall through.

    Missing key, transport errors, and non-verdict payloads all return None.
    """
    key = (config.AI_GATEWAY_API_KEY or "").strip()
    if not key:
        return None

    state: dict = {"message": message}
    if context:
        state["context"] = [
            {"author": turn.author, "user_id": turn.user_id, "text": turn.text}
            for turn in context
        ]

    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": config.JEV_MODEL,
            "state": state,
            "questions": _questions(),
        }
        url = _evaluation_url()
        if url.endswith("/evaluation-model"):
            headers["ai-evaluation-model-specification-version"] = "4"
            headers["ai-model-id"] = config.JEV_MODEL
            payload.pop("model", None)
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=_JEV_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logging.warning("Jev classification request failed: %s", e)
        return None

    if not response.ok:
        logging.warning(
            "Jev classification HTTP %s: %s",
            response.status_code,
            (response.text or "")[:300],
        )
        return None

    try:
        payload = response.json()
    except ValueError:
        logging.warning("Jev classification returned non-JSON")
        return None

    answers = payload.get("answers") if isinstance(payload, dict) else None
    verdict = answers.get("verdict") if isinstance(answers, dict) else None
    choice = verdict.get("choice") if isinstance(verdict, dict) else None
    if isinstance(choice, str) and choice.upper() in _VALID_VERDICTS:
        return choice.upper()
    logging.warning("Jev classification returned unusable verdict: %r", choice)
    return None
