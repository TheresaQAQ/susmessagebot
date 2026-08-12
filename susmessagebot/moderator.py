import base64
import io
import logging
import math
import os
import random
import re
import time

from openai import APIConnectionError, APIStatusError, OpenAI
from PIL import Image

from . import config
from .config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)
from .llm_utils import should_disable_thinking
from .prompt_loader import DEFAULT_PROMPT_ID, render_prompt
from .utils import normalize_text
from .vector_store import get_similar_examples

client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
    timeout=60.0,
    max_retries=0,
)
_text_client = client.with_options(max_retries=0)
dashscope_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY or "not-configured",
    base_url=config.DASHSCOPE_BASE_URL,
    timeout=60.0,
    max_retries=0,
)

PROMPT_ID = os.getenv("PROMPT_ID", DEFAULT_PROMPT_ID)
IMAGE_PROMPT_ID = os.getenv("IMAGE_PROMPT_ID", "v4_zh_multilingual")
_TEXT_REQUEST_TIMEOUT_SECONDS = 30.0
_TEXT_REQUEST_RETRIES = 3
_TEXT_RETRY_BASE_SECONDS = 2.0
_TEXT_RETRY_MAX_SECONDS = 60.0
_TEXT_RETRY_JITTER_RATIO = 0.25
_NON_RETRYABLE_TEXT_STATUS_CODES = {400, 401, 403}
_IMAGE_MAX_SIDE = 1568
# Qwen3-VL rejects images with either side smaller than 28.
_IMAGE_MIN_SIDE = 28


def _parse_verdict(content: str, prefer_last: bool = False) -> str:
    text = (content or "").strip().upper()
    if text in {"SAFE", "BAN"}:
        return text
    decided = list(re.finditer(
        r"(?:CLASSIFIED AS|CLASSIFICATION IS|SHOULD BE CLASSIFIED AS|ANSWER(?: SHOULD BE)?|FINAL(?: ANSWER)?|VERDICT|输出|判定)\s*[:=]?\s*(BAN|SAFE)",
        text,
    ))
    if decided:
        return decided[-1].group(1)
    matches = list(re.finditer(r"\b(BAN|SAFE)\b", text))
    if not matches:
        # Unparseable model output is a classification failure, not a SAFE pass.
        return "REVIEW"
    return matches[-1].group(1) if prefer_last else matches[0].group(1)


def _should_disable_thinking(model: str | None = None) -> bool:
    return should_disable_thinking(model or config.SILICONFLOW_MODEL)


def _verdict_from_response(response) -> tuple[str, str, str]:
    message_obj = response.choices[0].message
    content = message_obj.content or ""
    reasoning = (
        getattr(message_obj, "reasoning_content", None)
        or getattr(message_obj, "reasoning", None)
        or ""
    )
    result = _parse_verdict(content)
    if result == "REVIEW" and reasoning:
        reasoned = _parse_verdict(reasoning, prefer_last=True)
        if reasoned != "REVIEW":
            result = reasoned
    return result, content, reasoning


def _classify_with_dashscope(create_kwargs: dict, request_type: str) -> str:
    """Run one bounded DashScope fallback request."""
    if not config.DASHSCOPE_API_KEY:
        logging.warning(
            "%s classification fallback unavailable: DASHSCOPE_API_KEY is empty",
            request_type.capitalize(),
        )
        return "REVIEW"

    fallback_kwargs = dict(create_kwargs)
    fallback_kwargs["model"] = config.DASHSCOPE_VISION_MODEL
    fallback_kwargs.pop("extra_body", None)
    try:
        logging.info(
            "%s classification falling back to provider=dashscope model=%s",
            request_type.capitalize(),
            config.DASHSCOPE_VISION_MODEL,
        )
        response = dashscope_client.chat.completions.create(**fallback_kwargs)
        result, content, reasoning = _verdict_from_response(response)
        logging.info(
            "classify_%s provider=dashscope model=%s raw=%r "
            "reasoning_tail=%r verdict=%s",
            request_type,
            config.DASHSCOPE_VISION_MODEL,
            content[:80],
            str(reasoning)[-120:],
            result,
        )
        return result
    except Exception as e:
        logging.error("%s DashScope fallback error: %s", request_type.capitalize(), e)
        return "REVIEW"


def _extract_error_details(error: Exception) -> tuple[int | None, str, str]:
    """Read both OpenAI-style and SiliconFlow-style error payloads."""
    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    payload = body.get("error", body) if isinstance(body, dict) else body

    code = ""
    message = ""
    if isinstance(payload, dict):
        code = str(payload.get("code") or "")
        message = str(payload.get("message") or "")
    elif isinstance(payload, str):
        message = payload

    if isinstance(body, dict):
        code = code or str(body.get("code") or "")
        message = message or str(body.get("message") or "")

    return status_code, code, message or str(error)


def _is_retryable_text_error(error: Exception) -> bool:
    if isinstance(error, APIConnectionError):
        return True
    if not isinstance(error, APIStatusError):
        return False

    status_code, _, _ = _extract_error_details(error)
    return status_code not in _NON_RETRYABLE_TEXT_STATUS_CODES


def _provider_retry_delay_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    candidates = (
        (headers.get("retry-after-ms"), 1000.0),
        (headers.get("retry-after"), 1.0),
    )
    for raw_value, divisor in candidates:
        if raw_value is None:
            continue
        try:
            seconds = float(raw_value) / divisor
        except (TypeError, ValueError):
            continue
        if math.isfinite(seconds) and seconds >= 0:
            return min(seconds, _TEXT_RETRY_MAX_SECONDS)
    return None


def _text_retry_delay_seconds(error: Exception, failed_attempt: int) -> float:
    provider_delay = _provider_retry_delay_seconds(error)
    if provider_delay is not None:
        return provider_delay

    base_delay = min(
        _TEXT_RETRY_BASE_SECONDS * (2 ** (failed_attempt - 1)),
        _TEXT_RETRY_MAX_SECONDS,
    )
    return base_delay + random.uniform(0, base_delay * _TEXT_RETRY_JITTER_RATIO)


def _image_to_data_url(image_bytes: bytes) -> str:
    """Normalize attachment bytes into a JPEG data URL for vision APIs."""
    img = Image.open(io.BytesIO(image_bytes))
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    shortest = min(w, h)
    if longest > _IMAGE_MAX_SIDE:
        scale = _IMAGE_MAX_SIDE / longest
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
        w, h = img.size
        shortest = min(w, h)
    if shortest < _IMAGE_MIN_SIDE:
        scale = _IMAGE_MIN_SIDE / shortest
        img = img.resize(
            (max(_IMAGE_MIN_SIDE, int(w * scale)), max(_IMAGE_MIN_SIDE, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def classify_message(message: str) -> str:
    """
    Classifies a Discord message as SAFE, BAN, or REVIEW.

    REVIEW means classification failed and the message requires manual review.
    """
    normalized = normalize_text(message)
    if not normalized:
        return "SAFE"

    try:
        examples = get_similar_examples(normalized)
        system_prompt = render_prompt(PROMPT_ID, examples)

        model = config.SILICONFLOW_MODEL
        create_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<message>{normalized}</message>"},
            ],
            "max_tokens": 64,
            "temperature": 0,
        }
        if _should_disable_thinking(model):
            create_kwargs["extra_body"] = {"enable_thinking": False}
    except Exception as e:
        logging.error("Text classification setup error: %s", e)
        return "REVIEW"

    total_attempts = _TEXT_REQUEST_RETRIES + 1
    for attempt in range(1, total_attempts + 1):
        try:
            response = _text_client.chat.completions.create(
                **create_kwargs,
                timeout=_TEXT_REQUEST_TIMEOUT_SECONDS,
            )
            result, content, reasoning = _verdict_from_response(response)
        except Exception as e:
            status_code, error_code, error_message = _extract_error_details(e)
            retryable = _is_retryable_text_error(e)
            if retryable and attempt < total_attempts:
                delay = _text_retry_delay_seconds(e, attempt)
                logging.warning(
                    "Text classification request failed on attempt %s/%s "
                    "status=%s code=%s error=%s; retrying in %.2fs",
                    attempt,
                    total_attempts,
                    status_code,
                    error_code or "-",
                    error_message[:300],
                    delay,
                )
                time.sleep(delay)
                continue
            logging.error(
                "Text classification request failed on attempt %s/%s "
                "retryable=%s status=%s code=%s error=%s",
                attempt,
                total_attempts,
                retryable,
                status_code,
                error_code or "-",
                error_message[:300],
            )
            break

        logging.info(
            "classify_message prompt=%s model=%s attempt=%s/%s raw=%r "
            "reasoning_tail=%r verdict=%s",
            PROMPT_ID,
            model,
            attempt,
            total_attempts,
            content[:80],
            str(reasoning)[-120:],
            result,
        )
        if result != "REVIEW":
            return result
        logging.warning(
            "Text classification returned an unparseable SiliconFlow response; "
            "using DashScope fallback"
        )
        break

    return _classify_with_dashscope(create_kwargs, "message")


def classify_image(image_bytes: bytes) -> str:
    """
    Classify an image attachment via DashScope multimodal LLM (no OCR).
    Return REVIEW on errors so callers can request manual moderation.
    """
    if not config.DASHSCOPE_API_KEY:
        logging.warning(
            "Image classification unavailable: DASHSCOPE_API_KEY is empty"
        )
        return "REVIEW"

    try:
        data_url = _image_to_data_url(image_bytes)
        # Image moderation has no text query for RAG and uses its own prompt version.
        system_prompt = render_prompt(IMAGE_PROMPT_ID, "")
        vision_model = config.DASHSCOPE_VISION_MODEL
        create_kwargs = {
            "model": vision_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请审核这张社区图片（可能含文字、二维码、广告版式或引流视觉）。"
                                "按系统规则判定意图。只输出 SAFE 或 BAN。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            "max_tokens": 64,
            "temperature": 0,
        }
    except Exception as e:
        logging.error("Image classification setup error: %s", e)
        return "REVIEW"

    try:
        logging.info(
            "Image classification requesting provider=dashscope model=%s",
            vision_model,
        )
        response = dashscope_client.chat.completions.create(**create_kwargs)
        result, content, reasoning = _verdict_from_response(response)
        logging.info(
            "classify_image prompt=%s provider=%s model=%s raw=%r "
            "reasoning_tail=%r verdict=%s",
            IMAGE_PROMPT_ID,
            "dashscope",
            vision_model,
            content[:80],
            str(reasoning)[-120:],
            result,
        )
        return result
    except Exception as e:
        logging.error("Image DashScope classification error: %s", e)
        return "REVIEW"
