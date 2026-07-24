import base64
import io
import logging
import os
import re

from openai import OpenAI
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
)

PROMPT_ID = os.getenv("PROMPT_ID", DEFAULT_PROMPT_ID)
_IMAGE_MAX_SIDE = 1568


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


def _image_to_data_url(image_bytes: bytes) -> str:
    """Normalize attachment bytes into a JPEG data URL for vision APIs."""
    img = Image.open(io.BytesIO(image_bytes))
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > _IMAGE_MAX_SIDE:
        scale = _IMAGE_MAX_SIDE / longest
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
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

        response = client.chat.completions.create(**create_kwargs)
        result, content, reasoning = _verdict_from_response(response)
        logging.info(
            "classify_message prompt=%s model=%s raw=%r reasoning_tail=%r verdict=%s",
            PROMPT_ID,
            model,
            content[:80],
            str(reasoning)[-120:],
            result,
        )
        return result
    except Exception as e:
        logging.error("Text classification error: %s", e)
        return "REVIEW"


def classify_image(image_bytes: bytes) -> str:
    """
    Classify an image attachment via multimodal LLM (no OCR).
    Return REVIEW on errors so callers can request manual moderation.
    """
    try:
        data_url = _image_to_data_url(image_bytes)
        # No text for RAG; keep the same rules prompt with empty examples.
        system_prompt = render_prompt(PROMPT_ID, "")
        vision_model = config.SILICONFLOW_VISION_MODEL or config.SILICONFLOW_MODEL
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
        if _should_disable_thinking(vision_model):
            create_kwargs["extra_body"] = {"enable_thinking": False}

        response = client.chat.completions.create(**create_kwargs)
        result, content, reasoning = _verdict_from_response(response)
        logging.info(
            "classify_image prompt=%s model=%s raw=%r reasoning_tail=%r verdict=%s",
            PROMPT_ID,
            vision_model,
            content[:80],
            str(reasoning)[-120:],
            result,
        )
        return result
    except Exception as e:
        logging.error("Image classification error: %s", e)
        return "REVIEW"
