"""Compare text and image requests to Qwen/Qwen3.5-4B via the app client.

Usage:
  python -m scripts.demo_qwen35_multimodal
  python -m scripts.demo_qwen35_multimodal --image path/to/image.png --timeout 90
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from susmessagebot import config, moderator

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TIMEOUT_SECONDS = 60.0


def _sample_image_bytes() -> bytes:
    image = Image.new("RGB", (320, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 304, 144), outline="black", width=3)
    draw.text((48, 68), "Qwen image request test", fill="black")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def _response_text(response: Any) -> str:
    message = response.choices[0].message
    return (message.content or getattr(message, "reasoning_content", None) or "").strip()


def _error_details(error: Exception) -> dict[str, Any]:
    status_code, error_code, error_message = moderator._extract_error_details(error)
    cause = error.__cause__
    details: dict[str, Any] = {
        "error_type": type(error).__name__,
        "status_code": status_code,
        "error_code": error_code or None,
        "error": error_message[:500],
    }
    if cause is not None:
        details["cause_type"] = type(cause).__name__
        details["cause"] = str(cause)[:500]
    return details


def _request(kind: str, *, model: str, messages: list[dict], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0,
        }
        if moderator._should_disable_thinking(model):
            create_kwargs["extra_body"] = {"enable_thinking": False}

        response = moderator._text_client.chat.completions.create(
            **create_kwargs,
            timeout=timeout,
        )
        return {
            "kind": kind,
            "ok": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "request_id": getattr(response, "_request_id", None),
            "response": _response_text(response)[:500],
        }
    except Exception as error:
        return {
            "kind": kind,
            "ok": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            **_error_details(error),
        }


def run_demo(*, model: str, image_bytes: bytes, timeout: float) -> list[dict[str, Any]]:
    data_url = moderator._image_to_data_url(image_bytes)
    text_messages = [
        {"role": "user", "content": "只回复 TEXT_OK"},
    ]
    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "简短描述这张图片。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    print(
        json.dumps(
            {
                "event": "config",
                "model": model,
                "base_url": config.SILICONFLOW_BASE_URL,
                "timeout_seconds": timeout,
                "normalized_image_bytes": len(data_url),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results = []
    for kind, messages in (("text", text_messages), ("image", image_messages)):
        result = _request(kind, model=model, messages=messages, timeout=timeout)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare text and image chat.completions requests with the app client."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, help="Image file; omit to use a generated JPEG")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if not config.SILICONFLOW_API_KEY:
        parser.error("SILICONFLOW_API_KEY (or OPENROUTER_API_KEY) is required")

    image_bytes = args.image.read_bytes() if args.image else _sample_image_bytes()
    results = run_demo(model=args.model, image_bytes=image_bytes, timeout=args.timeout)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
