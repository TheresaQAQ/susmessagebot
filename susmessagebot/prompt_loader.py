"""Load versioned system prompts from prompts/*.md"""

from __future__ import annotations

import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT_ID = "v3_zh_general"


def list_prompt_ids() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))


def load_prompt(prompt_id: str = DEFAULT_PROMPT_ID) -> tuple[str, dict]:
    """
    Returns (prompt_body, meta).
    Prompt body must contain {examples} placeholder.
    """
    path = PROMPTS_DIR / f"{prompt_id}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    text = path.read_text(encoding="utf-8")
    meta: dict = {"id": prompt_id}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            front, body = parts[1], parts[2]
            for line in front.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()

    body = body.strip() + "\n"
    if "{examples}" not in body:
        raise ValueError(f"Prompt {prompt_id} missing {{examples}} placeholder")
    return body, meta


def render_prompt(prompt_id: str, examples: str) -> str:
    body, _ = load_prompt(prompt_id)
    # Avoid str.format breaking on other braces; only replace examples.
    return body.replace("{examples}", examples or "(none)")
