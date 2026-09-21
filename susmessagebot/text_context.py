from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextTurn:
    author: str
    user_id: int
    text: str


def format_text_classification_prompt(
    message: str,
    context: list[ContextTurn] | None = None,
    *,
    author: str = "",
    user_id: int | None = None,
) -> str:
    """Render optional channel context plus the current message for text models."""
    parts: list[str] = []
    if context:
        lines = [
            f"[{index}] {turn.author} (`{turn.user_id}`): {turn.text}"
            for index, turn in enumerate(context, start=1)
        ]
        parts.append("<context>\n" + "\n".join(lines) + "\n</context>")

    attrs = ""
    if author:
        attrs += f' author="{author}"'
    if user_id is not None:
        attrs += f' id="{user_id}"'
    parts.append(f"<message{attrs}>\n{message}\n</message>")
    return "\n".join(parts)
