"""Shared helpers for SiliconFlow / OpenAI-compatible chat models."""


def should_disable_thinking(model: str) -> bool:
    """Return True when the provider needs enable_thinking=false for this model.

    Some Qwen3-family models (e.g. Qwen3-VL Instruct) reject enable_thinking
    entirely — only send the flag for models that actually support it.
    """
    # Qwen3-VL / Omni Instruct endpoints do not accept enable_thinking at all.
    if ("-VL-" in model or "-Omni-" in model) and "Thinking" not in model:
        return False
    return (
        "Qwen3" in model
        or "GLM-4.5" in model
        or "GLM-4.6" in model
        or "GLM-4.7" in model
        or "DeepSeek-V3" in model
    )
