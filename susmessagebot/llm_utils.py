"""Shared helpers for SiliconFlow / OpenAI-compatible chat models."""


def should_disable_thinking(model: str) -> bool:
    """Return True when the provider needs enable_thinking=false for this model."""
    return (
        "Qwen3" in model
        or "GLM-4.5" in model
        or "GLM-4.6" in model
        or "GLM-4.7" in model
        or "DeepSeek-V3" in model
    )
