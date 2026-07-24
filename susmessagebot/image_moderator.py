"""Image moderation entrypoint — multimodal LLM, no OCR."""

from .moderator import classify_image

__all__ = ["classify_image"]
