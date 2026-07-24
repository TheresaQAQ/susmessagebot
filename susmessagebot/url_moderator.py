import re
import logging
import requests
from urllib.parse import urlparse
from openai import OpenAI
from . import config
from .config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)
from .llm_utils import should_disable_thinking

client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
    timeout=60.0,
)

# Blocklist cache
_blocklist: set = set()

BLOCKLIST_URL = "https://malware-filter.gitlab.io/malware-filter/urlhaus-filter-domains-online.txt"

URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)

SAFE_DOMAINS = {
    "github.com", "youtube.com", "google.com", "linkedin.com",
    "reddit.com", "twitter.com", "instagram.com", "facebook.com",
    "tiktok.com", "wikipedia.org",
}


def load_blocklist() -> None:
    """Download and cache the malware domain blocklist."""
    global _blocklist
    try:
        response = requests.get(BLOCKLIST_URL, timeout=10)
        response.raise_for_status()
        domains = set()
        for line in response.text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                domains.add(line.lower())
        if not domains:
            logging.error("Blocklist refresh returned no domains; keeping previous cache")
            return
        _blocklist = domains
    except Exception as e:
        logging.error(f"Failed to load blocklist: {e}")


def _extract_urls(text: str) -> list:
    """Extract all URLs from a message."""
    return URL_PATTERN.findall(text)


def _get_domain(url: str) -> str:
    """Extract hostname from URL (no port / userinfo)."""
    try:
        host = (urlparse(url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_on_blocklist(domain: str) -> bool:
    """Check if domain is in the cached blocklist."""
    return domain in _blocklist


def _classify_url_with_llm(url: str) -> str:
    """Use SiliconFlow to classify a URL as SAFE, BAN, or REVIEW."""
    try:
        model = config.SILICONFLOW_MODEL
        create_kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"""You moderate links for a game community. Treat all languages/regions equally.
Classify this URL as SAFE or BAN.

BAN if the URL looks like:
- Advertising, affiliate, or sales pages (accounts, boosts, currency, cheats, paid carries)
- Invite/recruitment to another Discord/Telegram/QQ/community for promo
- Phishing, scam, malware, fake giveaway/reward pages
- URL shorteners hiding a suspicious destination
- Random/auto-generated domains

SAFE if it is a normal well-known site or ordinary content shared in discussion
(e.g. youtube, steam, github, wikipedia, official game sites, patch notes).

URL: {url}

Respond with exactly one word: SAFE or BAN"""
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
        }
        if should_disable_thinking(model):
            create_kwargs["extra_body"] = {"enable_thinking": False}
        response = client.chat.completions.create(**create_kwargs)
        message_obj = response.choices[0].message
        content = message_obj.content or ""
        reasoning = (
            getattr(message_obj, "reasoning_content", None)
            or getattr(message_obj, "reasoning", None)
            or ""
        )
        text = content.strip().upper()
        if text in {"SAFE", "BAN"}:
            return text
        match = re.search(r"\b(BAN|SAFE)\b", (content or reasoning).upper())
        # Unparseable model output is a classification failure, not a SAFE pass.
        return match.group(1) if match else "REVIEW"
    except Exception as e:
        logging.error(f"URL LLM classification error: {e}")
        return "REVIEW"


def analyze_urls(text: str) -> str:
    """
    Extract and analyze all URLs in a message.

    Returns "BAN" if any URL is suspicious, "REVIEW" if classification failed
    for at least one URL without a later BAN, otherwise "SAFE".
    """
    urls = _extract_urls(text)
    if not urls:
        return "SAFE"

    saw_review = False
    for url in urls:
        # Never fetch user-controlled URLs from the bot host. Following redirects
        # here would allow messages to probe internal or cloud metadata services.
        domain = _get_domain(url)

        if not domain:
            continue

        # Skip known safe domains
        if domain in SAFE_DOMAINS:
            continue

        # Check blocklist first (fast)
        if _is_on_blocklist(domain):
            return "BAN"

        # Fall back to LLM classification
        result = _classify_url_with_llm(url)
        if result == "BAN":
            return "BAN"
        if result == "REVIEW":
            # Keep scanning: a later blocklisted/BAN URL must still win.
            saw_review = True

    return "REVIEW" if saw_review else "SAFE"
