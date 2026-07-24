import base64
import logging
import re

import requests

from .config import GITHUB_TOKEN, GITHUB_REPO, GITHUB_BRANCH

SEEDS_PATH = "susmessagebot/seeds.py"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{SEEDS_PATH}"
REQUEST_TIMEOUT = (5, 20)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def _quoted_message_variants(message: str) -> list[str]:
    """Quote forms that may already appear in seeds.py."""
    return [
        repr(message),
        '"' + message.replace("\\", "\\\\").replace('"', '\\"') + '"',
        "'" + message.replace("\\", "\\\\").replace("'", "\\'") + "'",
    ]


def _find_existing_entries(content: str, message: str) -> list[re.Match[str]]:
    """Locate every seeds.py tuple for this message text (including legacy dupes)."""
    quoted_alt = "|".join(re.escape(q) for q in _quoted_message_variants(message))
    pattern = r"\(\s*(?:" + quoted_alt + r')\s*,\s*"(SAFE|BAN)"\s*\)'
    return list(re.finditer(pattern, content))


def _find_existing_entry(content: str, message: str) -> re.Match[str] | None:
    """Locate the first seeds.py tuple for this message text."""
    matches = _find_existing_entries(content, message)
    return matches[0] if matches else None


def _remove_seed_entry(content: str, match: re.Match[str]) -> str:
    """Remove one seeds tuple, including a neighboring comma when present."""
    start, end = match.start(), match.end()
    trailing = re.match(r",[ \t]*(?:\r?\n)?", content[end:])
    if trailing:
        end += trailing.end()
    else:
        preceding = re.search(r",[ \t]*$", content[:start])
        if preceding:
            start = preceding.start()
    return content[:start] + content[end:]


def sync_example_to_github(message: str, label: str) -> bool:
    """
    Append or update an example in seeds.py in the GitHub repository.

    Best-effort side channel: returns False on failure instead of raising, so
    moderation enforcement is never blocked by GitHub outages.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logging.warning("GitHub sync skipped: GITHUB_TOKEN/GITHUB_REPO not configured")
        return False

    try:
        response = requests.get(
            API_URL,
            headers=HEADERS,
            params={"ref": GITHUB_BRANCH},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            logging.error(
                "Failed to fetch seeds.py: %s %s",
                response.status_code,
                response.text[:200],
            )
            return False

        data = response.json()
        sha = data["sha"]
        current_content = base64.b64decode(data["content"]).decode("utf-8")

        matches = _find_existing_entries(current_content, message)
        if matches:
            # Only skip when a single entry already has the desired label.
            # Legacy append-only dupes may still carry a later opposite label;
            # consolidate them so seeds.py remains the latest HITL truth.
            if len(matches) == 1 and matches[0].group(1) == label:
                logging.info(
                    "GitHub sync skipped: duplicate example already present with label %s",
                    label,
                )
                return True

            updated_content = current_content
            for match in reversed(matches[1:]):
                updated_content = _remove_seed_entry(updated_content, match)

            first = matches[0]
            if first.group(1) != label:
                updated_content = (
                    updated_content[: first.start(1)]
                    + label
                    + updated_content[first.end(1) :]
                )
            commit_message = f"Update example to {label} via HITL feedback"
        else:
            new_line = f'    ({repr(message)}, "{label}"),\n'
            updated_content = current_content.replace(
                "\n]\n",
                f"\n{new_line}]\n",
                1,
            )
            commit_message = f"Add {label} example via HITL feedback"

        encoded = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")
        payload = {
            "message": commit_message,
            "content": encoded,
            "sha": sha,
            "branch": GITHUB_BRANCH,
        }

        put_response = requests.put(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if put_response.status_code in (200, 201):
            logging.info("Successfully synced example to GitHub (%s)", commit_message)
            return True

        logging.error(
            "Failed to sync example to GitHub: %s %s",
            put_response.status_code,
            put_response.text[:200],
        )
        return False
    except Exception as e:
        logging.error("GitHub sync error: %s", e)
        return False
