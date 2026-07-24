import base64
import logging

import requests

from .config import GITHUB_TOKEN, GITHUB_REPO, GITHUB_BRANCH

SEEDS_PATH = "susmessagebot/seeds.py"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{SEEDS_PATH}"
REQUEST_TIMEOUT = (5, 20)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def sync_example_to_github(message: str, label: str) -> bool:
    """
    Append a new example to seeds.py in the GitHub repository.

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

        new_line = f'    ({repr(message)}, "{label}"),\n'
        updated_content = current_content.replace(
            "\n]\n",
            f"\n{new_line}]\n",
            1,
        )

        encoded = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")
        payload = {
            "message": f"Add {label} example via HITL feedback",
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
            logging.info("Successfully synced example to GitHub")
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
