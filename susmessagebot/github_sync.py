import base64
import requests
from .config import GITHUB_TOKEN, GITHUB_REPO, GITHUB_BRANCH

SEEDS_PATH = "susmessagebot/seeds.py"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{SEEDS_PATH}"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

def sync_example_to_github(message: str, label: str) -> None:
    """
    Appends a new example to seeds.py in the GitHub repository.

    Args:
        message: The example message text
        label: "SAFE" or "BAN"
    """
    # Get current seeds.py from GitHub
    response = requests.get(API_URL, headers=HEADERS, params={"ref": GITHUB_BRANCH})
    if response.status_code != 200:
        print(f"Failed to fetch seeds.py: {response.status_code}")
        return

    data = response.json()
    sha = data["sha"]
    current_content = base64.b64decode(data["content"]).decode("utf-8")

    # Append new example inside the examples list, before the closing ]
    new_line = f'    ({repr(message)}, "{label}"),\n'
    updated_content = current_content.replace(
        '\n]\n',
        f'\n{new_line}]\n',
        1
    )

    # Push updated file back to GitHub
    encoded = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")
    payload = {
        "message": f"Add {label} example via HITL feedback",
        "content": encoded,
        "sha": sha,
        "branch": GITHUB_BRANCH
    }

    put_response = requests.put(API_URL, headers=HEADERS, json=payload)
    if put_response.status_code in [200, 201]:
        print(f"Successfully synced example to GitHub")
    else:
        print(f"Failed to sync: {put_response.status_code} {put_response.text}")