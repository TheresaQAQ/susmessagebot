"""
One-shot: convert a dotenv file into config.yaml.

Usage:
  python -m scripts.migrate_env_to_yaml
  python -m scripts.migrate_env_to_yaml --env .env --out config.yaml
  python -m scripts.migrate_env_to_yaml --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_DEFAULTS = {
    "siliconflow": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "base_url": "https://api.siliconflow.cn/v1",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "vision_model": "qwen3-vl-flash",
    },
    "jev": {
        "base_url": "https://ai-gateway.vercel.sh/v1",
        "model": "typesafe-ai/jev",
        "text_classifier": "jev_cascade",
    },
    "github": {"branch": "main"},
    "prompts": {
        "text": "v7_zh_hard_gates",
        "image": "v4_zh_multilingual",
    },
    "runtime": {
        "data_dir": "./data",
        "health_port": 8001,
        "metrics_port": 8000,
    },
}


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _pick(env: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = env.get(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def _int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def env_to_config(env: dict[str, str]) -> dict:
    return {
        "discord": {
            "bot_token": _pick(env, "DISCORD_BOT_TOKEN"),
            "appeal_user_id": _pick(env, "APPEAL_DISCORD_USER_ID"),
        },
        "siliconflow": {
            "api_key": _pick(env, "SILICONFLOW_API_KEY", "OPENROUTER_API_KEY"),
            "model": _pick(
                env,
                "SILICONFLOW_MODEL",
                "OPENROUTER_MODEL",
                default=_DEFAULTS["siliconflow"]["model"],
            ),
            "base_url": _pick(
                env,
                "SILICONFLOW_BASE_URL",
                "OPENROUTER_BASE_URL",
                default=_DEFAULTS["siliconflow"]["base_url"],
            ),
        },
        "dashscope": {
            "api_key": _pick(env, "DASHSCOPE_API_KEY"),
            "base_url": _pick(
                env, "DASHSCOPE_BASE_URL", default=_DEFAULTS["dashscope"]["base_url"]
            ),
            "vision_model": _pick(
                env,
                "DASHSCOPE_VISION_MODEL",
                default=_DEFAULTS["dashscope"]["vision_model"],
            ),
        },
        "jev": {
            "api_key": _pick(env, "AI_GATEWAY_API_KEY"),
            "base_url": _pick(
                env, "AI_GATEWAY_BASE_URL", default=_DEFAULTS["jev"]["base_url"]
            ),
            "model": _pick(env, "JEV_MODEL", default=_DEFAULTS["jev"]["model"]),
            "text_classifier": _pick(
                env, "TEXT_CLASSIFIER", default=_DEFAULTS["jev"]["text_classifier"]
            ),
        },
        "github": {
            "token": _pick(env, "GITHUB_TOKEN"),
            "repo": _pick(env, "GITHUB_REPO"),
            "branch": _pick(env, "GITHUB_BRANCH", default=_DEFAULTS["github"]["branch"]),
        },
        "prompts": {
            "text": _pick(env, "PROMPT_ID", default=_DEFAULTS["prompts"]["text"]),
            "image": _pick(
                env, "IMAGE_PROMPT_ID", default=_DEFAULTS["prompts"]["image"]
            ),
        },
        "runtime": {
            "data_dir": _pick(env, "DATA_DIR", default=_DEFAULTS["runtime"]["data_dir"]),
            "health_port": _int(env, "HEALTH_PORT", _DEFAULTS["runtime"]["health_port"]),
            "metrics_port": _int(
                env, "METRICS_PORT", _DEFAULTS["runtime"]["metrics_port"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .env into config.yaml once")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--out", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing YAML")
    args = parser.parse_args()

    if not args.env.is_file():
        raise SystemExit(f"找不到环境文件: {args.env}")
    if args.out.exists() and not args.force:
        raise SystemExit(f"已存在 {args.out}，如需覆盖请加 --force")

    payload = env_to_config(parse_dotenv(args.env))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    filled = [
        "discord.bot_token" if payload["discord"]["bot_token"] else "",
        "siliconflow.api_key" if payload["siliconflow"]["api_key"] else "",
        "dashscope.api_key" if payload["dashscope"]["api_key"] else "",
        "jev.api_key" if payload["jev"]["api_key"] else "",
        "github.token" if payload["github"]["token"] else "",
    ]
    print(f"已写入 {args.out}")
    print("已填入密钥字段: " + ", ".join(name for name in filled if name))
    print(f"siliconflow.model={payload['siliconflow']['model']}")
    print(f"jev.text_classifier={payload['jev']['text_classifier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
