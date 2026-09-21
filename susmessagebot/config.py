from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .prompt_loader import DEFAULT_PROMPT_ID, list_prompt_ids

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = Path(PROJECT_ROOT) / "config.yaml"

_DEFAULT_SILICONFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_DASHSCOPE_VISION_MODEL = "qwen3-vl-flash"
_DEFAULT_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
_DEFAULT_JEV_MODEL = "typesafe-ai/jev"
_DEFAULT_TEXT_CLASSIFIER = "jev_cascade"
_DEFAULT_IMAGE_PROMPT_ID = "v4_zh_multilingual"

# Hardcoded retrieval settings; not part of the YAML file.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 1.0
MAX_EXAMPLES = 5

GITHUB_TOKEN = ""
GITHUB_REPO = ""
GITHUB_BRANCH = "main"
SILICONFLOW_API_KEY = ""
SILICONFLOW_BASE_URL = _DEFAULT_SILICONFLOW_BASE_URL
SILICONFLOW_MODEL = _DEFAULT_SILICONFLOW_MODEL
DASHSCOPE_API_KEY = ""
DASHSCOPE_BASE_URL = _DEFAULT_DASHSCOPE_BASE_URL
DASHSCOPE_VISION_MODEL = _DEFAULT_DASHSCOPE_VISION_MODEL
AI_GATEWAY_API_KEY = ""
AI_GATEWAY_BASE_URL = _DEFAULT_AI_GATEWAY_BASE_URL
JEV_MODEL = _DEFAULT_JEV_MODEL
TEXT_CLASSIFIER = _DEFAULT_TEXT_CLASSIFIER
DISCORD_BOT_TOKEN = ""
APPEAL_DISCORD_USER_ID = ""
PROMPT_ID = DEFAULT_PROMPT_ID
IMAGE_PROMPT_ID = _DEFAULT_IMAGE_PROMPT_ID
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
STATS_DB_PATH = os.path.join(DATA_DIR, "stats.db")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
HEALTH_PORT = 8001
METRICS_PORT = 8000


def _section(data: dict, name: str) -> dict:
    value = data.get(name)
    return value if isinstance(value, dict) else {}


def _str(section: dict, key: str, default: str = "") -> str:
    value = section.get(key, default)
    if value is None:
        return default
    return str(value)


def _int(section: dict, key: str, default: int) -> int:
    value = section.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    return Path(path) if path is not None else CONFIG_PATH


def _resolve_data_dir(raw: str) -> str:
    raw = (raw or "").strip() or "./data"
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return os.path.abspath(os.path.join(PROJECT_ROOT, raw))


def _apply_config(data: dict) -> None:
    global GITHUB_TOKEN, GITHUB_REPO, GITHUB_BRANCH
    global SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL
    global DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_VISION_MODEL
    global AI_GATEWAY_API_KEY, AI_GATEWAY_BASE_URL, JEV_MODEL, TEXT_CLASSIFIER
    global DISCORD_BOT_TOKEN, APPEAL_DISCORD_USER_ID
    global PROMPT_ID, IMAGE_PROMPT_ID
    global DATA_DIR, STATS_DB_PATH, CHROMA_DB_PATH, HEALTH_PORT, METRICS_PORT

    discord = _section(data, "discord")
    siliconflow = _section(data, "siliconflow")
    dashscope = _section(data, "dashscope")
    jev = _section(data, "jev")
    github = _section(data, "github")
    prompts = _section(data, "prompts")
    runtime = _section(data, "runtime")

    DISCORD_BOT_TOKEN = _str(discord, "bot_token")
    APPEAL_DISCORD_USER_ID = _str(discord, "appeal_user_id")

    SILICONFLOW_API_KEY = _str(siliconflow, "api_key")
    SILICONFLOW_MODEL = _str(siliconflow, "model", _DEFAULT_SILICONFLOW_MODEL)
    SILICONFLOW_BASE_URL = _str(
        siliconflow, "base_url", _DEFAULT_SILICONFLOW_BASE_URL
    )

    DASHSCOPE_API_KEY = _str(dashscope, "api_key")
    DASHSCOPE_BASE_URL = _str(dashscope, "base_url", _DEFAULT_DASHSCOPE_BASE_URL)
    DASHSCOPE_VISION_MODEL = _str(
        dashscope, "vision_model", _DEFAULT_DASHSCOPE_VISION_MODEL
    )

    AI_GATEWAY_API_KEY = _str(jev, "api_key")
    AI_GATEWAY_BASE_URL = _str(jev, "base_url", _DEFAULT_AI_GATEWAY_BASE_URL)
    JEV_MODEL = _str(jev, "model", _DEFAULT_JEV_MODEL)
    TEXT_CLASSIFIER = _str(jev, "text_classifier", _DEFAULT_TEXT_CLASSIFIER)

    GITHUB_TOKEN = _str(github, "token")
    GITHUB_REPO = _str(github, "repo")
    GITHUB_BRANCH = _str(github, "branch", "main") or "main"

    PROMPT_ID = _str(prompts, "text", DEFAULT_PROMPT_ID) or DEFAULT_PROMPT_ID
    IMAGE_PROMPT_ID = (
        _str(prompts, "image", _DEFAULT_IMAGE_PROMPT_ID) or _DEFAULT_IMAGE_PROMPT_ID
    )

    DATA_DIR = _resolve_data_dir(_str(runtime, "data_dir", "./data"))
    os.makedirs(DATA_DIR, exist_ok=True)
    STATS_DB_PATH = os.path.join(DATA_DIR, "stats.db")
    CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
    HEALTH_PORT = _int(runtime, "health_port", 8001)
    METRICS_PORT = _int(runtime, "metrics_port", 8000)


def _config_payload() -> dict:
    return {
        "discord": {
            "bot_token": DISCORD_BOT_TOKEN or "",
            "appeal_user_id": APPEAL_DISCORD_USER_ID or "",
        },
        "siliconflow": {
            "api_key": SILICONFLOW_API_KEY or "",
            "model": SILICONFLOW_MODEL or _DEFAULT_SILICONFLOW_MODEL,
            "base_url": SILICONFLOW_BASE_URL or _DEFAULT_SILICONFLOW_BASE_URL,
        },
        "dashscope": {
            "api_key": DASHSCOPE_API_KEY or "",
            "base_url": DASHSCOPE_BASE_URL or _DEFAULT_DASHSCOPE_BASE_URL,
            "vision_model": DASHSCOPE_VISION_MODEL or _DEFAULT_DASHSCOPE_VISION_MODEL,
        },
        "jev": {
            "api_key": AI_GATEWAY_API_KEY or "",
            "base_url": AI_GATEWAY_BASE_URL or _DEFAULT_AI_GATEWAY_BASE_URL,
            "model": JEV_MODEL or _DEFAULT_JEV_MODEL,
            "text_classifier": TEXT_CLASSIFIER or _DEFAULT_TEXT_CLASSIFIER,
        },
        "github": {
            "token": GITHUB_TOKEN or "",
            "repo": GITHUB_REPO or "",
            "branch": GITHUB_BRANCH or "main",
        },
        "prompts": {
            "text": PROMPT_ID or DEFAULT_PROMPT_ID,
            "image": IMAGE_PROMPT_ID or _DEFAULT_IMAGE_PROMPT_ID,
        },
        "runtime": {
            "data_dir": DATA_DIR,
            "health_port": int(HEALTH_PORT),
            "metrics_port": int(METRICS_PORT),
        },
    }


def load_config(path: str | os.PathLike[str] | None = None) -> None:
    """Load YAML into module-level settings. Missing file uses built-in defaults."""
    config_file = _resolve_config_path(path)
    data: dict = {}
    if config_file.is_file():
        with config_file.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is None:
            data = {}
        elif not isinstance(loaded, dict):
            raise ValueError(f"Config file must be a mapping: {config_file}")
        else:
            data = loaded
    _apply_config(data)


def save_config(path: str | os.PathLike[str] | None = None) -> None:
    """Write current module-level settings back to YAML."""
    config_file = _resolve_config_path(path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with config_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _config_payload(),
            handle,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


class ConfigError(ValueError):
    """Invalid /config key or value."""


@dataclass(frozen=True)
class ConfigField:
    key: str
    label: str
    attr: str
    kind: str
    live: bool
    choices: tuple[str, ...] | None = None
    choice_labels: dict[str, str] | None = None
    suggestions: tuple[str, ...] | None = None


FORBIDDEN_CONFIG_KEYS = frozenset({"discord.bot_token"})

TEXT_CLASSIFIER_CHOICES = ("jev_cascade", "siliconflow")
TEXT_CLASSIFIER_LABELS = {
    "jev_cascade": "Jev 级联",
    "siliconflow": "仅硅基",
}
SILICONFLOW_MODEL_SUGGESTIONS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-4B",
)
DASHSCOPE_VISION_SUGGESTIONS = ("qwen3-vl-flash",)
JEV_MODEL_SUGGESTIONS = ("typesafe-ai/jev",)

CONFIG_FIELDS: tuple[ConfigField, ...] = (
    ConfigField("siliconflow.api_key", "硅基密钥", "SILICONFLOW_API_KEY", "secret", True),
    ConfigField(
        "siliconflow.model",
        "硅基模型",
        "SILICONFLOW_MODEL",
        "text",
        True,
        suggestions=SILICONFLOW_MODEL_SUGGESTIONS,
    ),
    ConfigField(
        "siliconflow.base_url",
        "硅基接口",
        "SILICONFLOW_BASE_URL",
        "url",
        True,
    ),
    ConfigField("dashscope.api_key", "阿里云密钥", "DASHSCOPE_API_KEY", "secret", True),
    ConfigField(
        "dashscope.base_url",
        "阿里云接口",
        "DASHSCOPE_BASE_URL",
        "url",
        True,
    ),
    ConfigField(
        "dashscope.vision_model",
        "阿里云视觉模型",
        "DASHSCOPE_VISION_MODEL",
        "text",
        True,
        suggestions=DASHSCOPE_VISION_SUGGESTIONS,
    ),
    ConfigField("jev.api_key", "Jev 密钥", "AI_GATEWAY_API_KEY", "secret", True),
    ConfigField("jev.base_url", "Jev 接口", "AI_GATEWAY_BASE_URL", "url", True),
    ConfigField(
        "jev.model",
        "Jev 模型",
        "JEV_MODEL",
        "text",
        True,
        suggestions=JEV_MODEL_SUGGESTIONS,
    ),
    ConfigField(
        "jev.text_classifier",
        "文字分类器",
        "TEXT_CLASSIFIER",
        "choice",
        True,
        choices=TEXT_CLASSIFIER_CHOICES,
        choice_labels=TEXT_CLASSIFIER_LABELS,
    ),
    ConfigField("github.token", "GitHub 令牌", "GITHUB_TOKEN", "secret", True),
    ConfigField("github.repo", "GitHub 仓库", "GITHUB_REPO", "text", True),
    ConfigField("github.branch", "GitHub 分支", "GITHUB_BRANCH", "text", True),
    ConfigField("prompts.text", "文字提示词", "PROMPT_ID", "choice", True),
    ConfigField("prompts.image", "图片提示词", "IMAGE_PROMPT_ID", "choice", True),
    ConfigField(
        "discord.appeal_user_id",
        "申诉用户 ID",
        "APPEAL_DISCORD_USER_ID",
        "text",
        True,
    ),
    ConfigField("runtime.data_dir", "数据目录", "DATA_DIR", "text", False),
    ConfigField("runtime.health_port", "健康检查端口", "HEALTH_PORT", "int", False),
    ConfigField("runtime.metrics_port", "指标端口", "METRICS_PORT", "int", False),
)
CONFIG_FIELDS_BY_KEY = {field.key: field for field in CONFIG_FIELDS}


def mask_secret(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "（空）"
    if len(text) <= 4:
        return "****"
    return f"****{text[-4:]}"


def field_choices(field: ConfigField) -> tuple[str, ...] | None:
    if field.key in {"prompts.text", "prompts.image"}:
        return tuple(list_prompt_ids())
    if field.choices:
        return field.choices
    return None


def field_value(field: ConfigField) -> str:
    raw = globals().get(field.attr)
    if raw is None:
        return ""
    return str(raw)


def display_field_value(field: ConfigField) -> str:
    raw = field_value(field)
    if field.kind == "secret":
        return mask_secret(raw)
    return raw or "（空）"


def apply_runtime_config() -> None:
    """Rebuild live clients and prompt snapshots after module attrs change."""
    from . import moderator, url_moderator

    moderator.refresh_clients()
    moderator.PROMPT_ID = PROMPT_ID
    moderator.IMAGE_PROMPT_ID = IMAGE_PROMPT_ID
    url_moderator.refresh_clients()


def reload_runtime_config(path: str | os.PathLike[str] | None = None) -> None:
    """Reload YAML into memory and refresh live clients. Does not write disk."""
    load_config(path)
    apply_runtime_config()


def _parse_field_value(field: ConfigField, value: str):
    text = value if value is not None else ""
    if field.kind == "int":
        try:
            parsed = int(str(text).strip())
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"`{field.key}` 必须是整数。") from exc
        if parsed < 1 or parsed > 65535:
            raise ConfigError(f"`{field.key}` 端口必须在 1–65535。")
        return parsed
    if field.kind == "choice":
        normalized = str(text).strip()
        if field.key == "jev.text_classifier":
            normalized = normalized.lower()
        allowed = field_choices(field) or ()
        if normalized not in allowed:
            labels = []
            for item in allowed:
                title = (field.choice_labels or {}).get(item)
                labels.append(f"{item}（{title}）" if title else item)
            raise ConfigError(
                f"`{field.key}` 只能是：{'、'.join(labels) or '（无可用项）'}"
            )
        return normalized
    return str(text)


def set_config_value(
    key: str,
    value: str,
    *,
    persist: bool = True,
    path: str | os.PathLike[str] | None = None,
) -> ConfigField:
    """Update one editable field, optionally persist, and apply live clients."""
    key = (key or "").strip()
    if key in FORBIDDEN_CONFIG_KEYS:
        raise ConfigError("不能修改 discord.bot_token（改了也无法热换网关）。")
    field = CONFIG_FIELDS_BY_KEY.get(key)
    if field is None:
        raise ConfigError(f"未知配置项：{key}")

    parsed = _parse_field_value(field, value)
    globals()[field.attr] = parsed
    if field.live:
        apply_runtime_config()
    if persist:
        save_config(path)
    return field


load_config()
