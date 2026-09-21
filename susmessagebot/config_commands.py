"""Discord /config show|set|reload — admin-only hot reload of YAML settings."""

from __future__ import annotations

import logging
import sys

import discord
from discord import app_commands

from . import config
from .config import (
    CONFIG_FIELDS,
    CONFIG_FIELDS_BY_KEY,
    ConfigError,
    ConfigField,
    display_field_value,
    field_choices,
    field_value,
    reload_runtime_config,
    set_config_value,
)

_MAX_CHOICES = 25

config_group = app_commands.Group(
    name="config",
    description="查看或热更新机器人配置",
    default_permissions=discord.Permissions(administrator=True),
    allowed_contexts=app_commands.AppCommandContext(
        guild=False,
        dm_channel=True,
        private_channel=True,
    ),
)


def _choice_label(field: ConfigField, value: str) -> str:
    title = (field.choice_labels or {}).get(value)
    if title:
        return f"{title} ({value})"[:100]
    return value[:100]


def autocomplete_keys(current: str) -> list[app_commands.Choice[str]]:
    query = (current or "").strip().lower()
    choices: list[app_commands.Choice[str]] = []
    for field in CONFIG_FIELDS:
        label = f"{field.label} ({field.key})"
        haystack = f"{field.label} {field.key} {label}".lower()
        if query and query not in haystack:
            continue
        choices.append(app_commands.Choice(name=label[:100], value=field.key))
        if len(choices) >= _MAX_CHOICES:
            break
    return choices


def autocomplete_values(key: str, current: str) -> list[app_commands.Choice[str]]:
    field = CONFIG_FIELDS_BY_KEY.get((key or "").strip())
    if field is None or field.kind in {"secret", "url"}:
        return []

    query = (current or "").strip().lower()
    seen: set[str] = set()
    options: list[tuple[str, str]] = []

    allowed = field_choices(field)
    if allowed:
        for value in allowed:
            options.append((_choice_label(field, value), value))
    else:
        current_value = field_value(field)
        if current_value:
            options.append((f"当前值 ({current_value})"[:100], current_value))
        for suggestion in field.suggestions or ():
            if suggestion == current_value:
                continue
            options.append((suggestion[:100], suggestion))

    choices: list[app_commands.Choice[str]] = []
    for name, value in options:
        if value in seen:
            continue
        haystack = f"{name} {value}".lower()
        if query and query not in haystack:
            continue
        seen.add(value)
        choices.append(app_commands.Choice(name=name[:100], value=value))
        if len(choices) >= _MAX_CHOICES:
            break
    return choices


def format_config_show() -> str:
    lines = ["当前配置（密钥已掩码）："]
    for field in CONFIG_FIELDS:
        suffix = " — 改后需重启" if not field.live else ""
        lines.append(
            f"• {field.label} (`{field.key}`)：{display_field_value(field)}{suffix}"
        )
    return "\n".join(lines)


async def _user_is_shared_guild_admin(
    client: discord.Client,
    user_id: int,
) -> bool:
    """True if the user owns or has Administrator in any mutual guild."""
    for guild in getattr(client, "guilds", []):
        if getattr(guild, "owner_id", None) == user_id:
            return True
        member = None
        get_member = getattr(guild, "get_member", None)
        if callable(get_member):
            member = get_member(user_id)
        if member is None:
            fetch_member = getattr(guild, "fetch_member", None)
            if callable(fetch_member):
                try:
                    member = await fetch_member(user_id)
                except Exception:
                    continue
        if member is None:
            continue
        permissions = getattr(member, "guild_permissions", None)
        if getattr(permissions, "administrator", False):
            return True
    return False


def _loaded_bot_module(modules: dict | None = None):
    """Use the already-running bot module; never re-import susmessagebot.bot.

    `python -m susmessagebot.bot` loads bot.py as `__main__`. Importing
    `.bot` again would re-register Prometheus gauges and crash /config.
    """
    loaded = sys.modules if modules is None else modules
    bot_mod = loaded.get("susmessagebot.bot")
    if bot_mod is not None:
        return bot_mod
    main_mod = loaded.get("__main__")
    if callable(getattr(main_mod, "_is_application_owner", None)):
        return main_mod
    return None


def _runtime_is_application_owner(
    user_id: int,
    modules: dict | None = None,
) -> bool:
    module = _loaded_bot_module(modules)
    if module is None:
        return False
    checker = getattr(module, "_is_application_owner", None)
    return bool(checker(user_id)) if callable(checker) else False


async def _require_config_admin(interaction: discord.Interaction) -> bool:
    if interaction.guild_id is not None:
        await interaction.response.send_message(
            "请私聊机器人使用此命令。",
            ephemeral=True,
        )
        return False
    if _runtime_is_application_owner(interaction.user.id):
        return True
    if await _user_is_shared_guild_admin(interaction.client, interaction.user.id):
        return True
    await interaction.response.send_message(
        "仅管理员可操作。",
        ephemeral=True,
    )
    return False


@config_group.command(name="show", description="查看当前可热更新配置")
async def config_show(interaction: discord.Interaction) -> None:
    if not await _require_config_admin(interaction):
        return
    await interaction.response.send_message(format_config_show(), ephemeral=True)


@config_group.command(name="set", description="修改一项配置并立即写回 YAML")
@app_commands.describe(key="配置项", value="新值")
async def config_set(
    interaction: discord.Interaction,
    key: str,
    value: str,
) -> None:
    if not await _require_config_admin(interaction):
        return
    try:
        field = set_config_value(key, value)
    except ConfigError as exc:
        await interaction.response.send_message(str(exc), ephemeral=True)
        return
    except OSError as exc:
        logging.exception("Failed to persist config.yaml")
        await interaction.response.send_message(
            f"配置已改内存，但写盘失败：{exc}",
            ephemeral=True,
        )
        return

    if field.kind != "secret":
        logging.info("config set key=%s live=%s", field.key, field.live)
    else:
        logging.info("config set key=%s live=%s (secret masked)", field.key, field.live)

    if field.live:
        message = f"已更新 {field.label} (`{field.key}`)，立即生效。"
    else:
        message = (
            f"已写入 {field.label} (`{field.key}`)。"
            "数据目录和健康/指标端口已在启动时绑定，需重启后才切换。"
        )
    await interaction.response.send_message(message, ephemeral=True)


@config_set.autocomplete("key")
async def config_set_key_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    return autocomplete_keys(current)


@config_set.autocomplete("value")
async def config_set_value_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    key = ""
    namespace = getattr(interaction, "namespace", None)
    if namespace is not None:
        key = getattr(namespace, "key", "") or ""
    return autocomplete_values(key, current)


@config_group.command(name="reload", description="从磁盘 YAML 重载配置（不写盘）")
async def config_reload(interaction: discord.Interaction) -> None:
    if not await _require_config_admin(interaction):
        return
    try:
        reload_runtime_config()
    except Exception as exc:
        logging.exception("Failed to reload config.yaml")
        await interaction.response.send_message(
            f"重载失败：{exc}",
            ephemeral=True,
        )
        return
    await interaction.response.send_message(
        "已从磁盘重载配置并刷新运行时客户端。",
        ephemeral=True,
    )


def register_config_commands(tree: app_commands.CommandTree) -> None:
    if tree.get_command("config") is None:
        tree.add_command(config_group)
