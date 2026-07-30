import discord
from discord import app_commands
from .moderator import classify_message
from .image_moderator import classify_image
from .url_moderator import URL_PATTERN, analyze_urls, load_blocklist
from .config import (
    DISCORD_BOT_TOKEN,
    APPEAL_DISCORD_USER_ID,
    HEALTH_PORT,
    METRICS_PORT,
)
from .github_sync import sync_example_to_github
from .vector_store import add_example, ensure_normalized_index
from .strike_tracker import strikes, ban_notice_text
import logging
import asyncio
import io
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from prometheus_client import Gauge, start_http_server
from .stats import (
    init_db,
    get_stat,
    increment_stat,
    add_group,
    remove_group,
    get_groups_count,
    get_total_members,
    claim_review_decision,
    get_review_decision_owner,
    release_review_decision,
    store_review_evidence,
    get_review_evidence,
    get_review_reason,
    store_review_notification,
    claim_related_review_notifications,
    mark_review_notification_resolved,
    record_auto_ban,
    clear_auto_ban,
    take_reversible_auto_ban,
)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
_DISCORD_GIF_PROVIDER_PATHS = {
    "giphy.com": ("/gifs/",),
    "klipy.com": ("/gifs/",),
    "tenor.com": ("/view/",),
}
_DISCORD_CUSTOM_EMOJI_PATTERN = re.compile(r"<a?:[^:\s>]+:\d+>")
_BAN_DELETE_MESSAGE_SECONDS = 24 * 60 * 60

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Set True after Discord gateway ready; /health returns 503 until then.
_bot_ready = False
_BLOCKLIST_REFRESH_SECONDS = 6 * 60 * 60
_blocklist_refresh_task: asyncio.Task | None = None
_REQUIRED_GUILD_PERMISSIONS = (
    ("ban_members", "Ban Members"),
)
_REQUIRED_CHANNEL_PERMISSIONS = (
    ("view_channel", "View Channel"),
    ("read_message_history", "Read Message History"),
    ("manage_messages", "Manage Messages"),
)
_REQUIRED_FORUM_PERMISSIONS = (
    ("view_channel", "View Channel"),
    ("read_message_history", "Read Message History"),
    ("manage_messages", "Manage Messages"),
)


class SusMessageBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        self.add_dynamic_items(
            HITLBanButton,
            HITLDeleteButton,
            HITLFalseAlarmButton,
        )
        synced = await self.tree.sync()
        logging.info(f"Synced {len(synced)} commands globally")

client = SusMessageBot()
GROUPS_COUNT = Gauge('discord_groups_count_total', 'Number of Discord servers bot is in')
MEMBERS_PROTECTED = Gauge('discord_members_protected_total', 'Total Discord members protected')
MESSAGES_CLASSIFIED_SAFE = Gauge('discord_messages_classified_safe_total', 'SAFE classifications')
MESSAGES_CLASSIFIED_BAN = Gauge('discord_messages_classified_ban_total', 'BAN classifications')
BANS_CONFIRMED = Gauge('discord_bans_confirmed_total', 'Admin-confirmed correct bans')
FALSE_POSITIVES = Gauge('discord_false_positives_total', 'Admin-confirmed false positives')
FALSE_NEGATIVES = Gauge('discord_false_negatives_total', 'Admin-reported false negatives')


def is_ready() -> bool:
    """Return True once the Discord client has completed on_ready."""
    return _bot_ready


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            if is_ready():
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"NOT_READY")
        else:
            self.send_response(404)
            self.end_headers()

    def do_HEAD(self):
        if self.path == "/health":
            self.send_response(200 if is_ready() else 503)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_health_server(port: int = HEALTH_PORT) -> None:
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.info(f"Health server listening on :{port}/health")


def init_metrics() -> None:
    """Load persisted values from SQLite into Prometheus gauges."""
    MESSAGES_CLASSIFIED_SAFE.set(get_stat("messages_safe"))
    MESSAGES_CLASSIFIED_BAN.set(get_stat("messages_ban"))
    BANS_CONFIRMED.set(get_stat("bans_confirmed"))
    FALSE_POSITIVES.set(get_stat("false_positives"))
    FALSE_NEGATIVES.set(get_stat("false_negatives"))
    GROUPS_COUNT.set(get_groups_count())
    MEMBERS_PROTECTED.set(get_total_members())


def _refresh_guild_metrics() -> None:
    GROUPS_COUNT.set(get_groups_count())
    MEMBERS_PROTECTED.set(get_total_members())


def _guild_permission_issues(guild: discord.Guild) -> list[str]:
    """Return missing effective permissions for guild message surfaces."""
    member = guild.me
    if member is None:
        return ["Bot member information is unavailable"]

    issues: list[str] = []
    guild_missing = [
        label
        for attribute, label in _REQUIRED_GUILD_PERMISSIONS
        if not getattr(member.guild_permissions, attribute, False)
    ]
    if guild_missing:
        issues.append(f"Server role: {', '.join(guild_missing)}")

    channel_groups = (
        (guild.text_channels, _REQUIRED_CHANNEL_PERMISSIONS),
        (guild.forums, _REQUIRED_FORUM_PERMISSIONS),
    )
    for channels, required_permissions in channel_groups:
        for channel in channels:
            permissions = channel.permissions_for(member)
            missing = [
                label
                for attribute, label in required_permissions
                if not getattr(permissions, attribute, False)
            ]
            if missing:
                issues.append(
                    f"#{channel.name} ({channel.id}): {', '.join(missing)}"
                )
    return issues


async def _audit_guild_permissions(guild: discord.Guild) -> bool:
    """Log a guild permission audit and return whether all checks passed."""
    issues = _guild_permission_issues(guild)
    if not issues:
        logging.info(
            "Permission check passed for guild %s (%s)",
            guild.name,
            guild.id,
        )
        return True

    logging.warning(
        "Permission check failed for guild %s (%s): %s",
        guild.name,
        guild.id,
        "; ".join(issues),
    )
    return False


async def _refresh_blocklist_periodically() -> None:
    """Refresh the malware blocklist while the bot process remains alive."""
    while True:
        await asyncio.sleep(_BLOCKLIST_REFRESH_SECONDS)
        try:
            await asyncio.to_thread(load_blocklist)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.error(f"Unexpected blocklist refresh failure: {e}")


def _ensure_blocklist_refresh_task() -> None:
    global _blocklist_refresh_task
    if _blocklist_refresh_task is None or _blocklist_refresh_task.done():
        _blocklist_refresh_task = asyncio.create_task(
            _refresh_blocklist_periodically(),
            name="blocklist-refresh",
        )


@client.event
async def on_ready():
    global _bot_ready
    await asyncio.to_thread(load_blocklist)
    _ensure_blocklist_refresh_task()
    permission_failures = 0
    for guild in client.guilds:
        add_group(guild.id, guild.member_count)
        try:
            permission_ok = await _audit_guild_permissions(guild)
        except Exception as e:
            permission_ok = False
            logging.exception(
                "Unexpected permission check failure for guild %s (%s): %s",
                guild.name,
                guild.id,
                e,
            )
        if not permission_ok:
            permission_failures += 1
    _refresh_guild_metrics()
    _bot_ready = True
    logging.info(
        "Startup permission check complete: %s/%s guild(s) need attention",
        permission_failures,
        len(client.guilds),
    )
    logging.info(f"Bot ready — logged in as {client.user}")


@client.event
async def on_disconnect():
    global _bot_ready
    _bot_ready = False
    logging.warning("Discord gateway disconnected; health will report NOT_READY")


@client.event
async def on_resumed():
    global _bot_ready
    _bot_ready = True
    logging.info("Discord gateway resumed")


@client.event
async def on_guild_join(guild: discord.Guild):
    add_group(guild.id, guild.member_count)
    _refresh_guild_metrics()
    logging.info(f"Joined new server: {guild.name} ({guild.id}) with {guild.member_count} members")
    await _audit_guild_permissions(guild)


@client.event
async def on_guild_remove(guild: discord.Guild):
    remove_group(guild.id)
    _refresh_guild_metrics()
    logging.info(f"Removed from server: {guild.name} ({guild.id})")


@client.event
async def on_member_join(member: discord.Member):
    if member.guild:
        add_group(member.guild.id, member.guild.member_count or 0)
        _refresh_guild_metrics()


@client.event
async def on_member_remove(member: discord.Member):
    if member.guild:
        add_group(member.guild.id, member.guild.member_count or 0)
        _refresh_guild_metrics()

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    if message.guild is None:
        return

    if message.author.guild_permissions.administrator:
        return

    # Aggregate modalities with BAN > REVIEW > SAFE. Image REVIEW must not
    # short-circuit text/URL checks, or scam text attached to a broken image
    # would only enter soft review.
    loop = asyncio.get_event_loop()
    final = "SAFE"
    review_reason = "Moderation unavailable"
    evidence_images: list[tuple[str, bytes]] = []

    if message.attachments:
        for attachment in message.attachments:
            if not _is_image_attachment(attachment):
                continue
            try:
                image_bytes = bytes(await attachment.read())
            except Exception as e:
                logging.error(
                    "Error reading attachment %s: %s",
                    attachment.filename,
                    e,
                )
                final = "REVIEW"
                review_reason = "Image moderation unavailable"
                continue
            result = await loop.run_in_executor(None, classify_image, image_bytes)
            evidence_images.append((attachment.filename, image_bytes))
            if result == "BAN":
                await _ban_user(
                    message,
                    reason="Suspicious image",
                    evidence_images=[(attachment.filename, image_bytes)],
                )
                return
            if result == "REVIEW":
                final = "REVIEW"
                review_reason = "Image moderation unavailable"

    text = _moderation_text(message)
    if text:
        text_result = await loop.run_in_executor(None, classify_message, text)
        url_result = await loop.run_in_executor(None, analyze_urls, text)
        if text_result == "BAN" or url_result == "BAN":
            final = "BAN"
            review_reason = "Suspicious message"
        elif text_result == "REVIEW" or url_result == "REVIEW":
            if final != "BAN":
                final = "REVIEW"
                # Keep an earlier image REVIEW reason; otherwise report the true source.
                if review_reason == "Moderation unavailable":
                    if text_result == "REVIEW" and url_result == "REVIEW":
                        review_reason = "Text and URL moderation unavailable"
                    elif text_result == "REVIEW":
                        review_reason = "Text moderation unavailable"
                    else:
                        review_reason = "URL moderation unavailable"
        logging.info(
            "discord classify user=%s text=%r text_result=%s url_result=%s final=%s",
            message.author.id,
            text[:120],
            text_result,
            url_result,
            final,
        )
    elif not evidence_images and final == "SAFE":
        return

    if final == "BAN":
        ban_kwargs = {}
        if evidence_images:
            ban_kwargs["evidence_images"] = evidence_images
        await _ban_user(message, reason="Suspicious message", **ban_kwargs)
    elif final == "REVIEW":
        review_kwargs = {}
        if evidence_images:
            review_kwargs["evidence_images"] = evidence_images
        await _request_manual_review(
            message,
            reason=review_reason,
            **review_kwargs,
        )
    else:
        increment_stat('messages_safe')
        MESSAGES_CLASSIFIED_SAFE.set(get_stat('messages_safe'))


def _is_discord_picker_gif_url(url: str) -> bool:
    if not url or any(char.isspace() for char in url):
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    if host.startswith("www."):
        host = host[4:]
    for domain, path_prefixes in _DISCORD_GIF_PROVIDER_PATHS.items():
        if host != domain and not host.endswith(f".{domain}"):
            continue
        return any(parsed.path.startswith(prefix) for prefix in path_prefixes)
    return False


def _strip_discord_media_text(text: str) -> str:
    """Remove trusted GIF links and Discord custom emoji from text."""
    text = (text or "").strip()
    if not text:
        return ""

    text = _DISCORD_CUSTOM_EMOJI_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(
        lambda match: (
            " "
            if _is_discord_picker_gif_url(match.group(0))
            else match.group(0)
        ),
        text,
    )
    text = re.sub(r"<\s*>", " ", text)
    return " ".join(text.split())


def _moderation_text(message: discord.Message) -> str:
    """Return the user-authored text that should enter moderation."""
    return _strip_discord_media_text(message.content or "")


async def _admin_members(guild: discord.Guild) -> list[discord.Member]:
    """Guild owner + members with Administrator (excludes bots)."""
    if not getattr(guild, "chunked", True):
        try:
            await guild.chunk(cache=True)
        except Exception as e:
            logging.warning(f"Could not complete guild member cache: {e}")
    return [
        member
        for member in guild.members
        if not member.bot
        and (
            member.id == guild.owner_id
            or member.guild_permissions.administrator
        )
    ]


async def _dm_user(
    user: discord.abc.User,
    text: str,
    *,
    images: list[tuple[str, bytes]] | None = None,
) -> None:
    try:
        kwargs: dict = {"allowed_mentions": discord.AllowedMentions.none()}
        if images:
            kwargs["files"] = _discord_files(images)
        await user.send(text, **kwargs)
    except Exception as e:
        logging.warning(f"Could not DM user {user.id}: {e}")


def _user_notice_with_evidence(
    notice: str,
    *,
    guild_name: str,
    channel_name: str,
    message_id: int | None,
    content: str,
    images: list[tuple[str, bytes]] | None = None,
) -> str:
    """Add a compact source summary to a moderation notice."""
    original = (content or "").strip()
    if original == "[image]" or (not original and images):
        original = "See attached image"
    elif original == "[empty]" or not original:
        original = "No text content"
    else:
        original = " ".join(original.split())
        if len(original) > 500:
            original = original[:497] + "..."

    sent_at = "Unknown"
    if message_id is not None:
        try:
            sent_at = discord.utils.format_dt(
                discord.utils.snowflake_time(message_id),
                style="F",
            )
        except (TypeError, ValueError, OverflowError):
            pass

    return (
        f"{notice}\n\n"
        f"Server: {guild_name}\n"
        f"Channel: #{channel_name}\n"
        f"Time: {sent_at}\n"
        f"Message: {original}"
    )[:2000]


async def _execute_ban(
    *,
    guild: discord.Guild,
    user: discord.abc.User,
    reason: str,
    preserve_auto_ban_record: bool = False,
    automatic: bool = False,
    channel_name: str | None = None,
    evidence_message_id: int | None = None,
    evidence_text: str | None = None,
    evidence_images: list[tuple[str, bytes]] | None = None,
) -> None:
    """DM ban notice first (ban removes mutual servers), then ban and clear strikes."""
    notice = ban_notice_text(APPEAL_DISCORD_USER_ID, automatic=automatic)
    if evidence_text is not None or evidence_images:
        notice = _user_notice_with_evidence(
            notice,
            guild_name=getattr(guild, "name", "Unknown server"),
            channel_name=channel_name or "unknown",
            message_id=evidence_message_id,
            content=evidence_text or "",
            images=evidence_images,
        )
    await _dm_user(
        user,
        notice,
        images=evidence_images,
    )
    await guild.ban(
        discord.Object(id=user.id),
        reason=reason,
        delete_message_seconds=_BAN_DELETE_MESSAGE_SECONDS,
    )
    strikes.clear(guild.id, user.id)
    if not preserve_auto_ban_record:
        # Admin/manual bans must not be reversible by an unrelated false alarm.
        clear_auto_ban(guild.id, user.id)


def _message_text(message: discord.Message) -> str:
    text = (message.content or "").strip()
    if text:
        return text
    if message.attachments:
        return "[image]"
    return "[empty]"


def _trainable_text(text: str) -> str | None:
    """Return text suitable for RAG/GitHub training, or None for placeholders."""
    cleaned = (text or "").strip()
    if not cleaned or cleaned in {"[image]", "[empty]"}:
        return None
    return cleaned


def _record_training_example(text: str, label: str) -> None:
    """Persist a local training example and best-effort sync it to GitHub."""
    trainable = _trainable_text(_strip_discord_media_text(text))
    if not trainable:
        logging.info("Skipping text training for non-textual evidence (%r)", text)
        return
    add_example(trainable, label)
    if not sync_example_to_github(trainable, label):
        logging.warning("GitHub sync failed for %s example; local training kept", label)


def _is_image_attachment(attachment: discord.Attachment) -> bool:
    content_type = (getattr(attachment, "content_type", None) or "").lower()
    return (
        content_type.startswith("image/")
        or attachment.filename.lower().endswith(_IMAGE_EXTS)
    )


async def _snapshot_images(message: discord.Message) -> list[tuple[str, bytes]]:
    """Download image attachments before the message is deleted."""
    images: list[tuple[str, bytes]] = []
    for attachment in message.attachments:
        if not _is_image_attachment(attachment):
            continue
        try:
            images.append((attachment.filename, bytes(await attachment.read())))
        except Exception as e:
            logging.warning(f"Could not snapshot attachment {attachment.filename}: {e}")
    return images


def _discord_files(images: list[tuple[str, bytes]] | None) -> list[discord.File]:
    if not images:
        return []
    return [discord.File(io.BytesIO(data), filename=name) for name, data in images]


def _review_content_block(content: str) -> str:
    """Render text evidence while hiding internal image/empty placeholders."""
    cleaned = (content or "").strip()
    if cleaned in {"", "[image]", "[empty]"}:
        return ""
    preview = cleaned if len(cleaned) <= 1500 else cleaned[:1500] + "..."
    return f"\n\n📝 内容：\n{preview}"


async def _dm_admins_review(
    guild: discord.Guild,
    *,
    channel_name: str,
    author: discord.abc.User,
    content: str,
    reason: str,
    message_id: int,
    channel_id: int,
    images: list[tuple[str, bytes]] | None = None,
    removed: bool = True,
    status: str | None = None,
) -> int:
    """Send full content + Ban/False Alarm buttons to each admin via DM. Returns how many DMs succeeded."""
    store_review_evidence(guild.id, message_id, author.id, content, reason)
    if status is None:
        status = (
            "可疑内容已删除"
            if removed
            else "可疑内容已标记，但自动删除失败"
        )
    body = (
        f"⚠️ {status}｜**#{channel_name}**（{guild.name}）\n\n"
        f"👤 用户：{author} (`{author.id}`)"
        f"{_review_content_block(content)}"
    )
    notified = 0
    for admin in await _admin_members(guild):
        try:
            view = HITLView(
                guild_id=guild.id,
                user_id=author.id,
                username=str(author),
                text=content,
                reason=reason,
                message_id=message_id,
                channel_id=channel_id,
            )
            sent = await admin.send(body, view=view, files=_discord_files(images))
            notified += 1
            sent_id = getattr(sent, "id", None)
            sent_channel_id = getattr(getattr(sent, "channel", None), "id", None)
            if isinstance(sent_id, int) and isinstance(sent_channel_id, int):
                try:
                    store_review_notification(
                        guild.id,
                        message_id,
                        author.id,
                        admin.id,
                        sent_channel_id,
                        sent_id,
                    )
                except Exception as e:
                    logging.warning(
                        "Could not persist admin review DM %s for admin %s: %s",
                        sent_id,
                        admin.id,
                        e,
                    )
            else:
                logging.warning(
                    "Could not persist admin review DM reference for admin %s",
                    admin.id,
                )
        except Exception as e:
            logging.warning(f"Could not DM admin {admin.id}: {e}")
    return notified


async def _request_manual_review(
    message: discord.Message,
    reason: str,
    *,
    evidence_images: list[tuple[str, bytes]] | None = None,
):
    """
    REVIEW path: notify admins only.

    Do not delete the message, count messages_ban, record strikes, or write
    punishment state — classification failed and needs human eyes.
    """
    author = message.author
    guild = message.guild
    if not guild:
        return

    channel_name = getattr(message.channel, "name", "unknown")
    content = _message_text(message)
    images = evidence_images if evidence_images is not None else await _snapshot_images(message)
    notified = await _dm_admins_review(
        guild,
        channel_name=channel_name,
        author=author,
        content=content,
        reason=reason,
        message_id=message.id,
        channel_id=message.channel.id,
        images=images,
        removed=False,
        status="审核服务不可用，已请求人工复核（消息未删除）",
    )
    if notified == 0:
        logging.error(
            f"No admins notified for REVIEW in guild {guild.id} "
            f"(message {message.id}, user {author.id})"
        )
    else:
        logging.info(
            f"REVIEW requested for user {author.id} message {message.id} "
            f"({notified} admin DM(s))"
        )


async def _ban_user(
    message: discord.Message,
    reason: str,
    *,
    evidence_images: list[tuple[str, bytes]] | None = None,
):
    """Queue an AI BAN classification for admin review without enforcement."""
    increment_stat('messages_ban')
    MESSAGES_CLASSIFIED_BAN.set(get_stat('messages_ban'))
    author = message.author
    guild = message.guild
    if not guild:
        return

    channel_name = getattr(message.channel, "name", "unknown")
    content = _message_text(message)
    images = evidence_images if evidence_images is not None else await _snapshot_images(message)
    notified = await _dm_admins_review(
        guild,
        channel_name=channel_name,
        author=author,
        content=content,
        reason=reason,
        message_id=message.id,
        channel_id=message.channel.id,
        images=images,
        removed=False,
        status="检测到可疑内容，等待管理员处理（消息未删除）",
    )
    if notified == 0:
        logging.error(
            f"No admins notified for AI BAN in guild {guild.id} "
            f"(message {message.id}, user {author.id})"
        )
    else:
        logging.info(
            f"AI BAN queued for user {author.id} message {message.id} "
            f"({notified} admin DM(s), no automatic enforcement)"
        )


async def _claim_or_reject_review(
    interaction: discord.Interaction,
    *,
    guild_id: int,
    message_id: int,
    user_id: int,
    decision: str,
) -> str | None:
    """
    Claim a review decision or allow the same admin to retry enforcement.

    Returns:
      "new" — first successful claim
      "retry" — same admin/same decision may retry Discord enforcement
      None — rejected; an ephemeral reply was already sent
    """
    existing = claim_review_decision(
        guild_id,
        message_id,
        user_id,
        decision,
        interaction.user.id,
    )
    if existing is None:
        return "new"
    owner = get_review_decision_owner(guild_id, message_id, user_id)
    if owner and owner[0] == decision and owner[1] == interaction.user.id:
        return "retry"
    await interaction.response.send_message(
        f"该审核已被其他管理员处理（{existing}）。",
        ephemeral=True,
    )
    return None


async def _edit_review_message(
    interaction: discord.Interaction,
    content: str,
    *,
    view: discord.ui.View | None = None,
    keep_view: bool = False,
) -> None:
    """Mark the decision while retaining the review's source and evidence."""
    original = ((interaction.message.content if interaction.message else "") or "").strip()
    lines = original.splitlines()

    # Retry edits replace the previous status block instead of stacking it.
    if lines and lines[0].startswith("📌 处理状态："):
        lines.pop(0)
        if lines and lines[0].startswith("👮 处理管理员："):
            lines.pop(0)
        if lines and not lines[0].strip():
            lines.pop(0)

    # The initial AI-review line mixes pending state and source. Preserve only
    # the source once a concrete administrator decision is available.
    if lines and lines[0].startswith("⚠️ ") and "｜" in lines[0]:
        _, _, source = lines[0].partition("｜")
        lines[0] = f"📍 原始频道：{source}"

    moderator = interaction.user
    moderator_id = getattr(moderator, "id", "unknown")
    evidence = "\n".join(lines).strip()
    updated = (
        f"📌 处理状态：{content}\n"
        f"👮 处理管理员：{moderator} (`{moderator_id}`)"
    )
    if evidence:
        updated += f"\n\n{evidence}"
    kwargs: dict = {"content": updated[:2000]}
    if keep_view:
        # Leave existing components untouched.
        pass
    else:
        kwargs["view"] = view
    await interaction.edit_original_response(**kwargs)


def _related_ban_review_content(
    original: str,
    moderator: discord.abc.User,
) -> str:
    """Render a pending review card closed by a ban on a related card."""
    lines = (original or "").strip().splitlines()
    if lines and lines[0].startswith("📌 处理状态："):
        lines.pop(0)
        if lines and lines[0].startswith("👮 处理管理员："):
            lines.pop(0)
        if lines and not lines[0].strip():
            lines.pop(0)
    if lines and lines[0].startswith("⚠️ ") and "｜" in lines[0]:
        _, _, source = lines[0].partition("｜")
        lines[0] = f"📍 原始频道：{source}"

    moderator_id = getattr(moderator, "id", "unknown")
    updated = (
        "📌 处理状态：🚫 用户已封禁，该消息已随封禁操作删除"
        "（最近 24 小时清理范围）。\n"
        f"👮 处理管理员：{moderator} (`{moderator_id}`)"
    )
    evidence = "\n".join(lines).strip()
    if evidence:
        updated += f"\n\n{evidence}"
    return updated[:2000]


async def _close_related_ban_reviews(
    client: discord.Client,
    *,
    guild_id: int,
    user_id: int,
    current_message_id: int,
    moderator: discord.abc.User,
    already_edited_dm_message_id: int | None = None,
) -> tuple[int, int]:
    """Best-effort close every pending DM card covered by the 24-hour ban."""
    rows = claim_related_review_notifications(
        guild_id,
        user_id,
        current_message_id,
        moderator.id,
        max_age_seconds=_BAN_DELETE_MESSAGE_SECONDS,
    )
    updated = 0
    failed = 0
    for dm_channel_id, dm_message_id, _ in rows:
        if dm_message_id == already_edited_dm_message_id:
            mark_review_notification_resolved(dm_message_id)
            updated += 1
            continue
        try:
            channel = client.get_channel(dm_channel_id)
            if channel is None:
                channel = await client.fetch_channel(dm_channel_id)
            review_message = await channel.fetch_message(dm_message_id)
            await review_message.edit(
                content=_related_ban_review_content(
                    review_message.content,
                    moderator,
                ),
                view=None,
            )
            mark_review_notification_resolved(dm_message_id)
            updated += 1
        except discord.NotFound:
            mark_review_notification_resolved(dm_message_id)
        except Exception as e:
            failed += 1
            logging.warning(
                "Could not close related review DM %s in channel %s: %s",
                dm_message_id,
                dm_channel_id,
                e,
            )
    return updated, failed


async def _require_interaction_admin(
    interaction: discord.Interaction,
    guild_id: int,
) -> discord.Guild | None:
    guild = interaction.client.get_guild(guild_id)
    if not guild:
        await interaction.response.send_message(
            "未找到服务器（机器人可能已离开）。",
            ephemeral=True,
        )
        return None
    member = guild.get_member(interaction.user.id)
    if member is None:
        try:
            member = await guild.fetch_member(interaction.user.id)
        except Exception as e:
            logging.warning(f"Could not fetch interaction member {interaction.user.id}: {e}")
    if member is None or not (
        member.id == guild.owner_id
        or member.guild_permissions.administrator
    ):
        await interaction.response.send_message(
            "仅管理员可操作。",
            ephemeral=True,
        )
        return None
    return guild


def _interaction_review_text(
    interaction: discord.Interaction,
    *,
    guild_id: int,
    message_id: int,
    user_id: int,
) -> str:
    stored = get_review_evidence(guild_id, message_id, user_id)
    if stored:
        return stored
    message = interaction.message
    content = (message.content if message else "") or ""
    for marker in ("📝 内容：\n", "📝 Content:\n"):
        _, separator, review_text = content.partition(marker)
        if separator:
            return review_text.strip()
    return ""


async def _interaction_review_images(
    interaction: discord.Interaction,
) -> list[tuple[str, bytes]]:
    message = interaction.message
    if not message or not getattr(message, "attachments", None):
        return []
    return await _snapshot_images(message)


def _review_channel_name(guild: discord.Guild, channel_id: int) -> str:
    for accessor_name in ("get_channel_or_thread", "get_channel"):
        accessor = getattr(guild, accessor_name, None)
        if not callable(accessor):
            continue
        channel = accessor(channel_id)
        name = getattr(channel, "name", None) if channel is not None else None
        if name:
            return str(name)
    return f"unknown-{channel_id}"


async def _delete_reviewed_message(
    guild: discord.Guild,
    *,
    channel_id: int,
    message_id: int,
) -> str:
    """Delete the reviewed message and return deleted or missing."""
    try:
        channel = guild.get_channel_or_thread(channel_id)
        if channel is None:
            channel = await guild.fetch_channel(channel_id)
        message = await channel.fetch_message(message_id)
        await message.delete()
    except discord.NotFound:
        return "missing"
    return "deleted"


class HITLBanButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=(
        r"sm:h:b:(?P<guild_id>[0-9]+):(?P<user_id>[0-9]+):"
        r"(?P<message_id>[0-9]+):(?P<channel_id>[0-9]+)"
    ),
):
    def __init__(
        self,
        guild_id: int,
        user_id: int,
        message_id: int,
        channel_id: int,
    ):
        self.guild_id = guild_id
        self.user_id = user_id
        self.message_id = message_id
        self.channel_id = channel_id
        super().__init__(
            discord.ui.Button(
                label="🚫 删除并封禁",
                style=discord.ButtonStyle.danger,
                custom_id=(
                    f"sm:h:b:{guild_id}:{user_id}:"
                    f"{message_id}:{channel_id}"
                ),
            )
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(
            guild_id=int(match["guild_id"]),
            user_id=int(match["user_id"]),
            message_id=int(match["message_id"]),
            channel_id=int(match["channel_id"]),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        guild = await _require_interaction_admin(interaction, self.guild_id)
        if not guild:
            return
        text = _interaction_review_text(
            interaction,
            guild_id=self.guild_id,
            message_id=self.message_id,
            user_id=self.user_id,
        )
        if not text:
            await interaction.response.send_message(
                "审核上下文不可用。",
                ephemeral=True,
            )
            return
        claim_state = await _claim_or_reject_review(
            interaction,
            guild_id=self.guild_id,
            message_id=self.message_id,
            user_id=self.user_id,
            decision="ban",
        )
        if not claim_state:
            return
        await interaction.response.defer()
        evidence_images = await _interaction_review_images(interaction)
        channel_name = _review_channel_name(guild, self.channel_id)
        if claim_state == "new":
            try:
                _record_training_example(text, "BAN")
            except Exception as e:
                logging.error(f"Error recording ban training example: {e}")
            try:
                increment_stat('bans_confirmed')
                BANS_CONFIRMED.set(get_stat('bans_confirmed'))
            except Exception as e:
                logging.error(f"Error updating bans_confirmed metric: {e}")
        try:
            delete_state = await _delete_reviewed_message(
                guild,
                channel_id=self.channel_id,
                message_id=self.message_id,
            )
        except Exception as e:
            logging.warning(f"Could not delete message before ban: {e}")
            delete_state = "failed"

        try:
            user = await interaction.client.fetch_user(self.user_id)
            await _execute_ban(
                guild=guild,
                user=user,
                reason="Confirmed by admin",
                channel_name=channel_name,
                evidence_message_id=self.message_id,
                evidence_text=text,
                evidence_images=evidence_images,
            )
        except Exception as e:
            logging.error(f"Error banning user: {e}")
            await _edit_review_message(
                interaction,
                "⚠️ 封禁失败，请再点一次「删除并封禁」。",
                keep_view=True,
            )
            return
        if delete_state == "deleted":
            result = (
                "🚫 已删除该消息并封禁用户；Discord 同时清理了该用户最近 "
                "24 小时内的服务器消息。"
            )
        elif delete_state == "missing":
            result = (
                "🚫 原消息已不存在，已封禁用户；Discord 已清理该用户最近 "
                "24 小时内的服务器消息。"
            )
        else:
            result = (
                "🚫 已封禁用户；单条消息删除失败，但 Discord 已通过封禁清理"
                "该用户最近 24 小时内的服务器消息。"
            )
        current_edit_succeeded = False
        try:
            await _edit_review_message(interaction, result)
            current_edit_succeeded = True
        except Exception as e:
            logging.error(f"Ban succeeded but review message edit failed: {e}")
        current_dm_message_id = getattr(interaction.message, "id", None)
        if not isinstance(current_dm_message_id, int) or not current_edit_succeeded:
            current_dm_message_id = None
        try:
            updated, failed = await _close_related_ban_reviews(
                interaction.client,
                guild_id=self.guild_id,
                user_id=self.user_id,
                current_message_id=self.message_id,
                moderator=interaction.user,
                already_edited_dm_message_id=current_dm_message_id,
            )
            logging.info(
                "Closed related ban reviews for guild %s user %s: "
                "%s updated, %s failed",
                self.guild_id,
                self.user_id,
                updated,
                failed,
            )
        except Exception as e:
            logging.error(f"Could not close related ban reviews: {e}")


class HITLDeleteButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=(
        r"sm:h:d:(?P<guild_id>[0-9]+):(?P<user_id>[0-9]+):"
        r"(?P<message_id>[0-9]+):(?P<channel_id>[0-9]+)"
    ),
):
    def __init__(
        self,
        guild_id: int,
        user_id: int,
        message_id: int,
        channel_id: int,
    ):
        self.guild_id = guild_id
        self.user_id = user_id
        self.message_id = message_id
        self.channel_id = channel_id
        super().__init__(
            discord.ui.Button(
                label="🗑️ 删除",
                style=discord.ButtonStyle.primary,
                custom_id=(
                    f"sm:h:d:{guild_id}:{user_id}:"
                    f"{message_id}:{channel_id}"
                ),
            )
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(
            guild_id=int(match["guild_id"]),
            user_id=int(match["user_id"]),
            message_id=int(match["message_id"]),
            channel_id=int(match["channel_id"]),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        guild = await _require_interaction_admin(interaction, self.guild_id)
        if not guild:
            return
        text = _interaction_review_text(
            interaction,
            guild_id=self.guild_id,
            message_id=self.message_id,
            user_id=self.user_id,
        )
        if not text:
            await interaction.response.send_message(
                "审核上下文不可用。",
                ephemeral=True,
            )
            return
        claim_state = await _claim_or_reject_review(
            interaction,
            guild_id=self.guild_id,
            message_id=self.message_id,
            user_id=self.user_id,
            decision="delete",
        )
        if not claim_state:
            return
        await interaction.response.defer()
        evidence_images = await _interaction_review_images(interaction)
        channel_name = _review_channel_name(guild, self.channel_id)

        try:
            delete_state = await _delete_reviewed_message(
                guild,
                channel_id=self.channel_id,
                message_id=self.message_id,
            )
        except Exception as e:
            logging.error(f"Error deleting reviewed message: {e}")
            try:
                release_review_decision(
                    self.guild_id,
                    self.message_id,
                    self.user_id,
                    "delete",
                    interaction.user.id,
                )
            except Exception as release_error:
                logging.error(f"Could not release failed delete decision: {release_error}")
            await _edit_review_message(
                interaction,
                "⚠️ 删除失败，请重试；其他管理员也可以接手处理。",
                keep_view=True,
            )
            return

        if claim_state == "new":
            try:
                _record_training_example(text, "BAN")
            except Exception as e:
                logging.error(f"Error recording delete training example: {e}")

        if delete_state == "deleted":
            try:
                user = await interaction.client.fetch_user(self.user_id)
                await _dm_user(
                    user,
                    _user_notice_with_evidence(
                        "Your message was deleted after review by a server "
                        "administrator. You were not banned.",
                        guild_name=getattr(guild, "name", "Unknown server"),
                        channel_name=channel_name,
                        message_id=self.message_id,
                        content=text,
                        images=evidence_images,
                    ),
                    images=evidence_images,
                )
            except Exception as e:
                logging.warning(f"Could not notify user after admin deletion: {e}")
            result = "🗑️ 已删除该消息，未封禁用户。"
        else:
            result = "ℹ️ 原消息已不存在，未封禁用户，也未发送删除通知。"

        try:
            await _edit_review_message(interaction, result)
        except Exception as e:
            logging.error(f"Delete succeeded but review message edit failed: {e}")


class HITLFalseAlarmButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=(
        r"sm:h:f:(?P<guild_id>[0-9]+):(?P<user_id>[0-9]+):"
        r"(?P<message_id>[0-9]+):(?P<channel_id>[0-9]+)"
    ),
):
    def __init__(
        self,
        guild_id: int,
        user_id: int,
        message_id: int,
        channel_id: int,
    ):
        self.guild_id = guild_id
        self.user_id = user_id
        self.message_id = message_id
        self.channel_id = channel_id
        super().__init__(
            discord.ui.Button(
                label="❌ 误报",
                style=discord.ButtonStyle.secondary,
                custom_id=(
                    f"sm:h:f:{guild_id}:{user_id}:"
                    f"{message_id}:{channel_id}"
                ),
            )
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(
            guild_id=int(match["guild_id"]),
            user_id=int(match["user_id"]),
            message_id=int(match["message_id"]),
            channel_id=int(match["channel_id"]),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        guild = await _require_interaction_admin(interaction, self.guild_id)
        if not guild:
            return
        text = _interaction_review_text(
            interaction,
            guild_id=self.guild_id,
            message_id=self.message_id,
            user_id=self.user_id,
        )
        if not text:
            await interaction.response.send_message(
                "审核上下文不可用。",
                ephemeral=True,
            )
            return
        claim_state = await _claim_or_reject_review(
            interaction,
            guild_id=self.guild_id,
            message_id=self.message_id,
            user_id=self.user_id,
            decision="false_alarm",
        )
        if not claim_state:
            return
        await interaction.response.defer()
        if claim_state == "new":
            try:
                _record_training_example(text, "SAFE")
            except Exception as e:
                logging.error(f"Error recording false-alarm training example: {e}")
            try:
                review_reason = (
                    get_review_reason(self.guild_id, self.message_id, self.user_id)
                    or ""
                )
                # Classifier outages are not false positives.
                if "unavailable" not in review_reason.lower():
                    increment_stat('false_positives')
                    FALSE_POSITIVES.set(get_stat('false_positives'))
            except Exception as e:
                logging.error(f"Error updating false_positives metric: {e}")
        # Idempotent: also run on enforcement retries if an earlier clear failed.
        try:
            strikes.clear(self.guild_id, self.user_id)
        except Exception as e:
            logging.error(f"Error clearing strikes after false alarm: {e}")

        unban_note = "无自动封禁记录可撤销。"
        auto_ban_message_id = take_reversible_auto_ban(
            self.guild_id,
            self.user_id,
        )
        if auto_ban_message_id is not None:
            try:
                await guild.unban(
                    discord.Object(id=self.user_id),
                    reason="False alarm confirmed",
                )
                unban_note = "已撤销自动封禁。"
            except discord.NotFound:
                unban_note = "自动封禁记录已清除。"
            except Exception as e:
                logging.warning(
                    "Could not unban user %s after false alarm: %s",
                    self.user_id,
                    e,
                )
                record_auto_ban(
                    self.guild_id,
                    self.user_id,
                    auto_ban_message_id,
                )
                await _edit_review_message(
                    interaction,
                    "⚠️ 自动解封失败，请再点一次「误报」。",
                    keep_view=True,
                )
                return

        try:
            await _edit_review_message(
                interaction,
                f"❌ 判定为误报。{unban_note} 违规计数已清零。",
            )
        except Exception as e:
            logging.error(f"False alarm succeeded but review message edit failed: {e}")


class HITLView(discord.ui.View):
    def __init__(
        self,
        guild_id: int,
        user_id: int,
        username: str,
        text: str,
        message_id: int,
        channel_id: int,
        reason: str = "Suspicious message",
    ):
        super().__init__(timeout=None)
        self.add_item(
            HITLBanButton(guild_id, user_id, message_id, channel_id)
        )
        self.add_item(
            HITLDeleteButton(guild_id, user_id, message_id, channel_id)
        )
        self.add_item(
            HITLFalseAlarmButton(guild_id, user_id, message_id, channel_id)
        )


def main():
    if not DISCORD_BOT_TOKEN:
        print("DISCORD_BOT_TOKEN is not set. Exiting.")
        return
    init_db()
    ensure_normalized_index()
    init_metrics()
    start_health_server()
    start_http_server(METRICS_PORT)
    logging.info(f"Prometheus metrics listening on :{METRICS_PORT}")
    client.run(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    main()
