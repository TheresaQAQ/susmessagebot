import discord
from discord import app_commands
from .moderator import classify_message
from .image_moderator import classify_image
from .url_moderator import analyze_urls, load_blocklist
from .config import (
    DISCORD_BOT_TOKEN,
    APPEAL_DISCORD_USER_ID,
    HEALTH_PORT,
    METRICS_PORT,
)
from .github_sync import sync_example_to_github
from .vector_store import add_example
from .strike_tracker import strikes, remove_notice_text, ban_notice_text
import logging
import asyncio
import io
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
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
    store_review_evidence,
    get_review_evidence,
    get_review_reason,
    record_auto_ban,
    clear_auto_ban,
    take_reversible_auto_ban,
)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Set True after Discord gateway ready; /health returns 503 until then.
_bot_ready = False


class SusMessageBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        self.add_dynamic_items(
            HITLBanButton,
            HITLFalseAlarmButton,
            ReportConfirmButton,
            ReportDismissButton,
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


@client.event
async def on_ready():
    global _bot_ready
    load_blocklist()
    for guild in client.guilds:
        add_group(guild.id, guild.member_count)
    _refresh_guild_metrics()
    _bot_ready = True
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

    channel = guild.system_channel or next(
        (c for c in guild.text_channels if c.permissions_for(guild.me).send_messages), None
    )
    if channel:
        await channel.send(
            "👋 Thanks for adding SusMessageBot! I am an AI Anti-Scam Moderation Bot!\n\n"
            "Please ensure I have these permissions:\n"
            "✅ Ban Members\n"
            "✅ Manage Messages\n"
            "✅ View Channels\n"
            "✅ Send Messages\n"
            "✅ Read Message History\n\n"
            "Once set up, I'll automatically moderate scam messages and protect your group!"
        )


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

    # Handle @SusMessageBot mentions as reports
    if client.user in message.mentions and message.reference:
        try:
            reported_msg = await message.channel.fetch_message(
                message.reference.message_id
            )
        except Exception as e:
            logging.warning(f"Could not fetch reported message: {e}")
            await message.channel.send(
                "⚠️ I could not access the referenced message. "
                "It may have been deleted already."
            )
            return
        is_admin = message.author.guild_permissions.administrator

        if is_admin:
            text = _message_text(reported_msg)
            try:
                _record_training_example(text, "BAN")
                increment_stat('false_negatives')
                FALSE_NEGATIVES.set(get_stat('false_negatives'))
            except Exception as e:
                logging.error(f"Training persistence failed for admin report: {e}")
            try:
                await reported_msg.delete()
                await _execute_ban(
                    guild=message.guild,
                    user=reported_msg.author,
                    reason="Reported by admin",
                )
                await message.channel.send("✅ User banned.")
            except Exception as e:
                logging.error(f"Error banning reported user: {e}")
                await message.channel.send(
                    "⚠️ Could not ban user. Please retry or ban manually."
                )
        else:
            text = _message_text(reported_msg)
            images = await _snapshot_images(reported_msg)
            preview = text if len(text) <= 1500 else text[:1500] + "..."
            body = (
                f"🚨 Scam report by {message.author} in **#{message.channel.name}**\n\n"
                f"👤 Reported user: {reported_msg.author} (`{reported_msg.author.id}`)\n\n"
                f"📝 Content:\n{preview}"
            )
            store_review_evidence(
                message.guild.id,
                reported_msg.id,
                reported_msg.author.id,
                text,
                "User report",
            )
            notified = 0
            for admin in await _admin_members(message.guild):
                try:
                    view = ReportReviewView(
                        guild_id=message.guild.id,
                        user_id=reported_msg.author.id,
                        username=str(reported_msg.author),
                        text=text,
                        message_id=reported_msg.id,
                        channel_id=reported_msg.channel.id,
                    )
                    await admin.send(body, view=view, files=_discord_files(images))
                    notified += 1
                except Exception as e:
                    logging.warning(f"Could not DM admin {admin.id}: {e}")
            if notified:
                await message.channel.send(f"✅ Report sent to {notified} admin(s).")
            else:
                await message.channel.send(
                    "⚠️ Could not DM any admins — they may have DMs closed."
                )
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

    text = (message.content or "").strip()
    if text:
        text_result = await loop.run_in_executor(None, classify_message, text)
        url_result = await loop.run_in_executor(None, analyze_urls, text)
        if text_result == "BAN" or url_result == "BAN":
            final = "BAN"
            review_reason = "Suspicious message"
        elif text_result == "REVIEW" or url_result == "REVIEW":
            if final != "BAN":
                final = "REVIEW"
                if review_reason == "Moderation unavailable":
                    review_reason = "Text moderation unavailable"
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


async def _dm_user(user: discord.abc.User, text: str) -> None:
    try:
        await user.send(text)
    except Exception as e:
        logging.warning(f"Could not DM user {user.id}: {e}")


async def _execute_ban(
    *,
    guild: discord.Guild,
    user: discord.abc.User,
    reason: str,
    preserve_auto_ban_record: bool = False,
    automatic: bool = False,
) -> None:
    """DM ban notice first (ban removes mutual servers), then ban and clear strikes."""
    await _dm_user(
        user,
        ban_notice_text(APPEAL_DISCORD_USER_ID, automatic=automatic),
    )
    await guild.ban(discord.Object(id=user.id), reason=reason)
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
    trainable = _trainable_text(text)
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


async def _dm_admins_text(
    guild: discord.Guild,
    body: str,
    *,
    images: list[tuple[str, bytes]] | None = None,
) -> int:
    notified = 0
    for admin in await _admin_members(guild):
        try:
            await admin.send(body, files=_discord_files(images))
            notified += 1
        except Exception as e:
            logging.warning(f"Could not DM admin {admin.id}: {e}")
    return notified


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
    preview = content if len(content) <= 1500 else content[:1500] + "..."
    if status is None:
        status = (
            "Suspicious content removed"
            if removed
            else "Suspicious content flagged, but automatic removal failed"
        )
    body = (
        f"⚠️ {status} in **#{channel_name}** ({guild.name})\n\n"
        f"👤 User: {author} (`{author.id}`)\n\n"
        f"📝 Content:\n{preview}"
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
            await admin.send(body, view=view, files=_discord_files(images))
            notified += 1
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
        status="Moderation unavailable — manual review requested (message not removed)",
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
    record_strike: bool = True,
):
    """Delete and request review; only confirmed classifications count as strikes."""
    increment_stat('messages_ban')
    MESSAGES_CLASSIFIED_BAN.set(get_stat('messages_ban'))
    author = message.author
    guild = message.guild
    channel_name = getattr(message.channel, "name", "unknown")
    message_id = message.id
    channel_id = message.channel.id
    content = _message_text(message)
    images = evidence_images if evidence_images is not None else await _snapshot_images(message)

    try:
        await message.delete()
    except Exception as e:
        logging.error(f"Error deleting message: {e}")
        deleted = False
    else:
        deleted = True

    if not guild:
        return

    notified = await _dm_admins_review(
        guild,
        channel_name=channel_name,
        author=author,
        content=content,
        reason=reason,
        message_id=message_id,
        channel_id=channel_id,
        images=images,
        removed=deleted,
    )
    if notified == 0:
        logging.error(
            f"No admins notified for guild {guild.id} — "
            "not recording a strike because no review path is available"
        )
    elif record_strike:
        should_autoban = strikes.record(guild.id, author.id)
        if should_autoban:
            threshold = strikes.threshold
            window_min = strikes.window_seconds // 60
            logging.info(f"Auto-ban after {threshold} reviewed triggers: user {author.id}")
            # Do not treat machine auto-bans as admin-confirmed training/metrics.
            # Admins can still confirm/false-alarm via the review DM.
            record_auto_ban(guild.id, author.id, message_id)
            try:
                await _execute_ban(
                    guild=guild,
                    user=author,
                    reason=f"Auto-ban: {reason}",
                    preserve_auto_ban_record=True,
                    automatic=True,
                )
            except Exception as e:
                logging.error(f"Error auto-banning user: {e}")
                clear_auto_ban(guild.id, author.id)
                return
            await _dm_admins_text(
                guild,
                f"🚫 Auto-banned {author} (`{author.id}`) in **#{channel_name}** "
                f"after {threshold} reviewed triggers within {window_min} minutes.\n\n"
                f"📝 Last content:\n{content[:1500]}{'...' if len(content) > 1500 else ''}",
                images=images,
            )
            return

    if deleted:
        logging.info(f"Message removed (pending admin ban) for user {author.id}")
        await _dm_user(author, remove_notice_text(guild.id, author.id))
    else:
        logging.info(f"Message flagged but not removed (pending admin review) for user {author.id}")
        await _dm_user(
            author,
            "⚠️ Your message was flagged for moderation review, but I could not "
            "remove it automatically. Server admins have been notified.",
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
        f"Already handled by another admin ({existing}).",
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
    """Edit the review DM after the interaction has been deferred."""
    kwargs: dict = {"content": content}
    if keep_view:
        # Leave existing components untouched.
        pass
    else:
        kwargs["view"] = view
    await interaction.edit_original_response(**kwargs)


async def _require_interaction_admin(
    interaction: discord.Interaction,
    guild_id: int,
) -> discord.Guild | None:
    guild = interaction.client.get_guild(guild_id)
    if not guild:
        await interaction.response.send_message(
            "Server not found (bot may have left).",
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
            "Only admins can do this.",
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
    marker = "📝 Content:\n"
    _, separator, review_text = content.partition(marker)
    return review_text.strip() if separator else ""


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
                label="🚫 Ban",
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
                "Review context is unavailable.",
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
        if claim_state == "new":
            try:
                _record_training_example(text, "BAN")
                increment_stat('bans_confirmed')
                BANS_CONFIRMED.set(get_stat('bans_confirmed'))
            except Exception as e:
                logging.error(f"Error recording ban training example: {e}")
        try:
            channel = guild.get_channel(self.channel_id)
            if channel:
                try:
                    msg = await channel.fetch_message(self.message_id)
                    await msg.delete()
                except Exception:
                    pass
            user = await interaction.client.fetch_user(self.user_id)
            await _execute_ban(
                guild=guild,
                user=user,
                reason="Confirmed by admin",
            )
            await _edit_review_message(interaction, "🚫 User banned.")
        except Exception as e:
            logging.error(f"Error banning user: {e}")
            await _edit_review_message(
                interaction,
                "⚠️ Ban failed. Please try Ban again.",
                keep_view=True,
            )


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
                label="❌ False Alarm",
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
                "Review context is unavailable.",
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
                review_reason = (
                    get_review_reason(self.guild_id, self.message_id, self.user_id)
                    or ""
                )
                # Classifier outages are not false positives.
                if "unavailable" not in review_reason.lower():
                    increment_stat('false_positives')
                    FALSE_POSITIVES.set(get_stat('false_positives'))
            except Exception as e:
                logging.error(f"Error recording false-alarm training example: {e}")
            # Strike cleanup is mandatory even if training persistence fails.
            strikes.clear(self.guild_id, self.user_id)

        unban_note = "No auto-ban to reverse."
        if take_reversible_auto_ban(self.guild_id, self.user_id, self.message_id):
            try:
                await guild.unban(
                    discord.Object(id=self.user_id),
                    reason="False alarm confirmed",
                )
                unban_note = "Auto-ban reversed."
            except discord.NotFound:
                unban_note = "Auto-ban record cleared."
            except Exception as e:
                logging.warning(
                    "Could not unban user %s after false alarm: %s",
                    self.user_id,
                    e,
                )
                record_auto_ban(self.guild_id, self.user_id, self.message_id)
                await _edit_review_message(
                    interaction,
                    "⚠️ Could not unban automatically. Please try False Alarm again.",
                    keep_view=True,
                )
                return

        await _edit_review_message(
            interaction,
            f"❌ False alarm. {unban_note} Strikes cleared.",
        )


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
            HITLFalseAlarmButton(guild_id, user_id, message_id, channel_id)
        )


@client.tree.context_menu(name="Report to SusMessageBot")
async def report_context_menu(interaction: discord.Interaction, message: discord.Message):
    await _handle_report(interaction, message)


async def _handle_report(interaction: discord.Interaction, message: discord.Message):
    await interaction.response.defer(ephemeral=True)
    is_admin = interaction.user.guild_permissions.administrator

    if is_admin:
        text = _message_text(message)
        try:
            _record_training_example(text, "BAN")
            increment_stat('false_negatives')
            FALSE_NEGATIVES.set(get_stat('false_negatives'))
        except Exception as e:
            logging.error(f"Training persistence failed for admin report: {e}")
        try:
            await message.delete()
            await _execute_ban(
                guild=interaction.guild,
                user=message.author,
                reason="Reported by admin",
            )
            await interaction.followup.send("✅ User banned.", ephemeral=True)
        except Exception as e:
            logging.error(f"Error banning reported user: {e}")
            await interaction.followup.send(
                "⚠️ Could not ban user. Please retry or ban manually.",
                ephemeral=True,
            )
    else:
        text = _message_text(message)
        images = await _snapshot_images(message)
        store_review_evidence(
            interaction.guild.id,
            message.id,
            message.author.id,
            text,
            "User report",
        )
        preview = text if len(text) <= 1500 else text[:1500] + "..."
        body = (
            f"🚨 Scam report by {interaction.user} in **#{message.channel.name}**\n\n"
            f"👤 Reported user: {message.author} (`{message.author.id}`)\n\n"
            f"📝 Content:\n{preview}"
        )
        notified = 0
        for admin in await _admin_members(interaction.guild):
            try:
                view = ReportReviewView(
                    guild_id=interaction.guild.id,
                    user_id=message.author.id,
                    username=str(message.author),
                    text=text,
                    message_id=message.id,
                    channel_id=message.channel.id,
                )
                await admin.send(body, view=view, files=_discord_files(images))
                notified += 1
            except Exception as e:
                logging.warning(f"Could not DM admin {admin.id}: {e}")
        await interaction.followup.send(
            f"✅ Report sent to {notified} admin(s)." if notified else
            "⚠️ Could not DM any admins — they may have DMs closed.",
            ephemeral=True,
        )


class ReportConfirmButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=(
        r"sm:r:b:(?P<guild_id>[0-9]+):(?P<user_id>[0-9]+):"
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
                label="✅ Confirm Ban",
                style=discord.ButtonStyle.green,
                custom_id=(
                    f"sm:r:b:{guild_id}:{user_id}:"
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
                "Review context is unavailable.",
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
        if claim_state == "new":
            try:
                _record_training_example(text, "BAN")
                increment_stat('false_negatives')
                FALSE_NEGATIVES.set(get_stat('false_negatives'))
            except Exception as e:
                logging.error(f"Error recording report ban training example: {e}")
        try:
            channel = guild.get_channel(self.channel_id)
            if channel:
                try:
                    msg = await channel.fetch_message(self.message_id)
                    await msg.delete()
                except Exception:
                    pass
            user = await interaction.client.fetch_user(self.user_id)
            await _execute_ban(
                guild=guild,
                user=user,
                reason="Confirmed by admin",
            )
            await _edit_review_message(
                interaction,
                "✅ Report confirmed. User banned.",
            )
        except Exception as e:
            logging.error(f"Error banning reported user: {e}")
            await _edit_review_message(
                interaction,
                "⚠️ Ban failed. Please try Confirm Ban again.",
                keep_view=True,
            )


class ReportDismissButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=(
        r"sm:r:d:(?P<guild_id>[0-9]+):(?P<user_id>[0-9]+):"
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
                label="❌ Dismiss",
                style=discord.ButtonStyle.red,
                custom_id=(
                    f"sm:r:d:{guild_id}:{user_id}:"
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
        if not await _require_interaction_admin(interaction, self.guild_id):
            return
        if not await _claim_or_reject_review(
            interaction,
            guild_id=self.guild_id,
            message_id=self.message_id,
            user_id=self.user_id,
            decision="dismiss",
        ):
            return
        await interaction.response.defer()
        await _edit_review_message(interaction, "❌ Report dismissed.")


class ReportReviewView(discord.ui.View):
    def __init__(
        self,
        guild_id: int,
        user_id: int,
        username: str,
        text: str,
        message_id: int,
        channel_id: int,
    ):
        super().__init__(timeout=None)
        self.add_item(
            ReportConfirmButton(
                guild_id,
                user_id,
                message_id,
                channel_id,
            )
        )
        self.add_item(
            ReportDismissButton(
                guild_id,
                user_id,
                message_id,
                channel_id,
            )
        )


def main():
    if not DISCORD_BOT_TOKEN:
        print("DISCORD_BOT_TOKEN is not set. Exiting.")
        return
    init_db()
    init_metrics()
    start_health_server()
    start_http_server(METRICS_PORT)
    logging.info(f"Prometheus metrics listening on :{METRICS_PORT}")
    client.run(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    main()