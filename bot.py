from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, MessageHandler, CallbackQueryHandler, filters, ContextTypes, CommandHandler
from telegram.constants import ChatMemberStatus
from config import TELEGRAM_BOT_TOKEN, WEBHOOK_URL, USE_POLLING, APPEAL_DISCORD_USER_ID
from moderator import classify_message
from url_moderator import analyze_urls
from github_sync import sync_example_to_github
from strike_tracker import strikes, remove_notice_text, ban_notice_text
from prometheus_client import Gauge, start_http_server
from http.server import HTTPServer, BaseHTTPRequestHandler
from stats import init_db, get_stat, increment_stat, decrement_stat, add_group, update_group_member_count, get_groups_count, get_total_members, get_all_group_ids
import asyncio
import threading

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Prometheus metrics for Grafana 
MESSAGES_CLASSIFIED_SAFE = Gauge('messages_classified_safe_total', 'SAFE classifications')
MESSAGES_CLASSIFIED_BAN = Gauge('messages_classified_ban_total', 'BAN classifications')
BANS_CONFIRMED = Gauge('bans_confirmed_total', 'Admin-confirmed correct bans')
FALSE_POSITIVES = Gauge('false_positives_total', 'Admin-confirmed false positives')
FALSE_NEGATIVES = Gauge('false_negatives_total', 'Admin-reported false negatives')
# Other Generic Stats
GROUPS_COUNT = Gauge('groups_count_total', 'Number of groups bot is in')
MEMBERS_PROTECTED = Gauge('members_protected_total', 'Total members protected')

ACCURATE_CLASSIFICATIONS = Gauge('accurate_classifications_total', 'Accurately classified messages')

def init_metrics():
    """Load persisted values from SQLite into Prometheus gauges."""
    MESSAGES_CLASSIFIED_SAFE.set(get_stat('messages_safe'))
    MESSAGES_CLASSIFIED_BAN.set(get_stat('messages_ban'))
    BANS_CONFIRMED.set(get_stat('bans_confirmed'))
    FALSE_POSITIVES.set(get_stat('false_positives'))
    FALSE_NEGATIVES.set(get_stat('false_negatives'))
    GROUPS_COUNT.set(get_groups_count())
    MEMBERS_PROTECTED.set(get_total_members())
    ACCURATE_CLASSIFICATIONS.set(get_stat('accurate_classifications'))

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_HEAD(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

def start_health_server():
    server = HTTPServer(('0.0.0.0', 8001), HealthHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

banned_messages = {}


async def _tg_dm(bot, user_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id=user_id, text=text)
    except Exception as e:
        logging.warning(f"Could not DM user {user_id}: {e}")


async def _tg_execute_ban(bot, chat_id: int, user_id: int) -> None:
    """DM ban notice first, then ban and clear strikes."""
    await _tg_dm(bot, user_id, ban_notice_text(APPEAL_DISCORD_USER_ID))
    await bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
    strikes.clear(chat_id, user_id)


async def _tg_dm_admins(bot, chat_id: int, text: str, reply_markup=None) -> int:
    notified = 0
    try:
        admins = await bot.get_chat_administrators(chat_id)
    except Exception as e:
        logging.error(f"Could not fetch administrators: {e}")
        return 0
    for admin in admins:
        if admin.user.is_bot:
            continue
        try:
            await bot.send_message(
                chat_id=admin.user.id,
                text=text,
                reply_markup=reply_markup,
            )
            notified += 1
        except Exception as e:
            logging.warning(f"Could not DM admin {admin.user.id}: {e}")
    return notified


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles every incoming message.
    Classifies it; on BAN, deletes the message and notifies for admin review.

    Args:
        update: The incoming Telegram update
        context: The bot context
    """
    if not update.message or (not update.message.text and not update.message.caption):
        return

    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    message_id = update.message.message_id
    text = update.message.text or update.message.caption


    chat_member = await context.bot.get_chat_member(chat_id, user_id)
    if chat_member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER]:
        return
    # Track new groups
    member_count = await context.bot.get_chat_member_count(chat_id)
    is_new = add_group(chat_id, member_count)
    if is_new:
        GROUPS_COUNT.set(get_groups_count())
        MEMBERS_PROTECTED.set(get_total_members())
    text_result, url_result = await asyncio.gather(
        asyncio.to_thread(classify_message, text),
        asyncio.to_thread(analyze_urls, text),
    )
    if text_result == "BAN" or url_result == "BAN":
        result = "BAN"
    elif text_result == "REVIEW" or url_result == "REVIEW":
        result = "REVIEW"
    else:
        result = "SAFE"
    logging.info(
        "telegram classify user=%s text=%r text_result=%s url_result=%s final=%s",
        user_id,
        text[:120],
        text_result,
        url_result,
        result,
    )

    if result in {"BAN", "REVIEW"}:
        if result == "BAN":
            increment_stat('messages_ban')
            MESSAGES_CLASSIFIED_BAN.set(get_stat('messages_ban'))
        banned_messages[message_id] = {
            "user_id": user_id,
            "text": text,
            "verdict": result,
        }
        user = update.message.from_user
        chat_title = update.message.chat.title or str(chat_id)
        preview = text if len(text) <= 1500 else text[:1500] + "..."

        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception as e:
            logging.error(f"Error deleting message: {e}")
            return

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🚫 Ban", callback_data=f"correct|{message_id}|{chat_id}|{user_id}"),
                InlineKeyboardButton("❌ False Alarm", callback_data=f"false|{message_id}|{chat_id}|{user_id}")
            ]
        ])
        review_text = (
            f"⚠️ {'Suspicious message' if result == 'BAN' else 'Unclassified message'} "
            f"removed in {chat_title}\n\n"
            f"👤 User: {user.full_name}"
            f"{f' (@{user.username})' if user.username else ''}\n"
            f"🆔 ID: {user_id}\n\n"
            f"📝 Message:\n{preview}"
        )
        notified = await _tg_dm_admins(
            context.bot, chat_id, review_text, reply_markup=keyboard
        )
        if notified == 0:
            logging.error(
                f"No admins notified for chat {chat_id} — "
                "not recording a strike because no review path is available"
            )
            banned_messages.pop(message_id, None)
        elif result == "BAN":
            should_autoban = strikes.record(chat_id, user_id)
            if should_autoban:
                logging.info(f"Auto-ban after {strikes.threshold} reviewed triggers: user {user_id}")
                from vector_store import add_example
                add_example(text, "BAN")
                sync_example_to_github(text, "BAN")
                increment_stat('bans_confirmed')
                BANS_CONFIRMED.set(get_stat('bans_confirmed'))
                increment_stat('accurate_classifications')
                ACCURATE_CLASSIFICATIONS.set(get_stat('accurate_classifications'))
                try:
                    await _tg_execute_ban(context.bot, chat_id, user_id)
                except Exception as e:
                    logging.error(f"Error auto-banning user: {e}")
                    return
                await _tg_dm_admins(
                    context.bot,
                    chat_id,
                    f"🚫 Auto-banned {user.full_name} ({user_id}) in {chat_title} "
                    f"after {strikes.threshold} reviewed triggers within "
                    f"{strikes.window_seconds // 60} minutes.\n\n"
                    f"📝 Last message:\n{preview}",
                )
                banned_messages.pop(message_id, None)
                return

        logging.info(f"Message removed (pending admin ban) for user {user_id}")
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=remove_notice_text(chat_id, user_id),
                api_kwargs={"receiver_user_id": user_id},
            )
        except Exception as e:
            logging.warning(f"Could not send ephemeral notice to {user_id}: {e}")
    else:
        increment_stat('messages_safe')
        MESSAGES_CLASSIFIED_SAFE.set(get_stat('messages_safe'))
        increment_stat('accurate_classifications')
        ACCURATE_CLASSIFICATIONS.set(get_stat('accurate_classifications'))

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user_id = query.from_user.id

    data = query.data.split("|")
    action = data[0]
    message_id = int(data[1])
    group_chat_id = int(data[2])

    # Admin check against the group (callback may arrive from a private DM)
    chat_member = await context.bot.get_chat_member(group_chat_id, user_id)
    if chat_member.status not in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER]:
        await query.answer("Only admins can do this.")
        return

    banned_info = banned_messages.get(message_id)
    if not banned_info:
        await query.answer("Action expired.")
        return

    if action == "correct":
        banned_user_id = (
            int(data[3])
            if len(data) > 3
            else int(banned_info["user_id"])
        )
        from vector_store import add_example
        add_example(banned_info["text"], "BAN")
        sync_example_to_github(banned_info["text"], "BAN")
        increment_stat('bans_confirmed')
        BANS_CONFIRMED.set(get_stat('bans_confirmed'))
        increment_stat('accurate_classifications')
        ACCURATE_CLASSIFICATIONS.set(get_stat('accurate_classifications'))
        try:
            await _tg_execute_ban(context.bot, group_chat_id, banned_user_id)
            await query.edit_message_text("🚫 User banned.")
        except Exception as e:
            logging.error(f"Error banning user: {e}")
            await query.edit_message_text("✅ Added to training. Could not ban user — they may have already left.")
        
    elif action == "false":
        from vector_store import add_example
        add_example(banned_info["text"], "SAFE")
        sync_example_to_github(banned_info["text"], "SAFE")
        increment_stat('false_positives')
        FALSE_POSITIVES.set(get_stat('false_positives'))
        strikes.clear(group_chat_id, banned_info["user_id"])
        await query.edit_message_text("❌ False alarm. Message removed, user not banned.")

    del banned_messages[message_id]
    await query.answer()

reported_messages = {}

async def handle_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles /report command.
    Admins: immediately ban and add to training.
    Non-admins: send to admins for review via inline keyboard.
    """
    if not update.message:
        return

    chat_id = update.message.chat_id
    user_id = update.message.from_user.id

    # Check if it's a reply
    if not update.message.reply_to_message:
        await update.message.reply_text("Please reply to the scam message with /report.")
        return
    # Handle non-text messages
    if not update.message.reply_to_message.text:
        reported_user = update.message.reply_to_message.from_user
        reported_message_id = update.message.reply_to_message.message_id
        
        # Store minimal info for callback
        reported_messages[reported_message_id] = {
            "text": None,
            "user_id": reported_user.id,
            "chat_id": chat_id,
            "message_id": reported_message_id,
            "reported_by": update.message.from_user.username or str(user_id)
        }
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Ban", callback_data=f"report_confirm|{reported_message_id}|{chat_id}|{reported_user.id}"),
                InlineKeyboardButton("❌ Dismiss", callback_data=f"report_dismiss|{reported_message_id}|{chat_id}")
            ]
        ])
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🚨 Non-text report by @{update.message.from_user.username or str(user_id)}\n\n"
                f"👤 Reported user: {reported_user.full_name}"
                f"{f' (@{reported_user.username})' if reported_user.username else ''}\n"
                f"🆔 ID: {reported_user.id}\n\n"
                f"⚠️ Admin review required.",
            reply_markup=keyboard
        )
        return

    reported_text = update.message.reply_to_message.text
    reported_user_id = update.message.reply_to_message.from_user.id
    reported_message_id = update.message.reply_to_message.message_id

    chat_member = await context.bot.get_chat_member(chat_id, user_id)
    is_admin = chat_member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER]

    if is_admin:
        # Act immediately
        from vector_store import add_example
        add_example(reported_text, "BAN")
        sync_example_to_github(reported_text, "BAN")
        increment_stat('false_negatives')
        FALSE_NEGATIVES.set(get_stat('false_negatives'))
        decrement_stat('accurate_classifications')
        ACCURATE_CLASSIFICATIONS.set(get_stat('accurate_classifications'))
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=reported_message_id)
            await _tg_execute_ban(context.bot, chat_id, reported_user_id)
            await update.message.reply_text("✅ User banned.")
        except Exception as e:
            logging.error(f"Error banning reported user: {e}")
            await update.message.reply_text("✅ Message added to training examples. Could not ban user.")
    else:
        # Store and send to admins for review
        reported_messages[reported_message_id] = {
            "text": reported_text,
            "user_id": reported_user_id,
            "chat_id": chat_id,
            "message_id": reported_message_id,
            "reported_by": update.message.from_user.username or str(user_id)
        }
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Confirm Ban", callback_data=f"report_confirm|{reported_message_id}|{chat_id}"),
                InlineKeyboardButton("❌ Dismiss", callback_data=f"report_dismiss|{reported_message_id}|{chat_id}")
            ]
        ])
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🚨 Scam report by @{reported_messages[reported_message_id]['reported_by']}. Admin review required:",
            reply_markup=keyboard
        )

async def handle_report_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles admin confirmation/dismissal of user-reported scams."""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id

    # Check if admin
    chat_member = await context.bot.get_chat_member(chat_id, user_id)
    if chat_member.status not in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER]:
        await query.answer("Only admins can do this.")
        return

    data = query.data.split("|")
    action = data[0]
    message_id = int(data[1])

    report_info = reported_messages.get(message_id)
    if not report_info:
        await query.answer("Report expired — bot may have restarted. Please /report again.")
        return

    if action == "report_confirm":
        if report_info["text"] is not None:
            from vector_store import add_example
            add_example(report_info["text"], "BAN")
            sync_example_to_github(report_info["text"], "BAN")
            increment_stat('false_negatives')
            FALSE_NEGATIVES.set(get_stat('false_negatives'))
            decrement_stat('accurate_classifications')
            ACCURATE_CLASSIFICATIONS.set(get_stat('accurate_classifications'))
        try:
            await context.bot.delete_message(
                chat_id=report_info["chat_id"],
                message_id=report_info["message_id"],
            )
            await _tg_execute_ban(context.bot, report_info["chat_id"], report_info["user_id"])
            await query.edit_message_text("✅ Report confirmed. User banned.")
        except Exception as e:
            logging.error(f"Error banning reported user: {e}")
            await query.edit_message_text("✅ Could not ban user — they may have already left.")

    elif action == "report_dismiss":
        await query.edit_message_text("❌ Report dismissed.")

    del reported_messages[message_id]
    await query.answer()

async def update_member_counts(context: ContextTypes.DEFAULT_TYPE):
    """Background task to update member counts daily."""
    group_ids = get_all_group_ids()
    for chat_id in group_ids:
        try:
            count = await context.bot.get_chat_member_count(chat_id)
            update_group_member_count(chat_id, count)
        except Exception as e:
            logging.error(f"Error updating member count for {chat_id}: {e}")
    MEMBERS_PROTECTED.set(get_total_members())
    logging.info("Updated member counts for all groups")

async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /stats command — admins only."""
    if not update.message:
        return

    chat_id = update.message.chat_id
    user_id = update.message.from_user.id

    chat_member = await context.bot.get_chat_member(chat_id, user_id)
    if chat_member.status not in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER]:
        await update.message.reply_text("Only admins can use this command.")
        return

    accurate = get_stat('accurate_classifications')
    total = accurate + get_stat('false_positives') + get_stat('false_negatives')
    accuracy = (accurate / total * 100) if total > 0 else 0

    await update.message.reply_text(
        f"📊 Sus Message Bot Stats\n\n"
        f"👥 Groups protected: {get_groups_count()}\n"
        f"🛡️ Members protected: {get_total_members():,}\n"
        f"📨 Messages scanned: {get_stat('messages_safe') + get_stat('messages_ban'):,}\n"
        f"🚫 Total bans: {get_stat('bans_confirmed') + get_stat('false_positives')}\n"
        f"✅ Accuracy rate: {accuracy:.1f}%\n"
    )

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN is not set. Exiting.")
        return
    init_db() 
    init_metrics()
    start_health_server()
    start_http_server(8000)  # Prometheus metrics endpoint on port 8000
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback, pattern="^(correct|false)"))
    app.add_handler(CallbackQueryHandler(handle_report_callback, pattern="^(report_confirm|report_dismiss)"))
    app.add_handler(CommandHandler("report", handle_report))
    app.add_handler(CommandHandler("stats", handle_stats))
    app.job_queue.run_repeating(update_member_counts, interval=86400, first=86400)
    if USE_POLLING:
        logging.info("Starting bot in polling mode (local development)")
        app.run_polling(drop_pending_updates=True)
    else:
        logging.info("Starting bot in webhook mode")
        app.run_webhook(
            listen="0.0.0.0",
            port=80,
            webhook_url=WEBHOOK_URL,
            bootstrap_retries=-1
        )

if __name__ == "__main__":
    main() 