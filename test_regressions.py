import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import bot
import bot_discord
import moderator
import seeds
import url_moderator
from strike_tracker import StrikeTracker


class UrlModeratorRegressionTests(unittest.TestCase):
    @patch("url_moderator._classify_url_with_llm", return_value="BAN")
    def test_invite_domains_are_reviewed(self, classify_url):
        result = url_moderator.analyze_urls(
            "Join this promotional server: https://discord.gg/scam"
        )

        self.assertEqual(result, "BAN")
        classify_url.assert_called_once_with("https://discord.gg/scam")

    @patch("url_moderator._classify_url_with_llm", return_value="SAFE")
    @patch("url_moderator.requests.head")
    def test_user_urls_are_not_fetched_from_bot_host(self, head, classify_url):
        result = url_moderator.analyze_urls("See http://127.0.0.1:8001/health")

        self.assertEqual(result, "SAFE")
        head.assert_not_called()
        classify_url.assert_called_once_with("http://127.0.0.1:8001/health")


class ClassifierFailureRegressionTests(unittest.TestCase):
    @patch.object(
        moderator.client.chat.completions,
        "create",
        side_effect=RuntimeError("API unavailable"),
    )
    @patch.object(moderator, "render_prompt", return_value="rules")
    @patch.object(moderator, "get_similar_examples", return_value="")
    def test_text_api_error_requests_manual_review(
        self,
        get_examples,
        render_prompt,
        create,
    ):
        self.assertEqual(moderator.classify_message("hello"), "REVIEW")

    def test_invalid_image_requests_manual_review(self):
        self.assertEqual(moderator.classify_image(b"not an image"), "REVIEW")


class SeedDataRegressionTests(unittest.TestCase):
    def test_placeholder_messages_are_not_ban_training_examples(self):
        ban_messages = {
            message
            for message, verdict in seeds.examples
            if verdict == "BAN"
        }

        self.assertNotIn("", ban_messages)
        self.assertNotIn("hihi this is legit message", ban_messages)
        self.assertNotIn("final test", ban_messages)


class HandlerFailureRegressionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _discord_message(*, content="hello", attachments=None):
        author = SimpleNamespace(
            id=7,
            bot=False,
            guild_permissions=SimpleNamespace(administrator=False),
        )
        return SimpleNamespace(
            author=author,
            guild=SimpleNamespace(id=1, name="Test Guild"),
            channel=SimpleNamespace(name="general"),
            content=content,
            attachments=attachments or [],
            mentions=[],
            reference=None,
        )

    async def test_discord_text_error_is_sent_for_review_without_strike(self):
        message = self._discord_message()
        ban_user = AsyncMock()

        with (
            patch.object(bot_discord, "classify_message", return_value="REVIEW"),
            patch.object(bot_discord, "analyze_urls", return_value="SAFE"),
            patch.object(bot_discord, "_ban_user", ban_user),
        ):
            await bot_discord.on_message(message)

        ban_user.assert_awaited_once_with(
            message,
            reason="Text moderation unavailable",
            record_strike=False,
        )

    async def test_discord_invalid_image_is_sent_for_review_without_strike(self):
        attachment = SimpleNamespace(
            filename="broken.png",
            read=AsyncMock(return_value=b"not an image"),
        )
        message = self._discord_message(content="", attachments=[attachment])
        ban_user = AsyncMock()

        with (
            patch.object(bot_discord, "classify_image", return_value="REVIEW"),
            patch.object(bot_discord, "_ban_user", ban_user),
        ):
            await bot_discord.on_message(message)

        ban_user.assert_awaited_once_with(
            message,
            reason="Image moderation unavailable",
            evidence_images=[("broken.png", b"not an image")],
            record_strike=False,
        )

    async def test_discord_uses_image_content_type_without_file_extension(self):
        attachment = SimpleNamespace(
            filename="payload.bin",
            content_type="image/png",
            read=AsyncMock(return_value=b"image bytes"),
        )
        message = self._discord_message(content="", attachments=[attachment])
        ban_user = AsyncMock()

        with (
            patch.object(bot_discord, "classify_image", return_value="BAN"),
            patch.object(bot_discord, "_ban_user", ban_user),
        ):
            await bot_discord.on_message(message)

        ban_user.assert_awaited_once()

    async def test_telegram_text_error_is_sent_for_review_without_strike(self):
        tracker = StrikeTracker(threshold=3)
        dm_admins = AsyncMock(return_value=1)
        context = SimpleNamespace(
            bot=SimpleNamespace(
                get_chat_member=AsyncMock(
                    return_value=SimpleNamespace(status="member")
                ),
                get_chat_member_count=AsyncMock(return_value=10),
                delete_message=AsyncMock(),
                send_message=AsyncMock(),
            )
        )
        user = SimpleNamespace(id=7, full_name="Test User", username=None)
        message = SimpleNamespace(
            text="hello",
            caption=None,
            chat_id=1,
            message_id=1,
            from_user=user,
            chat=SimpleNamespace(title="Test Chat"),
        )

        with (
            patch.object(bot, "strikes", tracker),
            patch.object(bot, "classify_message", return_value="REVIEW"),
            patch.object(bot, "add_group", return_value=False),
            patch.object(bot, "increment_stat"),
            patch.object(bot, "get_stat", return_value=0),
            patch.object(bot, "_tg_dm_admins", dm_admins),
        ):
            bot.banned_messages.clear()
            await bot.handle_message(SimpleNamespace(message=message), context)

        dm_admins.assert_awaited_once()
        self.assertEqual(tracker.count(1, 7), 0)

    async def test_telegram_blocklisted_url_is_moderated(self):
        dm_admins = AsyncMock(return_value=1)
        context = SimpleNamespace(
            bot=SimpleNamespace(
                get_chat_member=AsyncMock(
                    return_value=SimpleNamespace(status="member")
                ),
                get_chat_member_count=AsyncMock(return_value=10),
                delete_message=AsyncMock(),
                send_message=AsyncMock(),
            )
        )
        user = SimpleNamespace(id=7, full_name="Test User", username=None)
        message = SimpleNamespace(
            text="See http://malware.example",
            caption=None,
            chat_id=1,
            message_id=1,
            from_user=user,
            chat=SimpleNamespace(title="Test Chat"),
        )

        with (
            patch.object(bot, "strikes", StrikeTracker(threshold=3)),
            patch.object(bot, "classify_message", return_value="SAFE"),
            patch.object(bot, "analyze_urls", return_value="BAN", create=True),
            patch.object(bot, "add_group", return_value=False),
            patch.object(bot, "increment_stat"),
            patch.object(bot, "get_stat", return_value=0),
            patch.object(bot, "_tg_dm_admins", dm_admins),
        ):
            bot.banned_messages.clear()
            await bot.handle_message(SimpleNamespace(message=message), context)

        context.bot.delete_message.assert_awaited_once_with(
            chat_id=1,
            message_id=1,
        )
        dm_admins.assert_awaited_once()


class DiscordHealthRegressionTests(unittest.TestCase):
    class _FakeHealthHandler(bot_discord.HealthHandler):
        def __init__(self, path="/health"):
            self.path = path
            self.status = None
            self.body = b""

        def send_response(self, code):
            self.status = code

        def end_headers(self):
            pass

        @property
        def wfile(self):
            handler = self

            class Writer:
                def write(self, data):
                    handler.body = data

            return Writer()

    def test_health_returns_503_until_ready(self):
        bot_discord._bot_ready = False
        self.assertFalse(bot_discord.is_ready())
        handler = self._FakeHealthHandler()
        handler.do_GET()
        self.assertEqual(handler.status, 503)
        self.assertEqual(handler.body, b"NOT_READY")

    def test_health_returns_200_when_ready(self):
        bot_discord._bot_ready = True
        try:
            self.assertTrue(bot_discord.is_ready())
            handler = self._FakeHealthHandler()
            handler.do_GET()
            self.assertEqual(handler.status, 200)
            self.assertEqual(handler.body, b"OK")
        finally:
            bot_discord._bot_ready = False


class DiscordOperationalRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_reported_message_is_handled(self):
        channel = SimpleNamespace(
            fetch_message=AsyncMock(side_effect=RuntimeError("message gone")),
            send=AsyncMock(),
        )
        message = SimpleNamespace(
            author=SimpleNamespace(bot=False),
            guild=SimpleNamespace(id=1),
            channel=channel,
            mentions=[bot_discord.client.user],
            reference=SimpleNamespace(message_id=99),
        )

        await bot_discord.on_message(message)

        channel.send.assert_awaited_once()

    async def test_admin_lookup_chunks_an_incomplete_member_cache(self):
        admin = SimpleNamespace(
            id=10,
            bot=False,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        members = []

        async def populate_cache(*, cache):
            self.assertTrue(cache)
            members.append(admin)

        guild = SimpleNamespace(
            owner_id=10,
            members=members,
            chunked=False,
            chunk=AsyncMock(side_effect=populate_cache),
        )

        admins = await bot_discord._admin_members(guild)

        self.assertEqual(admins, [admin])
        guild.chunk.assert_awaited_once_with(cache=True)

    def test_review_views_use_persistent_components(self):
        hitl = bot_discord.HITLView(
            guild_id=1,
            user_id=2,
            username="user",
            text="content",
        )
        report = bot_discord.ReportReviewView(
            guild_id=1,
            user_id=2,
            username="user",
            text="content",
            message_id=3,
            channel_id=4,
        )

        self.assertTrue(hitl.is_persistent())
        self.assertTrue(report.is_persistent())

    async def test_setup_registers_dynamic_review_components(self):
        with (
            patch.object(
                bot_discord.client.tree,
                "sync",
                AsyncMock(return_value=[]),
            ),
            patch.object(bot_discord.client, "add_dynamic_items") as register,
        ):
            await bot_discord.client.setup_hook()

        register.assert_called_once()


class TelegramCallbackRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_correct_callback_uses_stored_user_id(self):
        query = SimpleNamespace(
            from_user=SimpleNamespace(id=10),
            data="correct|1|100",
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )
        context = SimpleNamespace(
            bot=SimpleNamespace(
                get_chat_member=AsyncMock(
                    return_value=SimpleNamespace(
                        status=bot.ChatMemberStatus.ADMINISTRATOR
                    )
                )
            )
        )
        execute_ban = AsyncMock()

        with (
            patch.object(bot, "_tg_execute_ban", execute_ban),
            patch.object(bot, "increment_stat"),
            patch.object(bot, "get_stat", return_value=0),
            patch.object(bot, "sync_example_to_github"),
            patch("vector_store.add_example"),
        ):
            bot.banned_messages.clear()
            bot.banned_messages[1] = {"user_id": 7, "text": "scam"}
            await bot.handle_callback(
                SimpleNamespace(callback_query=query),
                context,
            )

        execute_ban.assert_awaited_once_with(context.bot, 100, 7)


class StrikeReviewRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_does_not_autoban_when_no_admin_can_review(self):
        context = SimpleNamespace(
            bot=SimpleNamespace(
                get_chat_member=AsyncMock(
                    return_value=SimpleNamespace(status="member")
                ),
                get_chat_member_count=AsyncMock(return_value=10),
                delete_message=AsyncMock(),
                send_message=AsyncMock(),
            )
        )
        execute_ban = AsyncMock()

        with (
            patch.object(bot, "strikes", StrikeTracker(threshold=3)),
            patch.object(bot, "classify_message", return_value="BAN"),
            patch.object(bot, "add_group", return_value=False),
            patch.object(bot, "increment_stat"),
            patch.object(bot, "get_stat", return_value=0),
            patch.object(bot, "_tg_dm_admins", AsyncMock(return_value=0)),
            patch.object(bot, "_tg_execute_ban", execute_ban),
            patch.object(bot, "sync_example_to_github"),
            patch("vector_store.add_example"),
        ):
            bot.banned_messages.clear()
            for message_id in range(1, 4):
                user = SimpleNamespace(
                    id=7,
                    full_name="Test User",
                    username=None,
                )
                message = SimpleNamespace(
                    text="scam",
                    caption=None,
                    chat_id=1,
                    message_id=message_id,
                    from_user=user,
                    chat=SimpleNamespace(title="Test Chat"),
                )
                await bot.handle_message(
                    SimpleNamespace(message=message),
                    context,
                )

        execute_ban.assert_not_awaited()

    async def test_discord_does_not_autoban_when_no_admin_can_review(self):
        execute_ban = AsyncMock()
        author = SimpleNamespace(id=7)
        guild = SimpleNamespace(id=1, name="Test Guild")
        channel = SimpleNamespace(name="general")

        with (
            patch.object(bot_discord, "strikes", StrikeTracker(threshold=3)),
            patch.object(bot_discord, "increment_stat"),
            patch.object(bot_discord, "_dm_user", AsyncMock()),
            patch.object(
                bot_discord,
                "_dm_admins_review",
                AsyncMock(return_value=0),
            ),
            patch.object(bot_discord, "_dm_admins_text", AsyncMock()),
            patch.object(bot_discord, "_execute_ban", execute_ban),
            patch.object(bot_discord, "add_example"),
            patch.object(bot_discord, "sync_example_to_github"),
        ):
            for _ in range(3):
                message = SimpleNamespace(
                    author=author,
                    guild=guild,
                    channel=channel,
                    content="scam",
                    attachments=[],
                    delete=AsyncMock(),
                )
                await bot_discord._ban_user(message, reason="test")

        execute_ban.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
