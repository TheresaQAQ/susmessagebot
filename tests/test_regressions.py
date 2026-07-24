import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from susmessagebot import bot as bot_discord
from susmessagebot import moderator, seeds, stats, url_moderator
from susmessagebot.strike_tracker import StrikeTracker


class UrlModeratorRegressionTests(unittest.TestCase):
    @patch(
        "susmessagebot.url_moderator.client.chat.completions.create",
        side_effect=RuntimeError("API unavailable"),
    )
    def test_url_api_error_requests_manual_review(self, create):
        result = url_moderator.analyze_urls("Check this link: https://unknown.example/path")

        self.assertEqual(result, "REVIEW")
        create.assert_called_once()

    @patch("susmessagebot.url_moderator._classify_url_with_llm", return_value="BAN")
    def test_invite_domains_are_reviewed(self, classify_url):
        result = url_moderator.analyze_urls(
            "Join this promotional server: https://discord.gg/scam"
        )

        self.assertEqual(result, "BAN")
        classify_url.assert_called_once_with("https://discord.gg/scam")

    @patch("susmessagebot.url_moderator._classify_url_with_llm", return_value="SAFE")
    @patch("susmessagebot.url_moderator.requests.head")
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


class StatsRegressionTests(unittest.TestCase):
    def test_existing_group_member_count_is_refreshed(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()

                self.assertTrue(stats.add_group(1, 10))
                self.assertFalse(stats.add_group(1, 25))

                self.assertEqual(stats.get_groups_count(), 1)
                self.assertEqual(stats.get_total_members(), 25)
        finally:
            stats.DB_PATH = old_db_path


class HandlerFailureRegressionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _discord_message(*, content="hello", attachments=None):
        author = SimpleNamespace(
            id=7,
            bot=False,
            guild_permissions=SimpleNamespace(administrator=False),
        )
        return SimpleNamespace(
            id=42,
            author=author,
            guild=SimpleNamespace(id=1, name="Test Guild"),
            channel=SimpleNamespace(id=9, name="general"),
            content=content,
            attachments=attachments or [],
            mentions=[],
            reference=None,
        )

    async def test_discord_text_error_is_sent_for_review_without_delete(self):
        message = self._discord_message()
        ban_user = AsyncMock()
        request_review = AsyncMock()

        with (
            patch.object(bot_discord, "classify_message", return_value="REVIEW"),
            patch.object(bot_discord, "analyze_urls", return_value="SAFE"),
            patch.object(bot_discord, "_ban_user", ban_user),
            patch.object(bot_discord, "_request_manual_review", request_review),
        ):
            await bot_discord.on_message(message)

        ban_user.assert_not_awaited()
        request_review.assert_awaited_once_with(
            message,
            reason="Text moderation unavailable",
        )

    async def test_discord_invalid_image_is_sent_for_review_without_delete(self):
        attachment = SimpleNamespace(
            filename="broken.png",
            read=AsyncMock(return_value=b"not an image"),
        )
        message = self._discord_message(content="", attachments=[attachment])
        ban_user = AsyncMock()
        request_review = AsyncMock()

        with (
            patch.object(bot_discord, "classify_image", return_value="REVIEW"),
            patch.object(bot_discord, "_ban_user", ban_user),
            patch.object(bot_discord, "_request_manual_review", request_review),
        ):
            await bot_discord.on_message(message)

        ban_user.assert_not_awaited()
        request_review.assert_awaited_once_with(
            message,
            reason="Image moderation unavailable",
            evidence_images=[("broken.png", b"not an image")],
        )

    async def test_manual_review_does_not_delete_or_count_ban(self):
        message = self._discord_message()
        message.delete = AsyncMock()
        dm_admins = AsyncMock(return_value=1)
        increment = MagicMock()

        with (
            patch.object(bot_discord, "_dm_admins_review", dm_admins),
            patch.object(bot_discord, "increment_stat", increment),
            patch.object(bot_discord, "_dm_user", AsyncMock()),
        ):
            await bot_discord._request_manual_review(
                message,
                reason="Text moderation unavailable",
            )

        message.delete.assert_not_awaited()
        increment.assert_not_called()
        dm_admins.assert_awaited_once()
        kwargs = dm_admins.await_args.kwargs
        self.assertFalse(kwargs["removed"])
        self.assertEqual(kwargs["message_id"], 42)
        self.assertEqual(kwargs["channel_id"], 9)
        self.assertIn("not removed", kwargs["status"])

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

    async def test_interaction_admin_check_fetches_uncached_member(self):
        admin = SimpleNamespace(
            id=10,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        guild = SimpleNamespace(
            owner_id=99,
            get_member=MagicMock(return_value=None),
            fetch_member=AsyncMock(return_value=admin),
        )
        interaction = SimpleNamespace(
            client=SimpleNamespace(get_guild=MagicMock(return_value=guild)),
            user=SimpleNamespace(id=10),
            response=SimpleNamespace(send_message=AsyncMock()),
        )

        result = await bot_discord._require_interaction_admin(interaction, 1)

        self.assertIs(result, guild)
        guild.get_member.assert_called_once_with(10)
        guild.fetch_member.assert_awaited_once_with(10)
        interaction.response.send_message.assert_not_awaited()

    def test_review_views_use_persistent_components(self):
        hitl = bot_discord.HITLView(
            guild_id=1,
            user_id=2,
            username="user",
            text="content",
            message_id=3,
            channel_id=4,
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


class DiscordStrikeReviewRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_failure_still_notifies_admins_and_user(self):
        author = SimpleNamespace(id=7)
        guild = SimpleNamespace(id=1, name="Test Guild")
        channel = SimpleNamespace(id=9, name="general")
        message = SimpleNamespace(
            id=42,
            author=author,
            guild=guild,
            channel=channel,
            content="scam",
            attachments=[],
            delete=AsyncMock(side_effect=RuntimeError("missing permissions")),
        )
        dm_admins = AsyncMock(return_value=1)
        dm_user = AsyncMock()

        with tempfile.NamedTemporaryFile(suffix=".db") as db:
            tracker = StrikeTracker(threshold=3, db_path=db.name)
            with (
                patch.object(bot_discord, "strikes", tracker),
                patch.object(bot_discord, "increment_stat"),
                patch.object(bot_discord, "get_stat", return_value=0),
                patch.object(bot_discord, "_dm_admins_review", dm_admins),
                patch.object(bot_discord, "_dm_user", dm_user),
            ):
                await bot_discord._ban_user(message, reason="test")

        message.delete.assert_awaited_once()
        dm_admins.assert_awaited_once_with(
            guild,
            channel_name="general",
            author=author,
            content="scam",
            reason="test",
            message_id=42,
            channel_id=9,
            images=[],
            removed=False,
        )
        dm_user.assert_awaited_once()
        self.assertIn("could not remove it automatically", dm_user.await_args.args[1])

    async def test_discord_does_not_autoban_when_no_admin_can_review(self):
        execute_ban = AsyncMock()
        author = SimpleNamespace(id=7)
        guild = SimpleNamespace(id=1, name="Test Guild")
        channel = SimpleNamespace(id=9, name="general")

        with tempfile.NamedTemporaryFile(suffix=".db") as db:
            tracker = StrikeTracker(threshold=3, db_path=db.name)
            with (
                patch.object(bot_discord, "strikes", tracker),
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
                for i in range(3):
                    message = SimpleNamespace(
                        id=100 + i,
                        author=author,
                        guild=guild,
                        channel=channel,
                        content="scam",
                        attachments=[],
                        delete=AsyncMock(),
                    )
                    await bot_discord._ban_user(message, reason="test")

        execute_ban.assert_not_awaited()


class ReviewDecisionAtomicityTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_admin_decision_is_rejected(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()

                first = stats.claim_review_decision(1, 42, 7, "false_alarm", 10)
                second = stats.claim_review_decision(1, 42, 7, "ban", 11)

                self.assertIsNone(first)
                self.assertEqual(second, "false_alarm")
                self.assertEqual(stats.get_review_decision(1, 42, 7), "false_alarm")
        finally:
            stats.DB_PATH = old_db_path

    async def test_hitl_false_alarm_blocks_later_ban(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()

                guild = SimpleNamespace(
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                )
                interaction = SimpleNamespace(
                    client=SimpleNamespace(get_guild=MagicMock(return_value=guild)),
                    user=SimpleNamespace(id=11),
                    response=SimpleNamespace(
                        send_message=AsyncMock(),
                        edit_message=AsyncMock(),
                    ),
                    message=SimpleNamespace(
                        content="📝 Content:\nscam text",
                    ),
                )
                button = bot_discord.HITLBanButton(
                    guild_id=1,
                    user_id=7,
                    message_id=42,
                    channel_id=9,
                )
                stats.claim_review_decision(1, 42, 7, "false_alarm", 10)

                with (
                    patch.object(bot_discord, "add_example") as add_example,
                    patch.object(bot_discord, "sync_example_to_github") as sync,
                    patch.object(bot_discord, "_execute_ban", AsyncMock()) as execute_ban,
                ):
                    await button.callback(interaction)

                interaction.response.send_message.assert_awaited_once()
                self.assertIn(
                    "Already handled",
                    interaction.response.send_message.await_args.args[0],
                )
                add_example.assert_not_called()
                sync.assert_not_called()
                execute_ban.assert_not_awaited()
        finally:
            stats.DB_PATH = old_db_path


class StrikePersistenceTests(unittest.TestCase):
    def test_strikes_survive_tracker_recreation(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as db:
            first = StrikeTracker(threshold=3, window_seconds=600, db_path=db.name)
            self.assertFalse(first.record(1, 7))
            self.assertFalse(first.record(1, 7))
            self.assertEqual(first.count(1, 7), 2)

            second = StrikeTracker(threshold=3, window_seconds=600, db_path=db.name)
            self.assertEqual(second.count(1, 7), 2)
            self.assertTrue(second.record(1, 7))

    def test_expired_strikes_are_pruned_from_sqlite(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as db:
            tracker = StrikeTracker(threshold=3, window_seconds=10, db_path=db.name)
            now = time.time()
            with patch("susmessagebot.strike_tracker.time.time", return_value=now):
                tracker.record(1, 7)
            with patch(
                "susmessagebot.strike_tracker.time.time",
                return_value=now + 11,
            ):
                self.assertEqual(tracker.count(1, 7), 0)


if __name__ == "__main__":
    unittest.main()
