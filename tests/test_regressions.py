import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import httpx
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from scripts import eval_accuracy
from susmessagebot import bot as bot_discord
from susmessagebot import github_sync, moderator, seeds, stats, url_moderator, utils
from susmessagebot.llm_utils import should_disable_thinking
from susmessagebot.strike_tracker import StrikeTracker, ban_notice_text
from susmessagebot import vector_store
from susmessagebot.vector_store import _example_id


def _openai_status_error(error_type, status_code, message, *, headers=None):
    request = httpx.Request(
        "POST",
        "https://api.siliconflow.cn/v1/chat/completions",
    )
    response = httpx.Response(
        status_code,
        request=request,
        headers=headers,
    )
    body = {
        "error": {
            "message": message,
            "type": "api_error",
            "param": None,
            "code": None,
        }
    }
    return error_type(message, response=response, body=body)


def _openai_connection_error(message="Connection failed"):
    request = httpx.Request(
        "POST",
        "https://api.siliconflow.cn/v1/chat/completions",
    )
    return APIConnectionError(message=message, request=request)


class UrlModeratorRegressionTests(unittest.TestCase):
    @patch(
        "susmessagebot.url_moderator.client.chat.completions.create",
        side_effect=RuntimeError("API unavailable"),
    )
    def test_url_api_error_requests_manual_review(self, create):
        result = url_moderator.analyze_urls("Check this link: https://unknown.example/path")

        self.assertEqual(result, "REVIEW")
        create.assert_called_once()

    def test_url_extraction_is_case_insensitive(self):
        self.assertEqual(
            url_moderator._extract_urls("See HTTPS://Evil.Example/path"),
            ["HTTPS://Evil.Example/path"],
        )

    def test_domain_strips_port_for_blocklist(self):
        self.assertEqual(
            url_moderator._get_domain("https://evil.example:443/x"),
            "evil.example",
        )

    @patch.object(url_moderator, "_blocklist", {"evil.example"})
    def test_uppercase_scheme_blocklist_url_is_banned(self):
        self.assertEqual(
            url_moderator.analyze_urls("See HTTPS://evil.example/path"),
            "BAN",
        )

    @patch("susmessagebot.config.SILICONFLOW_MODEL", "THUDM/GLM-4.6")
    @patch("susmessagebot.url_moderator.client.chat.completions.create")
    def test_url_moderator_disables_thinking_for_glm_models(self, create):
        create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="SAFE", reasoning=None)
                )
            ]
        )

        self.assertEqual(
            url_moderator._classify_url_with_llm("https://example.com"),
            "SAFE",
        )
        self.assertEqual(
            create.call_args.kwargs.get("extra_body"),
            {"enable_thinking": False},
        )
        self.assertEqual(create.call_args.kwargs.get("model"), "THUDM/GLM-4.6")

    @patch("susmessagebot.url_moderator.requests.get")
    def test_blocklist_refresh_keeps_cache_on_http_error(self, get):
        url_moderator._blocklist = {"evil.example"}
        get.return_value = SimpleNamespace(
            status_code=500,
            text="error",
            raise_for_status=MagicMock(side_effect=RuntimeError("500")),
        )

        url_moderator.load_blocklist()

        self.assertEqual(url_moderator._blocklist, {"evil.example"})

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


class TextNormalizationTests(unittest.TestCase):
    def test_normalize_folds_fullwidth_and_keeps_cjk(self):
        self.assertEqual(utils.normalize_text("ＳＩＰ　Ｔｒｕｎｋ"), "SIP Trunk")
        self.assertIn("免费代充", utils.normalize_text("免费代充活动"))

    def test_normalize_folds_mathematical_alnum_and_zero_width(self):
        fancy = "𝗦𝗜𝗣 𝗧𝗿𝘂𝗻𝗸𝘀"
        self.assertIn("SIP", utils.normalize_text(fancy))
        self.assertEqual(utils.normalize_text("a\u200bb"), "ab")

    @patch.object(
        moderator._text_client.chat.completions,
        "create",
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="SAFE", reasoning=None)
                )
            ]
        ),
    )
    @patch.object(moderator, "render_prompt", return_value="rules")
    @patch.object(moderator, "get_similar_examples", return_value="")
    def test_classify_message_uses_normalized_text(
        self,
        get_examples,
        render_prompt,
        create,
    ):
        self.assertEqual(
            moderator.classify_message("Please do not buy SIP trunks here; it is spam"),
            "SAFE",
        )
        create.assert_called_once()
        get_examples.assert_called_once()
        user_content = create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("SIP trunks", user_content)


class GithubSyncDedupTests(unittest.TestCase):
    @patch.object(github_sync, "GITHUB_TOKEN", "token")
    @patch.object(github_sync, "GITHUB_REPO", "TheresaQAQ/susmessagebot")
    @patch("susmessagebot.github_sync.requests.put")
    @patch("susmessagebot.github_sync.requests.get")
    def test_duplicate_example_is_not_rewritten(self, get, put):
        existing = 'SEED_EXAMPLES = [\n    ("hello", "BAN"),\n]\n'
        get.return_value = SimpleNamespace(
            status_code=200,
            text="ok",
            json=lambda: {
                "sha": "abc",
                "content": __import__("base64").b64encode(existing.encode()).decode(),
            },
        )

        self.assertTrue(github_sync.sync_example_to_github("hello", "BAN"))
        put.assert_not_called()

    @patch.object(github_sync, "GITHUB_TOKEN", "token")
    @patch.object(github_sync, "GITHUB_REPO", "TheresaQAQ/susmessagebot")
    @patch("susmessagebot.github_sync.requests.put")
    @patch("susmessagebot.github_sync.requests.get")
    def test_label_correction_rewrites_existing_seed(self, get, put):
        existing = 'SEED_EXAMPLES = [\n    ("hello", "BAN"),\n]\n'
        get.return_value = SimpleNamespace(
            status_code=200,
            text="ok",
            json=lambda: {
                "sha": "abc",
                "content": __import__("base64").b64encode(existing.encode()).decode(),
            },
        )
        put.return_value = SimpleNamespace(status_code=200, text="ok")

        self.assertTrue(github_sync.sync_example_to_github("hello", "SAFE"))
        put.assert_called_once()
        payload = put.call_args.kwargs["json"]
        decoded = __import__("base64").b64decode(payload["content"]).decode()
        self.assertIn('("hello", "SAFE")', decoded)
        self.assertNotIn('("hello", "BAN")', decoded)
        self.assertIn("Update example to SAFE", payload["message"])

    @patch.object(github_sync, "GITHUB_TOKEN", "token")
    @patch.object(github_sync, "GITHUB_REPO", "TheresaQAQ/susmessagebot")
    @patch("susmessagebot.github_sync.requests.put")
    @patch("susmessagebot.github_sync.requests.get")
    def test_legacy_conflicting_duplicates_are_consolidated(self, get, put):
        # First occurrence matches the new HITL label, but a later stale dupe
        # would otherwise win when seeds.py is applied in order.
        existing = (
            'examples = [\n'
            '    ("hello", "BAN"),\n'
            '    ("other", "SAFE"),\n'
            '    ("hello", "SAFE"),\n'
            ']\n'
        )
        get.return_value = SimpleNamespace(
            status_code=200,
            text="ok",
            json=lambda: {
                "sha": "abc",
                "content": __import__("base64").b64encode(existing.encode()).decode(),
            },
        )
        put.return_value = SimpleNamespace(status_code=200, text="ok")

        self.assertTrue(github_sync.sync_example_to_github("hello", "BAN"))
        put.assert_called_once()
        payload = put.call_args.kwargs["json"]
        decoded = __import__("base64").b64decode(payload["content"]).decode()
        self.assertEqual(decoded.count('("hello", "BAN")'), 1)
        self.assertNotIn('("hello", "SAFE")', decoded)
        self.assertIn('("other", "SAFE")', decoded)


class ClassifierFailureRegressionTests(unittest.TestCase):
    @patch.object(
        moderator._text_client.chat.completions,
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
        create.assert_called_once()

    @patch.object(moderator, "render_prompt", return_value="rules")
    @patch.object(moderator, "get_similar_examples", return_value="")
    def test_text_api_retries_three_times_with_thirty_second_timeout(
        self,
        get_examples,
        render_prompt,
    ):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="SAFE", reasoning=None)
                )
            ]
        )
        with patch.object(
            moderator._text_client.chat.completions,
            "create",
            side_effect=[
                _openai_connection_error("attempt 1"),
                _openai_connection_error("attempt 2"),
                _openai_connection_error("attempt 3"),
                response,
            ],
        ) as create, patch.object(moderator.time, "sleep") as sleep, patch.object(
            moderator.random,
            "uniform",
            return_value=0,
        ):
            self.assertEqual(
                moderator.classify_message("hello"),
                "SAFE",
            )

        self.assertEqual(moderator._text_client.max_retries, 0)
        self.assertEqual(create.call_count, 4)
        self.assertEqual(sleep.call_args_list, [call(2.0), call(4.0), call(8.0)])
        self.assertTrue(
            all(
                call.kwargs["timeout"] == 30.0
                for call in create.call_args_list
            )
        )

    @patch.object(moderator, "render_prompt", return_value="rules")
    @patch.object(moderator, "get_similar_examples", return_value="")
    def test_explicit_siliconflow_configuration_errors_do_not_retry(
        self,
        get_examples,
        render_prompt,
    ):
        errors = [
            _openai_status_error(BadRequestError, 400, "Invalid temperature"),
            _openai_status_error(AuthenticationError, 401, "Invalid API key"),
            _openai_status_error(PermissionDeniedError, 403, "Insufficient balance"),
        ]
        for error in errors:
            with self.subTest(status=error.status_code), patch.object(
                moderator._text_client.chat.completions,
                "create",
                side_effect=error,
            ) as create, patch.object(moderator.time, "sleep") as sleep:
                self.assertEqual(moderator.classify_message("hello"), "REVIEW")

            create.assert_called_once()
            sleep.assert_not_called()

    @patch.object(moderator, "render_prompt", return_value="rules")
    @patch.object(moderator, "get_similar_examples", return_value="")
    def test_minute_rate_limit_retries_with_provider_delay(
        self,
        get_examples,
        render_prompt,
    ):
        rate_limit = _openai_status_error(
            RateLimitError,
            429,
            "TPM rate limit exceeded",
            headers={"retry-after-ms": "1500"},
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="SAFE", reasoning=None)
                )
            ]
        )
        with patch.object(
            moderator._text_client.chat.completions,
            "create",
            side_effect=[rate_limit, response],
        ) as create, patch.object(moderator.time, "sleep") as sleep:
            self.assertEqual(moderator.classify_message("hello"), "SAFE")

        self.assertEqual(create.call_count, 2)
        sleep.assert_called_once_with(1.5)

    @patch.object(moderator, "render_prompt", return_value="rules")
    @patch.object(moderator, "get_similar_examples", return_value="")
    def test_all_rate_limits_retry(
        self,
        get_examples,
        render_prompt,
    ):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="SAFE", reasoning=None)
                )
            ]
        )
        messages = [
            "Rate limit exceeded for free-models-per-day",
            "RPH rate limit exceeded",
            "Request was rejected due to rate limiting",
        ]
        for message in messages:
            error = _openai_status_error(RateLimitError, 429, message)
            with self.subTest(message=message), patch.object(
                moderator._text_client.chat.completions,
                "create",
                side_effect=[error, response],
            ) as create, patch.object(moderator.time, "sleep") as sleep, patch.object(
                moderator.random,
                "uniform",
                return_value=0,
            ):
                self.assertEqual(moderator.classify_message("hello"), "SAFE")

            self.assertEqual(create.call_count, 2)
            sleep.assert_called_once_with(2.0)

    def test_top_level_siliconflow_error_payload_is_parsed(self):
        request = httpx.Request(
            "POST",
            "https://api.siliconflow.cn/v1/chat/completions",
        )
        response = httpx.Response(429, request=request)
        error = RateLimitError(
            "Daily limit reached",
            response=response,
            body={
                "code": 20001,
                "message": "RPD rate limit exceeded",
                "data": None,
            },
        )

        self.assertEqual(
            moderator._extract_error_details(error),
            (429, "20001", "RPD rate limit exceeded"),
        )
        self.assertTrue(moderator._is_retryable_text_error(error))

    @patch.object(moderator, "render_prompt", return_value="rules")
    @patch.object(moderator, "get_similar_examples", return_value="")
    def test_other_http_errors_retry(
        self,
        get_examples,
        render_prompt,
    ):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="SAFE", reasoning=None)
                )
            ]
        )
        for status_code in (408, 404, 500, 502, 503, 504):
            error = _openai_status_error(
                InternalServerError,
                status_code,
                "Server overloaded",
            )
            with self.subTest(status=status_code), patch.object(
                moderator._text_client.chat.completions,
                "create",
                side_effect=[error, response],
            ) as create, patch.object(moderator.time, "sleep") as sleep, patch.object(
                moderator.random,
                "uniform",
                return_value=0,
            ):
                self.assertEqual(moderator.classify_message("hello"), "SAFE")

            self.assertEqual(create.call_count, 2)
            sleep.assert_called_once_with(2.0)

    def test_evaluation_does_not_add_caller_level_retries(self):
        with patch(
            "susmessagebot.moderator.classify_message",
            return_value="REVIEW",
        ) as classify:
            with self.assertRaisesRegex(RuntimeError, "--resume"):
                eval_accuracy.classify_with_retry("hello")

        classify.assert_called_once_with("hello")


class EvaluationResumeRegressionTests(unittest.TestCase):
    def test_resume_reuses_only_the_same_evaluation_case(self):
        row = {
            "index": 62,
            "text": "old safe case",
            "expected": "SAFE",
            "tag": "en-chat",
        }

        self.assertTrue(
            eval_accuracy._matches_current_case(
                row,
                "old safe case",
                "SAFE",
                "en-chat",
            )
        )
        self.assertFalse(
            eval_accuracy._matches_current_case(
                row,
                "inserted ban case",
                "BAN",
                "ru-scam",
            )
        )

    def test_invalid_image_requests_manual_review(self):
        self.assertEqual(moderator.classify_image(b"not an image"), "REVIEW")

    def test_unparseable_verdict_requests_manual_review(self):
        self.assertEqual(moderator._parse_verdict(""), "REVIEW")
        self.assertEqual(moderator._parse_verdict("I cannot decide"), "REVIEW")

    @patch(
        "susmessagebot.url_moderator.client.chat.completions.create",
    )
    def test_unparseable_url_verdict_requests_manual_review(self, create):
        create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="maybe later", reasoning=None)
                )
            ]
        )
        self.assertEqual(
            url_moderator._classify_url_with_llm("https://unknown.example"),
            "REVIEW",
        )

    @patch.object(url_moderator, "_blocklist", {"evil.example"})
    @patch.object(url_moderator, "_classify_url_with_llm", return_value="REVIEW")
    def test_url_review_continues_to_later_blocklist_ban(self, classify_url):
        result = url_moderator.analyze_urls(
            "see https://unknown.example/a and https://evil.example/b"
        )

        self.assertEqual(result, "BAN")
        classify_url.assert_called_once_with("https://unknown.example/a")


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

    def test_remove_group_drops_stale_guild_row(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                stats.add_group(1, 10)
                self.assertTrue(stats.remove_group(1))
                self.assertEqual(stats.get_groups_count(), 0)
                self.assertEqual(stats.get_total_members(), 0)
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

    async def test_discord_picker_gif_only_message_skips_moderation(self):
        gif_url = "https://klipy.com/gifs/a-ty-ne-veril-serezno"
        message = self._discord_message(content=gif_url)
        message.embeds = [SimpleNamespace(type="gifv", url=gif_url)]
        classify_text = MagicMock(return_value="BAN")
        classify_urls = MagicMock(return_value="BAN")
        increment = MagicMock()

        with (
            patch.object(bot_discord, "classify_message", classify_text),
            patch.object(bot_discord, "analyze_urls", classify_urls),
            patch.object(bot_discord, "increment_stat", increment),
            patch.object(bot_discord, "_ban_user", AsyncMock()) as ban_user,
            patch.object(
                bot_discord,
                "_request_manual_review",
                AsyncMock(),
            ) as request_review,
        ):
            await bot_discord.on_message(message)

        classify_text.assert_not_called()
        classify_urls.assert_not_called()
        increment.assert_not_called()
        ban_user.assert_not_awaited()
        request_review.assert_not_awaited()

    async def test_picker_gif_without_embed_still_skips_moderation(self):
        gif_url = "https://klipy.com/gifs/embed-not-ready"
        message = self._discord_message(content=gif_url)
        message.embeds = []

        with (
            patch.object(bot_discord, "classify_message") as classify_text,
            patch.object(bot_discord, "analyze_urls") as classify_urls,
            patch.object(bot_discord, "_ban_user", AsyncMock()) as ban_user,
            patch.object(
                bot_discord,
                "_request_manual_review",
                AsyncMock(),
            ) as request_review,
        ):
            await bot_discord.on_message(message)

        classify_text.assert_not_called()
        classify_urls.assert_not_called()
        ban_user.assert_not_awaited()
        request_review.assert_not_awaited()

    async def test_discord_picker_gif_with_text_stays_in_moderation(self):
        gif_url = "https://klipy.com/gifs/reaction"
        content = f"{gif_url} cheap accounts for sale"
        message = self._discord_message(
            content=content
        )
        message.embeds = [SimpleNamespace(type="gifv", url=gif_url)]
        ban_user = AsyncMock()

        with (
            patch.object(
                bot_discord,
                "classify_message",
                return_value="BAN",
            ) as classify_text,
            patch.object(
                bot_discord,
                "analyze_urls",
                return_value="SAFE",
            ) as classify_urls,
            patch.object(bot_discord, "_ban_user", ban_user),
            patch.object(bot_discord, "_request_manual_review", AsyncMock()),
        ):
            await bot_discord.on_message(message)

        classify_text.assert_called_once_with("cheap accounts for sale")
        classify_urls.assert_called_once_with("cheap accounts for sale")
        ban_user.assert_awaited_once_with(
            message,
            reason="Suspicious message",
        )

    async def test_picker_gif_does_not_hide_other_url(self):
        gif_url = "https://klipy.com/gifs/reaction"
        other_url = "https://unknown.example/sale"
        message = self._discord_message(content=f"{gif_url} {other_url}")

        with (
            patch.object(
                bot_discord,
                "classify_message",
                return_value="SAFE",
            ) as classify_text,
            patch.object(
                bot_discord,
                "analyze_urls",
                return_value="REVIEW",
            ) as classify_urls,
            patch.object(bot_discord, "_ban_user", AsyncMock()),
            patch.object(
                bot_discord,
                "_request_manual_review",
                AsyncMock(),
            ) as request_review,
        ):
            await bot_discord.on_message(message)

        classify_text.assert_called_once_with(other_url)
        classify_urls.assert_called_once_with(other_url)
        request_review.assert_awaited_once_with(
            message,
            reason="URL moderation unavailable",
        )

    async def test_custom_emojis_only_skip_moderation(self):
        message = self._discord_message(
            content="<:static:123456789012345678> <a:dance:234567890123456789>"
        )

        with (
            patch.object(bot_discord, "classify_message") as classify_text,
            patch.object(bot_discord, "analyze_urls") as classify_urls,
            patch.object(bot_discord, "_ban_user", AsyncMock()) as ban_user,
            patch.object(
                bot_discord,
                "_request_manual_review",
                AsyncMock(),
            ) as request_review,
        ):
            await bot_discord.on_message(message)

        classify_text.assert_not_called()
        classify_urls.assert_not_called()
        ban_user.assert_not_awaited()
        request_review.assert_not_awaited()

    async def test_custom_emoji_does_not_hide_other_text(self):
        message = self._discord_message(
            content="<a:dance:234567890123456789> cheap accounts for sale"
        )
        ban_user = AsyncMock()

        with (
            patch.object(
                bot_discord,
                "classify_message",
                return_value="BAN",
            ) as classify_text,
            patch.object(
                bot_discord,
                "analyze_urls",
                return_value="SAFE",
            ) as classify_urls,
            patch.object(bot_discord, "_ban_user", ban_user),
            patch.object(bot_discord, "_request_manual_review", AsyncMock()),
        ):
            await bot_discord.on_message(message)

        classify_text.assert_called_once_with("cheap accounts for sale")
        classify_urls.assert_called_once_with("cheap accounts for sale")
        ban_user.assert_awaited_once_with(
            message,
            reason="Suspicious message",
        )

    async def test_native_sticker_only_skips_moderation(self):
        message = self._discord_message(content="")
        message.stickers = [SimpleNamespace(id=123, name="wave")]

        with (
            patch.object(bot_discord, "classify_message") as classify_text,
            patch.object(bot_discord, "analyze_urls") as classify_urls,
            patch.object(bot_discord, "_ban_user", AsyncMock()) as ban_user,
            patch.object(
                bot_discord,
                "_request_manual_review",
                AsyncMock(),
            ) as request_review,
        ):
            await bot_discord.on_message(message)

        classify_text.assert_not_called()
        classify_urls.assert_not_called()
        ban_user.assert_not_awaited()
        request_review.assert_not_awaited()

    async def test_native_sticker_does_not_hide_other_text(self):
        message = self._discord_message(content="cheap accounts for sale")
        message.stickers = [SimpleNamespace(id=123, name="wave")]
        ban_user = AsyncMock()

        with (
            patch.object(
                bot_discord,
                "classify_message",
                return_value="BAN",
            ) as classify_text,
            patch.object(
                bot_discord,
                "analyze_urls",
                return_value="SAFE",
            ) as classify_urls,
            patch.object(bot_discord, "_ban_user", ban_user),
            patch.object(bot_discord, "_request_manual_review", AsyncMock()),
        ):
            await bot_discord.on_message(message)

        classify_text.assert_called_once_with("cheap accounts for sale")
        classify_urls.assert_called_once_with("cheap accounts for sale")
        ban_user.assert_awaited_once_with(
            message,
            reason="Suspicious message",
        )

    def test_combined_discord_media_tokens_clean_to_empty(self):
        message = self._discord_message(
            content=(
                "<a:dance:234567890123456789> "
                "https://klipy.com/gifs/reaction"
            )
        )

        self.assertEqual(bot_discord._moderation_text(message), "")

    def test_training_example_uses_filtered_moderation_text(self):
        text = (
            "<a:dance:234567890123456789> cheap accounts for sale "
            "https://klipy.com/gifs/reaction"
        )

        with (
            patch.object(bot_discord, "add_example") as add_example,
            patch.object(
                bot_discord,
                "sync_example_to_github",
                return_value=True,
            ) as sync_example,
        ):
            bot_discord._record_training_example(text, "BAN")

        add_example.assert_called_once_with("cheap accounts for sale", "BAN")
        sync_example.assert_called_once_with("cheap accounts for sale", "BAN")

    def test_training_example_skips_discord_media_only_text(self):
        text = (
            "<:static:123456789012345678> "
            "https://klipy.com/gifs/reaction"
        )

        with (
            patch.object(bot_discord, "add_example") as add_example,
            patch.object(bot_discord, "sync_example_to_github") as sync_example,
        ):
            bot_discord._record_training_example(text, "SAFE")

        add_example.assert_not_called()
        sync_example.assert_not_called()

    async def test_uploaded_gif_still_uses_image_moderation(self):
        attachment = SimpleNamespace(
            filename="advertisement.gif",
            content_type="image/gif",
            read=AsyncMock(return_value=b"gif bytes"),
        )
        message = self._discord_message(content="", attachments=[attachment])
        ban_user = AsyncMock()

        with (
            patch.object(
                bot_discord,
                "classify_image",
                return_value="BAN",
            ) as classify_image,
            patch.object(bot_discord, "classify_message") as classify_text,
            patch.object(bot_discord, "analyze_urls") as classify_urls,
            patch.object(bot_discord, "_ban_user", ban_user),
            patch.object(bot_discord, "_request_manual_review", AsyncMock()),
        ):
            await bot_discord.on_message(message)

        classify_image.assert_called_once_with(b"gif bytes")
        classify_text.assert_not_called()
        classify_urls.assert_not_called()
        ban_user.assert_awaited_once_with(
            message,
            reason="Suspicious image",
            evidence_images=[("advertisement.gif", b"gif bytes")],
        )

    async def test_picker_domain_non_gif_path_is_still_moderated(self):
        url = "https://klipy.com/support"
        message = self._discord_message(content=url)
        message.embeds = []
        ban_user = AsyncMock()

        with (
            patch.object(
                bot_discord,
                "classify_message",
                return_value="SAFE",
            ) as classify_text,
            patch.object(
                bot_discord,
                "analyze_urls",
                return_value="BAN",
            ) as classify_urls,
            patch.object(bot_discord, "_ban_user", ban_user),
            patch.object(bot_discord, "_request_manual_review", AsyncMock()),
        ):
            await bot_discord.on_message(message)

        classify_text.assert_called_once_with(url)
        classify_urls.assert_called_once_with(url)
        ban_user.assert_awaited_once_with(
            message,
            reason="Suspicious message",
        )

    def test_picker_gif_url_accepts_only_known_share_paths(self):
        accepted = (
            "https://klipy.com/gifs/reaction",
            "https://tenor.com/view/reaction-gif-123",
            "https://giphy.com/gifs/reaction-abc123",
        )
        rejected = (
            "https://klipy.com/support",
            "https://klipy.com.evil.example/gifs/ad",
            "https://example.com/gifs/reaction",
            "https://klipy.com/gifs/reaction extra text",
        )

        for url in accepted:
            with self.subTest(url=url):
                self.assertTrue(bot_discord._is_discord_picker_gif_url(url))
        for url in rejected:
            with self.subTest(url=url):
                self.assertFalse(bot_discord._is_discord_picker_gif_url(url))

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

    async def test_url_review_reason_is_not_labeled_as_text_failure(self):
        message = self._discord_message(content="see https://unknown.example/x")
        request_review = AsyncMock()

        with (
            patch.object(bot_discord, "classify_message", return_value="SAFE"),
            patch.object(bot_discord, "analyze_urls", return_value="REVIEW"),
            patch.object(bot_discord, "_ban_user", AsyncMock()),
            patch.object(bot_discord, "_request_manual_review", request_review),
        ):
            await bot_discord.on_message(message)

        request_review.assert_awaited_once_with(
            message,
            reason="URL moderation unavailable",
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

    async def test_image_review_does_not_skip_text_ban(self):
        attachment = SimpleNamespace(
            filename="broken.png",
            content_type="image/png",
            read=AsyncMock(return_value=b"not an image"),
        )
        message = self._discord_message(
            content="cheap accounts for sale",
            attachments=[attachment],
        )
        ban_user = AsyncMock()
        request_review = AsyncMock()

        with (
            patch.object(bot_discord, "classify_image", return_value="REVIEW"),
            patch.object(bot_discord, "classify_message", return_value="BAN"),
            patch.object(bot_discord, "analyze_urls", return_value="SAFE"),
            patch.object(bot_discord, "_ban_user", ban_user),
            patch.object(bot_discord, "_request_manual_review", request_review),
        ):
            await bot_discord.on_message(message)

        request_review.assert_not_awaited()
        ban_user.assert_awaited_once_with(
            message,
            reason="Suspicious message",
            evidence_images=[("broken.png", b"not an image")],
        )

    async def test_safe_image_only_message_is_counted(self):
        attachment = SimpleNamespace(
            filename="ok.png",
            content_type="image/png",
            read=AsyncMock(return_value=b"image bytes"),
        )
        message = self._discord_message(content="", attachments=[attachment])
        increment = MagicMock()

        with (
            patch.object(bot_discord, "classify_image", return_value="SAFE"),
            patch.object(bot_discord, "increment_stat", increment),
            patch.object(bot_discord, "get_stat", return_value=1),
            patch.object(bot_discord, "_ban_user", AsyncMock()),
            patch.object(bot_discord, "_request_manual_review", AsyncMock()),
        ):
            await bot_discord.on_message(message)

        increment.assert_called_once_with("messages_safe")

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
        self.assertIn("消息未删除", kwargs["status"])

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


class DiscordHealthRegressionTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_disconnect_marks_health_not_ready(self):
        bot_discord._bot_ready = True
        try:
            await bot_discord.on_disconnect()
            self.assertFalse(bot_discord.is_ready())
            handler = self._FakeHealthHandler()
            handler.do_GET()
            self.assertEqual(handler.status, 503)
        finally:
            bot_discord._bot_ready = False

    async def test_blocklist_is_refreshed_periodically(self):
        sleep_calls = 0

        async def stop_after_one_refresh(_delay):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                raise asyncio.CancelledError

        with (
            patch.object(bot_discord, "load_blocklist") as refresh,
            patch.object(
                bot_discord.asyncio,
                "sleep",
                side_effect=stop_after_one_refresh,
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await bot_discord._refresh_blocklist_periodically()

        refresh.assert_called_once_with()


class DiscordPermissionAuditRegressionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _permissions(**overrides):
        values = {
            "ban_members": True,
            "view_channel": True,
            "send_messages": True,
            "send_messages_in_threads": True,
            "read_message_history": True,
            "manage_messages": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _guild(self, *, channel_permissions=None):
        member = SimpleNamespace(guild_permissions=self._permissions())
        channel = SimpleNamespace(
            id=9,
            name="question",
            permissions_for=MagicMock(
                return_value=channel_permissions or self._permissions()
            ),
            send=AsyncMock(),
        )
        owner = SimpleNamespace(id=10, send=AsyncMock())
        guild = SimpleNamespace(
            id=1,
            name="Annulus",
            owner_id=10,
            owner=owner,
            me=member,
            member_count=100,
            system_channel=channel,
            text_channels=[channel],
            forums=[],
        )
        return guild, channel, owner

    def test_permission_check_reports_effective_channel_overrides(self):
        guild, _, _ = self._guild(
            channel_permissions=self._permissions(manage_messages=False)
        )

        issues = bot_discord._guild_permission_issues(guild)

        self.assertEqual(
            issues,
            ["#question (9): Manage Messages"],
        )

    def test_permission_check_reports_forum_overrides(self):
        guild, _, _ = self._guild()
        forum = SimpleNamespace(
            id=12,
            name="support",
            permissions_for=MagicMock(
                return_value=self._permissions(view_channel=False)
            ),
        )
        guild.forums = [forum]

        issues = bot_discord._guild_permission_issues(guild)

        self.assertEqual(issues, ["#support (12): View Channel"])

    def test_permission_check_does_not_require_channel_send_permissions(self):
        guild, _, _ = self._guild(
            channel_permissions=self._permissions(
                send_messages=False,
                send_messages_in_threads=False,
            )
        )

        issues = bot_discord._guild_permission_issues(guild)

        self.assertEqual(issues, [])

    async def test_permission_audit_logs_without_notifying_owner(self):
        guild, _, owner = self._guild(
            channel_permissions=self._permissions(manage_messages=False)
        )
        with patch.object(bot_discord.logging, "warning") as warning:
            self.assertFalse(
                await bot_discord._audit_guild_permissions(guild)
            )

        owner.send.assert_not_awaited()
        warning.assert_called_once()
        self.assertIn("Manage Messages", str(warning.call_args))

    async def test_ready_audits_every_guild_without_failing_health(self):
        guild, _, _ = self._guild()
        audit = AsyncMock(return_value=False)
        bot_discord._bot_ready = False
        try:
            with (
                patch.object(
                    type(bot_discord.client),
                    "guilds",
                    new_callable=PropertyMock,
                    return_value=[guild],
                ),
                patch.object(bot_discord.asyncio, "to_thread", AsyncMock()),
                patch.object(bot_discord, "_ensure_blocklist_refresh_task"),
                patch.object(bot_discord, "add_group"),
                patch.object(bot_discord, "_refresh_guild_metrics"),
                patch.object(bot_discord, "_audit_guild_permissions", audit),
            ):
                await bot_discord.on_ready()
        finally:
            bot_discord._bot_ready = False

        audit.assert_awaited_once_with(guild)

    async def test_guild_join_never_sends_to_server_channels(self):
        guild, channel, _ = self._guild()
        audit = AsyncMock(return_value=True)

        with (
            patch.object(bot_discord, "add_group"),
            patch.object(bot_discord, "_refresh_guild_metrics"),
            patch.object(bot_discord, "_audit_guild_permissions", audit),
        ):
            await bot_discord.on_guild_join(guild)

        audit.assert_awaited_once_with(guild)
        channel.send.assert_not_awaited()


class DiscordOperationalRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_mention_is_not_treated_as_a_report(self):
        channel = SimpleNamespace(fetch_message=AsyncMock())
        message = SimpleNamespace(
            author=SimpleNamespace(
                bot=False,
                guild_permissions=SimpleNamespace(administrator=True),
            ),
            guild=SimpleNamespace(id=1),
            channel=channel,
            mentions=[bot_discord.client.user],
            reference=SimpleNamespace(message_id=99),
        )

        await bot_discord.on_message(message)

        channel.fetch_message.assert_not_awaited()

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
        self.assertTrue(hitl.is_persistent())
        self.assertEqual(
            [item.item.label for item in hitl.children],
            ["🚫 删除并封禁", "🗑️ 删除", "❌ 误报"],
        )

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

        register.assert_called_once_with(
            bot_discord.HITLBanButton,
            bot_discord.HITLDeleteButton,
            bot_discord.HITLFalseAlarmButton,
        )
        self.assertNotIn(
            "Report to SusMessageBot",
            [command.name for command in bot_discord.client.tree.get_commands()],
        )


class DiscordStrikeReviewRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_ai_ban_only_notifies_admins(self):
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
            delete=AsyncMock(),
        )
        dm_admins = AsyncMock(return_value=1)
        dm_user = AsyncMock()
        execute_ban = AsyncMock()
        tracker = MagicMock()

        with (
            patch.object(bot_discord, "strikes", tracker),
            patch.object(bot_discord, "increment_stat"),
            patch.object(bot_discord, "get_stat", return_value=0),
            patch.object(bot_discord, "_dm_admins_review", dm_admins),
            patch.object(bot_discord, "_dm_user", dm_user),
            patch.object(bot_discord, "_execute_ban", execute_ban),
        ):
            await bot_discord._ban_user(message, reason="test")

        message.delete.assert_not_awaited()
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
            status="检测到可疑内容，等待管理员处理（消息未删除）",
        )
        dm_user.assert_not_awaited()
        execute_ban.assert_not_awaited()
        tracker.record.assert_not_called()

    async def test_discord_does_not_autoban_when_no_admin_can_review(self):
        execute_ban = AsyncMock()
        dm_user = AsyncMock()
        author = SimpleNamespace(id=7)
        guild = SimpleNamespace(id=1, name="Test Guild")
        channel = SimpleNamespace(id=9, name="general")
        messages = []
        tracker = MagicMock()

        with (
            patch.object(bot_discord, "strikes", tracker),
            patch.object(bot_discord, "increment_stat"),
            patch.object(bot_discord, "get_stat", return_value=0),
            patch.object(bot_discord, "_dm_user", dm_user),
            patch.object(
                bot_discord,
                "_dm_admins_review",
                AsyncMock(return_value=0),
            ),
            patch.object(bot_discord, "_execute_ban", execute_ban),
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
                messages.append(message)
                await bot_discord._ban_user(message, reason="test")

        for message in messages:
            message.delete.assert_not_awaited()
        execute_ban.assert_not_awaited()
        dm_user.assert_not_awaited()
        tracker.record.assert_not_called()


class ReviewDecisionAtomicityTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _admin_interaction(
        guild,
        *,
        user_id=11,
        content="📝 Content:\nscam text",
        attachments=None,
    ):
        return SimpleNamespace(
            client=SimpleNamespace(
                get_guild=MagicMock(return_value=guild),
                fetch_user=AsyncMock(return_value=SimpleNamespace(id=7)),
            ),
            user=SimpleNamespace(id=user_id),
            response=SimpleNamespace(
                send_message=AsyncMock(),
                edit_message=AsyncMock(),
                defer=AsyncMock(),
            ),
            edit_original_response=AsyncMock(),
            message=SimpleNamespace(content=content, attachments=attachments or []),
        )

    async def test_image_only_review_hides_internal_placeholder(self):
        admin = SimpleNamespace(id=10, send=AsyncMock())
        author = SimpleNamespace(id=7)
        guild = SimpleNamespace(id=1, name="Test Guild")

        with (
            patch.object(
                bot_discord,
                "_admin_members",
                AsyncMock(return_value=[admin]),
            ),
            patch.object(bot_discord, "store_review_evidence") as store_evidence,
        ):
            notified = await bot_discord._dm_admins_review(
                guild,
                channel_name="ads",
                author=author,
                content="[image]",
                reason="Suspicious image",
                message_id=42,
                channel_id=9,
                images=[("proof.png", b"image")],
                removed=False,
                status="检测到可疑内容，等待管理员处理（消息未删除）",
            )

        self.assertEqual(notified, 1)
        body = admin.send.await_args.args[0]
        self.assertNotIn("[image]", body)
        self.assertNotIn("📝 内容：", body)
        self.assertEqual(
            admin.send.await_args.kwargs["files"][0].filename,
            "proof.png",
        )
        store_evidence.assert_called_once_with(
            1,
            42,
            7,
            "[image]",
            "Suspicious image",
        )

    async def test_review_dm_reference_is_persisted_after_send(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                sent = SimpleNamespace(
                    id=1001,
                    channel=SimpleNamespace(id=2001),
                )
                admin = SimpleNamespace(id=10, send=AsyncMock(return_value=sent))
                author = SimpleNamespace(id=7)
                guild = SimpleNamespace(id=1, name="Test Guild")
                source_message_id = stats._discord_snowflake_for_unix_time(
                    time.time() - 60
                )

                with patch.object(
                    bot_discord,
                    "_admin_members",
                    AsyncMock(return_value=[admin]),
                ):
                    notified = await bot_discord._dm_admins_review(
                        guild,
                        channel_name="ads",
                        author=author,
                        content="scam text",
                        reason="Suspicious message",
                        message_id=source_message_id,
                        channel_id=9,
                    )

                stats.claim_review_decision(
                    1,
                    source_message_id,
                    7,
                    "ban",
                    10,
                )
                rows = stats.claim_related_review_notifications(
                    1,
                    7,
                    source_message_id,
                    10,
                    max_age_seconds=86400,
                )
                self.assertEqual(notified, 1)
                self.assertEqual(
                    rows,
                    [(2001, 1001, source_message_id)],
                )
        finally:
            stats.DB_PATH = old_db_path

    async def test_successful_ban_closes_all_recent_related_review_dms(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                current_message_id = stats._discord_snowflake_for_unix_time(
                    time.time() - 60
                )
                related_message_id = stats._discord_snowflake_for_unix_time(
                    time.time() - 30
                )
                for source_message_id, dm_channel_id, dm_message_id, admin_id in (
                    (current_message_id, 2001, 1001, 10),
                    (current_message_id, 2002, 1002, 11),
                    (related_message_id, 2001, 1003, 10),
                    (related_message_id, 2002, 1004, 11),
                ):
                    stats.store_review_notification(
                        1,
                        source_message_id,
                        7,
                        admin_id,
                        dm_channel_id,
                        dm_message_id,
                    )
                stats.claim_review_decision(
                    1,
                    current_message_id,
                    7,
                    "ban",
                    10,
                )

                original = (
                    "⚠️ 检测到可疑内容，等待管理员处理（消息未删除）"
                    "｜**#ads**（Test Guild）\n\n"
                    "👤 用户：tester (`7`)\n\n"
                    "📝 内容：\nscam text"
                )
                messages = {
                    message_id: SimpleNamespace(content=original, edit=AsyncMock())
                    for message_id in (1002, 1003, 1004)
                }
                channels = {
                    2001: SimpleNamespace(
                        fetch_message=AsyncMock(
                            side_effect=lambda message_id: messages[message_id]
                        )
                    ),
                    2002: SimpleNamespace(
                        fetch_message=AsyncMock(
                            side_effect=lambda message_id: messages[message_id]
                        )
                    ),
                }
                client = SimpleNamespace(
                    get_channel=MagicMock(side_effect=lambda channel_id: channels[channel_id]),
                    fetch_channel=AsyncMock(),
                )
                moderator = SimpleNamespace(id=10)

                updated, failed = await bot_discord._close_related_ban_reviews(
                    client,
                    guild_id=1,
                    user_id=7,
                    current_message_id=current_message_id,
                    moderator=moderator,
                    already_edited_dm_message_id=1001,
                )

                self.assertEqual((updated, failed), (4, 0))
                self.assertEqual(
                    stats.get_review_decision(1, related_message_id, 7),
                    "covered_by_ban",
                )
                for message in messages.values():
                    message.edit.assert_awaited_once()
                    kwargs = message.edit.await_args.kwargs
                    self.assertIsNone(kwargs["view"])
                    self.assertIn(
                        "用户已封禁，该消息已随封禁操作删除",
                        kwargs["content"],
                    )

                second = await bot_discord._close_related_ban_reviews(
                    client,
                    guild_id=1,
                    user_id=7,
                    current_message_id=current_message_id,
                    moderator=moderator,
                )
                self.assertEqual(second, (0, 0))
        finally:
            stats.DB_PATH = old_db_path

    async def test_related_ban_leaves_reviews_older_than_cleanup_window_open(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                recent_message_id = stats._discord_snowflake_for_unix_time(
                    time.time() - 60
                )
                old_message_id = stats._discord_snowflake_for_unix_time(
                    time.time() - 86401
                )
                stats.store_review_notification(
                    1,
                    recent_message_id,
                    7,
                    10,
                    2001,
                    1001,
                )
                # The notification is new, but its original Discord message is old.
                stats.store_review_notification(
                    1,
                    old_message_id,
                    7,
                    10,
                    2001,
                    1002,
                )
                stats.claim_review_decision(
                    1,
                    recent_message_id,
                    7,
                    "ban",
                    10,
                )

                rows = stats.claim_related_review_notifications(
                    1,
                    7,
                    recent_message_id,
                    10,
                    max_age_seconds=86400,
                )

                self.assertEqual(
                    rows,
                    [(2001, 1001, recent_message_id)],
                )
                self.assertIsNone(
                    stats.get_review_decision(1, old_message_id, 7)
                )
        finally:
            stats.DB_PATH = old_db_path

    async def test_execute_ban_explicitly_cleans_last_24_hours(self):
        user = SimpleNamespace(id=7, send=AsyncMock())
        guild = SimpleNamespace(id=1, name="Test Guild", ban=AsyncMock())

        with (
            patch.object(bot_discord, "strikes", MagicMock()),
            patch.object(bot_discord, "clear_auto_ban"),
        ):
            await bot_discord._execute_ban(
                guild=guild,
                user=user,
                reason="Confirmed by admin",
            )

        self.assertEqual(
            guild.ban.await_args.kwargs["delete_message_seconds"],
            86400,
        )

    async def test_review_edit_retains_source_and_replaces_prior_status(self):
        interaction = self._admin_interaction(
            SimpleNamespace(),
            content=(
                "⚠️ 检测到可疑内容，等待管理员处理（消息未删除）"
                "｜**#ads**（Test Guild）\n\n"
                "👤 用户：tester (`7`)\n\n"
                "📝 内容：\nscam text"
            ),
        )

        await bot_discord._edit_review_message(
            interaction,
            "🗑️ 已删除该消息，未封禁用户。",
        )
        first = interaction.edit_original_response.await_args.kwargs["content"]

        self.assertIn("📌 处理状态：🗑️ 已删除该消息", first)
        self.assertIn("📍 原始频道：**#ads**（Test Guild）", first)
        self.assertIn("👤 用户：tester", first)
        self.assertIn("📝 内容：\nscam text", first)
        self.assertIsNone(
            interaction.edit_original_response.await_args.kwargs["view"]
        )

        interaction.message.content = first
        await bot_discord._edit_review_message(
            interaction,
            "⚠️ 删除失败，请重试。",
            keep_view=True,
        )
        second = interaction.edit_original_response.await_args.kwargs["content"]
        self.assertEqual(second.count("📌 处理状态："), 1)
        self.assertIn("📍 原始频道：**#ads**（Test Guild）", second)
        self.assertIn("📝 内容：\nscam text", second)
        self.assertNotIn(
            "view",
            interaction.edit_original_response.await_args.kwargs,
        )

    async def test_ban_notice_compactly_identifies_original_image(self):
        user = SimpleNamespace(id=7, send=AsyncMock())
        guild = SimpleNamespace(
            id=1,
            name="Test Guild",
            ban=AsyncMock(),
        )
        tracker = MagicMock()

        with (
            patch.object(bot_discord, "strikes", tracker),
            patch.object(bot_discord, "clear_auto_ban"),
        ):
            await bot_discord._execute_ban(
                guild=guild,
                user=user,
                reason="Confirmed by admin",
                channel_name="ads",
                evidence_message_id=42,
                evidence_text="[image]",
                evidence_images=[("proof.png", b"image")],
            )

        notice = user.send.await_args.args[0]
        self.assertIn("Server: Test Guild", notice)
        self.assertIn("Channel: #ads", notice)
        self.assertIn("Message: See attached image", notice)
        self.assertIn("Time: <t:1420070400:F>", notice)
        self.assertLess(notice.index("Time:"), notice.index("Message:"))
        self.assertNotIn("Original message ID", notice)
        self.assertNotIn("proof.png", notice)
        self.assertEqual(
            user.send.await_args.kwargs["files"][0].filename,
            "proof.png",
        )
        guild.ban.assert_awaited_once()
        tracker.clear.assert_called_once_with(1, 7)

    async def test_delete_reviewed_message_uses_cached_thread(self):
        message = SimpleNamespace(delete=AsyncMock())
        thread = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
        guild = SimpleNamespace(
            get_channel_or_thread=MagicMock(return_value=thread),
            fetch_channel=AsyncMock(),
        )

        result = await bot_discord._delete_reviewed_message(
            guild,
            channel_id=9,
            message_id=42,
        )

        self.assertEqual(result, "deleted")
        guild.get_channel_or_thread.assert_called_once_with(9)
        guild.fetch_channel.assert_not_awaited()
        thread.fetch_message.assert_awaited_once_with(42)
        message.delete.assert_awaited_once()

    async def test_delete_reviewed_message_fetches_uncached_thread(self):
        message = SimpleNamespace(delete=AsyncMock())
        thread = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
        guild = SimpleNamespace(
            get_channel_or_thread=MagicMock(return_value=None),
            fetch_channel=AsyncMock(return_value=thread),
        )

        result = await bot_discord._delete_reviewed_message(
            guild,
            channel_id=9,
            message_id=42,
        )

        self.assertEqual(result, "deleted")
        guild.fetch_channel.assert_awaited_once_with(9)
        thread.fetch_message.assert_awaited_once_with(42)
        message.delete.assert_awaited_once()

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
                interaction = self._admin_interaction(guild)
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
                    "该审核已被其他管理员处理",
                    interaction.response.send_message.await_args.args[0],
                )
                add_example.assert_not_called()
                sync.assert_not_called()
                execute_ban.assert_not_awaited()
        finally:
            stats.DB_PATH = old_db_path

    async def test_false_alarm_on_prior_strike_unbans_auto_banned_user(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                tracker = StrikeTracker(threshold=3, db_path=db.name)
                tracker.record(1, 7)
                tracker.record(1, 7)
                stats.store_review_evidence(1, 42, 7, "scam text", "Suspicious message")
                stats.record_auto_ban(1, 7, 44)

                guild = SimpleNamespace(
                    id=1,
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                    unban=AsyncMock(),
                )
                interaction = self._admin_interaction(guild)
                button = bot_discord.HITLFalseAlarmButton(1, 7, 42, 9)

                with (
                    patch.object(bot_discord, "strikes", tracker),
                    patch.object(bot_discord, "add_example"),
                    patch.object(bot_discord, "sync_example_to_github", return_value=True),
                    patch.object(bot_discord, "increment_stat"),
                    patch.object(bot_discord, "get_stat", return_value=1),
                ):
                    await button.callback(interaction)

                guild.unban.assert_awaited_once()
                self.assertEqual(tracker.count(1, 7), 0)
                interaction.response.defer.assert_awaited_once()
                body = interaction.edit_original_response.await_args.kwargs["content"]
                self.assertIn("已撤销自动封禁", body)
        finally:
            stats.DB_PATH = old_db_path

    async def test_false_alarm_does_not_unban_unrelated_admin_ban(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                stats.store_review_evidence(1, 99, 7, "other text", "Suspicious message")

                guild = SimpleNamespace(
                    id=1,
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                    unban=AsyncMock(),
                )
                interaction = self._admin_interaction(guild, content="📝 Content:\nother text")
                button = bot_discord.HITLFalseAlarmButton(1, 7, 99, 9)

                with (
                    patch.object(bot_discord, "strikes", StrikeTracker(db_path=db.name)),
                    patch.object(bot_discord, "add_example"),
                    patch.object(bot_discord, "sync_example_to_github", return_value=True),
                    patch.object(bot_discord, "increment_stat"),
                    patch.object(bot_discord, "get_stat", return_value=1),
                ):
                    await button.callback(interaction)

                guild.unban.assert_not_awaited()
                body = interaction.edit_original_response.await_args.kwargs["content"]
                self.assertIn("无自动封禁记录可撤销", body)
        finally:
            stats.DB_PATH = old_db_path

    async def test_failed_ban_enforcement_keeps_claim_for_same_admin_retry(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                stats.store_review_evidence(1, 42, 7, "scam text", "Suspicious message")

                guild = SimpleNamespace(
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                    get_channel_or_thread=MagicMock(return_value=None),
                    fetch_channel=AsyncMock(
                        side_effect=RuntimeError("channel unavailable")
                    ),
                )
                interaction = self._admin_interaction(guild)
                button = bot_discord.HITLBanButton(1, 7, 42, 9)

                with (
                    patch.object(bot_discord, "add_example") as add_example,
                    patch.object(bot_discord, "sync_example_to_github", return_value=True),
                    patch.object(bot_discord, "increment_stat") as increment,
                    patch.object(bot_discord, "get_stat", return_value=1),
                    patch.object(
                        bot_discord,
                        "_execute_ban",
                        AsyncMock(side_effect=RuntimeError("discord down")),
                    ),
                ):
                    await button.callback(interaction)

                self.assertEqual(stats.get_review_decision(1, 42, 7), "ban")
                add_example.assert_called_once_with("scam text", "BAN")
                increment.assert_called_once_with("bans_confirmed")
                body = interaction.edit_original_response.await_args.kwargs["content"]
                self.assertIn("再点一次「删除并封禁」", body)
                self.assertTrue(
                    interaction.edit_original_response.await_args.kwargs.get("keep_view")
                    or "view" not in interaction.edit_original_response.await_args.kwargs
                )

                interaction2 = self._admin_interaction(guild)
                with (
                    patch.object(bot_discord, "add_example") as add_example2,
                    patch.object(bot_discord, "sync_example_to_github", return_value=True),
                    patch.object(bot_discord, "increment_stat") as increment2,
                    patch.object(bot_discord, "get_stat", return_value=1),
                    patch.object(bot_discord, "_execute_ban", AsyncMock()) as execute_ban2,
                ):
                    await button.callback(interaction2)

                add_example2.assert_not_called()
                increment2.assert_not_called()
                execute_ban2.assert_awaited_once()
                self.assertEqual(stats.get_review_decision(1, 42, 7), "ban")
        finally:
            stats.DB_PATH = old_db_path

    async def test_delete_only_removes_message_and_notifies_user(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                stats.store_review_evidence(1, 42, 7, "scam text", "Suspicious message")

                message = SimpleNamespace(delete=AsyncMock())
                channel = SimpleNamespace(
                    name="ads",
                    fetch_message=AsyncMock(return_value=message),
                )
                guild = SimpleNamespace(
                    id=1,
                    name="Test Guild",
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                    get_channel_or_thread=MagicMock(return_value=channel),
                )
                attachment = SimpleNamespace(
                    filename="proof.png",
                    content_type="image/png",
                    read=AsyncMock(return_value=b"image"),
                )
                interaction = self._admin_interaction(
                    guild,
                    content=(
                        "⚠️ 检测到可疑内容，等待管理员处理（消息未删除）"
                        "｜**#ads**（Test Guild）\n\n"
                        "👤 用户：tester (`7`)\n\n"
                        "📝 内容：\nscam text"
                    ),
                    attachments=[attachment],
                )
                button = bot_discord.HITLDeleteButton(1, 7, 42, 9)
                dm_user = AsyncMock()

                with (
                    patch.object(bot_discord, "add_example") as add_example,
                    patch.object(bot_discord, "sync_example_to_github", return_value=True),
                    patch.object(bot_discord, "increment_stat") as increment,
                    patch.object(bot_discord, "_dm_user", dm_user),
                    patch.object(bot_discord, "_execute_ban", AsyncMock()) as execute_ban,
                ):
                    await button.callback(interaction)

                message.delete.assert_awaited_once()
                add_example.assert_called_once_with("scam text", "BAN")
                increment.assert_not_called()
                execute_ban.assert_not_awaited()
                dm_user.assert_awaited_once()
                notice = dm_user.await_args.args[1]
                self.assertIn("Server: Test Guild", notice)
                self.assertIn("Channel: #ads", notice)
                self.assertIn("Message: scam text", notice)
                self.assertIn("Time: <t:1420070400:F>", notice)
                self.assertLess(notice.index("Time:"), notice.index("Message:"))
                self.assertNotIn("Original message ID", notice)
                self.assertNotIn("proof.png", notice)
                self.assertEqual(
                    dm_user.await_args.kwargs["images"],
                    [("proof.png", b"image")],
                )
                self.assertEqual(stats.get_review_decision(1, 42, 7), "delete")
                body = interaction.edit_original_response.await_args.kwargs["content"]
                self.assertIn("未封禁用户", body)
                self.assertIn("📍 原始频道：**#ads**（Test Guild）", body)
                self.assertIn("📝 内容：\nscam text", body)
        finally:
            stats.DB_PATH = old_db_path

    async def test_delete_failure_releases_decision_for_another_admin(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                stats.store_review_evidence(1, 42, 7, "scam text", "Suspicious message")

                channel = SimpleNamespace(
                    fetch_message=AsyncMock(side_effect=RuntimeError("discord down"))
                )
                guild = SimpleNamespace(
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                    get_channel_or_thread=MagicMock(return_value=channel),
                )
                interaction = self._admin_interaction(guild)
                button = bot_discord.HITLDeleteButton(1, 7, 42, 9)

                with (
                    patch.object(bot_discord, "add_example") as add_example,
                    patch.object(bot_discord, "_dm_user", AsyncMock()) as dm_user,
                    patch.object(bot_discord, "_execute_ban", AsyncMock()) as execute_ban,
                ):
                    await button.callback(interaction)

                self.assertIsNone(stats.get_review_decision(1, 42, 7))
                add_example.assert_not_called()
                dm_user.assert_not_awaited()
                execute_ban.assert_not_awaited()
                body = interaction.edit_original_response.await_args.kwargs["content"]
                self.assertIn("其他管理员也可以接手", body)
        finally:
            stats.DB_PATH = old_db_path

    async def test_delete_and_ban_reports_partial_success(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                stats.store_review_evidence(1, 42, 7, "scam text", "Suspicious message")

                channel = SimpleNamespace(
                    fetch_message=AsyncMock(side_effect=RuntimeError("missing permission"))
                )
                guild = SimpleNamespace(
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                    get_channel_or_thread=MagicMock(return_value=channel),
                )
                interaction = self._admin_interaction(guild)
                button = bot_discord.HITLBanButton(1, 7, 42, 9)

                with (
                    patch.object(bot_discord, "add_example"),
                    patch.object(bot_discord, "sync_example_to_github", return_value=True),
                    patch.object(bot_discord, "increment_stat"),
                    patch.object(bot_discord, "get_stat", return_value=1),
                    patch.object(bot_discord, "_execute_ban", AsyncMock()) as execute_ban,
                ):
                    await button.callback(interaction)

                execute_ban.assert_awaited_once()
                body = interaction.edit_original_response.await_args.kwargs["content"]
                self.assertIn("单条消息删除失败", body)
                self.assertIn("最近 24 小时内的服务器消息", body)
        finally:
            stats.DB_PATH = old_db_path

    async def test_false_alarm_clears_strikes_even_if_training_fails(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                stats.store_review_evidence(1, 42, 7, "scam text", "Suspicious message")
                tracker = StrikeTracker(threshold=3, db_path=db.name)
                tracker.record(1, 7)

                guild = SimpleNamespace(
                    id=1,
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                    unban=AsyncMock(),
                )
                interaction = self._admin_interaction(guild)
                button = bot_discord.HITLFalseAlarmButton(1, 7, 42, 9)

                with (
                    patch.object(bot_discord, "strikes", tracker),
                    patch.object(
                        bot_discord,
                        "add_example",
                        side_effect=RuntimeError("chroma down"),
                    ),
                    patch.object(bot_discord, "increment_stat"),
                    patch.object(bot_discord, "get_stat", return_value=1),
                ):
                    await button.callback(interaction)

                self.assertEqual(tracker.count(1, 7), 0)
        finally:
            stats.DB_PATH = old_db_path

    async def test_review_uses_persisted_full_evidence_not_preview(self):
        old_db_path = stats.DB_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as db:
                stats.DB_PATH = db.name
                stats.init_db()
                full = "A" * 2000
                stats.store_review_evidence(1, 42, 7, full, "Suspicious message")

                guild = SimpleNamespace(
                    owner_id=10,
                    get_member=MagicMock(
                        return_value=SimpleNamespace(
                            id=11,
                            guild_permissions=SimpleNamespace(administrator=True),
                        )
                    ),
                    get_channel_or_thread=MagicMock(return_value=None),
                    fetch_channel=AsyncMock(
                        side_effect=RuntimeError("channel unavailable")
                    ),
                )
                interaction = self._admin_interaction(
                    guild,
                    content="📝 Content:\n" + full[:1500] + "...",
                )
                button = bot_discord.HITLBanButton(1, 7, 42, 9)

                with (
                    patch.object(bot_discord, "add_example") as add_example,
                    patch.object(bot_discord, "sync_example_to_github", return_value=True),
                    patch.object(bot_discord, "increment_stat"),
                    patch.object(bot_discord, "get_stat", return_value=1),
                    patch.object(bot_discord, "_execute_ban", AsyncMock()),
                ):
                    await button.callback(interaction)

                add_example.assert_called_once_with(full, "BAN")
        finally:
            stats.DB_PATH = old_db_path

    def test_image_placeholder_is_not_trained(self):
        with (
            patch.object(bot_discord, "add_example") as add_example,
            patch.object(bot_discord, "sync_example_to_github") as sync,
        ):
            bot_discord._record_training_example("[image]", "BAN")

        add_example.assert_not_called()
        sync.assert_not_called()

class BanNoticeTests(unittest.TestCase):
    def test_manual_ban_notice_is_not_auto_ban_copy(self):
        text = ban_notice_text(automatic=False)
        self.assertIn("moderator confirmed", text.lower())
        self.assertNotIn("automatically banned", text.lower())
        self.assertNotIn("你因", text)
        self.assertNotIn("Вы были", text)

    def test_automatic_ban_notice_mentions_auto_ban(self):
        text = ban_notice_text(automatic=True)
        self.assertIn("automatically banned", text.lower())
        self.assertNotIn("你因", text)
        self.assertNotIn("Вы были", text)

    def test_remove_notice_is_english_only(self):
        from susmessagebot.strike_tracker import remove_notice_text

        with tempfile.NamedTemporaryFile(suffix=".db") as db:
            tracker = StrikeTracker(threshold=3, db_path=db.name)
            with patch("susmessagebot.strike_tracker.strikes", tracker):
                tracker.record(1, 7)
                text = remove_notice_text(1, 7)

        self.assertIn("violating community rules", text.lower())
        self.assertNotIn("社区规则", text)
        self.assertNotIn("Ваше сообщение", text)


class ConfigDefaultTests(unittest.TestCase):
    def test_default_text_model_matches_readme(self):
        root = Path(__file__).resolve().parents[1]
        config_src = (root / "susmessagebot" / "config.py").read_text(encoding="utf-8")
        env_example = (root / ".env.example").read_text(encoding="utf-8")
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn('"Qwen/Qwen2.5-7B-Instruct"', config_src)
        self.assertIn("SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct", env_example)
        self.assertIn(
            "SILICONFLOW_VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct",
            env_example,
        )
        self.assertIn("defaults to `Qwen/Qwen2.5-7B-Instruct`", readme)


class DeployWorkflowTests(unittest.TestCase):
    def test_rollback_waits_for_previous_image_to_be_healthy(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'wait_for_healthy "${PREVIOUS_IMAGE}" "Rollback"',
            workflow,
        )
        self.assertNotIn("compose up -d --remove-orphans || true", workflow)


class ThinkingFlagTests(unittest.TestCase):
    def test_url_and_text_moderators_share_thinking_disable_rules(self):
        for model in (
            "Qwen/Qwen3-8B",
            "THUDM/GLM-4.5",
            "THUDM/GLM-4.6",
            "THUDM/GLM-4.7",
            "deepseek-ai/DeepSeek-V3",
        ):
            self.assertTrue(should_disable_thinking(model), model)
            self.assertTrue(moderator._should_disable_thinking(model), model)

        self.assertFalse(should_disable_thinking("Qwen/Qwen2.5-7B-Instruct"))
        # VL Instruct rejects enable_thinking; Thinking VL may still need the flag.
        self.assertFalse(should_disable_thinking("Qwen/Qwen3-VL-8B-Instruct"))
        self.assertTrue(should_disable_thinking("Qwen/Qwen3-VL-8B-Thinking"))
        self.assertTrue(should_disable_thinking("Qwen/Qwen3.5-9B"))


class ModelOverrideTests(unittest.TestCase):
    @patch("susmessagebot.moderator.get_similar_examples", return_value="")
    @patch("susmessagebot.moderator.render_prompt", return_value="rules")
    @patch("susmessagebot.moderator._text_client.chat.completions.create")
    def test_classify_message_reads_config_model_at_call_time(
        self,
        create,
        render_prompt,
        get_examples,
    ):
        from susmessagebot import config

        create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="SAFE", reasoning_content=None)
                )
            ]
        )
        previous = config.SILICONFLOW_MODEL
        try:
            config.SILICONFLOW_MODEL = "bakeoff/Override-Model"
            self.assertEqual(moderator.classify_message("hello"), "SAFE")
        finally:
            config.SILICONFLOW_MODEL = previous

        self.assertEqual(create.call_args.kwargs["model"], "bakeoff/Override-Model")


class ExampleIdTests(unittest.TestCase):
    def test_example_id_is_stable_sha256_hex(self):
        first = _example_id("cheap accounts")
        second = _example_id("cheap accounts")
        other = _example_id("legit hello")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 64)
        self.assertEqual(
            first,
            __import__("hashlib").sha256(b"cheap accounts").hexdigest(),
        )
        self.assertNotEqual(first, str(hash("cheap accounts")))

    @patch.object(vector_store, "ensure_normalized_index")
    @patch.object(vector_store, "embedding_model")
    @patch.object(vector_store, "collection")
    def test_add_example_keeps_raw_text_id_after_normalization(
        self,
        collection,
        embedding_model,
        ensure_index,
    ):
        raw = "a\u200bb"
        normalized = utils.normalize_text(raw)
        self.assertEqual(normalized, "ab")
        self.assertNotEqual(_example_id(raw), _example_id(normalized))
        encoded = MagicMock()
        encoded.tolist.return_value = [0.1, 0.2, 0.3]
        embedding_model.encode.return_value = encoded
        # No siblings share this normalized form yet.
        collection.get.side_effect = [
            {"ids": []},  # where norm_key=
            {"ids": [], "documents": [], "metadatas": []},  # fallback scan
        ]

        vector_store.add_example(raw, "SAFE")

        ensure_index.assert_called_once()
        collection.delete.assert_not_called()
        collection.upsert.assert_called_once()
        kwargs = collection.upsert.call_args.kwargs
        self.assertEqual(kwargs["ids"], [_example_id(raw)])
        self.assertEqual(kwargs["documents"], [normalized])
        self.assertEqual(
            kwargs["metadatas"],
            [{"label": "SAFE", "norm_key": normalized}],
        )
        embedding_model.encode.assert_called_once_with(normalized)

    @patch.object(vector_store, "ensure_normalized_index")
    @patch.object(vector_store, "embedding_model")
    @patch.object(vector_store, "collection")
    def test_add_example_reconciles_all_normalized_variants(
        self,
        collection,
        embedding_model,
        ensure_index,
    ):
        first = "ＳＩＰ"
        second = "𝕊𝕀ℙ"
        normalized = utils.normalize_text(first)
        self.assertEqual(normalized, utils.normalize_text(second))
        self.assertEqual(normalized, "SIP")
        first_id = _example_id(first)
        second_id = _example_id(second)
        plain_id = _example_id(normalized)
        self.assertNotEqual(first_id, second_id)
        self.assertNotEqual(first_id, plain_id)

        encoded = MagicMock()
        encoded.tolist.return_value = [0.1, 0.2, 0.3]
        embedding_model.encode.return_value = encoded
        # Existing non-plain variants share the normalized document; neither
        # is stored under the plain SIP id.
        collection.get.side_effect = [
            {"ids": []},  # where norm_key= (pre-metadata)
            {
                "ids": [first_id, second_id],
                "documents": [normalized, normalized],
                "metadatas": [
                    {"label": "BAN"},
                    {"label": "SAFE"},
                ],
            },
        ]

        vector_store.add_example(second, "SAFE")

        ensure_index.assert_called_once()
        collection.delete.assert_not_called()
        collection.upsert.assert_called_once()
        kwargs = collection.upsert.call_args.kwargs
        self.assertEqual(set(kwargs["ids"]), {second_id, first_id})
        self.assertEqual(kwargs["documents"], [normalized, normalized])
        self.assertEqual(
            kwargs["metadatas"],
            [
                {"label": "SAFE", "norm_key": normalized},
                {"label": "SAFE", "norm_key": normalized},
            ],
        )

    @patch.object(vector_store, "ensure_normalized_index")
    @patch.object(vector_store, "embedding_model")
    @patch.object(vector_store, "collection")
    def test_get_similar_examples_omits_conflicting_duplicate_docs(
        self,
        collection,
        embedding_model,
        ensure_index,
    ):
        encoded = MagicMock()
        encoded.tolist.return_value = [0.1, 0.2, 0.3]
        embedding_model.encode.return_value = encoded
        collection.count.return_value = 2
        collection.query.return_value = {
            "documents": [["SIP", "SIP"]],
            "metadatas": [[{"label": "BAN"}, {"label": "SAFE"}]],
            "distances": [[0.1, 0.2]],
        }

        self.assertEqual(vector_store.get_similar_examples("SIP"), "")
        ensure_index.assert_called_once()

    @patch.object(vector_store, "embedding_model")
    @patch.object(vector_store, "collection")
    def test_ensure_normalized_index_reembeds_persisted_rows(
        self,
        collection,
        embedding_model,
    ):
        encoded = MagicMock()
        encoded.tolist.return_value = [0.4, 0.5, 0.6]
        embedding_model.encode.return_value = encoded
        collection.get.return_value = {
            "ids": ["row1"],
            "documents": ["ＳＩＰ"],
            "metadatas": [{"label": "BAN"}],
        }

        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / ".schema_normalized-v1"
            with (
                patch.object(vector_store, "_SCHEMA_MARKER", str(marker)),
                patch.object(vector_store, "_index_ready", False),
            ):
                vector_store.ensure_normalized_index()
                vector_store.ensure_normalized_index()  # second call is no-op

            self.assertTrue(marker.is_file())

        embedding_model.encode.assert_called_once_with("SIP")
        collection.upsert.assert_called_once()
        kwargs = collection.upsert.call_args.kwargs
        self.assertEqual(kwargs["ids"], ["row1"])
        self.assertEqual(kwargs["documents"], ["SIP"])
        self.assertEqual(
            kwargs["metadatas"],
            [{"label": "BAN", "norm_key": "SIP"}],
        )

    @patch.object(vector_store, "collection")
    def test_ensure_normalized_index_does_not_mark_failed_reindex(self, collection):
        collection.get.side_effect = RuntimeError("chroma unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / ".schema_normalized-v1"
            with (
                patch.object(vector_store, "_SCHEMA_MARKER", str(marker)),
                patch.object(vector_store, "_index_ready", False),
            ):
                vector_store.ensure_normalized_index()
                self.assertFalse(marker.exists())
                self.assertFalse(vector_store._index_ready)

    @patch.object(vector_store, "ensure_normalized_index")
    @patch.object(vector_store, "embedding_model")
    @patch.object(vector_store, "collection")
    def test_sibling_lookup_skips_full_scan_after_metadata_hit(
        self,
        collection,
        embedding_model,
        ensure_index,
    ):
        raw = "ＳＩＰ"
        normalized = "SIP"
        variant_id = _example_id(raw)
        encoded = MagicMock()
        encoded.tolist.return_value = [0.1, 0.2, 0.3]
        embedding_model.encode.return_value = encoded
        collection.get.return_value = {"ids": [variant_id]}

        vector_store.add_example(raw, "BAN")

        collection.get.assert_called_once_with(
            where={"norm_key": normalized},
            include=[],
        )
        kwargs = collection.upsert.call_args.kwargs
        self.assertEqual(kwargs["ids"], [variant_id])


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
