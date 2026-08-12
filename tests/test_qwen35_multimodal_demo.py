import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import demo_qwen35_multimodal as demo


class Qwen35MultimodalDemoTests(unittest.TestCase):
    @patch.object(demo.moderator._text_client.chat.completions, "create")
    def test_demo_uses_existing_chat_completions_shape_for_text_and_image(self, create):
        create.side_effect = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="TEXT_OK"))],
                _request_id="text-id",
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="a test image"))],
                _request_id="image-id",
            ),
        ]

        results = demo.run_demo(
            model="Qwen/Qwen3.5-4B",
            image_bytes=demo._sample_image_bytes(),
            timeout=12.5,
        )

        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual(create.call_count, 2)
        text_call, image_call = create.call_args_list
        self.assertEqual(text_call.kwargs["model"], "Qwen/Qwen3.5-4B")
        self.assertEqual(text_call.kwargs["timeout"], 12.5)
        self.assertEqual(
            text_call.kwargs["extra_body"],
            {"enable_thinking": False},
        )
        image_content = image_call.kwargs["messages"][0]["content"]
        self.assertEqual(image_content[1]["type"], "image_url")
        self.assertTrue(image_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    @patch.object(
        demo.moderator._text_client.chat.completions,
        "create",
        side_effect=TimeoutError("timed out"),
    )
    def test_request_reports_timeout_type_and_elapsed_time(self, create):
        result = demo._request(
            "image",
            model="Qwen/Qwen3.5-4B",
            messages=[{"role": "user", "content": "test"}],
            timeout=1.0,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertIn("elapsed_seconds", result)
        create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
