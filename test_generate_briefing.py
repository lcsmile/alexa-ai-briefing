import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import generate_briefing as briefing


def article(title, published, link=None, source="Test AI"):
    return {
        "source": source,
        "title": title,
        "summary": f"Concrete details about {title}.",
        "link": link or f"https://example.com/{title.lower().replace(' ', '-')}",
        "published": published,
    }


class FakeGeminiClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.models = self
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.response_text)


class BriefingTests(unittest.TestCase):
    def test_collect_articles_sorts_undated_after_dated_without_leaking_datetime(self):
        newer = SimpleNamespace(
            title="Newer dated news",
            summary="Newer",
            link="https://example.com/newer",
            published_parsed=(2026, 7, 13, 16, 0, 0, 0, 0, 0),
            updated_parsed=None,
        )
        older = SimpleNamespace(
            title="Older dated news",
            summary="Older",
            link="https://example.com/older",
            published_parsed=(2026, 7, 12, 16, 0, 0, 0, 0, 0),
            updated_parsed=None,
        )
        undated = SimpleNamespace(
            title="Undated news",
            summary="Undated",
            link="https://example.com/undated",
            published_parsed=None,
            updated_parsed=None,
        )
        parsed_feed = SimpleNamespace(
            entries=[undated, older, newer],
            status=200,
        )

        with patch.object(briefing.feedparser, "parse", return_value=parsed_feed):
            with patch.object(
                briefing,
                "FEEDS",
                {"Test AI": "https://example.com/feed"},
            ):
                with patch.object(briefing, "LOOKBACK_HOURS", 24 * 3650):
                    articles = briefing.collect_articles()

        self.assertEqual(
            [item["title"] for item in articles],
            ["Newer dated news", "Older dated news", "Undated news"],
        )
        self.assertEqual(
            articles[-1]["published"],
            "Publication time unavailable",
        )
        json.dumps(articles)
        self.assertFalse(
            any(
                isinstance(value, datetime)
                for item in articles
                for value in item.values()
            )
        )

    def test_dashboard_filter_rejects_low_news_value_title_phrases(self):
        for phrase in briefing.LOW_NEWS_VALUE_TITLE_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertFalse(
                    briefing.is_dashboard_candidate(
                        article(
                            f"An {phrase.title()} for developers",
                            "2026-07-13T16:00:00+00:00",
                        )
                    )
                )

        self.assertTrue(
            briefing.is_dashboard_candidate(
                article(
                    "New frontier model launches",
                    "2026-07-13T16:00:00+00:00",
                )
            )
        )

    def test_gemini_selects_three_filtered_articles(self):
        articles = [
            article(
                "Getting started with ChatGPT",
                "2026-07-13T18:00:00+00:00",
            ),
            article("New frontier model launches", "2026-07-13T17:00:00+00:00"),
            article("AI safety benchmark released", "2026-07-12T17:00:00+00:00"),
            article("Major AI acquisition announced", "2026-07-11T17:00:00+00:00"),
            article("Undated product article", "Publication time unavailable"),
        ]
        client = FakeGeminiClient(
            json.dumps({"selected_article_indexes": [0, 1, 2]})
        )

        selected = briefing.select_dashboard_articles(articles, client=client)

        self.assertEqual(
            [item["title"] for item in selected],
            [
                "New frontier model launches",
                "AI safety benchmark released",
                "Major AI acquisition announced",
            ],
        )
        prompt = client.calls[0]["contents"]
        self.assertNotIn("Getting started with ChatGPT", prompt)
        self.assertIn("Undated product article", prompt)
        self.assertEqual(
            client.calls[0]["config"]["response_mime_type"],
            "application/json",
        )
        self.assertIn("response_json_schema", client.calls[0]["config"])

    def test_invalid_gemini_selection_falls_back_to_filtered_dated_news(self):
        articles = [
            article(
                "Getting started with ChatGPT",
                "2026-07-13T18:00:00+00:00",
            ),
            article("Dated news one", "2026-07-13T17:00:00+00:00"),
            article("Dated news two", "2026-07-12T17:00:00+00:00"),
            article("Dated news three", "2026-07-11T17:00:00+00:00"),
            article("Undated news", "Publication time unavailable"),
        ]
        client = FakeGeminiClient(
            json.dumps({"selected_article_indexes": [0, 0, 999]})
        )

        selected = briefing.select_dashboard_articles(articles, client=client)

        self.assertEqual(
            [item["title"] for item in selected],
            ["Dated news one", "Dated news two", "Dated news three"],
        )

    def test_dashboard_png_is_expected_grayscale_size(self):
        stories = [
            article("New frontier model launches", "2026-07-13T17:00:00+00:00"),
            article("AI safety benchmark released", "2026-07-12T17:00:00+00:00"),
            article("Major AI acquisition announced", "2026-07-11T17:00:00+00:00"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "dashboard.png"
            briefing.create_kindle_dashboard(stories, output_path)

            with Image.open(output_path) as image:
                self.assertEqual(image.size, (1072, 1448))
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
