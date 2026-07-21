from datetime import datetime, timedelta, timezone

import pytest

from article_collector import (
    is_low_value_title,
    normalize_url,
)
from quality_checks import (
    validate_briefing,
    validate_feed_payload,
)
from story_selector import (
    deterministic_fallback,
)


def make_article(
    source: str,
    index: int,
    hours_old: int = 1,
    source_type: str = "primary",
) -> dict:
    now = datetime.now(
        timezone.utc
    )

    published = now - timedelta(
        hours=hours_old
    )

    return {
        "source": source,
        "category": "Test category",
        "source_type": source_type,
        "title": f"Test story {index}",
        "summary": "Test summary.",
        "article_text":
            "Detailed test article content.",
        "link":
            f"https://example.com/{source}/{index}",
        "normalized_link":
            f"https://example.com/{source}/{index}",
        "published": published.isoformat(),
        "age_label": (
            "recent"
            if hours_old <= 36
            else "earlier this week"
        ),
    }


def make_valid_summary() -> str:
    opening = (
        "Good morning. Here is your curated AI "
        "briefing for Monday, July 20, 2026."
    )

    closing = (
        "That is your curated AI briefing "
        "for today."
    )

    middle_words = [
        "development"
        for _ in range(670)
    ]

    return (
        opening
        + " "
        + " ".join(middle_words)
        + " "
        + closing
    )


def test_low_value_title_filter():
    assert is_low_value_title(
        "Getting started with an AI tool"
    )

    assert is_low_value_title(
        "Customer story: Example Corporation"
    )

    assert not is_low_value_title(
        "New model released with benchmark results"
    )


def test_url_normalization_removes_tracking():
    normalized = normalize_url(
        "https://EXAMPLE.com/story/"
        "?utm_source=x&useful=1#section"
    )

    assert normalized == (
        "https://example.com/story?useful=1"
    )


def test_deterministic_fallback_caps_publishers():
    articles = [
        make_article("OpenAI", 1),
        make_article("OpenAI", 2),
        make_article("OpenAI", 3),
        make_article("DeepMind", 4),
        make_article("Microsoft", 5),
        make_article("NVIDIA", 6),
        make_article("AWS", 7),
        make_article(
            "TechCrunch",
            8,
            source_type="editorial",
        ),
        make_article(
            "Ars Technica",
            9,
            source_type="editorial",
        ),
    ]

    selected = deterministic_fallback(
        articles
    )

    assert len(selected) == 8

    assert (
        sum(
            article["source"] == "OpenAI"
            for article in selected
        )
        <= 2
    )

    assert any(
        article["source_type"] == "editorial"
        for article in selected
    )


def test_valid_feed_payload():
    payload = [
        {
            "uid": "123",
            "updateDate":
                "2026-07-20T12:00:00.0Z",
            "titleText":
                "My Curated AI Sources",
            "mainText":
                "A valid briefing.",
            "redirectionUrl":
                "https://example.com/sources.html",
        }
    ]

    validate_feed_payload(payload)


def test_feed_rejects_empty_main_text():
    payload = [
        {
            "uid": "123",
            "updateDate":
                "2026-07-20T12:00:00.0Z",
            "titleText":
                "My Curated AI Sources",
            "mainText": "",
            "redirectionUrl":
                "https://example.com/sources.html",
        }
    ]

    with pytest.raises(ValueError):
        validate_feed_payload(payload)


def test_valid_briefing_passes():
    now = datetime.now(
        timezone.utc
    )

    selected = [
        make_article("OpenAI", 1),
        make_article("DeepMind", 2),
        make_article("Microsoft", 3),
        make_article("NVIDIA", 4),
        make_article("AWS", 5),
        make_article(
            "TechCrunch",
            6,
            source_type="editorial",
        ),
        make_article(
            "Ars Technica",
            7,
            source_type="editorial",
        ),
        make_article("Hugging Face", 8),
    ]

    validate_briefing(
        summary=make_valid_summary(),
        selected_articles=selected,
        all_articles=selected,
        covered_urls=set(),
        now=now,
    )


def test_briefing_rejects_old_article():
    now = datetime.now(
        timezone.utc
    )

    selected = [
        make_article("OpenAI", 1, hours_old=80),
        make_article("DeepMind", 2),
        make_article("Microsoft", 3),
        make_article("NVIDIA", 4),
    ]

    with pytest.raises(ValueError):
        validate_briefing(
            summary=make_valid_summary(),
            selected_articles=selected,
            all_articles=selected,
            covered_urls=set(),
            now=now,
        )


def test_briefing_rejects_history_duplicate():
    now = datetime.now(
        timezone.utc
    )

    selected = [
        make_article("OpenAI", 1),
        make_article("DeepMind", 2),
        make_article("Microsoft", 3),
        make_article("NVIDIA", 4),
    ]

    covered_urls = {
        selected[0]["normalized_link"]
    }

    with pytest.raises(ValueError):
        validate_briefing(
            summary=make_valid_summary(),
            selected_articles=selected,
            all_articles=selected,
            covered_urls=covered_urls,
            now=now,
        )
