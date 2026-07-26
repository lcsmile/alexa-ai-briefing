import json
from collections import Counter
from datetime import datetime, timedelta, timezone


TARGET_MIN_WORDS = 650
HARD_MIN_WORDS = 600
MAX_WORDS = 950

MIN_ORGANIZATIONS = 3
MAX_PER_ORGANIZATION = 2
MAX_ARTICLE_AGE_HOURS = 72

OPENING_PREFIX = (
    "Good morning. Here is your curated AI briefing for"
)

CLOSING_SENTENCE = (
    "That is your curated AI briefing for today."
)


def parse_datetime(value: str):
    """Parse an ISO date safely."""
    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
    ):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def minimum_words_for_story_count(
    story_count: int,
) -> int:
    """
    Use a realistic minimum on low-news days.

    Five stories should not be forced to contain as much filler
    as eight stories.
    """
    if story_count >= 8:
        return 650

    if story_count == 7:
        return 625

    return HARD_MIN_WORDS


def validate_briefing(
    summary: str,
    selected_articles: list[dict],
    all_articles: list[dict],
    covered_urls: set[str],
    now: datetime | None = None,
) -> None:
    """Reject briefings that fail editorial or data checks."""
    now = now or datetime.now(
        timezone.utc
    )

    errors = []
    warnings = []

    word_count = len(
        summary.split()
    )

    required_minimum = (
        minimum_words_for_story_count(
            len(selected_articles)
        )
    )

    if word_count < required_minimum:
        errors.append(
            f"Briefing has {word_count} words. "
            f"Minimum for "
            f"{len(selected_articles)} stories is "
            f"{required_minimum}."
        )

    elif word_count < TARGET_MIN_WORDS:
        warnings.append(
            f"Briefing has {word_count} words, "
            f"slightly below the preferred "
            f"{TARGET_MIN_WORDS}-word target."
        )

    if word_count > MAX_WORDS:
        errors.append(
            f"Briefing has {word_count} words. "
            f"Maximum is {MAX_WORDS}."
        )

    if not summary.startswith(
        OPENING_PREFIX
    ):
        errors.append(
            "Required opening sentence is missing."
        )

    if not summary.endswith(
        CLOSING_SENTENCE
    ):
        errors.append(
            "Required closing sentence is missing."
        )

    if not selected_articles:
        errors.append(
            "No stories were selected."
        )

    publisher_counts = Counter(
        article["source"]
        for article in selected_articles
    )

    available_publishers = len(
        {
            article["source"]
            for article in all_articles
        }
    )

    required_publishers = min(
        MIN_ORGANIZATIONS,
        available_publishers,
        len(selected_articles),
    )

    if (
        len(publisher_counts)
        < required_publishers
    ):
        errors.append(
            "The selected stories do not use "
            "enough available publishers."
        )

    for publisher, count in (
        publisher_counts.items()
    ):
        if count > MAX_PER_ORGANIZATION:
            errors.append(
                f"{publisher} has {count} selected stories. "
                f"Maximum is {MAX_PER_ORGANIZATION}."
            )

    collected_urls = {
        article["normalized_link"]
        for article in all_articles
    }

    for article in selected_articles:
        normalized_link = article[
            "normalized_link"
        ]

        if normalized_link not in collected_urls:
            errors.append(
                "A selected URL was not present "
                f"in the collected candidates: "
                f"{article['link']}"
            )

        if normalized_link in covered_urls:
            errors.append(
                "A selected story was already covered: "
                f"{article['link']}"
            )

        published = parse_datetime(
            article.get("published")
        )

        if published is None:
            errors.append(
                "A selected article has no valid date: "
                f"{article['title']}"
            )
            continue

        age = now - published

        if age > timedelta(
            hours=MAX_ARTICLE_AGE_HOURS,
            minutes=5,
        ):
            errors.append(
                "A selected article is older than "
                f"{MAX_ARTICLE_AGE_HOURS} hours: "
                f"{article['title']}"
            )

    for warning in warnings:
        print(
            f"Quality warning: {warning}"
        )

    if errors:
        raise ValueError(
            "\n".join(errors)
        )


def validate_feed_payload(
    payload: list[dict],
) -> None:
    """Verify the Alexa JSON feed before replacing it."""
    try:
        encoded = json.dumps(
            payload
        )
        decoded = json.loads(
            encoded
        )
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            "The feed payload is not valid JSON."
        ) from error

    if (
        not isinstance(decoded, list)
        or len(decoded) != 1
    ):
        raise ValueError(
            "The Alexa feed must contain exactly one item."
        )

    item = decoded[0]

    required_fields = {
        "uid",
        "updateDate",
        "titleText",
        "mainText",
        "redirectionUrl",
    }

    missing_fields = (
        required_fields - set(item)
    )

    if missing_fields:
        raise ValueError(
            "The Alexa feed is missing these fields: "
            + ", ".join(
                sorted(missing_fields)
            )
        )

    if (
        not isinstance(item["mainText"], str)
        or not item["mainText"].strip()
    ):
        raise ValueError(
            "The Alexa feed mainText is empty."
        )
