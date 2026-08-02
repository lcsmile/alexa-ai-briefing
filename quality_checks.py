import json
from collections import Counter
from datetime import datetime, timedelta, timezone


PREFERRED_MIN_WORDS = 650
PREFERRED_MAX_WORDS = 950

# Only content below this threshold is considered unusably short.
HARD_MIN_WORDS = 350
HARD_MAX_WORDS = 1200

PREFERRED_MIN_ORGANIZATIONS = 4
PREFERRED_MAX_PER_ORGANIZATION = 2
PREFERRED_MAX_ARTICLE_AGE_HOURS = 72

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


def print_warning(message: str) -> None:
    """Print a clearly labeled nonfatal quality warning."""
    print(f"QUALITY WARNING: {message}")


def validate_briefing(
    summary: str,
    selected_articles: list[dict],
    all_articles: list[dict],
    covered_urls: set[str],
    now: datetime | None = None,
) -> None:
    """
    Validate the briefing without rejecting usable content.

    Only structural or severely unusable output raises an error.
    Editorial preferences produce warnings.
    """
    now = now or datetime.now(
        timezone.utc
    )

    fatal_errors = []

    if not isinstance(summary, str):
        fatal_errors.append(
            "The briefing is not text."
        )
        summary = ""

    summary = summary.strip()
    word_count = len(summary.split())

    if not summary:
        fatal_errors.append(
            "The briefing is empty."
        )

    if word_count < HARD_MIN_WORDS:
        fatal_errors.append(
            f"The briefing has only {word_count} words. "
            f"The hard minimum is {HARD_MIN_WORDS}."
        )
    elif word_count < PREFERRED_MIN_WORDS:
        print_warning(
            f"The briefing has {word_count} words. "
            f"The preferred minimum is "
            f"{PREFERRED_MIN_WORDS}, but the briefing "
            "is still usable."
        )

    if word_count > HARD_MAX_WORDS:
        fatal_errors.append(
            f"The briefing has {word_count} words. "
            f"The hard maximum is {HARD_MAX_WORDS}."
        )
    elif word_count > PREFERRED_MAX_WORDS:
        print_warning(
            f"The briefing has {word_count} words. "
            f"The preferred maximum is "
            f"{PREFERRED_MAX_WORDS}."
        )

    if not summary.startswith(
        OPENING_PREFIX
    ):
        print_warning(
            "The preferred opening sentence is missing."
        )

    if not summary.endswith(
        CLOSING_SENTENCE
    ):
        print_warning(
            "The preferred closing sentence is missing."
        )

    if not selected_articles:
        fatal_errors.append(
            "No stories were selected."
        )

    publisher_counts = Counter(
        article.get("source", "Unknown")
        for article in selected_articles
    )

    available_publishers = len(
        {
            article.get("source", "Unknown")
            for article in all_articles
        }
    )

    preferred_publishers = min(
        PREFERRED_MIN_ORGANIZATIONS,
        available_publishers,
        len(selected_articles),
    )

    if (
        selected_articles
        and len(publisher_counts)
        < preferred_publishers
    ):
        print_warning(
            f"The briefing uses "
            f"{len(publisher_counts)} publishers. "
            f"The preferred number for the available "
            f"material is {preferred_publishers}."
        )

    for publisher, count in (
        publisher_counts.items()
    ):
        if count > PREFERRED_MAX_PER_ORGANIZATION:
            print_warning(
                f"{publisher} has {count} stories. "
                f"The preferred maximum is "
                f"{PREFERRED_MAX_PER_ORGANIZATION}."
            )

    collected_urls = {
        article.get("normalized_link")
        for article in all_articles
        if article.get("normalized_link")
    }

    for article in selected_articles:
        title = article.get(
            "title",
            "Untitled article",
        )

        normalized_link = article.get(
            "normalized_link"
        )

        if not normalized_link:
            fatal_errors.append(
                f"A selected article has no URL: {title}"
            )
            continue

        if normalized_link not in collected_urls:
            fatal_errors.append(
                "A selected URL was not present in the "
                f"collected candidates: {normalized_link}"
            )

        if normalized_link in covered_urls:
            fatal_errors.append(
                f"A previously covered story was selected: {title}"
            )

        published = parse_datetime(
            article.get("published")
        )

        if published is None:
            print_warning(
                f"A selected article has no valid date: "
                f"{title}"
            )
            continue

        age = now - published

        if age > timedelta(
            hours=PREFERRED_MAX_ARTICLE_AGE_HOURS,
            minutes=5,
        ):
            fatal_errors.append(
                f"A selected article is older than "
                f"{PREFERRED_MAX_ARTICLE_AGE_HOURS} hours: "
                f"{title}"
            )

    if fatal_errors:
        raise ValueError(
            "\n".join(fatal_errors)
        )

    print(
        f"Briefing quality check passed: "
        f"{word_count} words, "
        f"{len(selected_articles)} stories, "
        f"{len(publisher_counts)} publishers."
    )


def validate_feed_payload(
    payload: list[dict],
) -> None:
    """Verify Alexa's JSON feed before replacing the live feed."""
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

    if not isinstance(item, dict):
        raise ValueError(
            "The Alexa feed item must be an object."
        )

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

    for field in required_fields:
        if (
            not isinstance(item[field], str)
            or not item[field].strip()
        ):
            raise ValueError(
                f"The Alexa feed field "
                f"'{field}' is empty or invalid."
            )

    main_text = item["mainText"]

    if len(main_text) >= 4500:
        raise ValueError(
            "The Alexa feed mainText must be fewer than 4,500 characters."
        )

    if any(char in main_text for char in "<>\n\r"):
        raise ValueError(
            "The Alexa feed mainText must be plain single-line text."
        )
