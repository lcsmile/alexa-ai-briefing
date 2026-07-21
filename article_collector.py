import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import feedparser

from sources import SOURCES


PRIMARY_LOOKBACK_HOURS = 36
FALLBACK_LOOKBACK_HOURS = 72
MINIMUM_CANDIDATES = 8
MAX_ARTICLES_PER_SOURCE = 5
MAX_TOTAL_ARTICLES = 40

HISTORY_PATH = Path("covered_stories.json")

LOW_VALUE_TITLE_PHRASES = (
    "getting started",
    "how to",
    "beginner guide",
    "tutorial",
    "webinar",
    "event recap",
    "customer story",
    "case study",
    "careers",
    "hiring",
    "partner spotlight",
    "community spotlight",
    "weekly roundup",
    "course",
    "certification",
)


def clean_text(text):
    """Remove HTML tags and repeated whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_url(url):
    """
    Remove fragments and common tracking parameters.

    This helps identify the same story when a feed changes its tracking URL.
    """
    if not url:
        return ""

    parts = urlsplit(url.strip())

    tracking_prefixes = (
        "utm_",
        "ref",
        "source",
        "campaign",
        "fbclid",
        "gclid",
    )

    kept_parameters = []

    if parts.query:
        for parameter in parts.query.split("&"):
            parameter_name = parameter.split("=", 1)[0].casefold()

            if any(
                parameter_name.startswith(prefix)
                for prefix in tracking_prefixes
            ):
                continue

            kept_parameters.append(parameter)

    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path.rstrip("/"),
            "&".join(kept_parameters),
            "",
        )
    )


def parse_entry_date(entry):
    """Return an aware UTC datetime from an RSS entry when available."""
    parsed = getattr(entry, "published_parsed", None)

    if not parsed:
        parsed = getattr(entry, "updated_parsed", None)

    if not parsed:
        return None

    return datetime(*parsed[:6], tzinfo=timezone.utc)


def load_covered_urls(path=HISTORY_PATH):
    """Load normalized URLs already used in recent briefings."""
    path = Path(path)

    if not path.exists():
        return set()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("History file could not be read. Continuing without history.")
        return set()

    stories = payload.get("stories", [])

    if not isinstance(stories, list):
        return set()

    covered_urls = set()

    for story in stories:
        if not isinstance(story, dict):
            continue

        normalized = normalize_url(story.get("link", ""))

        if normalized:
            covered_urls.add(normalized)

    return covered_urls


def is_low_value_title(title):
    """Return True for routine tutorials, promotions and similar material."""
    normalized_title = clean_text(title).casefold()

    if not normalized_title:
        return True

    return any(
        phrase in normalized_title
        for phrase in LOW_VALUE_TITLE_PHRASES
    )


def collect_with_cutoff(
    cutoff,
    covered_urls,
    now=None,
):
    """Collect candidate articles newer than the supplied cutoff."""
    now = now or datetime.now(timezone.utc)

    articles = []
    seen_titles = set()
    seen_urls = set()

    for source in SOURCES:
        source_name = source["name"]
        feed_url = source["feed"]

        feed = feedparser.parse(feed_url)

        print(
            f"{source_name}: "
            f"status={getattr(feed, 'status', 'unknown')}, "
            f"entries={len(feed.entries)}"
        )

        source_articles = []

        for entry in feed.entries[:30]:
            title = clean_text(getattr(entry, "title", ""))
            link = clean_text(getattr(entry, "link", ""))
            normalized_link = normalize_url(link)
            published = parse_entry_date(entry)

            if not title or not link:
                continue

            if published is None:
                print(f"Skipped undated article: {source_name} — {title}")
                continue

            if published < cutoff or published > now + timedelta(hours=2):
                continue

            if is_low_value_title(title):
                print(f"Skipped low-value title: {source_name} — {title}")
                continue

            if normalized_link in covered_urls:
                print(f"Skipped previously covered URL: {title}")
                continue

            title_key = title.casefold()

            if title_key in seen_titles:
                continue

            if normalized_link in seen_urls:
                continue

            age_hours = (now - published).total_seconds() / 3600

            article = {
                "source": source_name,
                "category": source.get("category", "Other"),
                "source_type": source.get("source_type", "primary"),
                "title": title,
                "summary": clean_text(
                    getattr(entry, "summary", "")
                )[:1000],
                "link": link,
                "normalized_link": normalized_link,
                "published": published.isoformat(),
                "age_label": (
                    "recent"
                    if age_hours <= PRIMARY_LOOKBACK_HOURS
                    else "earlier this week"
                ),
            }

            source_articles.append(article)
            seen_titles.add(title_key)
            seen_urls.add(normalized_link)

        source_articles.sort(
            key=lambda article: article["published"],
            reverse=True,
        )

        articles.extend(
            source_articles[:MAX_ARTICLES_PER_SOURCE]
        )

    articles.sort(
        key=lambda article: article["published"],
        reverse=True,
    )

    return articles[:MAX_TOTAL_ARTICLES]


def collect_articles(now=None, history_path=HISTORY_PATH):
    """
    Use a 36-hour window first.

    Expand to 72 hours only when fewer than eight useful candidates exist.
    Previously covered stories are never reused.
    """
    now = now or datetime.now(timezone.utc)
    covered_urls = load_covered_urls(history_path)

    primary_cutoff = now - timedelta(
        hours=PRIMARY_LOOKBACK_HOURS
    )

    articles = collect_with_cutoff(
        cutoff=primary_cutoff,
        covered_urls=covered_urls,
        now=now,
    )

    window_used = PRIMARY_LOOKBACK_HOURS

    if len(articles) < MINIMUM_CANDIDATES:
        print(
            f"Only {len(articles)} candidates found in "
            f"{PRIMARY_LOOKBACK_HOURS} hours. "
            f"Expanding to {FALLBACK_LOOKBACK_HOURS} hours."
        )

        fallback_cutoff = now - timedelta(
            hours=FALLBACK_LOOKBACK_HOURS
        )

        articles = collect_with_cutoff(
            cutoff=fallback_cutoff,
            covered_urls=covered_urls,
            now=now,
        )

        window_used = FALLBACK_LOOKBACK_HOURS

    print(
        f"Collection complete: {len(articles)} candidates, "
        f"{window_used}-hour window."
    )

    return articles, window_used
