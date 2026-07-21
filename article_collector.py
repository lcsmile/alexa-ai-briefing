import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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

TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}


def clean_text(text: str) -> str:
    """Remove HTML tags and repeated whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_url(url: str) -> str:
    """Remove fragments and common tracking parameters."""
    if not url:
        return ""

    parts = urlsplit(url.strip())
    retained_parameters = []

    for key, value in parse_qsl(
        parts.query,
        keep_blank_values=True,
    ):
        normalized_key = key.casefold()

        if normalized_key.startswith("utm_"):
            continue

        if normalized_key in TRACKING_PARAMETERS:
            continue

        retained_parameters.append((key, value))

    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path.rstrip("/"),
            urlencode(retained_parameters, doseq=True),
            "",
        )
    )


def parse_entry_date(entry):
    """Read the publication or update date from an RSS entry."""
    parsed = getattr(entry, "published_parsed", None)

    if not parsed:
        parsed = getattr(entry, "updated_parsed", None)

    if not parsed:
        return None

    return datetime(
        *parsed[:6],
        tzinfo=timezone.utc,
    )


def is_low_value_title(title: str) -> bool:
    """Identify routine tutorials, webinars and promotional articles."""
    normalized_title = clean_text(title).casefold()

    if not normalized_title:
        return True

    return any(
        phrase in normalized_title
        for phrase in LOW_VALUE_TITLE_PHRASES
    )


def load_covered_urls(
    path: Path = HISTORY_PATH,
) -> set[str]:
    """Load URLs used in recent briefings."""
    if not path.exists():
        return set()

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        print(
            "Warning: covered_stories.json could not be read. "
            "Continuing without history."
        )
        return set()

    stories = payload.get("stories", [])

    if not isinstance(stories, list):
        return set()

    covered_urls = set()

    for story in stories:
        if not isinstance(story, dict):
            continue

        normalized_link = normalize_url(
            story.get("link", "")
        )

        if normalized_link:
            covered_urls.add(normalized_link)

    return covered_urls


def collect_with_cutoff(
    cutoff: datetime,
    covered_urls: set[str],
    now: datetime,
) -> list[dict]:
    """Collect articles newer than a supplied date."""
    articles = []
    seen_urls = set()
    seen_titles = set()

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
            title = clean_text(
                getattr(entry, "title", "")
            )
            link = clean_text(
                getattr(entry, "link", "")
            )
            published = parse_entry_date(entry)
            normalized_link = normalize_url(link)

            if not title or not link:
                continue

            if published is None:
                print(
                    f"Skipped undated article: "
                    f"{source_name} — {title}"
                )
                continue

            if published < cutoff:
                continue

            if published > now + timedelta(hours=2):
                print(
                    f"Skipped future-dated article: "
                    f"{source_name} — {title}"
                )
                continue

            if is_low_value_title(title):
                print(
                    f"Skipped low-value title: "
                    f"{source_name} — {title}"
                )
                continue

            if normalized_link in covered_urls:
                print(
                    f"Skipped previously covered article: "
                    f"{title}"
                )
                continue

            if normalized_link in seen_urls:
                continue

            title_key = title.casefold()

            if title_key in seen_titles:
                continue

            age_hours = (
                now - published
            ).total_seconds() / 3600

            article = {
                "source": source_name,
                "category": source.get(
                    "category",
                    "Other",
                ),
                "source_type": source.get(
                    "source_type",
                    "primary",
                ),
                "title": title,
                "summary": clean_text(
                    getattr(entry, "summary", "")
                )[:1200],
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
            seen_urls.add(normalized_link)
            seen_titles.add(title_key)

        source_articles.sort(
            key=lambda article: article["published"],
            reverse=True,
        )

        articles.extend(
            source_articles[
                :MAX_ARTICLES_PER_SOURCE
            ]
        )

    articles.sort(
        key=lambda article: article["published"],
        reverse=True,
    )

    return articles[:MAX_TOTAL_ARTICLES]


def collect_articles(
    now: datetime | None = None,
    history_path: Path = HISTORY_PATH,
) -> tuple[list[dict], int]:
    """
    Search the last 36 hours first.

    Expand to 72 hours when fewer than eight useful
    candidates are available.
    """
    now = now or datetime.now(timezone.utc)

    covered_urls = load_covered_urls(
        history_path
    )

    primary_articles = collect_with_cutoff(
        cutoff=now - timedelta(
            hours=PRIMARY_LOOKBACK_HOURS
        ),
        covered_urls=covered_urls,
        now=now,
    )

    if len(primary_articles) >= MINIMUM_CANDIDATES:
        print(
            f"Using {PRIMARY_LOOKBACK_HOURS}-hour window "
            f"with {len(primary_articles)} candidates."
        )

        return (
            primary_articles,
            PRIMARY_LOOKBACK_HOURS,
        )

    fallback_articles = collect_with_cutoff(
        cutoff=now - timedelta(
            hours=FALLBACK_LOOKBACK_HOURS
        ),
        covered_urls=covered_urls,
        now=now,
    )

    print(
        f"Only {len(primary_articles)} candidates were found "
        f"in {PRIMARY_LOOKBACK_HOURS} hours. "
        f"Using the {FALLBACK_LOOKBACK_HOURS}-hour window "
        f"with {len(fallback_articles)} candidates."
    )

    return (
        fallback_articles,
        FALLBACK_LOOKBACK_HOURS,
    )
