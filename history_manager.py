import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


HISTORY_PATH = Path("covered_stories.json")
RETENTION_DAYS = 14


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


def load_history(
    path: Path = HISTORY_PATH,
) -> dict:
    """Read the existing history file."""
    if not path.exists():
        return {
            "stories": []
        }

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {
            "stories": []
        }

    if not isinstance(payload, dict):
        return {
            "stories": []
        }

    stories = payload.get("stories")

    if not isinstance(stories, list):
        return {
            "stories": []
        }

    return {
        "stories": stories
    }


def update_history(
    selected_articles: list[dict],
    now: datetime | None = None,
    path: Path = HISTORY_PATH,
) -> None:
    """Retain 14 days of history and add newly used stories."""
    now = now or datetime.now(
        timezone.utc
    )

    cutoff = now - timedelta(
        days=RETENTION_DAYS
    )

    payload = load_history(path)

    retained_stories = []
    seen_links = set()

    for story in payload["stories"]:
        if not isinstance(story, dict):
            continue

        covered_at = parse_datetime(
            story.get("covered_at")
        )
        link = story.get("link")

        if not covered_at or covered_at < cutoff:
            continue

        if not link or link in seen_links:
            continue

        retained_stories.append(story)
        seen_links.add(link)

    for article in selected_articles:
        link = article["normalized_link"]

        if link in seen_links:
            continue

        retained_stories.append(
            {
                "title": article["title"],
                "source": article["source"],
                "link": link,
                "published": article["published"],
                "covered_at": now.isoformat(),
            }
        )

        seen_links.add(link)

    path.write_text(
        json.dumps(
            {
                "stories": retained_stories
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
