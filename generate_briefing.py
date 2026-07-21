import html
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from google import genai
from PIL import Image, ImageDraw, ImageFont

from article_collector import (
    collect_articles,
    load_covered_urls,
)
from article_extractor import enrich_articles
from history_manager import update_history
from quality_checks import (
    validate_briefing,
    validate_feed_payload,
)
from story_selector import select_stories


MODEL_NAME = "gemini-3.1-flash-lite"

FEED_PATH = Path("feed.json")
SOURCES_PAGE_PATH = Path("sources.html")
DASHBOARD_PATH = Path(
    "kindle-dashboard.png"
)

LOCAL_TIMEZONE = ZoneInfo(
    "America/Los_Angeles"
)

DASHBOARD_WIDTH = 1072
DASHBOARD_HEIGHT = 1448
DASHBOARD_STORIES = 3

FONT_PATHS = {
    "regular": (
        "/usr/share/fonts/truetype/"
        "dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/"
        "Supplemental/Arial.ttf",
    ),
    "bold": (
        "/usr/share/fonts/truetype/"
        "dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/"
        "Supplemental/Arial Bold.ttf",
    ),
}


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def create_summary(
    selected_articles: list[dict],
    now: datetime | None = None,
    client=None,
) -> str:
    """Create the final spoken briefing."""
    if not selected_articles:
        raise RuntimeError(
            "No stories were selected."
        )

    now = now or datetime.now(
        timezone.utc
    )

    if client is None:
        api_key = os.environ.get(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY was not found."
            )

        client = genai.Client(
            api_key=api_key
        )

    article_blocks = []

    for article in selected_articles:
        article_blocks.append(
            f"Publisher: {article['source']}\n"
            f"Publisher type: "
            f"{article['source_type']}\n"
            f"Category: {article['category']}\n"
            f"Title: {article['title']}\n"
            f"Published: {article['published']}\n"
            f"Age label: {article['age_label']}\n"
            f"Article content: "
            f"{article['article_text']}"
        )

    article_text = "\n\n".join(
        article_blocks
    )

    spoken_date = (
        now.astimezone(LOCAL_TIMEZONE)
        .strftime("%A, %B %d, %Y")
        .replace(" 0", " ")
    )

    prompt = f"""
Write a neutral spoken artificial-intelligence news
briefing between 700 and 900 words.

Use only the supplied material.

The exact first sentence must be:
Good morning. Here is your curated AI briefing for
{spoken_date}.

The exact final sentence must be:
That is your curated AI briefing for today.

Required structure:
- Begin with the most important story and explain why
  it matters.
- Cover two additional major developments with useful
  context.
- Cover the remaining selected stories more concisely.
- Include research, infrastructure, policy, safety, or
  business context when the material supports it.
- Include a brief paragraph about what to watch next
  before the required final sentence.

Rules:
- Cover every supplied selected story once.
- Use neutral and factual language.
- Do not copy company marketing language.
- Attribute company claims clearly.
- Distinguish primary announcements from independent
  reporting.
- Explain technical terms briefly.
- Mention meaningful limitations, availability,
  uncertainty, or missing details.
- Do not invent facts, dates, prices, benchmarks, or
  conclusions.
- Refer to an article as “earlier this week” when its
  age label says “earlier this week.”
- Do not use headings, bullet points, markdown, or URLs.
- Do not provide a source list inside the spoken text.
- Avoid generic phrases such as “significant step
  forward.”
- Expand abbreviations when that improves speech.

Selected material:

{article_text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    summary = clean_text(
        response.text
    )

    if not summary:
        raise RuntimeError(
            "Gemini returned an empty summary."
        )

    return summary


def build_feed_payload(
    summary: str,
    now: datetime | None = None,
) -> list[dict]:
    """Create Alexa's Flash Briefing JSON structure."""
    now = now or datetime.now(
        timezone.utc
    )

    return [
        {
            "uid": str(uuid.uuid4()),
            "updateDate": now.strftime(
                "%Y-%m-%dT%H:%M:%S.0Z"
            ),
            "titleText":
                "My Curated AI Sources",
            "mainText": summary,
            "redirectionUrl": (
                "https://lcsmile.github.io/"
                "alexa-ai-briefing/"
                "sources.html"
            ),
        }
    ]


def write_sources_page(
    selected_articles: list[dict],
    now: datetime | None = None,
    output_path: Path = SOURCES_PAGE_PATH,
) -> None:
    """Create a public page listing the sources used."""
    now = now or datetime.now(
        timezone.utc
    )

    date_text = (
        now.astimezone(LOCAL_TIMEZONE)
        .strftime("%B %d, %Y")
        .replace(" 0", " ")
    )

    list_items = []

    for article in selected_articles:
        published = datetime.fromisoformat(
            article["published"].replace(
                "Z",
                "+00:00",
            )
        ).astimezone(LOCAL_TIMEZONE)

        published_text = (
            published.strftime(
                "%b %d, %Y %-I:%M %p %Z"
            ).replace(" 0", " ")
        )

        list_items.append(
            "<li>"
            f'<a href="'
            f'{html.escape(article["link"], quote=True)}'
            f'">'
            f'{html.escape(article["title"])}'
            f"</a>"
            f"<br><span>"
            f'{html.escape(article["source"])}'
            f" · {published_text}"
            f"</span>"
            f"</li>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
content="width=device-width, initial-scale=1">
<title>My Curated AI Sources</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 760px;
    margin: 40px auto;
    padding: 0 20px;
    line-height: 1.5;
}}
li {{
    margin-bottom: 18px;
}}
span {{
    color: #555;
}}
a {{
    color: #174ea6;
}}
</style>
</head>
<body>
<h1>My Curated AI Sources</h1>
<p>
Sources used for the briefing published
{html.escape(date_text)}.
</p>
<ol>
{''.join(list_items)}
</ol>
</body>
</html>
"""

    output_path.write_text(
        document,
        encoding="utf-8",
    )


def load_font(
    style: str,
    size: int,
):
    """Load an available dashboard font."""
    for font_path in FONT_PATHS[style]:
        if os.path.exists(font_path):
            return ImageFont.truetype(
                font_path,
                size=size,
            )

    raise RuntimeError(
        f"No {style} dashboard font was found."
    )


def fit_lines(
    draw,
    text: str,
    font,
    max_width: int,
    max_lines: int,
) -> list[str]:
    """Fit text into a limited number of image lines."""
    words = clean_text(text).split()
    lines = []
    current = ""

    while words and len(lines) < max_lines:
        word = words.pop(0)
        candidate = (
            f"{current} {word}".strip()
        )

        if (
            draw.textlength(
                candidate,
                font=font,
            )
            <= max_width
        ):
            current = candidate
            continue

        if current:
            lines.append(current)
            current = word
        else:
            shortened = word

            while (
                shortened
                and draw.textlength(
                    f"{shortened}…",
                    font=font,
                )
                > max_width
            ):
                shortened = shortened[:-1]

            lines.append(
                f"{shortened}…"
            )
            current = ""

    if (
        current
        and len(lines) < max_lines
    ):
        lines.append(current)

    if words and lines:
        last_line = lines[-1]

        while (
            last_line
            and draw.textlength(
                f"{last_line}…",
                font=font,
            )
            > max_width
        ):
            last_line = last_line[
                :-1
            ].rstrip()

        lines[-1] = f"{last_line}…"

    return lines


def shorten_summary(
    summary: str,
    max_characters: int = 300,
) -> str:
    """Create a compact dashboard summary."""
    summary = clean_text(summary)

    if not summary:
        return (
            "No summary is available "
            "for this story."
        )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        summary,
    )

    shortened = " ".join(
        sentences[:2]
    )

    if len(shortened) <= max_characters:
        return shortened

    shortened = shortened[
        :max_characters + 1
    ].rsplit(
        " ",
        1,
    )[0]

    return f"{shortened}…"


def format_publication_time(
    value: str,
) -> str:
    """Format a UTC article date for the dashboard."""
    try:
        published = datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00",
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return (
            "Publication time unavailable"
        )

    return (
        published
        .astimezone(LOCAL_TIMEZONE)
        .strftime("%b %d, %-I:%M %p %Z")
        .replace(" 0", " ")
    )


def draw_lines(
    draw,
    lines: list[str],
    position: tuple[int, int],
    font,
    fill: int,
    spacing: int,
) -> int:
    """Draw several text lines."""
    x, y = position

    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
        )
        y += spacing

    return y


def create_kindle_dashboard(
    stories: list[dict],
    output_path: Path = DASHBOARD_PATH,
) -> None:
    """Create the existing grayscale Kindle image."""
    output_path = Path(output_path)

    image = Image.new(
        "L",
        (
            DASHBOARD_WIDTH,
            DASHBOARD_HEIGHT,
        ),
        color=255,
    )

    draw = ImageDraw.Draw(image)

    fonts = {
        "masthead": load_font(
            "bold",
            58,
        ),
        "date": load_font(
            "regular",
            29,
        ),
        "metadata": load_font(
            "bold",
            22,
        ),
        "title": load_font(
            "bold",
            34,
        ),
        "summary": load_font(
            "regular",
            27,
        ),
        "footer": load_font(
            "regular",
            24,
        ),
    }

    now = datetime.now(
        timezone.utc
    ).astimezone(
        LOCAL_TIMEZONE
    )

    margin = 58
    content_width = (
        DASHBOARD_WIDTH
        - 2 * margin
    )
    footer_top = 1384

    draw.text(
        (margin, 42),
        "CURATED AI SOURCES",
        font=fonts["masthead"],
        fill=0,
    )

    display_date = (
        now.strftime(
            "%A, %B %d, %Y"
        ).replace(" 0", " ")
    )

    draw.text(
        (margin, 116),
        display_date,
        font=fonts["date"],
        fill=64,
    )

    draw.line(
        (
            margin,
            172,
            DASHBOARD_WIDTH - margin,
            172,
        ),
        fill=0,
        width=4,
    )

    visible_stories = stories[
        :DASHBOARD_STORIES
    ]

    if visible_stories:
        content_top = 188

        story_height = (
            footer_top - content_top
        ) // len(visible_stories)

        for index, article in enumerate(
            visible_stories
        ):
            story_top = (
                content_top
                + index * story_height
            )

            metadata = (
                f"{article['source'].upper()} • "
                f"{format_publication_time(article['published'])}"
            )

            draw.text(
                (
                    margin,
                    story_top + 8,
                ),
                metadata,
                font=fonts["metadata"],
                fill=72,
            )

            title_lines = fit_lines(
                draw,
                article["title"],
                fonts["title"],
                content_width,
                2,
            )

            title_bottom = draw_lines(
                draw,
                title_lines,
                (
                    margin,
                    story_top + 39,
                ),
                fonts["title"],
                0,
                40,
            )

            summary_source = (
                article.get(
                    "article_text"
                )
                or article.get(
                    "summary",
                    "",
                )
            )

            summary_lines = fit_lines(
                draw,
                shorten_summary(
                    summary_source
                ),
                fonts["summary"],
                content_width,
                3,
            )

            draw_lines(
                draw,
                summary_lines,
                (
                    margin,
                    title_bottom + 7,
                ),
                fonts["summary"],
                42,
                33,
            )

            if (
                index
                < len(visible_stories) - 1
            ):
                divider_y = (
                    story_top
                    + story_height
                    - 8
                )

                draw.line(
                    (
                        margin,
                        divider_y,
                        DASHBOARD_WIDTH
                        - margin,
                        divider_y,
                    ),
                    fill=190,
                    width=2,
                )

    else:
        message = (
            "No recent AI stories were found "
            "during the latest check."
        )

        message_lines = fit_lines(
            draw,
            message,
            fonts["title"],
            content_width,
            3,
        )

        draw_lines(
            draw,
            message_lines,
            (
                margin,
                240,
            ),
            fonts["title"],
            0,
            44,
        )

    draw.line(
        (
            margin,
            footer_top,
            DASHBOARD_WIDTH - margin,
            footer_top,
        ),
        fill=0,
        width=3,
    )

    updated_text = now.strftime(
        "Updated %-I:%M %p %Z"
    )

    draw.text(
        (
            margin,
            footer_top + 18,
        ),
        updated_text,
        font=fonts["footer"],
        fill=48,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=(
                f".{output_path.stem}."
            ),
            suffix=".tmp.png",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

        image.save(
            temporary_path,
            format="PNG",
            optimize=True,
        )

        with Image.open(
            temporary_path
        ) as validation_image:
            if validation_image.size != (
                DASHBOARD_WIDTH,
                DASHBOARD_HEIGHT,
            ):
                raise RuntimeError(
                    "Dashboard dimensions are incorrect."
                )

            if validation_image.mode != "L":
                raise RuntimeError(
                    "Dashboard is not grayscale."
                )

            validation_image.verify()

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if (
            temporary_path
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def write_feed_atomically(
    payload: list[dict],
    output_path: Path = FEED_PATH,
) -> None:
    """Replace feed.json only after validation succeeds."""
    temporary_path = output_path.with_suffix(
        ".json.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        output_path,
    )


def main() -> None:
    now = datetime.now(
        timezone.utc
    )

    articles, window_used = collect_articles(
        now=now
    )

    print(
        f"Collected {len(articles)} candidates "
        f"using a {window_used}-hour window."
    )

    if not articles:
        raise RuntimeError(
            "No eligible articles were collected. "
            "The existing feed will remain unchanged."
        )

    enriched_articles = enrich_articles(
        articles
    )

    selected_articles = select_stories(
        enriched_articles
    )

    print(
        f"Selected {len(selected_articles)} stories "
        f"from "
        f"{len({article['source'] for article in selected_articles})} "
        f"publishers."
    )

    summary = create_summary(
        selected_articles,
        now=now,
    )

    covered_urls = load_covered_urls()

    validate_briefing(
        summary=summary,
        selected_articles=selected_articles,
        all_articles=enriched_articles,
        covered_urls=covered_urls,
        now=now,
    )

    feed_payload = build_feed_payload(
        summary=summary,
        now=now,
    )

    validate_feed_payload(
        feed_payload
    )

    # Live outputs change only after validation passes.
    write_sources_page(
        selected_articles,
        now=now,
    )

    create_kindle_dashboard(
        selected_articles
    )

    write_feed_atomically(
        feed_payload
    )

    update_history(
        selected_articles,
        now=now,
    )

    print(
        f"Briefing word count: "
        f"{len(summary.split())}"
    )

    print(
        "feed.json, sources.html, "
        "kindle-dashboard.png, and "
        "covered_stories.json were updated."
    )


if __name__ == "__main__":
    main()
