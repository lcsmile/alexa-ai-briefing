import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
from google import genai
from PIL import Image, ImageDraw, ImageFont

FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Microsoft AI": "https://blogs.microsoft.com/ai/feed/",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
}

LOOKBACK_HOURS = 168
MAX_ARTICLES = 20
DASHBOARD_WIDTH = 1072
DASHBOARD_HEIGHT = 1448
DASHBOARD_STORIES = 5
DASHBOARD_TIMEZONE = ZoneInfo("America/Los_Angeles")
DASHBOARD_PATH = Path("kindle-dashboard.png")
FONT_PATHS = {
    "regular": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ),
    "bold": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
}


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def collect_articles():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    articles = []
    seen_titles = set()

    for source_name, feed_url in FEEDS.items():
        feed = feedparser.parse(feed_url)

        print(
            f"{source_name}: status={getattr(feed, 'status', 'unknown')}, "
            f"entries={len(feed.entries)}"
        )

        for entry in feed.entries[:15]:
            published = None

            if getattr(entry, "published_parsed", None):
                published = datetime(
                    *entry.published_parsed[:6],
                    tzinfo=timezone.utc,
                )
            elif getattr(entry, "updated_parsed", None):
                published = datetime(
                    *entry.updated_parsed[:6],
                    tzinfo=timezone.utc,
                )

            if published and published < cutoff:
                continue

            title = clean_text(getattr(entry, "title", ""))

            if not title or title.lower() in seen_titles:
                continue

            seen_titles.add(title.lower())

            articles.append(
                {
                    "source": source_name,
                    "title": title,
                    "summary": clean_text(
                        getattr(entry, "summary", "")
                    )[:600],
                    "link": getattr(entry, "link", ""),
                    "published": (
                        published.isoformat()
                        if published
                        else "Publication time unavailable"
                    ),
                }
            )

    articles.sort(key=lambda item: item["published"], reverse=True)
    return articles[:MAX_ARTICLES]


def create_summary(articles):
    if not articles:
        return (
            "Good morning. No new articles were found from the selected "
            "artificial intelligence sources during the latest check."
        )

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found.")

    client = genai.Client(api_key=api_key)

    article_text = "\n\n".join(
        (
            f"Source: {article['source']}\n"
            f"Title: {article['title']}\n"
            f"Published: {article['published']}\n"
            f"Description: {article['summary']}"
        )
        for article in articles
    )

    prompt = f"""
Create a spoken morning briefing based only on the supplied articles.

Requirements:
- Select up to five important AI developments.
- Remove duplicate stories.
- Mention the source company.
- Use natural spoken English.
- Do not use bullets, markdown, URLs or headings.
- Keep it between 250 and 400 words.
- Start with: Good morning. Here is your AI briefing.
- End with: That is your AI briefing for today.

Articles:
{article_text}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    summary = clean_text(response.text)

    if not summary:
        raise RuntimeError("Gemini returned an empty summary.")

    return summary


def create_alexa_feed(summary, articles):
    now = datetime.now(timezone.utc)
    source_link = articles[0]["link"] if articles else "https://openai.com/news/"

    feed = [
        {
            "uid": str(uuid.uuid4()),
            "updateDate": now.strftime("%Y-%m-%dT%H:%M:%S.0Z"),
            "titleText": "Daily AI Briefing",
            "mainText": summary,
            "redirectionUrl": source_link,
        }
    ]

    with open("feed.json", "w", encoding="utf-8") as file:
        json.dump(feed, file, ensure_ascii=False, indent=2)


def load_font(style, size):
    for font_path in FONT_PATHS[style]:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)

    raise RuntimeError(
        f"No {style} dashboard font was found. Install DejaVu Sans."
    )


def fit_lines(draw, text, font, max_width, max_lines):
    words = clean_text(text).split()
    lines = []
    current = ""

    while words and len(lines) < max_lines:
        word = words.pop(0)
        candidate = f"{current} {word}".strip()

        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = word
        else:
            shortened = word
            while (
                shortened
                and draw.textlength(f"{shortened}…", font=font) > max_width
            ):
                shortened = shortened[:-1]
            lines.append(f"{shortened}…")
            current = ""

    if current and len(lines) < max_lines:
        lines.append(current)

    if words or current and len(lines) == max_lines:
        last_line = lines[-1]
        while (
            last_line
            and draw.textlength(f"{last_line}…", font=font) > max_width
        ):
            last_line = last_line[:-1].rstrip()
        lines[-1] = f"{last_line}…"

    return lines


def shorten_summary(summary, max_characters=260):
    summary = clean_text(summary)

    if not summary:
        return "No summary is available for this story."

    sentences = re.split(r"(?<=[.!?])\s+", summary)
    shortened = " ".join(sentences[:2])

    if len(shortened) <= max_characters:
        return shortened

    shortened = shortened[: max_characters + 1].rsplit(" ", 1)[0]
    return f"{shortened}…"


def format_publication_time(value):
    if value == "Publication time unavailable":
        return value

    try:
        published = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "Publication time unavailable"

    return published.astimezone(DASHBOARD_TIMEZONE).strftime(
        "%b %d, %-I:%M %p %Z"
    ).replace(" 0", " ")


def draw_lines(draw, lines, position, font, fill, spacing):
    x, y = position

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += spacing

    return y


def create_kindle_dashboard(articles, output_path=DASHBOARD_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (DASHBOARD_WIDTH, DASHBOARD_HEIGHT), color=255)
    draw = ImageDraw.Draw(image)
    fonts = {
        "masthead": load_font("bold", 58),
        "date": load_font("regular", 29),
        "metadata": load_font("bold", 22),
        "title": load_font("bold", 34),
        "summary": load_font("regular", 27),
        "footer": load_font("regular", 24),
    }

    now = datetime.now(timezone.utc).astimezone(DASHBOARD_TIMEZONE)
    margin = 58
    content_width = DASHBOARD_WIDTH - 2 * margin
    footer_top = 1384

    draw.text((margin, 42), "AI NEWS BRIEFING", font=fonts["masthead"], fill=0)
    display_date = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
    draw.text((margin, 116), display_date, font=fonts["date"], fill=64)
    draw.line((margin, 172, DASHBOARD_WIDTH - margin, 172), fill=0, width=4)

    stories = articles[:DASHBOARD_STORIES]

    if stories:
        content_top = 188
        story_height = (footer_top - content_top) // len(stories)

        for index, article in enumerate(stories):
            story_top = content_top + index * story_height
            metadata = (
                f"{article['source'].upper()}  •  "
                f"{format_publication_time(article['published'])}"
            )
            draw.text(
                (margin, story_top + 8),
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
            summary_lines = fit_lines(
                draw,
                shorten_summary(article["summary"]),
                fonts["summary"],
                content_width,
                3,
            )
            title_bottom = draw_lines(
                draw,
                title_lines,
                (margin, story_top + 39),
                fonts["title"],
                0,
                40,
            )
            draw_lines(
                draw,
                summary_lines,
                (margin, title_bottom + 7),
                fonts["summary"],
                42,
                33,
            )

            if index < len(stories) - 1:
                divider_y = story_top + story_height - 8
                draw.line(
                    (margin, divider_y, DASHBOARD_WIDTH - margin, divider_y),
                    fill=190,
                    width=2,
                )
    else:
        message = "No recent AI stories were found during the latest check."
        lines = fit_lines(
            draw,
            message,
            fonts["title"],
            content_width,
            3,
        )
        draw_lines(draw, lines, (margin, 240), fonts["title"], 0, 44)

    draw.line(
        (margin, footer_top, DASHBOARD_WIDTH - margin, footer_top),
        fill=0,
        width=3,
    )
    updated = now.strftime("Updated %-I:%M %p %Z")
    draw.text((margin, footer_top + 18), updated, font=fonts["footer"], fill=48)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".tmp.png",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        image.save(temporary_path, format="PNG", optimize=True)

        with Image.open(temporary_path) as validation_image:
            if validation_image.size != (DASHBOARD_WIDTH, DASHBOARD_HEIGHT):
                raise RuntimeError("Dashboard PNG has incorrect dimensions.")
            if validation_image.mode != "L":
                raise RuntimeError("Dashboard PNG is not 8-bit grayscale.")
            validation_image.verify()

        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def main():
    articles = collect_articles()
    print(f"Found {len(articles)} recent articles.")

    summary = create_summary(articles)
    create_alexa_feed(summary, articles)
    create_kindle_dashboard(articles)

    print("feed.json and kindle-dashboard.png were created successfully.")


if __name__ == "__main__":
    main()
