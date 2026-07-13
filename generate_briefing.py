import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import feedparser
from google import genai

FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Microsoft AI": "https://blogs.microsoft.com/ai/feed/",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
}

LOOKBACK_HOURS = 168
MAX_ARTICLES = 20


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


def main():
    articles = collect_articles()
    print(f"Found {len(articles)} recent articles.")

    summary = create_summary(articles)
    create_alexa_feed(summary, articles)

    print("feed.json was created successfully.")


if __name__ == "__main__":
    main()
