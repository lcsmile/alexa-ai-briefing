import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser
from google import genai

# Official company websites to monitor through Google News RSS.
SOURCES = {
    "OpenAI": "openai.com",
    "Anthropic": "anthropic.com",
    "Google DeepMind": "deepmind.google",
    "Microsoft AI": "blogs.microsoft.com",
    "Meta AI": "ai.meta.com",
    "Hugging Face": "huggingface.co",
}

LOOKBACK_HOURS = 36
MAX_ARTICLES = 20


def clean_text(text):
    """Remove HTML and extra spaces."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def collect_articles():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    articles = []
    seen_titles = set()

    for source_name, domain in SOURCES.items():
        query = quote_plus(f"site:{domain} artificial intelligence")
        feed_url = (
            "https://news.google.com/rss/search"
            f"?q={query}&hl=en-US&gl=US&ceid=US:en"
        )

        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:10]:
            published = None

            if getattr(entry, "published_parsed", None):
                published = datetime(
                    *entry.published_parsed[:6],
                    tzinfo=timezone.utc,
                )

            if published and published < cutoff:
                continue

            title = clean_text(getattr(entry, "title", ""))

            # Google News sometimes adds the publisher after a dash.
            title = re.sub(r"\s+-\s+[^-]+$", "", title).strip()

            if not title or title.lower() in seen_titles:
                continue

            seen_titles.add(title.lower())

            articles.append(
                {
                    "source": source_name,
                    "title": title,
                    "summary": clean_text(
                        getattr(entry, "summary", "")
                    )[:500],
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
            "Good morning. No major updates were found from the selected "
            "artificial intelligence company sources during the latest check."
        )

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found.")

    client = genai.Client(api_key=api_key)

    article_text = "\n\n".join(
        [
            (
                f"Source: {article['source']}\n"
                f"Title: {article['title']}\n"
                f"Published: {article['published']}\n"
                f"Description: {article['summary']}"
            )
            for article in articles
        ]
    )

    prompt = f"""
Create a spoken morning briefing about the most important artificial
intelligence developments in the material below.

Requirements:
- Use only the supplied material.
- Select no more than five important developments.
- Remove duplicate stories.
- Mention the company or source for each development.
- Use natural spoken English.
- Do not use bullet symbols, markdown, URLs, headings or abbreviations.
- Explain technical terms briefly.
- Keep the briefing between 350 and 500 words.
- Start with: Good morning. Here is your AI briefing.
- End with: That is your AI briefing for today.
- Do not claim that an article was published within the past 24 hours unless
  its supplied date confirms that.

Material:
{article_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    summary = clean_text(response.text)

    if not summary:
        raise RuntimeError("Gemini returned an empty summary.")

    return summary


def create_alexa_feed(summary, articles):
    now = datetime.now(timezone.utc)
    source_link = articles[0]["link"] if articles else "https://news.google.com/"

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
