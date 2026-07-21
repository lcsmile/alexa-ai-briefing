import re

import requests
import trafilatura
from bs4 import BeautifulSoup


REQUEST_TIMEOUT_SECONDS = 12
MAX_ARTICLE_CHARACTERS = 4000
MIN_EXTRACTED_CHARACTERS = 300

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; CuratedAIBriefing/1.0; "
        "+https://lcsmile.github.io/"
        "alexa-ai-briefing/)"
    )
}


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def extract_with_beautifulsoup(
    html_content: str,
) -> str:
    """Use HTML structure as a secondary extraction method."""
    soup = BeautifulSoup(
        html_content,
        "html.parser",
    )

    unwanted_tags = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "noscript",
        "svg",
    ]

    for tag in soup(unwanted_tags):
        tag.decompose()

    target = (
        soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    return clean_text(
        target.get_text(
            " ",
            strip=True,
        )
    )


def extract_article_text(
    article: dict,
) -> dict:
    """
    Extract one article's main text.

    The RSS description is used when page extraction fails.
    """
    enriched_article = dict(article)

    fallback_text = clean_text(
        article.get("summary", "")
    )

    try:
        response = requests.get(
            article["link"],
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).casefold()

        if "html" not in content_type:
            raise ValueError(
                f"Response was not HTML: {content_type}"
            )

        extracted_text = trafilatura.extract(
            response.text,
            url=response.url,
            include_comments=False,
            include_tables=False,
            include_links=False,
            include_images=False,
            favor_precision=True,
            deduplicate=True,
        )

        extracted_text = clean_text(
            extracted_text
        )
        extraction_method = "trafilatura"

        if (
            len(extracted_text)
            < MIN_EXTRACTED_CHARACTERS
        ):
            extracted_text = (
                extract_with_beautifulsoup(
                    response.text
                )
            )
            extraction_method = "beautifulsoup"

        if (
            len(extracted_text)
            < MIN_EXTRACTED_CHARACTERS
        ):
            raise ValueError(
                "Extracted article text was too short."
            )

        enriched_article["article_text"] = (
            extracted_text[
                :MAX_ARTICLE_CHARACTERS
            ]
        )
        enriched_article["extraction_method"] = (
            extraction_method
        )

        print(
            f"Extracted "
            f"{len(enriched_article['article_text'])} "
            f"characters from "
            f"{article['source']}: "
            f"{article['title']}"
        )

    except (
        requests.RequestException,
        ValueError,
        KeyError,
    ) as error:
        enriched_article["article_text"] = (
            fallback_text[
                :MAX_ARTICLE_CHARACTERS
            ]
        )
        enriched_article["extraction_method"] = (
            "rss-fallback"
        )

        print(
            f"Used RSS fallback for "
            f"{article.get('title', 'unknown article')}: "
            f"{error}"
        )

    return enriched_article


def enrich_articles(
    articles: list[dict],
) -> list[dict]:
    """Extract page content for all collected candidates."""
    enriched_articles = []

    for article in articles:
        enriched_articles.append(
            extract_article_text(article)
        )

    return enriched_articles
