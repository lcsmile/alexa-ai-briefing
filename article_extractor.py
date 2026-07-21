import re

import requests
import trafilatura
from bs4 import BeautifulSoup


REQUEST_TIMEOUT_SECONDS = 12
MAX_ARTICLE_CHARACTERS = 3000

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 compatible; "
        "AlexaAIBriefing/1.0; "
        "article summary generator"
    )
}


def clean_text(text):
    """Normalize whitespace in extracted article text."""
    return re.sub(r"\s+", " ", text or "").strip()


def extract_with_trafilatura(html, url):
    """Extract the main article content through Trafilatura."""
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        include_links=False,
        include_images=False,
        favor_precision=True,
        deduplicate=True,
    )

    return clean_text(extracted)


def extract_with_beautifulsoup(html):
    """Use a simpler HTML extraction method as a secondary fallback."""
    soup = BeautifulSoup(html, "html.parser")

    for unwanted in soup(
        [
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
    ):
        unwanted.decompose()

    article_element = soup.find("article")

    if article_element:
        text = article_element.get_text(" ", strip=True)
    else:
        main_element = soup.find("main")
        target = main_element or soup.body or soup
        text = target.get_text(" ", strip=True)

    return clean_text(text)


def download_article(url):
    """Download an article with a timeout and basic validation."""
    response = requests.get(
        url,
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
            f"URL did not return HTML: {content_type}"
        )

    return response.text, response.url


def extract_article_text(article):
    """
    Add article_text and extraction_method to one article dictionary.

    RSS summary text remains available when page extraction fails.
    """
    article = dict(article)
    fallback = clean_text(article.get("summary", ""))

    try:
        html, final_url = download_article(article["link"])

        text = extract_with_trafilatura(
            html,
            final_url,
        )

        method = "trafilatura"

        if len(text) < 300:
            text = extract_with_beautifulsoup(html)
            method = "beautifulsoup"

        if len(text) < 300:
            raise ValueError(
                "Extracted article text was too short."
            )

        article["article_text"] = text[
            :MAX_ARTICLE_CHARACTERS
        ]
        article["extraction_method"] = method

        print(
            f"Extracted {len(article['article_text'])} characters "
            f"from {article['source']}: {article['title']}"
        )

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        article["article_text"] = fallback[
            :MAX_ARTICLE_CHARACTERS
        ]
        article["extraction_method"] = "rss-fallback"

        print(
            f"Article extraction failed for "
            f"{article.get('title', 'unknown article')}: {error}"
        )

    return article


def enrich_articles(articles):
    """Extract full text for every collected candidate."""
    enriched = []

    for article in articles:
        enriched.append(extract_article_text(article))

    return enriched
