"""Search-driven article discovery with a trusted-domain allowlist."""

import os
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/news/search"
SEARCH_TIMEOUT_SECONDS = 10
MAX_RESULTS_PER_QUERY = 6
MAX_ARTICLES = 25

SEARCH_QUERIES = (
    "artificial intelligence model release",
    "artificial intelligence research safety",
    "artificial intelligence regulation policy",
    "artificial intelligence infrastructure chips",
    "artificial intelligence major product business",
)

ALLOWED_DOMAINS = {
    "openai.com",
    "deepmind.google",
    "microsoft.com",
    "huggingface.co",
    "ai.meta.com",
    "anthropic.com",
    "techcrunch.com",
    "technologyreview.com",
    "arstechnica.com",
    "theregister.com",
}


def normalize_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def clean_text(value: str) -> str:
    return " ".join((value or "").split())


def allowed_domain(hostname: str) -> bool:
    hostname = (hostname or "").casefold().removeprefix("www.")
    return any(hostname == domain or hostname.endswith("." + domain) for domain in ALLOWED_DOMAINS)


def fetch_page_context(url: str) -> str:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "alexa-ai-briefing/1.0"},
            timeout=SEARCH_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        response.raise_for_status()
        if "html" not in response.headers.get("Content-Type", "").casefold():
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        description = soup.find("meta", attrs={"name": "description"})
        paragraphs = " ".join(
            clean_text(node.get_text(" ", strip=True))
            for node in soup.select("article p, main p, p")[:12]
        )
        return clean_text(
            (description.get("content", "") if description else "")
            + " " + paragraphs
        )[:1200]
    except requests.RequestException as error:
        print(f"Search result fetch failed for {url}: {error}")
        return ""


def search_articles() -> list[dict]:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        print("BRAVE_SEARCH_API_KEY is not set; using RSS fallback.")
        return []

    articles = []
    seen_urls = set()
    seen_titles = set()
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}

    for query in SEARCH_QUERIES:
        try:
            response = requests.get(
                BRAVE_SEARCH_URL,
                headers=headers,
                params={
                    "q": query,
                    "freshness": "pd",
                    "count": MAX_RESULTS_PER_QUERY,
                    "extra_snippets": "true",
                },
                timeout=SEARCH_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (requests.RequestException, ValueError) as error:
            print(f"Search failed for '{query}': {error}")
            continue

        for result in results:
            url = normalize_url(result.get("url"))
            title = clean_text(result.get("title", ""))
            hostname = (result.get("meta_url") or {}).get("hostname", "")
            if (
                not url
                or not title
                or url in seen_urls
                or title.casefold() in seen_titles
                or not allowed_domain(hostname)
            ):
                continue

            articles.append(
                {
                    "source": hostname.removeprefix("www.") or "Web search",
                    "title": title,
                    "summary": fetch_page_context(url) or clean_text(result.get("description", "")),
                    "link": url,
                    "published": datetime.now(timezone.utc).isoformat(),
                }
            )
            seen_urls.add(url)
            seen_titles.add(title.casefold())
            if len(articles) >= MAX_ARTICLES:
                return articles

    print(f"Search discovery found {len(articles)} articles.")
    return articles
