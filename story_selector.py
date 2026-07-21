import json
import os
from collections import Counter

from google import genai


MODEL_NAME = "gemini-3.1-flash-lite"

TARGET_STORIES = 8
MIN_ORGANIZATIONS = 4
MAX_PER_ORGANIZATION = 2


def prepare_candidates(
    articles: list[dict],
) -> str:
    """Convert candidate articles into text for Gemini."""
    blocks = []

    for index, article in enumerate(articles):
        blocks.append(
            f"Article index: {index}\n"
            f"Publisher: {article['source']}\n"
            f"Publisher type: "
            f"{article['source_type']}\n"
            f"Category: {article['category']}\n"
            f"Title: {article['title']}\n"
            f"Published: {article['published']}\n"
            f"Age label: {article['age_label']}\n"
            f"Content: "
            f"{article['article_text'][:2500]}"
        )

    return "\n\n".join(blocks)


def deterministic_fallback(
    articles: list[dict],
    count: int = TARGET_STORIES,
) -> list[dict]:
    """
    Select stories without Gemini.

    Articles are already sorted by publication date.
    """
    selected = []
    publisher_counts = Counter()

    target_count = min(
        count,
        len(articles),
    )

    for article in articles:
        publisher = article["source"]

        if (
            publisher_counts[publisher]
            >= MAX_PER_ORGANIZATION
        ):
            continue

        selected.append(article)
        publisher_counts[publisher] += 1

        if len(selected) >= target_count:
            break

    editorial_available = any(
        article["source_type"] == "editorial"
        for article in articles
    )

    editorial_selected = any(
        article["source_type"] == "editorial"
        for article in selected
    )

    if (
        editorial_available
        and not editorial_selected
        and selected
    ):
        replacement = next(
            (
                article
                for article in articles
                if (
                    article["source_type"]
                    == "editorial"
                    and article not in selected
                )
            ),
            None,
        )

        if replacement is not None:
            selected[-1] = replacement

    return selected


def validate_selection(
    payload: dict,
    articles: list[dict],
) -> list[dict]:
    """Validate indexes returned by Gemini."""
    if not isinstance(payload, dict):
        raise ValueError(
            "Gemini selection must be a JSON object."
        )

    selection_items = payload.get(
        "selected_stories"
    )

    if not isinstance(selection_items, list):
        raise ValueError(
            "selected_stories must be a list."
        )

    expected_count = min(
        TARGET_STORIES,
        len(articles),
    )

    selected_indexes = []

    for item in selection_items:
        if not isinstance(item, dict):
            raise ValueError(
                "Each selection must be an object."
            )

        index = item.get("article_index")

        if (
            isinstance(index, bool)
            or not isinstance(index, int)
        ):
            raise ValueError(
                "article_index must be an integer."
            )

        if index < 0 or index >= len(articles):
            raise ValueError(
                "article_index is outside the candidate list."
            )

        selected_indexes.append(index)

    if len(selected_indexes) != expected_count:
        raise ValueError(
            f"Expected {expected_count} selections, "
            f"received {len(selected_indexes)}."
        )

    if len(set(selected_indexes)) != len(
        selected_indexes
    ):
        raise ValueError(
            "Gemini selected a duplicate article."
        )

    selected_articles = [
        articles[index]
        for index in selected_indexes
    ]

    publisher_counts = Counter(
        article["source"]
        for article in selected_articles
    )

    if any(
        count > MAX_PER_ORGANIZATION
        for count in publisher_counts.values()
    ):
        raise ValueError(
            "A publisher exceeds the two-story limit."
        )

    available_publishers = len(
        {
            article["source"]
            for article in articles
        }
    )

    required_publishers = min(
        MIN_ORGANIZATIONS,
        available_publishers,
        expected_count,
    )

    if (
        len(publisher_counts)
        < required_publishers
    ):
        raise ValueError(
            "Selection does not use enough publishers."
        )

    editorial_available = any(
        article["source_type"] == "editorial"
        for article in articles
    )

    editorial_selected = any(
        article["source_type"] == "editorial"
        for article in selected_articles
    )

    if (
        editorial_available
        and not editorial_selected
    ):
        raise ValueError(
            "An editorial source was available "
            "but was not selected."
        )

    return selected_articles


def select_stories(
    articles: list[dict],
    client=None,
) -> list[dict]:
    """Use Gemini to rank and select the briefing stories."""
    if not articles:
        return []

    if len(articles) <= TARGET_STORIES:
        return deterministic_fallback(articles)

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

    response_schema = {
        "type": "object",
        "properties": {
            "selected_stories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_index": {
                            "type": "integer"
                        },
                        "importance_score": {
                            "type": "integer"
                        },
                        "topic": {
                            "type": "string"
                        },
                        "reason": {
                            "type": "string"
                        },
                    },
                    "required": [
                        "article_index",
                        "importance_score",
                        "topic",
                        "reason",
                    ],
                },
            }
        },
        "required": [
            "selected_stories"
        ],
    }

    prompt = f"""
Select exactly {min(TARGET_STORIES, len(articles))}
stories for a spoken artificial-intelligence news
briefing.

Rules:
- Use at least four publishers when four are available.
- Use no more than two stories from one publisher.
- Include at least one independent editorial source
  when one is available.
- Prefer important model releases, research, safety,
  regulation, infrastructure, major business changes,
  and widely used product launches.
- Reject duplicate reports about the same event.
- Reject routine marketing material, tutorials,
  webinars, customer promotions, and minor updates.
- Rank stories by importance, not only publication time.
- Return only JSON matching the required schema.

Candidate articles:

{prepare_candidates(articles)}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type":
                    "application/json",
                "response_json_schema":
                    response_schema,
            },
        )

        payload = json.loads(response.text)

        return validate_selection(
            payload,
            articles,
        )

    except Exception as error:
        print(
            "Gemini story selection failed. "
            "Using deterministic fallback. "
            f"Error: {error}"
        )

        return deterministic_fallback(
            articles
        )
