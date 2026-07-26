import json
import os
from collections import Counter

from google import genai


MODEL_NAME = "gemini-3.1-flash-lite"

TARGET_STORIES = 8
MIN_STORIES = 4
MIN_ORGANIZATIONS = 3
MAX_PER_ORGANIZATION = 2


def feasible_story_count(articles: list[dict]) -> int:
    """
    Calculate how many stories can actually be selected while respecting
    the two-story publisher limit.
    """
    publisher_counts = Counter(
        article["source"]
        for article in articles
    )

    maximum_under_cap = sum(
        min(count, MAX_PER_ORGANIZATION)
        for count in publisher_counts.values()
    )

    return min(
        TARGET_STORIES,
        maximum_under_cap,
        len(articles),
    )


def required_organization_count(
    articles: list[dict],
    selection_count: int,
) -> int:
    """
    Require source diversity only when the available candidates support it.
    """
    available_publishers = len(
        {
            article["source"]
            for article in articles
        }
    )

    return min(
        MIN_ORGANIZATIONS,
        available_publishers,
        selection_count,
    )


def prepare_candidates(
    articles: list[dict],
) -> str:
    """Convert candidate articles into text for Gemini."""
    blocks = []

    for index, article in enumerate(articles):
        blocks.append(
            f"Article index: {index}\n"
            f"Publisher: {article['source']}\n"
            f"Publisher type: {article['source_type']}\n"
            f"Category: {article['category']}\n"
            f"Title: {article['title']}\n"
            f"Published: {article['published']}\n"
            f"Age label: {article['age_label']}\n"
            f"Content: {article['article_text'][:2500]}"
        )

    return "\n\n".join(blocks)


def deterministic_fallback(
    articles: list[dict],
) -> list[dict]:
    """
    Select the best feasible number of stories without Gemini.

    The article list is expected to be ordered by recency.
    """
    target_count = feasible_story_count(articles)

    selected = []
    publisher_counts = Counter()

    # First pass: maximize publisher diversity.
    publishers_used = set()

    for article in articles:
        publisher = article["source"]

        if publisher in publishers_used:
            continue

        selected.append(article)
        publishers_used.add(publisher)
        publisher_counts[publisher] += 1

        if len(selected) >= target_count:
            break

    # Second pass: fill remaining positions while keeping the two-story cap.
    if len(selected) < target_count:
        for article in articles:
            if article in selected:
                continue

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

    # Ensure an editorial item is included when possible.
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
                    article["source_type"] == "editorial"
                    and article not in selected
                )
            ),
            None,
        )

        if replacement is not None:
            replace_index = next(
                (
                    index
                    for index in range(
                        len(selected) - 1,
                        -1,
                        -1,
                    )
                    if (
                        publisher_counts[
                            selected[index]["source"]
                        ]
                        > 1
                    )
                ),
                len(selected) - 1,
            )

            removed = selected[replace_index]
            publisher_counts[
                removed["source"]
            ] -= 1

            selected[replace_index] = replacement
            publisher_counts[
                replacement["source"]
            ] += 1

    return selected


def validate_selection(
    payload: dict,
    articles: list[dict],
) -> list[dict]:
    """Validate the indexes returned by Gemini."""
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

    expected_count = feasible_story_count(
        articles
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

    if (
        len(set(selected_indexes))
        != len(selected_indexes)
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

    required_publishers = (
        required_organization_count(
            articles,
            expected_count,
        )
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
    """Use Gemini to rank and select a feasible set of stories."""
    if not articles:
        return []

    target_count = feasible_story_count(
        articles
    )

    if target_count < MIN_STORIES:
        print(
            f"Only {target_count} stories can satisfy "
            "the publisher limits."
        )

    if len(articles) <= target_count:
        return deterministic_fallback(
            articles
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

    required_publishers = (
        required_organization_count(
            articles,
            target_count,
        )
    )

    prompt = f"""
Select exactly {target_count} stories for a spoken
artificial-intelligence news briefing.

Rules:
- Use at least {required_publishers} publishers.
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

        payload = json.loads(
            response.text
        )

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
