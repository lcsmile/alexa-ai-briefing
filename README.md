# alexa-ai-briefing

Creates a daily spoken AI-news briefing and Kindle dashboard for Alexa.

## How discovery works

The workflow searches the web through Brave News Search, then fetches the
matching pages from an allowlist of AI-news domains. RSS feeds remain as a
fallback when `BRAVE_SEARCH_API_KEY` is unavailable or search returns no
usable results.

Configure these GitHub Actions secrets:

- `GEMINI_API_KEY` — required for ranking, summarization, and dashboard selection.
- `BRAVE_SEARCH_API_KEY` — recommended for website-based discovery.

The generated Alexa item is kept below the platform's 4,500-character text
limit and written atomically only after the content has been generated.
