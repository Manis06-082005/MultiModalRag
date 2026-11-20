"""Small Tavily client used only when the user enables web search."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TavilyWebSearch:
    """Fetch a few web results and format them as source passages."""

    endpoint = "https://api.tavily.com/search"

    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, str]]:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is missing. Add it to .env and restart the app.")
        payload = json.dumps({
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Web search failed ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach Tavily web search: {error.reason}") from error

        passages: list[dict[str, str]] = []
        for result in data.get("results", []):
            title = str(result.get("title") or result.get("url") or "Web result").strip()
            url = str(result.get("url") or "").strip()
            content = str(result.get("content") or "").strip()
            if not url or not content:
                continue
            passages.append({"source": title, "url": url, "text": content[:4000]})
        return passages
