from __future__ import annotations

from urllib.parse import quote_plus

import requests


SEARCH_URL = "https://openlibrary.org/search.json"
COVER_URL = "https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"


def cover_url_from_id(cover_id: int | float | str | None) -> str:
    if cover_id is None or cover_id == "":
        return ""
    try:
        return COVER_URL.format(cover_id=int(cover_id))
    except (TypeError, ValueError):
        return ""


def search_books(query: str, limit: int = 12) -> list[dict[str, object]]:
    query = query.strip()
    if not query:
        return []
    response = requests.get(
        SEARCH_URL,
        params={
            "q": query,
            "limit": limit,
            "fields": "key,title,author_name,first_publish_year,cover_i,subject,ratings_average,ratings_count",
        },
        timeout=8,
    )
    response.raise_for_status()
    results = []
    for doc in response.json().get("docs", []):
        authors = doc.get("author_name") or ["Unknown author"]
        subjects = doc.get("subject") or []
        results.append(
            {
                "title": doc.get("title", "Untitled"),
                "author": ", ".join(authors[:2]),
                "year": doc.get("first_publish_year", ""),
                "cover_url": cover_url_from_id(doc.get("cover_i")),
                "subjects": ", ".join(subjects[:4]),
                "rating": doc.get("ratings_average"),
                "ratings_count": doc.get("ratings_count"),
                "open_library_url": f"https://openlibrary.org{doc.get('key', '')}",
            }
        )
    return results


def find_cover(title: str, author: str) -> str:
    query = quote_plus(f"{title} {author}")
    try:
        results = search_books(query, limit=1)
    except requests.RequestException:
        return ""
    if not results:
        return ""
    return str(results[0].get("cover_url") or "")

