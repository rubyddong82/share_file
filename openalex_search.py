"""
OpenAlex Search — citation lookup for arXiv-only papers.

No API key required. Providing your email gets you into the "polite pool"
for higher rate limits (100 req/s vs 10 req/s).

Usage:
  export OPENALEX_EMAIL="your@email.com"   # optional but recommended
"""

import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL       = "https://api.openalex.org"
OA_MAX_WORKERS = 10   # safe within 100 req/s polite-pool limit


OPENALEX_EMAIL = "sc21d2k@leeds.ac.uk"


def _get(endpoint: str, params: dict, retries: int = 3) -> dict:
    email = os.environ.get("OPENALEX_EMAIL", OPENALEX_EMAIL)
    if email:
        params = {**params, "mailto": email}

    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"  [OpenAlex] Rate limited. Waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                resp.raise_for_status()
        except requests.exceptions.Timeout:
            print(f"  [OpenAlex] Timeout (attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                time.sleep(2)
    raise RuntimeError(f"[OpenAlex] Failed after {retries} retries for {url}")


def _fetch_one_citation(arxiv_id: str) -> tuple[str, dict | None]:
    """
    Fetch citation data for a single arXiv ID from OpenAlex.
    Returns (arxiv_id, result_dict) or (arxiv_id, None) if not found.
    """
    params = {
        "filter": f"locations.landing_page_url:https://arxiv.org/abs/{arxiv_id}",
        "select": "id,title,cited_by_count,ids",
        "per_page": 1,
    }
    try:
        data = _get("works", params)
        for work in data.get("results", []):
            work_ids  = work.get("ids") or {}
            arxiv_url = work_ids.get("arxiv")
            if not arxiv_url:
                continue
            found_id = arxiv_url.split("/abs/")[-1].split("v")[0]
            return arxiv_id, {
                "citations":   work.get("cited_by_count", 0),
                "openalex_id": work.get("id"),
                "title":       work.get("title"),
            }
    except Exception:
        pass
    return arxiv_id, None


def batch_get_citations(arxiv_ids: list[str]) -> dict[str, dict]:
    """
    Fetch citation counts for arXiv IDs from OpenAlex in parallel.
    Used as last-resort fallback for papers not found in S2 batch lookup.

    Fires up to OA_MAX_WORKERS=10 concurrent requests — well within the
    100 req/s polite-pool limit (requires mailto in params, set via OPENALEX_EMAIL).

    OpenAlex does not support ids.arxiv filter, so we use
    locations.landing_page_url per paper (one request each).
    """
    if not arxiv_ids:
        return {}

    results = {}
    with ThreadPoolExecutor(max_workers=OA_MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_one_citation, aid): aid for aid in arxiv_ids}
        for future in as_completed(futures):
            aid, data = future.result()
            if data:
                results[aid] = data

    return results


def search_papers(
    query: str,
    filters: dict,
    pool_size: int = 50,
) -> list[dict]:
    """
    Search OpenAlex directly (used as an optional supplementary source).
    Returns standardized paper dicts.

    filters keys:
      year_from        : int
      year_to          : int
      open_access_only : bool
    """
    params = {
        "search":   query,
        "select":   "id,title,authorships,publication_year,primary_location,cited_by_count,ids,open_access,abstract_inverted_index",
        "per_page": min(pool_size, 200),
        "sort":     "cited_by_count:desc",
    }

    filter_parts = ["type:article"]
    year_from = filters.get("year_from")
    year_to   = filters.get("year_to")
    if year_from:
        filter_parts.append(f"publication_year:>{year_from - 1}")
    if year_to:
        filter_parts.append(f"publication_year:<{year_to + 1}")
    if filters.get("open_access_only"):
        filter_parts.append("is_oa:true")

    params["filter"] = ",".join(filter_parts)

    data = _get("works", params)
    results = []
    for work in data.get("results", []):
        work_ids = work.get("ids") or {}
        arxiv_url = work_ids.get("arxiv")
        arxiv_id  = arxiv_url.split("/abs/")[-1].split("v")[0] if arxiv_url else None
        doi       = work_ids.get("doi", "").replace("https://doi.org/", "") or None

        authors = [
            a["author"].get("display_name", "")
            for a in (work.get("authorships") or [])
            if a.get("author")
        ]

        loc = work.get("primary_location") or {}
        source = loc.get("source") or {}
        pdf_url = loc.get("pdf_url")

        oa = work.get("open_access") or {}

        results.append({
            "source":               "openalex",
            "arxiv_id":             arxiv_id,
            "s2_id":                None,
            "openalex_id":          work.get("id"),
            "title":                work.get("title") or "",
            "authors":              authors,
            "year":                 work.get("publication_year"),
            "published":            str(work.get("publication_year")) if work.get("publication_year") else None,
            "venue":                source.get("display_name"),
            "citations":            work.get("cited_by_count", 0),
            "influential_citations": 0,
            "is_open_access":       oa.get("is_oa", False),
            "pdf_url":              oa.get("oa_url") or pdf_url,
            "abstract":             "",   # OpenAlex uses inverted index; skip for now
            "doi":                  doi,
            "arxiv_url":            arxiv_url,
            "s2_url":               None,
            "categories":           None,
            "publication_types":    None,
            "fields_of_study":      None,
        })

    return results
