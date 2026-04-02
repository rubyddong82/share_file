"""
Semantic Scholar Search — fetch pool for unified search pipeline.

API key (optional but strongly recommended to avoid 429s):
  Register: https://www.semanticscholar.org/product/api
  Usage:    export S2_API_KEY="your_key"

Fields of study:  "Computer Science", "Mathematics", "Physics", "Engineering",
                  "Biology", "Medicine", "Economics", "Neuroscience"
Publication types: "JournalArticle", "Conference", "Review", "Book",
                   "Dataset", "MetaAnalysis", "Study"
"""

import os
import time
import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"

PAPER_FIELDS = ",".join([
    "title", "abstract", "year", "authors",
    "citationCount", "referenceCount", "influentialCitationCount",
    "isOpenAccess", "openAccessPdf", "publicationTypes",
    "publicationDate", "venue", "externalIds", "fieldsOfStudy", "url",
])


def _get(endpoint: str, params: dict, api_key: str = None, retries: int = 3) -> dict:
    headers = {"x-api-key": api_key} if api_key else {}
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** attempt * 3
                print(f"  [S2] Rate limited. Waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                resp.raise_for_status()
        except requests.exceptions.Timeout:
            print(f"  [S2] Timeout (attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                time.sleep(2)
    raise RuntimeError(f"[S2] Failed after {retries} retries")


def _parse_paper(p: dict) -> dict:
    """Convert raw S2 API paper dict to standardized format."""
    pdf_url = p["openAccessPdf"].get("url") if p.get("openAccessPdf") else None
    ext_ids  = p.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv")
    pub_date = (p.get("publicationDate") or "")[:10]
    abstract = p.get("abstract") or ""
    return {
        "source":               "s2",
        "arxiv_id":             arxiv_id,
        "s2_id":                p.get("paperId"),
        "openalex_id":          None,
        "title":                p.get("title") or "",
        "authors":              [a["name"] for a in (p.get("authors") or [])],
        "year":                 p.get("year"),
        "published":            pub_date,
        "venue":                p.get("venue"),
        "citations":            p.get("citationCount") or 0,
        "influential_citations": p.get("influentialCitationCount") or 0,
        "is_open_access":       p.get("isOpenAccess") or False,
        "pdf_url":              pdf_url,
        "abstract":             abstract,
        "doi":                  ext_ids.get("DOI"),
        "arxiv_url":            f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "s2_url":               p.get("url"),
        "categories":           None,
        "publication_types":    p.get("publicationTypes"),
        "fields_of_study":      p.get("fieldsOfStudy"),
    }


def fetch_pool(
    query: str,
    sort_by: str,           # "relevance" or "citationCount"
    filters: dict,
    pool_size: int = 50,
    api_key: str = None,
) -> list[dict]:
    """
    Fetch one candidate pool from S2 with given native sort.
    Returns standardized paper dicts.

    filters keys (all optional):
      year_from        : int
      year_to          : int
      open_access_only : bool
      fields_of_study  : list[str]
      publication_types: list[str]
      venue            : list[str]
      min_citations    : int
    """
    key = api_key or os.environ.get("S2_API_KEY")

    params = {
        "query":  query,
        "limit":  min(pool_size, 100),
        "fields": PAPER_FIELDS,
    }

    if sort_by != "relevance":
        params["sort"] = sort_by

    year_from = filters.get("year_from")
    year_to   = filters.get("year_to")
    if year_from and year_to:
        params["year"] = f"{year_from}-{year_to}"
    elif year_from:
        params["year"] = f"{year_from}-"
    elif year_to:
        params["year"] = f"-{year_to}"

    fos = filters.get("fields_of_study")
    if fos:
        params["fieldsOfStudy"] = ",".join(fos)
    if filters.get("open_access_only"):
        params["openAccessPdf"] = ""
    pub_types = filters.get("publication_types")
    if pub_types:
        params["publicationTypes"] = ",".join(pub_types)
    venue = filters.get("venue")
    if venue:
        params["venue"] = ",".join(venue) if isinstance(venue, list) else venue
    min_cit = filters.get("min_citations")
    if min_cit:
        params["minCitationCount"] = min_cit

    print(f"  [S2] sort={sort_by} | pool_size={pool_size}")

    results = []
    offset  = 0
    page_size = 100  # S2 search endpoint hard cap per request

    while len(results) < pool_size:
        params["limit"]  = min(page_size, pool_size - len(results))
        params["offset"] = offset
        data  = _get("paper/search", params, api_key=key)
        batch = data.get("data", [])
        if not batch:
            break
        results.extend(_parse_paper(p) for p in batch)
        offset += len(batch)
        # Stop if S2 has no more results
        total = data.get("total", 0)
        if offset >= total:
            break

    print(f"  [S2] Retrieved {len(results)} papers")
    return results


def batch_get_citations(
    arxiv_ids: list[str],
    batch_size: int = 100,
    api_key: str = None,
) -> dict[str, dict]:
    """
    Batch-fetch citation counts for arXiv IDs using S2's batch paper endpoint.
    Returns {arxiv_id: {"citations": int, "influential_citations": int, "s2_id": str}}

    Much more reliable than OpenAlex for CS/AI papers.
    S2 batch endpoint accepts up to 500 IDs per request.
    """
    import requests as _requests

    key = api_key or os.environ.get("S2_API_KEY")
    headers = {"x-api-key": key} if key else {}
    results = {}

    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i : i + batch_size]
        ids   = [f"arXiv:{aid}" for aid in batch]

        for attempt in range(3):
            try:
                resp = _requests.post(
                    f"{BASE_URL}/paper/batch",
                    params={"fields": "paperId,citationCount,influentialCitationCount"},
                    json={"ids": ids},
                    headers=headers,
                    timeout=15,
                )
                if resp.status_code == 200:
                    for arxiv_id, paper in zip(batch, resp.json()):
                        if paper:   # None means not found in S2
                            results[arxiv_id] = {
                                "citations":             paper.get("citationCount") or 0,
                                "influential_citations": paper.get("influentialCitationCount") or 0,
                                "s2_id":                paper.get("paperId"),
                            }
                    break
                elif resp.status_code == 429:
                    wait = 2 ** attempt * 3
                    print(f"  [S2 batch] Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [S2 batch] Error {resp.status_code}")
                    break
            except Exception as e:
                print(f"  [S2 batch] Exception: {e}")
                break

        if i + batch_size < len(arxiv_ids):
            time.sleep(1)

    return results


def get_paper_citations(s2_id: str, max_results: int = 10, api_key: str = None) -> list[dict]:
    """Fetch papers that cite a given S2 paper ID."""
    key = api_key or os.environ.get("S2_API_KEY")
    data = _get(
        f"paper/{s2_id}/citations",
        {
            "limit": min(max_results, 1000),
            "fields": "title,year,citationCount,influentialCitationCount,authors,venue,"
                      "abstract,externalIds,url,openAccessPdf,isOpenAccess",
        },
        api_key=key,
    )
    results = []
    for item in data.get("data", []):
        p = item["citingPaper"]
        ext = p.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv")
        doi      = ext.get("DOI")
        oa_pdf   = (p.get("openAccessPdf") or {}).get("url")
        results.append({
            "title":                 p.get("title"),
            "year":                  p.get("year"),
            "citations":             p.get("citationCount"),
            "influential_citations": p.get("influentialCitationCount"),
            "venue":                 p.get("venue"),
            "authors":               [a["name"] for a in (p.get("authors") or [])],
            "abstract":              p.get("abstract"),
            "s2_id":                 p.get("paperId"),
            "arxiv_id":              arxiv_id,
            "doi":                   doi,
            "arxiv_url":             f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            "s2_url":                p.get("url"),
            "pdf_url":               oa_pdf,
            "is_open_access":        p.get("isOpenAccess"),
            "source":                "s2",
        })
    return results
