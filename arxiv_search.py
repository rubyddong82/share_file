"""
arXiv Search — fetch pool for unified search pipeline.

Category reference:
  cs.AI   – Artificial Intelligence      cs.LG   – Machine Learning
  cs.DS   – Data Structures/Algorithms   cs.CV   – Computer Vision
  cs.CL   – NLP                          cs.NE   – Neural/Evolutionary Computing
  cs.IR   – Information Retrieval        cs.RO   – Robotics
  stat.ML – ML (Statistics)              math.OC – Optimization
  eess.SP – Signal Processing

Speed note:
  arXiv rate-limits per IP — parallel requests trigger 429s and make things worse.
  The main speedup lever is page_size: setting it equal to pool_size means a single
  HTTP request with zero inter-page delays, vs multiple requests with 3s waits each.
"""

import time
import arxiv
from datetime import datetime, timezone

# Singleton client — preserves _last_request_dt across searches so the 3s
# inter-request gap is respected even between separate fetch_pool() calls.
_client = arxiv.Client(
    page_size=200,
    delay_seconds=3.0,
    num_retries=3,
)


def fetch_pool(
    query: str,
    filters: dict,
    pool_size: int = 50,
) -> list[dict]:
    """
    Fetch papers from arXiv by relevance. Returns standardized paper dicts.

    Uses a module-level singleton Client so _last_request_dt persists between
    calls — prevents 429s when multiple searches are run in the same session.

    filters keys (all optional):
      year_from       : int
      year_to         : int
      open_access_only: bool  (ignored — arXiv is always open access)
      categories      : list[str]   e.g. ["cs.LG", "cs.AI"]
      title_only      : bool
      abstract_only   : bool
    """
    # Build query string
    if filters.get("title_only"):
        base = f"ti:{query}"
    elif filters.get("abstract_only"):
        base = f"abs:{query}"
    else:
        base = f"all:{query}"

    categories = filters.get("categories") or []
    if categories:
        # All categories in one OR clause — single request
        base += " AND (" + " OR ".join(f"cat:{c}" for c in categories) + ")"

    print(f"  [arXiv] Query: {base} | pool_size={pool_size}")
    search = arxiv.Search(
        query=base,
        max_results=pool_size,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    year_from = filters.get("year_from")
    year_to   = filters.get("year_to")

    results = []
    for attempt in range(4):
        try:
            results = []
            for paper in _client.results(search):
                pub = paper.published
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)

                year = pub.year
                if year_from and year < year_from:
                    continue
                if year_to and year > year_to:
                    continue

                raw_id   = paper.entry_id.split("/abs/")[-1]
                arxiv_id = raw_id.split("v")[0]

                results.append({
                    "source":                "arxiv",
                    "arxiv_id":              arxiv_id,
                    "s2_id":                 None,
                    "openalex_id":           None,
                    "title":                 paper.title,
                    "authors":               [a.name for a in paper.authors],
                    "year":                  year,
                    "published":             pub.strftime("%Y-%m-%d"),
                    "venue":                 None,
                    "citations":             0,
                    "influential_citations": 0,
                    "is_open_access":        True,
                    "pdf_url":               paper.pdf_url,
                    "abstract":              paper.summary,
                    "doi":                   paper.doi,
                    "arxiv_url":             f"https://arxiv.org/abs/{arxiv_id}",
                    "s2_url":                None,
                    "categories":            paper.categories,
                    "publication_types":     None,
                    "fields_of_study":       None,
                })
            break  # success

        except Exception as e:
            if "429" in str(e) and attempt < 3:
                wait = 5 * (2 ** attempt)   # 5s, 10s, 20s
                print(f"\n  [arXiv] Rate limited. Retrying in {wait}s (attempt {attempt + 1}/4)...")
                time.sleep(wait)
            else:
                raise

    print(f"  [arXiv] Retrieved {len(results)} papers")
    return results
