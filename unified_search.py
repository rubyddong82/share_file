"""
Unified Paper Search — interactive REPL combining arXiv, Semantic Scholar, and OpenAlex.

Pipeline:
  1. Fetch arXiv pool   (relevance, N papers)
  2. Fetch S2 pools     (relevance pool + citationCount pool, N papers each)
  3. Deduplicate        (by arXiv ID → S2 ID → title similarity fallback)
  4. OpenAlex lookup    (citation counts for arXiv-only papers)
  5. Composite score    (citations + recency + title match)
  6. Return top outputs

Commands:
  search <query>            – run a search with current settings
  show <n>                  – show full details of result n
  pdf <n>                   – download and open PDF of result n
  citation <n> [m]          – show m citing papers for result n (default 10, requires S2 ID)
  show_citation <n>         – show full details for citation n
  pdf_citation <n>          – download PDF for citation n
  get_reference <n>         – Harvard reference for result n
  get_citation_reference <n> – Harvard reference for citation n
  limit <outputs> [pool]    – set output count and pool size per source
  filter <key> [values...]  – set a filter (see 'help filters')
  sort <mode>               – relevance | citations | composite
  weights [c=0.5 r=0.25 t=0.25] – composite weights (citations/recency/title_match)
  email <your@email.com>    – set email for OpenAlex polite pool
  settings                  – show current settings
  help                      – show this help
  quit                      – exit
"""

import os
import sys
import math
import subprocess
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich.rule import Rule

console = Console()

sys.path.insert(0, os.path.dirname(__file__))
import arxiv_search as arxiv_mod
import semantic_scholar_search as s2_mod
import openalex_search as oa_mod

# ─────────────────────────────────────────────────────────────────────────────
# STOP WORDS (excluded from title match scoring)
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = {
    # Articles & determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Prepositions
    "of", "in", "on", "at", "to", "for", "with", "by", "as", "from",
    "into", "onto", "upon", "over", "under", "about", "across", "through",
    "between", "among", "within", "without", "against", "along", "beyond",
    "during", "before", "after", "above", "below", "up", "down",
    # Conjunctions ("and" kept in STOP_WORDS so it's filtered from text tokens;
    # query parser catches it before stop-word filtering and uses it as a boundary)
    "and", "or", "but", "nor", "yet", "so", "both", "either", "neither",
    "whether", "while", "although", "though", "whereas",
    # Auxiliary / linking verbs
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    # Pronouns
    "it", "its", "we", "our", "they", "their", "i", "you", "he", "she",
    "who", "which", "what", "where", "when", "how",
    # Common filler
    "not", "no", "also", "just", "only", "such", "than", "then", "thus",
    "hence", "however", "therefore", "each", "every", "any", "all",
    "more", "most", "very", "too", "here", "there",
}

RECENCY_LAMBDA = 0.15   # exponential decay rate (~5 year half-life)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT STATE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FILTERS = {
    "year_from":         None,
    "year_to":           None,
    "open_access_only":  False,
    # arXiv-specific
    "categories":        [],          # e.g. ["cs.LG", "cs.AI"]
    "title_only":        False,
    "abstract_only":     False,
    # S2-specific
    "fields_of_study":   [],          # e.g. ["Computer Science"]
    "publication_types": [],          # e.g. ["Conference", "JournalArticle"]
    "venue":             [],          # e.g. ["NeurIPS", "ICML"]
    "min_citations":     None,
}

DEFAULT_WEIGHTS = {"citations": 0.4, "recency": 0.25, "title_match": 0.35}


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def merge_pools(*pools: list[dict]) -> list[dict]:
    """
    Deduplicate papers across pools.
    Priority: arXiv ID match → S2 ID match → title similarity (>0.85).
    When merging, prefer the entry with more citation data (S2 over arXiv).
    """
    seen_arxiv = {}   # arxiv_id  → index in result
    seen_s2    = {}   # s2_id     → index in result
    result     = []

    for pool in pools:
        for paper in pool:
            arxiv_id = paper.get("arxiv_id")
            s2_id    = paper.get("s2_id")

            # 1. Dedupe by arXiv ID
            if arxiv_id and arxiv_id in seen_arxiv:
                existing = result[seen_arxiv[arxiv_id]]
                # Merge: take the better citation data
                if paper.get("citations", 0) > existing.get("citations", 0):
                    existing["citations"]             = paper["citations"]
                    existing["influential_citations"] = paper.get("influential_citations", 0)
                # Absorb S2 metadata if missing
                if paper.get("s2_id") and not existing.get("s2_id"):
                    existing["s2_id"]  = paper["s2_id"]
                    existing["s2_url"] = paper.get("s2_url")
                if paper.get("venue") and not existing.get("venue"):
                    existing["venue"] = paper["venue"]
                if paper.get("pdf_url") and not existing.get("pdf_url"):
                    existing["pdf_url"] = paper["pdf_url"]
                existing["source"] = "both"
                continue

            # 2. Dedupe by S2 ID
            if s2_id and s2_id in seen_s2:
                continue

            # 3. Title similarity fallback
            title = paper.get("title", "")
            dup_found = False
            for idx, existing in enumerate(result):
                if _title_similarity(title, existing.get("title", "")) > 0.85:
                    # Merge citation data
                    if paper.get("citations", 0) > existing.get("citations", 0):
                        existing["citations"]             = paper["citations"]
                        existing["influential_citations"] = paper.get("influential_citations", 0)
                    if paper.get("arxiv_id") and not existing.get("arxiv_id"):
                        existing["arxiv_id"]  = paper["arxiv_id"]
                        existing["arxiv_url"] = paper.get("arxiv_url")
                        seen_arxiv[paper["arxiv_id"]] = idx
                    if paper.get("s2_id") and not existing.get("s2_id"):
                        existing["s2_id"]  = paper["s2_id"]
                        existing["s2_url"] = paper.get("s2_url")
                        seen_s2[paper["s2_id"]] = idx
                    existing["source"] = "both"
                    dup_found = True
                    break

            if dup_found:
                continue

            # New unique paper
            idx = len(result)
            result.append(paper)
            if arxiv_id:
                seen_arxiv[arxiv_id] = idx
            if s2_id:
                seen_s2[s2_id] = idx

    return result


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, split, strip punctuation, remove stop words."""
    import re
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def _normalize_token(t: str) -> str:
    """
    Minimal suffix normalization so singular/plural variants match.
    Rules (in order):
      - strip trailing 's'  if len > 4 and doesn't end in 'ss'
        e.g. spaces→space, ecosystems→ecosystem, methods→method
    Intentionally simple: avoids over-stemming short words like 'is', 'was'.
    """
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _citation_score(citations: int, influential: int, max_cit: int, max_inf: int) -> float:
    """
    Log-normalized citation score.
    Combines raw citations (65%) and influential citations (35%).
    log1p prevents log(0) and compresses extreme outliers.
    """
    cit = math.log1p(citations)  / math.log1p(max_cit)
    inf = math.log1p(influential) / math.log1p(max_inf)
    return cit * 0.65 + inf * 0.35


def _recency_score(year: int) -> float:
    """
    Exponential decay from current year.
    score = exp(-λ × age)   where λ = RECENCY_LAMBDA (~5yr half-life)
    A 2026 paper → 1.0, a 2021 paper → ~0.47, a 2016 paper → ~0.22.
    """
    import datetime
    current_year = datetime.datetime.now().year
    age = max(current_year - year, 0)
    return math.exp(-RECENCY_LAMBDA * age)


def _plain_query(raw: str) -> str:
    """Strip bracket notation and 'and' separators to build a plain API query string."""
    import re
    plain = re.sub(r'[\[\]]', ' ', raw)
    plain = re.sub(r'\band\b', ' ', plain, flags=re.IGNORECASE)
    return ' '.join(plain.split())


def parse_query_for_scoring(raw: str):
    """
    Parse a raw query string (may contain [bracket] groups) into term groups.

    Returns:
      ind_tokens : list[str] — all scorable tokens with duplicates preserved.
                   Stop words and "and" excluded.
                   "data data" → ["data","data"] so "data" can score up to 2.
      groups     : list[{"terms": list[str], "depth": int}]

    Scoring rules:
      - "and" is the only boundary token. It isolates groups at every level —
        no pair can span an "and" boundary.
      - Terms outside any bracket, split by "and" → depth-0 groups.
        Consecutive terms in a group form pairs automatically (weight = 2n × 2^0).
      - [a b c]  → depth-1 group. Consecutive terms form pairs (weight = 2n × 2^1).
      - [[a b]]  → depth-2 group. Weight = 2n × 2^2.
      - "and" inside brackets also splits into isolated sub-groups.
      - Single words are always weight 1 regardless of depth.

    Example:
      "industrial data and data space"
        → groups: [terms=["industrial","data"], depth=0],
                  [terms=["data","space"],      depth=0]
      "[industrial data] and [data space]"
        → groups: [terms=["industrial","data"], depth=1],
                  [terms=["data","space"],      depth=1]
    """
    import re

    def _terms(text):
        toks = re.findall(r'[a-z0-9]+', text.lower())
        return [t for t in toks if t not in STOP_WORDS and t != 'and']

    def _split_and(text):
        return re.split(r'\band\b', text, flags=re.IGNORECASE)

    groups = []
    outside_chars = []
    i, n = 0, len(raw)

    while i < n:
        if raw[i] == '[':
            depth = 0
            j = i
            while j < n and raw[j] == '[':
                depth += 1
                j += 1
            close = ']' * depth
            end = raw.find(close, j)
            if end == -1:
                end = n
            content = raw[j:end]
            i = end + depth
            import re as _re
            if _re.search(r'\band\b', content, flags=_re.IGNORECASE):
                raise ValueError(
                    f'"and" is not allowed inside brackets. '
                    f'Use "and" outside brackets to separate groups.\n'
                    f'  Got: [{content}]'
                )
            terms = _terms(content)
            if terms:
                groups.append({"terms": terms, "depth": depth})
        else:
            outside_chars.append(raw[i])
            i += 1

    # Text outside brackets: split by "and" → depth-0 groups
    for part in _split_and(''.join(outside_chars)):
        terms = _terms(part)
        if terms:
            groups.append({"terms": terms, "depth": 0})

    ind_tokens = [t for g in groups for t in g["terms"]]
    return ind_tokens, groups


def _find_pair(norm_terms: list[str], norm_tokens: list[str]) -> bool:
    """
    Return True if norm_terms appear in-order in norm_tokens with at most
    1 intervening word between each consecutive pair of terms.
    """
    n_terms = len(norm_terms)
    n_tok   = len(norm_tokens)
    for start in range(n_tok):
        if norm_tokens[start] == norm_terms[0]:
            pos, ok = start, True
            for j in range(1, n_terms):
                found = False
                for gap in (1, 2):   # adjacent (gap=1) or 1 word between (gap=2)
                    nxt = pos + gap
                    if nxt < n_tok and norm_tokens[nxt] == norm_terms[j]:
                        pos, found = nxt, True
                        break
                if not found:
                    ok = False
                    break
            if ok:
                return True
    return False


def _relevance_score(
    ind_tokens:      list[str],
    groups:          list[dict],
    title_tokens:    list[str],
    abstract_tokens: list[str],
) -> float:
    """
    Score a paper against the parsed query. Returns value in [0, 1].

    Individual tokens (weight 1 each):
      score += min(text_count(t), query_count(t)) for each unique token t.
      "data" twice in query can score up to 2.

    Pair scoring for all consecutive sub-sequences of length n≥2 within each group:
      weight = 2n × 2^depth
        depth=0 (no brackets): 2-pair=4, 3-pair=6, 4-pair=8
        depth=1 [brackets]:    2-pair=8, 3-pair=12, 4-pair=16
        depth=2 [[brackets]]:  2-pair=16, 3-pair=24, 4-pair=32
      Terms must appear in-order in title OR abstract with ≤1 word between
      each consecutive term.
    """
    from collections import Counter

    if not ind_tokens and not groups:
        return 0.0

    norm_title    = [_normalize_token(t) for t in title_tokens]
    norm_abstract = [_normalize_token(t) for t in abstract_tokens]
    norm_all      = norm_title + norm_abstract

    # ── Individual token scoring ────────────────────────────────────────────
    query_cnt = Counter(_normalize_token(t) for t in ind_tokens)
    text_cnt  = Counter(norm_all)
    ind_score = sum(min(text_cnt[t], c) for t, c in query_cnt.items())
    ind_max   = sum(query_cnt.values())

    # ── Pair scoring: all consecutive sub-sequences of length 2..m ──────────
    pair_score = 0
    pair_max   = 0
    for g in groups:
        terms = [_normalize_token(t) for t in g["terms"]]
        depth = g["depth"]
        m = len(terms)
        for length in range(2, m + 1):
            weight = 2 * length * (2 ** depth)
            for start in range(m - length + 1):
                pair_max += weight
                ngram = terms[start : start + length]
                if _find_pair(ngram, norm_title) or _find_pair(ngram, norm_abstract):
                    pair_score += weight

    total     = ind_score + pair_score
    total_max = ind_max + pair_max
    return total / total_max if total_max > 0 else 0.0


def composite_sort(
    papers: list[dict],
    query: str,
    weights: dict,
    sort_by: str,
    top_n: int,
) -> list[dict]:
    """
    Score and rank papers.
    sort_by: "relevance" — pool order, truncated
             "citations" — raw citation count descending
             "composite" — weighted combination of citation, recency, title match
    """
    if sort_by == "relevance":
        return papers[:top_n]

    if sort_by == "citations":
        papers.sort(key=lambda x: x.get("citations", 0), reverse=True)
        return papers[:top_n]

    # ── Composite ──────────────────────────────────────────────────────────────
    max_cit = max((p.get("citations", 0)             for p in papers), default=1) or 1
    max_inf = max((p.get("influential_citations", 0)  for p in papers), default=1) or 1

    ind_tokens, groups = parse_query_for_scoring(query)

    for p in papers:
        cit_score = _citation_score(
            p.get("citations", 0),
            p.get("influential_citations", 0),
            max_cit, max_inf,
        )

        year      = p.get("year") or 2000
        rec_score = _recency_score(year)

        title_tokens    = _tokenize(p.get("title") or "")
        abstract_tokens = _tokenize(p.get("abstract") or "")
        rel_score = _relevance_score(ind_tokens, groups, title_tokens, abstract_tokens)

        p["_score"] = (
            cit_score * weights.get("citations",   0.4)  +
            rec_score * weights.get("recency",     0.25) +
            rel_score * weights.get("title_match", 0.35)
        )
        p["_score_detail"] = {
            "citation":    round(cit_score, 3),
            "recency":     round(rec_score, 3),
            "title_match": round(rel_score, 3),
            "total":       round(p["_score"], 3),
        }

    papers.sort(key=lambda x: x.get("_score", 0), reverse=True)
    return papers[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_search(query: str, state: dict) -> list[dict]:
    pool_size = state["limit_pool"]
    filters   = state["filters"]
    plain_q   = _plain_query(query)   # strip brackets/and for API calls

    console.print()
    from rich.markup import escape as rich_escape
    console.print(Rule(f"[bold cyan]Search: {rich_escape(query)}[/bold cyan]"))
    console.print(
        f"  Pool [dim]per source[/dim]: [yellow]{pool_size}[/yellow]  "
        f"Output: [yellow]{state['limit_outputs']}[/yellow]  "
        f"Sort: [yellow]{state['sort_by']}[/yellow]"
    )
    console.print()

    # 1. arXiv pool (single request — page_size=pool_size avoids pagination delays)
    console.print("[bold][[1/4]][/bold] Fetching [cyan]arXiv[/cyan] pool...", end=" ")
    try:
        arxiv_pool = arxiv_mod.fetch_pool(plain_q, filters, pool_size)
        console.print(f"[green]✓[/green] {len(arxiv_pool)} papers")
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        arxiv_pool = []

    # 2. S2 pools — relevance + citationCount in parallel (2 threads)
    console.print("[bold][[2/4]][/bold] Fetching [cyan]Semantic Scholar[/cyan] pools in parallel...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_rel = ex.submit(s2_mod.fetch_pool, plain_q, "relevance",     filters, pool_size)
        fut_cit = ex.submit(s2_mod.fetch_pool, plain_q, "citationCount", filters, pool_size)
        s2_rel, s2_cit = [], []
        try:
            s2_rel = fut_rel.result()
        except Exception as e:
            console.print(f"  [red]✗ S2 relevance: {e}[/red]")
        try:
            s2_cit = fut_cit.result()
        except Exception as e:
            console.print(f"  [red]✗ S2 citations: {e}[/red]")
    console.print(f"  [green]✓[/green] {len(s2_rel)} relevance + {len(s2_cit)} citation")

    # 3. Merge & deduplicate
    console.print("[bold][[3/4]][/bold] Merging and deduplicating...", end=" ")
    merged  = merge_pools(arxiv_pool, s2_rel, s2_cit)
    n_arxiv = sum(1 for p in merged if p["source"] == "arxiv")
    n_both  = sum(1 for p in merged if p["source"] == "both")
    n_s2    = sum(1 for p in merged if p["source"] == "s2")
    console.print(
        f"[green]✓[/green] [bold]{len(merged)}[/bold] unique  "
        f"[dim](arXiv-only: {n_arxiv}  S2-only: {n_s2}  both: {n_both})[/dim]"
    )

    # 4. Citation enrichment for arXiv-only papers
    arxiv_only = [p for p in merged if p["source"] == "arxiv" and p.get("arxiv_id")]
    if arxiv_only:
        arxiv_only_ids = [p["arxiv_id"] for p in arxiv_only]
        console.print(
            f"[bold][[4/4]][/bold] Citation enrichment for "
            f"[yellow]{len(arxiv_only)}[/yellow] arXiv-only papers..."
        )

        # Primary: S2 batch (single POST, fast)
        s2_enriched   = 0
        still_missing = []
        try:
            s2_batch = s2_mod.batch_get_citations(arxiv_only_ids)
            for p in arxiv_only:
                if p["arxiv_id"] in s2_batch:
                    hit = s2_batch[p["arxiv_id"]]
                    p["citations"]             = hit["citations"]
                    p["influential_citations"] = hit["influential_citations"]
                    if hit.get("s2_id") and not p.get("s2_id"):
                        p["s2_id"] = hit["s2_id"]
                    p["source"] = "both"
                    s2_enriched += 1
                else:
                    still_missing.append(p)
            console.print(f"  [cyan]S2 batch[/cyan]: [green]{s2_enriched}[/green]/{len(arxiv_only)} enriched")
        except Exception as e:
            console.print(f"  [cyan]S2 batch[/cyan]: [red]✗ {e}[/red]")
            still_missing = arxiv_only

        # Fallback: OpenAlex parallel (10 workers, well within 100 req/s limit)
        if still_missing:
            console.print(
                f"  [cyan]OpenAlex fallback[/cyan]: {len(still_missing)} papers "
                f"[dim](10 parallel workers)[/dim]...", end=" "
            )
            try:
                oa_data     = oa_mod.batch_get_citations([p["arxiv_id"] for p in still_missing])
                oa_enriched = 0
                for p in still_missing:
                    if p["arxiv_id"] in oa_data:
                        p["citations"]   = oa_data[p["arxiv_id"]]["citations"]
                        p["openalex_id"] = oa_data[p["arxiv_id"]]["openalex_id"]
                        oa_enriched += 1
                console.print(f"[green]{oa_enriched}[/green] enriched")
            except Exception as e:
                console.print(f"[red]✗ {e}[/red]")
    else:
        console.print("[bold][[4/4]][/bold] No arXiv-only papers — skipping enrichment")

    # 5. Composite scoring & rank
    results = composite_sort(merged, query, state["weights"], state["sort_by"], state["limit_outputs"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_pdf_url(paper: dict) -> str | None:
    """
    Resolve the best available PDF URL for a paper.
    Priority: explicit pdf_url → arXiv PDF → DOI → None
    """
    if paper.get("pdf_url"):
        return paper["pdf_url"]
    if paper.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{paper['arxiv_id']}"
    if paper.get("doi"):
        return f"https://doi.org/{paper['doi']}"
    return None

def _score_bar(score: float, width: int = 10) -> str:
    """Render a mini progress bar for a 0-1 score."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _score_color(score: float) -> str:
    if score >= 0.75:   return "green"
    elif score >= 0.5:  return "yellow"
    else:               return "red"


def _src_badge(src: str) -> Text:
    badges = {
        "both":    ("[both]",    "bold green"),
        "s2":      ("[S2]",      "bold blue"),
        "arxiv":   ("[arXiv]",   "bold magenta"),
        "openalex":("[OA]",      "bold cyan"),
    }
    label, style = badges.get(src, (f"[{src}]", "dim"))
    return Text(label, style=style)


def print_summary(results: list[dict]):
    if not results:
        console.print("\n[yellow]No results.[/yellow]")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
        padding=(0, 1),
    )
    table.add_column("#",       style="bold white", width=3,  justify="right")
    table.add_column("Title",                       width=48)
    table.add_column("Year",    style="dim",         width=6,  justify="center")
    table.add_column("Cites",                        width=7,  justify="right")
    table.add_column("PDF",                          width=4,  justify="center")
    table.add_column("Src",                          width=8,  justify="center")
    table.add_column("Score",                        width=22, justify="left")

    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        year  = str(r.get("year") or "N/A")
        cites = r.get("citations", 0)
        src   = r.get("source", "?")

        # Citation count coloring
        if cites >= 1000:   cite_style = "bold green"
        elif cites >= 100:  cite_style = "green"
        elif cites >= 10:   cite_style = "yellow"
        else:               cite_style = "dim"

        # Score column: bar + breakdown if composite
        if "_score" in r:
            total = r["_score"]
            bar   = _score_bar(total)
            score_text = Text()
            score_text.append(bar, style=_score_color(total))
            score_text.append(f" {total:.2f}", style="bold " + _score_color(total))
            if "_score_detail" in r:
                d = r["_score_detail"]
                score_text.append(
                    f"  c:{d['citation']:.2f} r:{d['recency']:.2f} t:{d['title_match']:.2f}",
                    style="dim"
                )
        else:
            score_text = Text("—", style="dim")

        pdf_avail = Text("PDF", style="bold green") if get_pdf_url(r) else Text("—", style="dim")

        table.add_row(
            str(i),
            title,
            year,
            Text(str(cites), style=cite_style),
            pdf_avail,
            _src_badge(src),
            score_text,
        )

    console.print()
    console.print(table)
    console.print(f"  [dim]{len(results)} result(s) — use [bold]show <n>[/bold] for details[/dim]")


def print_detail(paper: dict, index: int):
    title   = paper.get("title") or "Unknown"
    authors = paper.get("authors") or []
    d       = paper.get("_score_detail")

    # Header panel
    console.print()
    console.print(Panel(
        f"[bold white]{title}[/bold white]",
        title=f"[cyan]#{index}[/cyan]",
        border_style="cyan",
        padding=(0, 1),
    ))

    # Metadata grid
    meta = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    meta.add_column(style="dim", width=16)
    meta.add_column()

    authors_str = ", ".join(authors[:5]) + ("..." if len(authors) > 5 else "")
    meta.add_row("Authors",    authors_str)
    meta.add_row("Year",       f"{paper.get('year') or 'N/A'}  (published: {paper.get('published') or 'N/A'})")
    meta.add_row("Venue",      paper.get("venue") or "N/A")
    meta.add_row("Source",     f"{_src_badge(paper.get('source','?'))}  |  Open Access: {paper.get('is_open_access')}")
    meta.add_row("Citations",  f"[green]{paper.get('citations', 0)}[/green]  (influential: {paper.get('influential_citations', 0)})")

    cats = paper.get("categories") or paper.get("fields_of_study") or []
    if cats:
        meta.add_row("Fields", ", ".join(cats[:6]))

    console.print(meta)

    # Score breakdown panel (only when composite scored)
    if d:
        score_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                            padding=(0, 2), expand=False)
        score_table.add_column("Component",  style="dim")
        score_table.add_column("Score",      justify="right")
        score_table.add_column("Bar",        width=12)
        score_table.add_column("Weight",     justify="right", style="dim")
        score_table.add_column("Meaning")

        rows = [
            ("Citation",    d["citation"],    "citations (log-normalized + influential)",    "0.40"),
            ("Recency",     d["recency"],     "pub year (exponential decay, λ=0.15)",       "0.25"),
            ("Relevance",   d["title_match"], "token+pair match in title & abstract",       "0.35"),
        ]
        for name, val, meaning, weight in rows:
            score_table.add_row(
                name,
                f"[{_score_color(val)}]{val:.3f}[/{_score_color(val)}]",
                Text(_score_bar(val, 12), style=_score_color(val)),
                weight,
                f"[dim]{meaning}[/dim]",
            )

        total = d["total"]
        score_table.add_row(
            "[bold]Total[/bold]",
            f"[bold {_score_color(total)}]{total:.3f}[/bold {_score_color(total)}]",
            Text(_score_bar(total, 12), style="bold " + _score_color(total)),
            "",
            f"[dim]weighted composite[/dim]",
        )

        console.print(Panel(score_table, title="[dim]Score Breakdown[/dim]",
                            border_style="dim", padding=(0, 0)))

    # Links
    links = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    links.add_column(style="dim", width=16)
    links.add_column()
    links.add_row("arXiv ID",  paper.get("arxiv_id") or "N/A")
    links.add_row("S2 ID",     paper.get("s2_id") or "N/A")
    links.add_row("DOI",       paper.get("doi") or "N/A")
    links.add_row("arXiv URL", f"[link]{paper.get('arxiv_url') or 'N/A'}[/link]")
    links.add_row("S2 URL",    f"[link]{paper.get('s2_url') or 'N/A'}[/link]")
    resolved_pdf = get_pdf_url(paper)
    links.add_row(
        "PDF URL",
        Text(resolved_pdf, style="bold green") if resolved_pdf else Text("N/A", style="dim")
    )
    console.print(links)

    # Abstract
    abstract = (paper.get("abstract") or "N/A")
    console.print(Panel(
        abstract,
        title="[dim]Abstract[/dim]",
        border_style="dim",
        padding=(0, 1),
    ))
    console.print()


def print_help():
    print("""
Commands:
  search <query>              Search across arXiv + Semantic Scholar
  show <n>                    Full details for result n
  pdf <n>                     Download PDF for result n
  citation <n> [m]            Show m papers citing result n (default 10, needs S2 ID)
  show_citation <n>           Full details for citation n (abstract, DOI, IDs, link)
  pdf_citation <n>            Download PDF for citation n
  get_reference <n>           Harvard reference for result n
  get_citation_reference <n>  Harvard reference for citation n
  limit <outputs> [pool]      Set output count and pool size per source
                              e.g. 'limit 10 50' → top 10 from 50-paper pools
  sort <mode>                 relevance | citations | composite
  weights [c=N r=N t=N]       Composite weights (must sum to 1.0)
                              c=citations, r=recency, t=title_match
                              e.g. 'weights c=0.6 r=0.2 t=0.2'
  email <addr>                Set email for OpenAlex polite pool (faster)
  settings                    Show current settings and filters
  help filters                Show filter command details
  quit / exit                 Exit

Filters (use 'filter <key> [values]'):
  filter year <from> [to]     e.g. 'filter year 2022 2024'
  filter open_access true     Only open-access papers
  filter categories cs.LG cs.AI ...    arXiv categories
  filter fields "Computer Science" Mathematics    S2 fields of study
  filter pub_types Conference JournalArticle      S2 publication types
  filter venue NeurIPS ICML                       S2 venue filter
  filter min_citations <n>    Minimum citation count (S2 only)
  filter title_only true      Match query in title only
  filter clear                Reset all filters
  filter show                 Display active filters
""")


def print_filter_help():
    print("""
Filter keys and accepted values:

  year <from> [to]
      e.g. filter year 2022 2024  →  papers from 2022 to 2024
           filter year 2023       →  papers from 2023 onwards

  open_access true|false
      Only return open-access papers

  categories <cat1> <cat2> ...    (arXiv only)
      cs.AI  cs.LG  cs.DS  cs.CV  cs.CL  cs.NE  cs.IR  cs.RO
      stat.ML  math.OC  eess.SP

  fields <field1> <field2> ...    (S2 only)
      Use underscores for spaces: Computer_Science  Machine_Learning
      Valid: Computer_Science  Mathematics  Physics  Engineering
             Biology  Medicine  Economics  Neuroscience

  pub_types <type1> ...           (S2 only)
      JournalArticle  Conference  Review  Book  Dataset  MetaAnalysis

  venue <venue1> ...              (S2 only)
      NeurIPS  ICML  ICLR  CVPR  ACL  EMNLP  Nature  Science

  min_citations <n>               (S2 only)
      Minimum citation count

  title_only true|false           Restrict arXiv query match to title
  abstract_only true|false        Restrict arXiv query match to abstract

  clear    Reset all filters to defaults
  show     Display current active filters
""")


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_index(args: list[str], state: dict) -> int | None:
    """Parse and validate a 1-based paper index from args."""
    if not args:
        print("Usage: <command> <n>")
        return None
    try:
        n = int(args[0])
    except ValueError:
        print(f"Invalid index: '{args[0]}' — must be an integer.")
        return None
    results = state["last_results"]
    if not results:
        print("No search results yet. Run 'search <query>' first.")
        return None
    if n < 1 or n > len(results):
        print(f"Index out of range: {n}. Valid range: 1–{len(results)}.")
        return None
    return n


def cmd_search(args: list[str], state: dict):
    query = " ".join(args)
    if not query:
        print("Usage: search <query>")
        return
    try:
        results = run_search(query, state)
    except ValueError as e:
        console.print(f"\n[bold red]Query error:[/bold red] {e}")
        return
    state["last_results"] = results
    state["last_query"]   = query
    print_summary(results)


def cmd_show(args: list[str], state: dict):
    n = parse_index(args, state)
    if n is None:
        return
    print_detail(state["last_results"][n - 1], n)


def _resolve_pdf_url(url: str) -> str | None:
    """
    Ensure `url` points to a direct PDF file.

    1. HEAD request → Content-Type application/pdf  → return url as-is.
    2. Otherwise GET the page and scan for a PDF link:
         - <a href> ending in .pdf
         - <a> whose text or href contains "pdf", "download", "full text", "view"
         - <meta name="citation_pdf_url"> (common in academic sites)
    Returns the resolved direct PDF URL, or None if nothing found.
    """
    import re
    import requests
    from urllib.parse import urljoin

    headers = {"User-Agent": "Mozilla/5.0 (compatible; paper-search/1.0)"}

    try:
        # Step 1: is it already a direct PDF?
        head = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        ct   = head.headers.get("Content-Type", "")
        if "application/pdf" in ct or url.lower().rstrip("?").endswith(".pdf"):
            return head.url   # use final URL after any redirects

        # Step 2: IEEE Xplore — PDF is JS-rendered, construct stamp URL from arnumber
        # Check both the original URL and the final redirected URL (e.g. DOI links)
        resolved_url = head.url
        ieee_m = re.search(r'ieeexplore\.ieee\.org/(?:document|abstract)/(\d+)', resolved_url) \
              or re.search(r'ieeexplore\.ieee\.org/(?:document|abstract)/(\d+)', url)
        if ieee_m:
            arnumber = ieee_m.group(1)
            return f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}"

        # Step 3: fetch the landing page and look for PDF links
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html     = resp.text
        base_url = resp.url   # final URL after redirects

        def abs_url(href):
            return urljoin(base_url, href)

        # Priority order for PDF link candidates
        candidates = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # <meta name="citation_pdf_url"> — used by Google Scholar / most publishers
            meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if meta and meta.get("content"):
                candidates.append(abs_url(meta["content"]))

            # <a> tags: prioritise those whose href or text looks like a PDF link
            pdf_re = re.compile(r'pdf|download|full.?text|view.?article', re.IGNORECASE)
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                text = a.get_text(" ", strip=True)
                if href.lower().endswith(".pdf"):
                    candidates.insert(0, abs_url(href))  # highest priority
                elif pdf_re.search(href) or pdf_re.search(text):
                    candidates.append(abs_url(href))

        except ImportError:
            # bs4 not available — fall back to regex
            # <meta name="citation_pdf_url" content="...">
            for m in re.finditer(
                r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE
            ):
                candidates.append(abs_url(m.group(1)))

            # href ending in .pdf
            for m in re.finditer(r'href=["\']([^"\']*\.pdf[^"\']*)["\']', html, re.IGNORECASE):
                candidates.insert(0, abs_url(m.group(1)))

            # href or nearby text containing pdf/download
            for m in re.finditer(
                r'href=["\']([^"\']+)["\'][^>]*>[^<]{0,60}(?:pdf|download|full.text)[^<]{0,30}<',
                html, re.IGNORECASE
            ):
                candidates.append(abs_url(m.group(1)))

        # Verify the first candidate is actually a PDF
        for candidate in candidates:
            try:
                h = requests.head(candidate, headers=headers, timeout=8, allow_redirects=True)
                if "application/pdf" in h.headers.get("Content-Type", "") \
                        or candidate.lower().rstrip("?").endswith(".pdf"):
                    return h.url
            except Exception:
                continue

        return None

    except Exception:
        return None


def cmd_pdf(args: list[str], state: dict):
    n = parse_index(args, state)
    if n is None:
        return

    paper       = state["last_results"][n - 1]
    initial_url = get_pdf_url(paper)

    if not initial_url:
        print("No PDF URL available for this paper.")
        print(f"  arXiv: {paper.get('arxiv_url') or 'N/A'}")
        print(f"  S2   : {paper.get('s2_url') or 'N/A'}")
        return

    import requests

    # Resolve to a direct PDF link (follows landing pages if needed)
    console.print(f"  Resolving: [dim]{initial_url}[/dim]")
    pdf_url = _resolve_pdf_url(initial_url)

    if not pdf_url:
        console.print(f"[yellow]  Could not locate a direct PDF from:[/yellow] {initial_url}")
        console.print(f"  Open manually: [link]{initial_url}[/link]")
        return

    if pdf_url != initial_url:
        console.print(f"  Resolved PDF: [green]{pdf_url}[/green]")

    download_dir = os.path.expanduser("~/pdf")
    os.makedirs(download_dir, exist_ok=True)

    safe_id  = (paper.get("arxiv_id") or paper.get("s2_id") or "paper").replace("/", "_")
    path     = os.path.join(download_dir, f"{safe_id}.pdf")

    console.print(f"  Downloading...")
    try:
        resp = requests.get(pdf_url, timeout=60, stream=True,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; paper-search/1.0)"})
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = os.path.getsize(path) // 1024
        console.print(f"  [green]Saved[/green] ({size_kb} KB): {path}")

        for opener in ["wslview", "xdg-open"]:
            try:
                subprocess.Popen([opener, path],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                break
            except FileNotFoundError:
                continue

    except Exception as e:
        console.print(f"  [red]Download failed:[/red] {e}")
        console.print(f"  Direct URL: [link]{pdf_url}[/link]")


def _fetch_and_store_citations(paper: dict, max_results: int, state: dict) -> list[dict] | None:
    """Shared helper: fetch citations for a paper, print table, store in state."""
    s2_id = paper.get("s2_id")
    title = paper.get("title", "Unknown")
    cites = paper.get("citations", 0)

    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"Total citations: [green]{cites}[/green]")

    if not s2_id:
        console.print("[yellow]  Detailed citing papers require a Semantic Scholar ID.[/yellow]")
        oa_id    = paper.get("openalex_id")
        arxiv_id = paper.get("arxiv_id")
        if oa_id:
            console.print(f"  OpenAlex: https://openalex.org/works/{oa_id.split('/')[-1]}")
        if arxiv_id:
            console.print(f"  S2 search: https://www.semanticscholar.org/search?q={arxiv_id}")
        return None

    console.print(f"Fetching up to {max_results} citing papers from Semantic Scholar...")
    try:
        citing = s2_mod.get_paper_citations(s2_id, max_results=max_results)
        if not citing:
            console.print("[dim]  No citing papers found.[/dim]")
            return []

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", padding=(0, 1))
        table.add_column("#",      width=3,  justify="right")
        table.add_column("Title",  width=52)
        table.add_column("Year",   width=6,  justify="center", style="dim")
        table.add_column("Cites",  width=7,  justify="right")
        table.add_column("Venue",  width=16, style="dim")

        for i, c in enumerate(citing, 1):
            cit_n = c.get("citations") or 0
            cit_style = "green" if cit_n >= 100 else ("yellow" if cit_n >= 10 else "dim")
            table.add_row(
                str(i),
                c.get("title") or "",
                str(c.get("year") or "N/A"),
                Text(str(cit_n), style=cit_style),
                c.get("venue") or "N/A",
            )
        console.print(table)
        console.print(f"  [dim]{len(citing)} result(s) — use [bold]show_citation <n>[/bold] for details[/dim]")
        state["last_citations"] = citing
        return citing
    except Exception as e:
        console.print(f"[red]  Failed to fetch citations: {e}[/red]")
        return None


def _parse_citation_index(args: list[str], state: dict) -> int | None:
    citations = state.get("last_citations")
    if not citations:
        print("No citation results yet. Run 'citation <n> [m]' first.")
        return None
    if not args:
        print("Usage: show_citation <n>")
        return None
    try:
        n = int(args[0])
    except ValueError:
        print(f"Invalid index: '{args[0]}'")
        return None
    if n < 1 or n > len(citations):
        print(f"Index out of range: {n}. Valid range: 1–{len(citations)}.")
        return None
    return n


def cmd_citation(args: list[str], state: dict):
    # citation <n> [m]  — n=result index, m=max citations to fetch (default 10)
    if not args:
        print("Usage: citation <n> [m]")
        return
    try:
        n = int(args[0])
    except ValueError:
        print(f"Invalid index: '{args[0]}'")
        return
    results = state.get("last_results", [])
    if not results:
        print("No search results yet. Run 'search <query>' first.")
        return
    if n < 1 or n > len(results):
        print(f"Index out of range: {n}. Valid range: 1–{len(results)}.")
        return

    max_results = 10
    if len(args) >= 2:
        try:
            max_results = int(args[1])
        except ValueError:
            print(f"Invalid count: '{args[1]}'")
            return

    _fetch_and_store_citations(results[n - 1], max_results, state)


def cmd_show_citation(args: list[str], state: dict):
    n = _parse_citation_index(args, state)
    if n is None:
        return
    print_detail(state["last_citations"][n - 1], n)


def cmd_pdf_citation(args: list[str], state: dict):
    n = _parse_citation_index(args, state)
    if n is None:
        return

    paper       = state["last_citations"][n - 1]
    initial_url = get_pdf_url(paper)

    if not initial_url:
        print("No PDF URL available for this paper.")
        print(f"  arXiv: {paper.get('arxiv_url') or 'N/A'}")
        print(f"  S2   : {paper.get('s2_url') or 'N/A'}")
        return

    import requests

    console.print(f"  Resolving: [dim]{initial_url}[/dim]")
    pdf_url = _resolve_pdf_url(initial_url)

    if not pdf_url:
        console.print(f"[yellow]  Could not locate a direct PDF from:[/yellow] {initial_url}")
        console.print(f"  Open manually: [link]{initial_url}[/link]")
        return

    if pdf_url != initial_url:
        console.print(f"  Resolved PDF: [green]{pdf_url}[/green]")

    download_dir = os.path.expanduser("~/tmp/papers")
    os.makedirs(download_dir, exist_ok=True)

    safe_id = (paper.get("arxiv_id") or paper.get("s2_id") or "paper").replace("/", "_")
    path    = os.path.join(download_dir, f"{safe_id}.pdf")

    console.print(f"  Downloading...")
    try:
        resp = requests.get(pdf_url, timeout=60, stream=True,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; paper-search/1.0)"})
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = os.path.getsize(path) // 1024
        console.print(f"  [green]Saved[/green] ({size_kb} KB): {path}")

        for opener in ["wslview", "xdg-open"]:
            try:
                subprocess.Popen([opener, path],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                break
            except FileNotFoundError:
                continue

    except Exception as e:
        console.print(f"  [red]Download failed:[/red] {e}")
        console.print(f"  Direct URL: [link]{pdf_url}[/link]")


def _format_harvard(paper: dict) -> str:
    """Format a paper dict as a Harvard-style full reference string."""
    import re

    def _fmt_author(name: str) -> str:
        name = name.strip()
        if "," in name:
            return name  # already "Last, F." form
        parts = name.split()
        if len(parts) == 1:
            return parts[0]
        last = parts[-1]
        initials = ". ".join(p[0].upper() for p in parts[:-1] if p) + "."
        return f"{last}, {initials}"

    authors = paper.get("authors") or []
    if len(authors) == 0:
        author_str = "Anon."
    elif len(authors) <= 3:
        fmt = [_fmt_author(a) for a in authors]
        if len(fmt) == 1:
            author_str = fmt[0]
        elif len(fmt) == 2:
            author_str = f"{fmt[0]} and {fmt[1]}"
        else:
            author_str = f"{fmt[0]}, {fmt[1]} and {fmt[2]}"
    else:
        author_str = f"{_fmt_author(authors[0])} et al."

    year  = paper.get("year") or "n.d."
    title = paper.get("title") or "Untitled"
    venue = paper.get("venue") or ""
    doi   = paper.get("doi") or ""
    arxiv = paper.get("arxiv_id") or ""
    s2_url = paper.get("s2_url") or ""

    ref = f"{author_str} ({year}) '{title}'"
    if venue:
        ref += f", *{venue}*"
    ref += "."
    if doi:
        ref += f" doi: {doi}."
    elif arxiv:
        ref += f" Available at: https://arxiv.org/abs/{arxiv}."
    elif s2_url:
        ref += f" Available at: {s2_url}."
    return ref


def cmd_get_reference(args: list[str], state: dict):
    n = parse_index(args, state)
    if n is None:
        return
    paper = state["last_results"][n - 1]
    ref = _format_harvard(paper)
    console.print()
    console.print(Panel(ref, title=f"[cyan]Harvard Reference #{n}[/cyan]",
                        border_style="cyan", padding=(0, 1)))


def cmd_get_citation_reference(args: list[str], state: dict):
    n = _parse_citation_index(args, state)
    if n is None:
        return
    paper = state["last_citations"][n - 1]
    ref = _format_harvard(paper)
    console.print()
    console.print(Panel(ref, title=f"[cyan]Harvard Reference (Citation #{n})[/cyan]",
                        border_style="cyan", padding=(0, 1)))


def cmd_limit(args: list[str], state: dict):
    if not args:
        print(f"Current: outputs={state['limit_outputs']}  pool={state['limit_pool']}")
        print("Usage: limit <outputs> [pool]")
        return
    try:
        state["limit_outputs"] = int(args[0])
        if len(args) >= 2:
            state["limit_pool"] = int(args[1])
        print(f"Set: outputs={state['limit_outputs']}  pool={state['limit_pool']}")
    except ValueError:
        print("Usage: limit <outputs> [pool]  — both must be integers")


def cmd_filter(args: list[str], state: dict):
    filters = state["filters"]

    if not args or args[0] == "show":
        active = {k: v for k, v in filters.items() if v not in (None, False, [], "")}
        if not active:
            print("No active filters.")
        else:
            print("Active filters:")
            for k, v in active.items():
                print(f"  {k}: {v}")
        return

    if args[0] == "clear":
        import copy
        state["filters"] = copy.deepcopy(DEFAULT_FILTERS)
        print("All filters cleared.")
        return

    key  = args[0]
    vals = args[1:]

    def _to_bool(s):
        return s.lower() in ("true", "yes", "1", "on")

    def _underscore_to_space(lst):
        return [v.replace("_", " ") for v in lst]

    if key == "year":
        if not vals:
            print("Usage: filter year <from> [to]")
            return
        filters["year_from"] = int(vals[0])
        filters["year_to"]   = int(vals[1]) if len(vals) >= 2 else None
        print(f"Year: {filters['year_from']} – {filters['year_to'] or 'present'}")

    elif key == "open_access":
        filters["open_access_only"] = _to_bool(vals[0]) if vals else True
        print(f"open_access_only = {filters['open_access_only']}")

    elif key == "categories":
        filters["categories"] = vals
        print(f"categories = {vals}")

    elif key == "fields":
        filters["fields_of_study"] = _underscore_to_space(vals)
        print(f"fields_of_study = {filters['fields_of_study']}")

    elif key == "pub_types":
        filters["publication_types"] = vals
        print(f"publication_types = {vals}")

    elif key == "venue":
        filters["venue"] = vals
        print(f"venue = {vals}")

    elif key == "min_citations":
        filters["min_citations"] = int(vals[0]) if vals else None
        print(f"min_citations = {filters['min_citations']}")

    elif key == "title_only":
        filters["title_only"]    = _to_bool(vals[0]) if vals else True
        filters["abstract_only"] = False
        print(f"title_only = {filters['title_only']}")

    elif key == "abstract_only":
        filters["abstract_only"] = _to_bool(vals[0]) if vals else True
        filters["title_only"]    = False
        print(f"abstract_only = {filters['abstract_only']}")

    else:
        print(f"Unknown filter key: '{key}'. Run 'help filters' for valid keys.")


def cmd_sort(args: list[str], state: dict):
    valid = {"relevance", "citations", "composite"}
    if not args or args[0] not in valid:
        print(f"Usage: sort <{'|'.join(valid)}>")
        print(f"Current: {state['sort_by']}")
        return
    state["sort_by"] = args[0]
    print(f"Sort mode: {args[0]}")


def cmd_weights(args: list[str], state: dict):
    """
    Usage: weights c=0.5 r=0.25 t=0.25
    Keys: c=citations, r=recency, t=title_match
    """
    if not args:
        w = state["weights"]
        print(f"Current weights: citations={w['citations']}  recency={w['recency']}  title_match={w['title_match']}")
        print("Usage: weights c=<n> r=<n> t=<n>   (must sum to 1.0)")
        return

    mapping = {"c": "citations", "r": "recency", "t": "title_match"}
    new_w = dict(state["weights"])

    for arg in args:
        if "=" not in arg:
            print(f"Invalid: '{arg}'. Expected format: c=0.5")
            return
        k, v = arg.split("=", 1)
        key  = mapping.get(k.strip())
        if not key:
            print(f"Unknown weight key '{k}'. Valid: c, r, t")
            return
        new_w[key] = float(v)

    total = sum(new_w.values())
    if abs(total - 1.0) > 0.01:
        print(f"Warning: weights sum to {total:.3f}, not 1.0. Normalizing...")
        new_w = {k: v / total for k, v in new_w.items()}

    state["weights"] = new_w
    print(f"Weights: citations={new_w['citations']:.3f}  "
          f"recency={new_w['recency']:.3f}  title_match={new_w['title_match']:.3f}")


def cmd_settings(state: dict):
    print(f"\nSettings:")
    print(f"  limit_outputs : {state['limit_outputs']}")
    print(f"  limit_pool    : {state['limit_pool']}")
    print(f"  sort_by       : {state['sort_by']}")
    w = state["weights"]
    print(f"  weights       : citations={w['citations']}  recency={w['recency']}  title_match={w['title_match']}")
    print(f"  email         : {state.get('email') or '(not set)'}")
    filters = state["filters"]
    active  = {k: v for k, v in filters.items() if v not in (None, False, [], "")}
    if active:
        print(f"  filters:")
        for k, v in active.items():
            print(f"    {k}: {v}")
    else:
        print(f"  filters       : (none active)")


# ─────────────────────────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────────────────────────

def repl():
    import copy
    state = {
        "last_results":   [],
        "last_citations": [],
        "last_query":     "",
        "limit_outputs": 10,
        "limit_pool":    50,
        "sort_by":       "composite",
        "weights":       dict(DEFAULT_WEIGHTS),
        "filters":       copy.deepcopy(DEFAULT_FILTERS),
        "email":         os.environ.get("OPENALEX_EMAIL", "sc21d2k@leeds.ac.uk"),
    }

    console.print(Panel(
        "[bold cyan]Unified Paper Search[/bold cyan]  [dim]arXiv + Semantic Scholar + OpenAlex[/dim]\n"
        "[dim]Type [bold]help[/bold] for commands, [bold]quit[/bold] to exit.[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))

    # Enable readline history if available
    try:
        import readline
        readline.set_history_length(200)
    except ImportError:
        pass

    while True:
        try:
            line = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        parts = line.split()
        cmd   = parts[0].lower()
        args  = parts[1:]

        if cmd in ("quit", "exit", "q"):
            print("Bye.")
            break
        elif cmd == "search":
            cmd_search(args, state)
        elif cmd == "show":
            cmd_show(args, state)
        elif cmd == "pdf":
            cmd_pdf(args, state)
        elif cmd == "citation":
            cmd_citation(args, state)
        elif cmd == "show_citation":
            cmd_show_citation(args, state)
        elif cmd == "pdf_citation":
            cmd_pdf_citation(args, state)
        elif cmd == "get_reference":
            cmd_get_reference(args, state)
        elif cmd == "get_citation_reference":
            cmd_get_citation_reference(args, state)
        elif cmd == "limit":
            cmd_limit(args, state)
        elif cmd == "filter":
            cmd_filter(args, state)
        elif cmd == "sort":
            cmd_sort(args, state)
        elif cmd == "weights":
            cmd_weights(args, state)
        elif cmd == "settings":
            cmd_settings(state)
        elif cmd == "email":
            if args:
                state["email"] = args[0]
                os.environ["OPENALEX_EMAIL"] = args[0]
                print(f"Email set: {args[0]}")
            else:
                print(f"Current email: {state.get('email') or '(not set)'}")
        elif cmd == "help":
            if args and args[0] == "filters":
                print_filter_help()
            else:
                print_help()
        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for commands.")


if __name__ == "__main__":
    repl()
