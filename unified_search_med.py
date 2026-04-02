"""
Unified Paper Search (Medical Edition)
arXiv + Semantic Scholar + OpenAlex + PubMed

Pipeline:
  1. Fetch arXiv pool
  2. Fetch S2 pools (relevance + citationCount, parallel)
  3. Fetch PubMed pool
  4. Deduplicate (arXiv ID → S2 ID → DOI → title similarity)
  5. Citation enrichment:
       arXiv-only  → S2 batch → OpenAlex fallback
       PubMed-only → OpenAlex DOI lookup
  6. Composite score → top outputs

New commands vs unified_search:
  apikey <key>   — set NCBI API key (or export NCBI_API_KEY before launch)

All other commands are identical to unified_search.
"""

import os
import sys
import copy
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.style import Style
from rich.rule import Rule
from rich.markup import escape as rich_escape

# ── reuse shared utilities from unified_search ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import unified_search as _us
import arxiv_search as arxiv_mod
import semantic_scholar_search as s2_mod
import openalex_search as oa_mod
import pubmed_search as pm_mod

console = _us.console

# Re-export constants so they are available in this module's namespace
STOP_WORDS      = _us.STOP_WORDS
RECENCY_LAMBDA  = _us.RECENCY_LAMBDA
DEFAULT_WEIGHTS = _us.DEFAULT_WEIGHTS

DEFAULT_FILTERS = {
    **_us.DEFAULT_FILTERS,
    # PubMed has no extra filter keys for now
}

# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED DEDUPLICATION (adds DOI matching on top of unified_search logic)
# ─────────────────────────────────────────────────────────────────────────────

def merge_pools(*pools: list[dict]) -> list[dict]:
    """
    Deduplicate across all pools.
    Pass 1: arXiv ID → S2 ID → title similarity (via unified_search).
    Pass 2: DOI matching for any remaining duplicates (catches PubMed ↔ S2/OA).
    """
    # Pass 1: standard dedup
    intermediate = _us.merge_pools(*pools)

    # Pass 2: DOI dedup
    seen_doi: dict[str, int] = {}
    final: list[dict] = []

    for paper in intermediate:
        doi = (paper.get("doi") or "").strip().lower()
        if doi and doi in seen_doi:
            existing = final[seen_doi[doi]]
            # Merge citation data
            if paper.get("citations", 0) > existing.get("citations", 0):
                existing["citations"]             = paper["citations"]
                existing["influential_citations"] = paper.get("influential_citations", 0)
            # Absorb PubMed-specific fields if missing
            for field in ("pmid", "pmc_id", "pubmed_url"):
                if paper.get(field) and not existing.get(field):
                    existing[field] = paper[field]
            # Absorb pdf_url if missing and PubMed has a PMC one
            if paper.get("pdf_url") and not existing.get("pdf_url"):
                existing["pdf_url"] = paper["pdf_url"]
            existing["source"] = "both"
        else:
            idx = len(final)
            final.append(paper)
            if doi:
                seen_doi[doi] = idx

    return final


# ─────────────────────────────────────────────────────────────────────────────
# OPENALEX DOI ENRICHMENT (for PubMed-only papers)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_oa_by_doi(doi: str) -> tuple[str, dict | None]:
    """Look up a single DOI on OpenAlex to get citation count."""
    import requests, time
    email = os.environ.get("OPENALEX_EMAIL", "")
    params: dict = {
        "filter": f"doi:{doi}",
        "select": "id,cited_by_count",
        "per_page": 1,
    }
    if email:
        params["mailto"] = email
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return doi, {
                "citations":   results[0].get("cited_by_count", 0),
                "openalex_id": results[0].get("id"),
            }
    except Exception:
        pass
    return doi, None


def batch_get_citations_doi(dois: list[str]) -> dict[str, dict]:
    """Fetch citation counts for a list of DOIs via OpenAlex (parallel)."""
    if not dois:
        return {}
    results = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch_oa_by_doi, doi): doi for doi in dois}
        for future in as_completed(futures):
            doi, data = future.result()
            if data:
                results[doi] = data
    return results


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY OVERRIDES
# ─────────────────────────────────────────────────────────────────────────────

def _src_badge(src: str) -> Text:
    badges = {
        "both":    ("[both]",   "bold green"),
        "s2":      ("[S2]",     "bold blue"),
        "arxiv":   ("[arXiv]",  "bold magenta"),
        "pubmed":  ("[PubMed]", "bold yellow"),
        "openalex":("[OA]",     "bold cyan"),
    }
    label, style = badges.get(src, (f"[{src}]", "dim"))
    return Text(label, style=style)


def print_detail(paper: dict, index: int):
    """Extended print_detail that shows PMID / PubMed URL for PubMed papers."""
    title   = paper.get("title") or "Unknown"
    authors = paper.get("authors") or []
    d       = paper.get("_score_detail")

    console.print()
    console.print(Panel(
        f"[bold white]{title}[/bold white]",
        title=f"[cyan]#{index}[/cyan]",
        border_style="cyan",
        padding=(0, 1),
    ))

    meta = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    meta.add_column(style="dim", width=16)
    meta.add_column()

    authors_str = ", ".join(authors[:5]) + ("..." if len(authors) > 5 else "")
    meta.add_row("Authors",   authors_str)
    meta.add_row("Year",      f"{paper.get('year') or 'N/A'}  (published: {paper.get('published') or 'N/A'})")
    meta.add_row("Venue",     paper.get("venue") or "N/A")
    meta.add_row("Source",    f"{_src_badge(paper.get('source','?'))}  |  Open Access: {paper.get('is_open_access')}")
    meta.add_row("Citations", f"[green]{paper.get('citations', 0)}[/green]  (influential: {paper.get('influential_citations', 0)})")

    cats = paper.get("categories") or paper.get("fields_of_study") or []
    if cats:
        meta.add_row("Fields", ", ".join(cats[:6]))

    console.print(meta)

    if d:
        score_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                            padding=(0, 2), expand=False)
        score_table.add_column("Component", style="dim")
        score_table.add_column("Score",     justify="right")
        score_table.add_column("Bar",       width=12)
        score_table.add_column("Weight",    justify="right", style="dim")
        score_table.add_column("Meaning")

        rows = [
            ("Citation",  d["citation"],    "citations (log-normalized + influential)", "0.40"),
            ("Recency",   d["recency"],     "pub year (exponential decay, λ=0.15)",     "0.25"),
            ("Relevance", d["title_match"], "token+pair match in title & abstract",     "0.35"),
        ]
        for name, val, meaning, weight in rows:
            c = _us._score_color(val)
            score_table.add_row(
                name,
                f"[{c}]{val:.3f}[/{c}]",
                Text(_us._score_bar(val, 12), style=c),
                weight,
                f"[dim]{meaning}[/dim]",
            )

        total = d["total"]
        c = _us._score_color(total)
        score_table.add_row(
            "[bold]Total[/bold]",
            f"[bold {c}]{total:.3f}[/bold {c}]",
            Text(_us._score_bar(total, 12), style="bold " + c),
            "",
            "[dim]weighted composite[/dim]",
        )
        console.print(Panel(score_table, title="[dim]Score Breakdown[/dim]",
                            border_style="dim", padding=(0, 0)))

    links = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    links.add_column(style="dim", width=16)
    links.add_column()
    links.add_row("arXiv ID",   paper.get("arxiv_id") or "N/A")
    links.add_row("S2 ID",      paper.get("s2_id") or "N/A")
    links.add_row("PMID",       paper.get("pmid") or "N/A")
    links.add_row("PMC ID",     paper.get("pmc_id") or "N/A")
    links.add_row("DOI",        paper.get("doi") or "N/A")
    links.add_row("arXiv URL",  f"[link]{paper.get('arxiv_url') or 'N/A'}[/link]")
    links.add_row("S2 URL",     f"[link]{paper.get('s2_url') or 'N/A'}[/link]")
    links.add_row("PubMed URL", f"[link]{paper.get('pubmed_url') or 'N/A'}[/link]")
    resolved_pdf = _us.get_pdf_url(paper)
    links.add_row(
        "PDF URL",
        Text(resolved_pdf, style="bold green") if resolved_pdf else Text("N/A", style="dim"),
    )
    console.print(links)

    abstract = paper.get("abstract") or "N/A"
    console.print(Panel(
        abstract,
        title="[dim]Abstract[/dim]",
        border_style="dim",
        padding=(0, 1),
    ))
    console.print()


def print_summary(results: list[dict]):
    """Like unified_search.print_summary but uses local _src_badge."""
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
    table.add_column("#",      style="bold white", width=3,  justify="right")
    table.add_column("Title",                      width=48)
    table.add_column("Year",   style="dim",         width=6,  justify="center")
    table.add_column("Cites",                       width=7,  justify="right")
    table.add_column("PDF",                         width=4,  justify="center")
    table.add_column("Src",                         width=9,  justify="center")
    table.add_column("Score",                       width=22, justify="left")

    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        year  = str(r.get("year") or "N/A")
        cites = r.get("citations", 0)
        src   = r.get("source", "?")

        if cites >= 1000:  cite_style = "bold green"
        elif cites >= 100: cite_style = "green"
        elif cites >= 10:  cite_style = "yellow"
        else:              cite_style = "dim"

        if "_score" in r:
            total = r["_score"]
            bar   = _us._score_bar(total)
            score_text = Text()
            score_text.append(bar, style=_us._score_color(total))
            score_text.append(f" {total:.2f}", style="bold " + _us._score_color(total))
            if "_score_detail" in r:
                d = r["_score_detail"]
                score_text.append(
                    f"  c:{d['citation']:.2f} r:{d['recency']:.2f} t:{d['title_match']:.2f}",
                    style="dim",
                )
        else:
            score_text = Text("—", style="dim")

        pdf_avail = Text("PDF", style="bold green") if _us.get_pdf_url(r) else Text("—", style="dim")

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


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH ORCHESTRATION (adds PubMed as step 3)
# ─────────────────────────────────────────────────────────────────────────────

def run_search(query: str, state: dict) -> list[dict]:
    pool_size = state["limit_pool"]
    filters   = state["filters"]
    plain_q   = _us._plain_query(query)

    console.print()
    console.print(Rule(f"[bold cyan]Search: {rich_escape(query)}[/bold cyan]"))
    console.print(
        f"  Pool [dim]per source[/dim]: [yellow]{pool_size}[/yellow]  "
        f"Output: [yellow]{state['limit_outputs']}[/yellow]  "
        f"Sort: [yellow]{state['sort_by']}[/yellow]"
    )
    console.print()

    # 1. arXiv
    console.print("[bold][[1/5]][/bold] Fetching [cyan]arXiv[/cyan] pool...", end=" ")
    try:
        arxiv_pool = arxiv_mod.fetch_pool(plain_q, filters, pool_size)
        console.print(f"[green]✓[/green] {len(arxiv_pool)} papers")
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        arxiv_pool = []

    # 2. Semantic Scholar (relevance + citationCount in parallel)
    console.print("[bold][[2/5]][/bold] Fetching [cyan]Semantic Scholar[/cyan] pools in parallel...")
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

    # 3. PubMed
    console.print("[bold][[3/5]][/bold] Fetching [yellow]PubMed[/yellow] pool...", end=" ")
    try:
        pubmed_pool = pm_mod.fetch_pool(plain_q, filters, pool_size)
        console.print(f"[green]✓[/green] {len(pubmed_pool)} papers")
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        pubmed_pool = []

    # 4. Merge & deduplicate (arXiv ID → S2 ID → DOI → title similarity)
    console.print("[bold][[4/5]][/bold] Merging and deduplicating...", end=" ")
    merged  = merge_pools(arxiv_pool, s2_rel, s2_cit, pubmed_pool)
    n_arxiv  = sum(1 for p in merged if p["source"] == "arxiv")
    n_both   = sum(1 for p in merged if p["source"] == "both")
    n_s2     = sum(1 for p in merged if p["source"] == "s2")
    n_pubmed = sum(1 for p in merged if p["source"] == "pubmed")
    console.print(
        f"[green]✓[/green] [bold]{len(merged)}[/bold] unique  "
        f"[dim](arXiv: {n_arxiv}  S2: {n_s2}  PubMed: {n_pubmed}  both: {n_both})[/dim]"
    )

    # 5a. Citation enrichment — arXiv-only papers (S2 batch → OA fallback)
    arxiv_only = [p for p in merged if p["source"] == "arxiv" and p.get("arxiv_id")]
    # 5b. Citation enrichment — PubMed-only papers with a DOI (OA DOI lookup)
    pubmed_only_doi = [
        p for p in merged
        if p["source"] == "pubmed" and p.get("doi")
    ]

    if arxiv_only or pubmed_only_doi:
        console.print(
            f"[bold][[5/5]][/bold] Citation enrichment — "
            f"[yellow]{len(arxiv_only)}[/yellow] arXiv-only, "
            f"[yellow]{len(pubmed_only_doi)}[/yellow] PubMed-only papers..."
        )

        # arXiv: S2 batch first
        still_missing = []
        if arxiv_only:
            arxiv_only_ids = [p["arxiv_id"] for p in arxiv_only]
            s2_enriched    = 0
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
                console.print(
                    f"  [cyan]S2 batch[/cyan]: "
                    f"[green]{s2_enriched}[/green]/{len(arxiv_only)} enriched"
                )
            except Exception as e:
                console.print(f"  [cyan]S2 batch[/cyan]: [red]✗ {e}[/red]")
                still_missing = arxiv_only

        # arXiv: OpenAlex fallback
        if still_missing:
            console.print(
                f"  [cyan]OpenAlex arXiv fallback[/cyan]: {len(still_missing)} papers...",
                end=" ",
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

        # PubMed: OpenAlex DOI lookup
        if pubmed_only_doi:
            console.print(
                f"  [cyan]OpenAlex DOI lookup[/cyan]: {len(pubmed_only_doi)} PubMed papers...",
                end=" ",
            )
            try:
                doi_map     = batch_get_citations_doi([p["doi"] for p in pubmed_only_doi])
                doi_enriched = 0
                for p in pubmed_only_doi:
                    hit = doi_map.get((p["doi"] or "").strip().lower()) or doi_map.get(p["doi"] or "")
                    if hit:
                        p["citations"]   = hit["citations"]
                        p["openalex_id"] = hit.get("openalex_id")
                        doi_enriched += 1
                console.print(f"[green]{doi_enriched}[/green] enriched")
            except Exception as e:
                console.print(f"[red]✗ {e}[/red]")
    else:
        console.print("[bold][[5/5]][/bold] No arXiv/PubMed-only papers — skipping enrichment")

    results = _us.composite_sort(merged, query, state["weights"], state["sort_by"], state["limit_outputs"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS (pass-through or overrides)
# ─────────────────────────────────────────────────────────────────────────────

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
    n = _us.parse_index(args, state)
    if n is None:
        return
    print_detail(state["last_results"][n - 1], n)


def cmd_show_citation(args: list[str], state: dict):
    n = _us._parse_citation_index(args, state)
    if n is None:
        return
    print_detail(state["last_citations"][n - 1], n)


def print_help():
    print("""
Commands:
  search <query>              Search arXiv + Semantic Scholar + PubMed
  show <n>                    Full details for result n
  pdf <n>                     Download PDF for result n
  citation <n> [m]            Show m papers citing result n (default 10, needs S2 ID)
  show_citation <n>           Full details for citation n
  pdf_citation <n>            Download PDF for citation n
  get_reference <n>           Harvard reference for result n
  get_citation_reference <n>  Harvard reference for citation n
  limit <outputs> [pool]      Set output count and pool size per source
  sort <mode>                 relevance | citations | composite
  weights [c=N r=N t=N]       Composite weights (must sum to 1.0)
  email <addr>                Set email for OpenAlex polite pool
  settings                    Show current settings and filters
  help filters                Show filter command details
  quit / exit                 Exit

Filters (use 'filter <key> [values]'):
  filter year <from> [to]     e.g. 'filter year 2022 2024'
  filter open_access true     Only open-access papers
  filter categories cs.LG ... arXiv categories
  filter fields "Computer Science" ...  S2 fields of study
  filter pub_types Conference ...       S2 publication types
  filter venue NeurIPS ...              S2 venue filter
  filter min_citations <n>    Minimum citation count (S2 only)
  filter title_only true      Match query in title only
  filter clear                Reset all filters
  filter show                 Display active filters
""")


# ─────────────────────────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────────────────────────

def repl():
    state = {
        "last_results":   [],
        "last_citations": [],
        "last_query":     "",
        "limit_outputs":  10,
        "limit_pool":     50,
        "sort_by":        "composite",
        "weights":        dict(DEFAULT_WEIGHTS),
        "filters":        copy.deepcopy(DEFAULT_FILTERS),
        "email":          os.environ.get("OPENALEX_EMAIL", ""),
    }

    console.print(Panel(
        "[bold cyan]Unified Paper Search — Medical Edition[/bold cyan]\n"
        "[dim]arXiv + Semantic Scholar + OpenAlex + [yellow]PubMed[/yellow][/dim]\n"
        "[dim]Type [bold]help[/bold] for commands, [bold]quit[/bold] to exit.[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))

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
            _us.cmd_pdf(args, state)
        elif cmd == "citation":
            _us.cmd_citation(args, state)
        elif cmd == "show_citation":
            cmd_show_citation(args, state)
        elif cmd == "pdf_citation":
            _us.cmd_pdf_citation(args, state)
        elif cmd == "get_reference":
            _us.cmd_get_reference(args, state)
        elif cmd == "get_citation_reference":
            _us.cmd_get_citation_reference(args, state)
        elif cmd == "limit":
            _us.cmd_limit(args, state)
        elif cmd == "filter":
            _us.cmd_filter(args, state)
        elif cmd == "sort":
            _us.cmd_sort(args, state)
        elif cmd == "weights":
            _us.cmd_weights(args, state)
        elif cmd == "settings":
            _us.cmd_settings(state)
        elif cmd == "email":
            if args:
                state["email"] = args[0]
                os.environ["OPENALEX_EMAIL"] = args[0]
                print(f"Email set: {args[0]}")
            else:
                print(f"Current email: {state.get('email') or '(not set)'}")
        elif cmd == "help":
            if args and args[0] == "filters":
                _us.print_filter_help()
            else:
                print_help()
        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for commands.")


if __name__ == "__main__":
    repl()
