"""
PubMed Paper Search — interactive REPL powered by NCBI E-utilities.

Pipeline:
  1. Fetch PubMed pool
  2. Citation enrichment via OpenAlex DOI lookup
  3. Composite score → top outputs

Commands:
  search <query>            – run a search with current settings
  show <n>                  – show full details of result n
  pdf <n>                   – download and open PDF of result n
  get_reference <n>         – Harvard reference for result n
  limit <outputs> [pool]    – set output count and pool size
  filter <key> [values...]  – set a filter (see 'help filters')
  sort <mode>               – relevance | citations | composite
  weights [c=0.4 r=0.25 t=0.35] – composite weights
  email <your@email.com>    – set email for OpenAlex polite pool
  settings                  – show current settings
  help                      – show this help
  quit                      – exit
"""

import os
import sys
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.rule import Rule
from rich.markup import escape as rich_escape

sys.path.insert(0, os.path.dirname(__file__))

import unified_search as _us
import pubmed_search as pm_mod

console = _us.console

DEFAULT_WEIGHTS = _us.DEFAULT_WEIGHTS
DEFAULT_FILTERS = {
    "year_from":  None,
    "year_to":    None,
    "open_access_only": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# OPENALEX DOI ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_oa_by_doi(doi: str) -> tuple[str, dict | None]:
    import requests
    email = os.environ.get("OPENALEX_EMAIL", "")
    params: dict = {
        "filter":   f"doi:{doi}",
        "select":   "id,cited_by_count",
        "per_page": 1,
    }
    if email:
        params["mailto"] = email
    try:
        resp = requests.get("https://api.openalex.org/works", params=params, timeout=15)
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
    if not dois:
        return {}
    out = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch_oa_by_doi, doi): doi for doi in dois}
        for f in as_completed(futures):
            doi, data = f.result()
            if data:
                out[doi] = data
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

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
    table.add_column("#",     style="bold white", width=3,  justify="right")
    table.add_column("Title",                     width=52)
    table.add_column("Year",  style="dim",         width=6,  justify="center")
    table.add_column("Cites",                      width=7,  justify="right")
    table.add_column("OA",                         width=4,  justify="center")
    table.add_column("Score",                      width=22, justify="left")

    for i, r in enumerate(results, 1):
        cites = r.get("citations", 0)
        if cites >= 1000:  cite_style = "bold green"
        elif cites >= 100: cite_style = "green"
        elif cites >= 10:  cite_style = "yellow"
        else:              cite_style = "dim"

        if "_score" in r:
            total = r["_score"]
            score_text = Text()
            score_text.append(_us._score_bar(total), style=_us._score_color(total))
            score_text.append(f" {total:.2f}", style="bold " + _us._score_color(total))
            if "_score_detail" in r:
                d = r["_score_detail"]
                score_text.append(
                    f"  c:{d['citation']:.2f} r:{d['recency']:.2f} t:{d['title_match']:.2f}",
                    style="dim",
                )
        else:
            score_text = Text("—", style="dim")

        oa = Text("OA", style="bold green") if r.get("is_open_access") else Text("—", style="dim")

        table.add_row(
            str(i),
            r.get("title") or "",
            str(r.get("year") or "N/A"),
            Text(str(cites), style=cite_style),
            oa,
            score_text,
        )

    console.print()
    console.print(table)
    console.print(f"  [dim]{len(results)} result(s) — use [bold]show <n>[/bold] for details[/dim]")


def print_detail(paper: dict, index: int):
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
    meta.add_row("Authors",    authors_str)
    meta.add_row("Year",       str(paper.get("year") or "N/A"))
    meta.add_row("Journal",    paper.get("venue") or "N/A")
    meta.add_row("Open Access",str(paper.get("is_open_access", False)))
    meta.add_row("Citations",  f"[green]{paper.get('citations', 0)}[/green]")
    console.print(meta)

    if d:
        score_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                            padding=(0, 2), expand=False)
        score_table.add_column("Component", style="dim")
        score_table.add_column("Score",     justify="right")
        score_table.add_column("Bar",       width=12)
        score_table.add_column("Weight",    justify="right", style="dim")
        score_table.add_column("Meaning")

        for name, val, meaning, weight in [
            ("Citation",  d["citation"],    "citations (log-normalized)", "0.40"),
            ("Recency",   d["recency"],     "pub year (exp decay λ=0.15)", "0.25"),
            ("Relevance", d["title_match"], "token+pair match in title & abstract", "0.35"),
        ]:
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
            "", "[dim]weighted composite[/dim]",
        )
        console.print(Panel(score_table, title="[dim]Score Breakdown[/dim]",
                            border_style="dim", padding=(0, 0)))

    links = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    links.add_column(style="dim", width=16)
    links.add_column()
    links.add_row("PMID",       paper.get("pmid") or "N/A")
    links.add_row("PMC ID",     paper.get("pmc_id") or "N/A")
    links.add_row("DOI",        paper.get("doi") or "N/A")
    links.add_row("PubMed URL", f"[link]{paper.get('pubmed_url') or 'N/A'}[/link]")
    pdf = _us.get_pdf_url(paper)
    links.add_row(
        "PDF URL",
        Text(pdf, style="bold green") if pdf else Text("N/A", style="dim"),
    )
    console.print(links)

    console.print(Panel(
        paper.get("abstract") or "N/A",
        title="[dim]Abstract[/dim]",
        border_style="dim",
        padding=(0, 1),
    ))
    console.print()


def print_help():
    print("""
Commands:
  search <query>             Search PubMed
  show <n>                   Full details for result n
  pdf <n>                    Download PDF for result n
  get_reference <n>          Harvard reference for result n
  limit <outputs> [pool]     Set output count and pool size
                             e.g. 'limit 10 50' → top 10 from 50-paper pool
  sort <mode>                relevance | citations | composite
  weights [c=N r=N t=N]      Composite weights (must sum to 1.0)
                             c=citations, r=recency, t=title_match
  email <addr>               Set email for OpenAlex polite pool (faster enrichment)
  settings                   Show current settings and filters
  help filters               Show filter command details
  quit / exit                Exit

Filters (use 'filter <key> [values]'):
  filter year <from> [to]    e.g. 'filter year 2022 2024'
  filter open_access true    Only open-access (PMC) papers
  filter clear               Reset all filters
  filter show                Display active filters
""")


def print_filter_help():
    print("""
Filter keys:

  year <from> [to]
      e.g. filter year 2020 2024  →  papers published 2020–2024
           filter year 2022       →  papers from 2022 onwards

  open_access true|false
      Only return papers available in PubMed Central (free full text)

  clear    Reset all filters to defaults
  show     Display current active filters
""")


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_search(query: str, state: dict) -> list[dict]:
    pool_size = state["limit_pool"]
    filters   = state["filters"]
    plain_q   = _us._plain_query(query)

    console.print()
    console.print(Rule(f"[bold cyan]Search: {rich_escape(query)}[/bold cyan]"))
    console.print(
        f"  Pool: [yellow]{pool_size}[/yellow]  "
        f"Output: [yellow]{state['limit_outputs']}[/yellow]  "
        f"Sort: [yellow]{state['sort_by']}[/yellow]"
    )
    console.print()

    # 1. PubMed
    console.print("[bold][[1/2]][/bold] Fetching [yellow]PubMed[/yellow] pool...", end=" ")
    try:
        papers = pm_mod.fetch_pool(plain_q, filters, pool_size)
        console.print(f"[green]✓[/green] {len(papers)} papers")
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        return []

    # 2. Citation enrichment via OpenAlex DOI lookup
    with_doi = [p for p in papers if p.get("doi")]
    if with_doi:
        console.print(
            f"[bold][[2/2]][/bold] Citation enrichment — "
            f"[yellow]{len(with_doi)}[/yellow] papers with DOI...",
            end=" ",
        )
        try:
            doi_map  = batch_get_citations_doi([p["doi"] for p in with_doi])
            enriched = 0
            for p in with_doi:
                hit = doi_map.get(p["doi"])
                if hit:
                    p["citations"]   = hit["citations"]
                    p["openalex_id"] = hit.get("openalex_id")
                    enriched += 1
            console.print(f"[green]{enriched}[/green] enriched")
        except Exception as e:
            console.print(f"[red]✗ {e}[/red]")
    else:
        console.print("[bold][[2/2]][/bold] No DOIs found — skipping enrichment")

    return _us.composite_sort(papers, query, state["weights"], state["sort_by"], state["limit_outputs"])


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
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


def cmd_filter(args: list[str], state: dict):
    filters = state["filters"]

    if not args or args[0] == "show":
        active = {k: v for k, v in filters.items() if v not in (None, False)}
        if not active:
            print("No active filters.")
        else:
            print("Active filters:")
            for k, v in active.items():
                print(f"  {k}: {v}")
        return

    if args[0] == "clear":
        state["filters"] = copy.deepcopy(DEFAULT_FILTERS)
        print("All filters cleared.")
        return

    key  = args[0]
    vals = args[1:]

    if key == "year":
        if not vals:
            print("Usage: filter year <from> [to]")
            return
        filters["year_from"] = int(vals[0])
        filters["year_to"]   = int(vals[1]) if len(vals) >= 2 else None
        print(f"Year: {filters['year_from']} – {filters['year_to'] or 'present'}")
    elif key == "open_access":
        filters["open_access_only"] = (vals[0].lower() in ("true", "yes", "1")) if vals else True
        print(f"open_access_only = {filters['open_access_only']}")
    else:
        print(f"Unknown filter key: '{key}'. Run 'help filters' for valid keys.")


def cmd_settings(state: dict):
    print(f"\nSettings:")
    print(f"  limit_outputs : {state['limit_outputs']}")
    print(f"  limit_pool    : {state['limit_pool']}")
    print(f"  sort_by       : {state['sort_by']}")
    w = state["weights"]
    print(f"  weights       : citations={w['citations']}  recency={w['recency']}  title_match={w['title_match']}")
    print(f"  email         : {state.get('email') or '(not set)'}")
    active = {k: v for k, v in state["filters"].items() if v not in (None, False)}
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
    state = {
        "last_results": [],
        "last_query":   "",
        "limit_outputs": 10,
        "limit_pool":    50,
        "sort_by":       "composite",
        "weights":       dict(DEFAULT_WEIGHTS),
        "filters":       copy.deepcopy(DEFAULT_FILTERS),
        "email":         os.environ.get("OPENALEX_EMAIL", ""),
    }

    console.print(Panel(
        "[bold cyan]PubMed Paper Search[/bold cyan]  "
        "[dim]powered by NCBI E-utilities + OpenAlex[/dim]\n"
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
        elif cmd == "get_reference":
            _us.cmd_get_reference(args, state)
        elif cmd == "limit":
            _us.cmd_limit(args, state)
        elif cmd == "filter":
            cmd_filter(args, state)
        elif cmd == "sort":
            _us.cmd_sort(args, state)
        elif cmd == "weights":
            _us.cmd_weights(args, state)
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
