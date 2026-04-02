"""
PubMed E-utilities search module.

Uses NCBI Entrez API (esearch + efetch).
Set NCBI_API_KEY env var to get 10 req/s instead of 3 req/s.
"""

import os
import re
import time
import xml.etree.ElementTree as ET

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _api_key() -> str:
    return os.environ.get("NCBI_API_KEY", "")


def _get(endpoint: str, params: dict) -> requests.Response:
    key = _api_key()
    if key:
        params = {**params, "api_key": key}
    delay = 0.11 if key else 0.35   # 10 req/s with key, 3 req/s without
    resp = requests.get(endpoint, params=params, timeout=20)
    resp.raise_for_status()
    time.sleep(delay)
    return resp


def fetch_pool(query: str, filters: dict, pool_size: int) -> list[dict]:
    """Search PubMed and return normalized paper dicts."""
    params: dict = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  pool_size,
        "retmode": "json",
        "sort":    "relevance",
    }
    year_from = filters.get("year_from")
    year_to   = filters.get("year_to")
    if year_from or year_to:
        params["mindate"]  = f"{year_from or 1900}/01/01"
        params["maxdate"]  = f"{year_to or 2100}/12/31"
        params["datetype"] = "pdat"

    data  = _get(f"{EUTILS_BASE}/esearch.fcgi", params).json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    # efetch in chunks of 200 to avoid overloading the API
    papers = []
    chunk_size = 200
    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i : i + chunk_size]
        resp  = _get(f"{EUTILS_BASE}/efetch.fcgi", {
            "db":      "pubmed",
            "id":      ",".join(chunk),
            "rettype": "abstract",
            "retmode": "xml",
        })
        papers.extend(_parse_xml(resp.text))
    return papers


def _parse_xml(xml_text: str) -> list[dict]:
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return papers
    for article in root.findall(".//PubmedArticle"):
        p = _parse_article(article)
        if p:
            papers.append(p)
    return papers


def _parse_article(article) -> dict | None:
    medline = article.find("MedlineCitation")
    if medline is None:
        return None

    pmid_el = medline.find("PMID")
    pmid    = pmid_el.text.strip() if pmid_el is not None else None

    art = medline.find("Article")
    if art is None:
        return None

    title_el = art.find("ArticleTitle")
    title    = "".join(title_el.itertext()) if title_el is not None else ""

    authors = []
    for au in art.findall(".//Author"):
        last       = au.findtext("LastName") or ""
        fore       = au.findtext("ForeName") or au.findtext("Initials") or ""
        collective = au.findtext("CollectiveName")
        if last:
            initial = (fore[0] + ".") if fore else ""
            authors.append(f"{last}, {initial}" if initial else last)
        elif collective:
            authors.append(collective)

    abstract_parts = []
    for el in art.findall(".//AbstractText"):
        label = el.get("Label")
        text  = "".join(el.itertext())
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = " ".join(abstract_parts)

    journal = (art.findtext(".//Journal/Title") or
               art.findtext(".//Journal/ISOAbbreviation") or "")

    year = None
    pub_date = art.find(".//Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year_text = pub_date.findtext("Year")
        if year_text:
            try:
                year = int(year_text)
            except ValueError:
                pass
        if year is None:
            medline_date = pub_date.findtext("MedlineDate")
            if medline_date:
                m = re.search(r"\b(19|20)\d{2}\b", medline_date)
                if m:
                    year = int(m.group())

    doi    = None
    pmc_id = None
    for id_el in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = id_el.get("IdType")
        if id_type == "doi":
            doi = id_el.text
        elif id_type == "pmc":
            pmc_id = id_el.text  # e.g. "PMC1234567"

    pdf_url = None
    if pmc_id:
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"

    return {
        "title":                 title,
        "authors":               authors,
        "year":                  year,
        "published":             str(year) if year else None,
        "abstract":              abstract,
        "citations":             0,
        "influential_citations": 0,
        "source":                "pubmed",
        "pmid":                  pmid,
        "pmc_id":                pmc_id,
        "doi":                   doi,
        "venue":                 journal,
        "pubmed_url":            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
        "pdf_url":               pdf_url,
        "is_open_access":        bool(pmc_id),
        "fields_of_study":       [],
    }
