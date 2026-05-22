"""
Paper-source backends.

Each source implements `BaseSource.search(query, max_results) -> list[Paper]`.
All are free APIs; some support optional API keys for higher rate limits.

To add a new source, subclass `BaseSource` and register it in `SOURCES` below.
"""

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


@dataclass
class Paper:
    """Normalized paper record returned by every source."""

    title: str
    abstract: str
    authors: list
    url: str
    date: str  # YYYY-MM-DD
    source: str
    citations: Optional[int] = None
    peer_reviewed: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class BaseSource(ABC):
    """Abstract paper-source backend."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.opts = kwargs

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list:
        """Return a list of Paper objects."""
        ...

    def _log(self, msg: str):
        print(f"      [{self.name}] {msg}")


# ── arxiv ─────────────────────────────────────────────────────────────────────


class ArxivSource(BaseSource):
    name = "arxiv"

    def search(self, query, max_results=5):
        q = urllib.parse.quote(query)
        url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query=all:{q}&max_results={max_results}&sortBy=relevance"
        )
        try:
            with urllib.request.urlopen(url, timeout=30, context=_SSL_CTX) as resp:
                tree = ET.parse(resp)
        except Exception as e:
            self._log(f"failed: {e}")
            return []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in tree.findall("atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            link = entry.findtext("atom:id", "", ns).strip()
            date = entry.findtext("atom:published", "", ns)[:10]
            if title:
                papers.append(
                    Paper(
                        title=title,
                        abstract=abstract[:500],
                        authors=authors[:3],
                        url=link,
                        date=date,
                        source="arxiv",
                        citations=None,
                        peer_reviewed=False,
                    )
                )
        return papers


# ── Semantic Scholar ──────────────────────────────────────────────────────────


class SemanticScholarSource(BaseSource):
    name = "semantic_scholar"

    def search(self, query, max_results=5):
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        q = urllib.parse.quote(query)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={q}&limit={max_results}"
            f"&fields=title,abstract,authors,year,citationCount,externalIds,openAccessPdf"
        )
        headers = {"User-Agent": "LitReviewAgents/0.1"}
        if api_key:
            headers["x-api-key"] = api_key

        data = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                    data = json.loads(resp.read())
                break
            except Exception as e:
                if "429" in str(e):
                    wait = (5 if api_key else 20) * (attempt + 1)
                    self._log(f"rate limited — waiting {wait}s ({'keyed' if api_key else 'no API key set'})")
                    time.sleep(wait)
                    continue
                self._log(f"failed: {e}")
                return []
        if data is None:
            self._log("gave up after 3 attempts — disable in config if persistent")
            return []

        papers = []
        for p in data.get("data", []):
            title = p.get("title", "").strip()
            abstract = (p.get("abstract") or "")[:500]
            authors = [a.get("name", "") for a in p.get("authors", [])[:3]]
            year = p.get("year")
            date = f"{year}-01-01" if year else "2000-01-01"
            citations = p.get("citationCount", 0)
            ext_ids = p.get("externalIds", {})
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")
            paper_id = p.get("paperId", "")
            if doi:
                url_out = f"https://doi.org/{doi}"
            elif arxiv_id:
                url_out = f"https://arxiv.org/abs/{arxiv_id}"
            else:
                url_out = f"https://www.semanticscholar.org/paper/{paper_id}"
            peer_reviewed = bool(doi and not arxiv_id)
            if title:
                papers.append(
                    Paper(
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        url=url_out,
                        date=date,
                        source="semantic_scholar",
                        citations=citations,
                        peer_reviewed=peer_reviewed,
                    )
                )
        return papers


# ── CrossRef ──────────────────────────────────────────────────────────────────


class CrossrefSource(BaseSource):
    name = "crossref"

    def search(self, query, max_results=5):
        q = urllib.parse.quote(query)
        mailto = os.environ.get("CROSSREF_MAILTO", "litreviewagents@example.com")
        url = (
            f"https://api.crossref.org/works"
            f"?query={q}&rows={max_results}&sort=relevance"
            f"&select=title,abstract,author,published,DOI,container-title,is-referenced-by-count"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": f"LitReviewAgents/0.1 (mailto:{mailto})"}
            )
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            self._log(f"failed: {e}")
            return []
        papers = []
        for item in data.get("message", {}).get("items", []):
            titles = item.get("title", [])
            title = titles[0].strip() if titles else ""
            abstract = (item.get("abstract") or "")[:500]
            abstract = abstract.replace("<jats:p>", "").replace("</jats:p>", "").strip()
            authors_raw = item.get("author", [])
            authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors_raw[:3]]
            pub_date = item.get("published", {}).get("date-parts", [[2000]])[0]
            date = f"{pub_date[0]}-{pub_date[1]:02d}-01" if len(pub_date) > 1 else f"{pub_date[0]}-01-01"
            doi = item.get("DOI", "")
            url_out = f"https://doi.org/{doi}" if doi else ""
            journal = item.get("container-title", [""])[0]
            citations = item.get("is-referenced-by-count", 0)
            if title and url_out:
                papers.append(
                    Paper(
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        url=url_out,
                        date=date,
                        source=f"crossref ({journal})" if journal else "crossref",
                        citations=citations,
                        peer_reviewed=True,
                    )
                )
        return papers


# ── PubMed ────────────────────────────────────────────────────────────────────


class PubMedSource(BaseSource):
    name = "pubmed"

    def search(self, query, max_results=5):
        q = urllib.parse.quote(query)
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={q}&retmax={max_results}&retmode=json&sort=relevance"
        )
        try:
            with urllib.request.urlopen(search_url, timeout=30, context=_SSL_CTX) as resp:
                search_data = json.loads(resp.read())
            ids = search_data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
        except Exception as e:
            self._log(f"search failed: {e}")
            return []

        ids_str = ",".join(ids)
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
        )
        try:
            with urllib.request.urlopen(fetch_url, timeout=30, context=_SSL_CTX) as resp:
                fetch_data = json.loads(resp.read())
        except Exception as e:
            self._log(f"fetch failed: {e}")
            return []

        papers = []
        result = fetch_data.get("result", {})
        for uid in ids:
            item = result.get(uid, {})
            title = item.get("title", "").strip()
            authors = [a.get("name", "") for a in item.get("authors", [])[:3]]
            date = item.get("pubdate", "2000")[:10].replace(" ", "0")[:10]
            if len(date) < 10:
                date = date[:4] + "-01-01"
            doi = next(
                (
                    id_obj.get("value", "")
                    for id_obj in item.get("articleids", [])
                    if id_obj.get("idtype") == "doi"
                ),
                "",
            )
            url_out = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
            if title:
                papers.append(
                    Paper(
                        title=title,
                        abstract="(See PubMed for abstract)",
                        authors=authors,
                        url=url_out,
                        date=date,
                        source="pubmed",
                        citations=None,
                        peer_reviewed=True,
                    )
                )
        return papers


# ── Registry ──────────────────────────────────────────────────────────────────

SOURCES = {
    "arxiv": ArxivSource,
    "semantic_scholar": SemanticScholarSource,
    "crossref": CrossrefSource,
    "pubmed": PubMedSource,
}


def get_source(name: str) -> BaseSource:
    """Instantiate a source backend by name."""
    if name not in SOURCES:
        raise ValueError(f"Unknown source: {name}. Available: {list(SOURCES.keys())}")
    return SOURCES[name]()


def register_source(name: str, cls):
    """Register a custom source backend at runtime."""
    if not issubclass(cls, BaseSource):
        raise TypeError(f"{cls} must inherit from BaseSource")
    SOURCES[name] = cls
