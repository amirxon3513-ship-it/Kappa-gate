"""Tavily web search as the live source of the Kappa Gate. Standard library only.

    export TAVILY_API_KEY=tvly-...
    python -c "from tavily_source import tavily; print(tavily('capital of Uzbekistan'))"

Every result page is ONE independent source, named by its domain. The domain also decides the source class,
and the class decides the weight that goes into K (the same table as the knowledge base):

    official    +3   government, universities, international bodies, major journals
    external    +1   everything else (news, company sites, encyclopedias)
    unverified   0   forums, social networks, blog platforms: K = 0, so the statement is quarantined (set U)

This is a transparent heuristic, not a truth oracle. Two pages of one site are still one source name here.
"""
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Callable, Dict, List

API = os.environ.get("TAVILY_URL", "https://api.tavily.com/search")   # TAVILY_URL only for tests

OFFICIAL_SUFFIXES = (".gov", ".edu", ".int", ".mil")
OFFICIAL_TLD_PARTS = (".gov.", ".edu.", ".ac.", ".go.")
OFFICIAL_HOSTS = {
    "who.int", "un.org", "europa.eu", "worldbank.org", "imf.org", "oecd.org", "nih.gov", "cdc.gov", "nasa.gov",
    "nature.com", "science.org", "thelancet.com", "nejm.org", "bmj.com", "ieee.org", "acm.org", "iso.org",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "britannica.com",
}
UNVERIFIED_HOSTS = {
    "reddit.com", "quora.com", "medium.com", "facebook.com", "x.com", "twitter.com", "tiktok.com", "instagram.com",
    "youtube.com", "pinterest.com", "blogspot.com", "wordpress.com", "substack.com", "tumblr.com", "t.me",
    "vk.com", "answers.yahoo.com", "linkedin.com", "stackexchange.com",
}


def host_of(url: str) -> str:
    h = (urllib.parse.urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def _under(host: str, domains) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def classify_url(url: str) -> str:
    """-> 'official' | 'external' | 'unverified'"""
    h = host_of(url)
    if not h:
        return "unverified"
    if _under(h, UNVERIFIED_HOSTS):
        return "unverified"
    if h.endswith(OFFICIAL_SUFFIXES) or any(p in "." + h for p in OFFICIAL_TLD_PARTS) or _under(h, OFFICIAL_HOSTS):
        return "official"
    return "external"


def _post(url: str, headers: Dict[str, str], body: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def search(query: str, n: int = 5, depth: str = "basic", key: str = "", post: Callable = _post) -> List[dict]:
    """Raw Tavily call -> [{"title", "url", "content", "score"}]"""
    key = key or os.environ.get("TAVILY_API_KEY", "")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    r = post(API, {"Authorization": f"Bearer {key}"},
             {"query": query[:400], "search_depth": depth, "max_results": max(1, min(int(n), 10)),
              "include_answer": False, "include_raw_content": False})
    return [x for x in r.get("results", []) if x.get("url") and x.get("content")]


def tavily(query: str, lang: str = "en", n: int = 5, get=None, key: str = "", post: Callable = _post) -> List[dict]:
    """Connector with the same shape as live_gateway.wikipedia -> [{"source", "url", "text", "cls"}].
    `lang` and `get` are accepted for interface compatibility."""
    out = []
    for x in search(query, n, key=key, post=post):
        out.append({"source": host_of(x["url"]) or x.get("title", "web"), "url": x["url"],
                    "text": re.sub(r"\s+", " ", x["content"]).strip(), "cls": classify_url(x["url"]),
                    "title": x.get("title", ""), "score": x.get("score")})
    return out
