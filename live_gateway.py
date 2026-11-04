"""Live mode: the gateway answers ONLY from external sources it searches at question time. No stored knowledge base.

For every question:
  1. search the web with Tavily (set TAVILY_API_KEY; without a key it falls back to Wikipedia),
  2. cut the results into sentences and keep those that relate to the question,
  3. every source page counts as one independent source; identical statements from several sources add up in K,
     conflicting ones lower it (same engine as the site),
  4. the gateway decides what is allowed (K >= 1), the model only words the answer, unsupported claims are dropped.

    python live_ask.py "Столица Узбекистана — Ташкент?"
    python live_ask.py "Сколько месяцев длится гарантия?" --provider nebius     # Nemotron on Nebius Token Factory words the answer

Needs internet. Standard library only. Written against the documented MediaWiki API and tested on a recorded
fixture only: the author's environment cannot reach wikipedia.org, so run it once and check the output.
Limits: pages of one site are separate sources but not fully independent; sentences are matched to the question by
word overlap; Wikipedia is not a primary scientific source. Add Crossref/PubMed/etc. as more entries in SOURCES.
"""
import json
import re
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Tuple

from contextual_gateway import ContextualGateway, RuleBasedLLM

UA = {"User-Agent": "KappaGate/0.1 (research prototype)"}


def http_get(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


# ---------------------------------------------------------------- source connectors
def wikipedia(query: str, lang: str = "ru", n: int = 5, get: Callable = http_get) -> List[dict]:
    """-> [{"source": title, "url": ..., "text": intro extract}]"""
    url = (f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": n, "prop": "extracts|info",
        "exintro": 1, "explaintext": 1, "inprop": "url", "format": "json"}))
    pages = get(url).get("query", {}).get("pages", {})
    out = []
    for p in sorted(pages.values(), key=lambda x: x.get("index", 0)):
        if p.get("extract"):
            out.append({"source": f"{p['title']} (Википедия)", "url": p.get("fullurl", ""), "text": p["extract"]})
    return out


import os
try:
    from tavily_source import tavily
except ImportError:        # tavily_source.py not copied next to this file
    tavily = None
# Tavily when a key is set, otherwise Wikipedia. Add more connectors with the same signature.
SOURCES: List[Callable] = [tavily] if (tavily and os.environ.get("TAVILY_API_KEY")) else [wikipedia]


# ---------------------------------------------------------------- sentences related to the question
def _stems(s: str) -> set:
    try:
        import cl_ml
        return cl_ml.stems(s)
    except ImportError:
        return {w[:5] for w in re.findall(r"[a-zа-яё]+", s.lower()) if len(w) >= 4}


def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9«\"])|\n+", text) if len(s.strip()) >= 25]


def related(query: str, docs: List[dict], per_doc: int = 2, min_overlap: int = 2) -> List[Tuple[str, dict]]:
    q = _stems(query)
    picked = []
    for d in docs:
        scored = sorted(((len(q & _stems(s)), s) for s in sentences(d["text"])), reverse=True)
        picked += [(s, d) for sc, s in scored[:per_doc] if sc >= min_overlap]
    return picked


# ---------------------------------------------------------------- the live gateway
class LiveGateway:
    def __init__(self, llm=None, lang: str = "ru", get: Callable = http_get, sources=None) -> None:
        self.llm = llm or RuleBasedLLM()
        self.lang, self.get = lang, get
        self.sources = SOURCES if sources is None else sources
        self.errors: List[str] = []     # connector failures of the last ask(), so callers can tell "no data" from "search is down"

    def ask(self, query: str):
        gw = ContextualGateway(self.llm)            # a fresh, empty gateway for every question
        docs: List[dict] = []
        self.errors = []
        for src in self.sources:
            try:
                docs += src(query, self.lang, 5, self.get)
            except Exception as e:                  # one source failing must not break the answer
                self.errors.append(str(e))
                print(f"[источник недоступен: {e}]")
        facts: Dict[str, dict] = {}
        for i, (text, d) in enumerate(related(query, docs), 1):
            fid = gw.ingest(f"live_{i}", text, d["source"], d.get("cls", "external"))
            facts.setdefault(fid, {"text": gw.text[fid], "urls": []})["urls"].append((d["source"], d["url"]))
        v = gw.ask(query)
        return v, gw, facts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--provider", choices=["none", "nebius"], default="none")
    a = ap.parse_args()
    llm = None
    if a.provider != "none":
        from llm_adapters import make_llm
        llm = make_llm(a.provider)
    v, gw, facts = LiveGateway(llm, a.lang).ask(a.query)
    print(f"статус: {v.status}   K = {v.K}   множество: {v.set}")
    print(v.answer or {"paradox": "Информации нет: запрос парадоксален.", "ungrounded": "Информации нет: запрос ссылается на себя.",
                       "refuted": "Утверждение опровергнуто найденными источниками.", "no_info": "Информации нет: подтверждающих источников не найдено."}[v.status])
    for fid, f in facts.items():
        print(f"  K = {gw.system.K(fid):+d}  {f['text']}")
        for name, url in f["urls"]:
            print(f"      источник: {name}  {url}")
