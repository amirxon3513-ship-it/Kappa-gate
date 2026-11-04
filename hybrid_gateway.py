"""Three modes in one place: kb (your knowledge base only), live (external search only), hybrid (both).

    python hybrid_gateway.py state.json "Сколько дней действует возврат?" --mode hybrid
    python hybrid_gateway.py state.json "Столица Узбекистана — Ташкент?" --mode live
    python hybrid_gateway.py state.json "Гарантия на технику?" --mode kb

state.json is the file exported from the Kappa Gate site (Интеграция -> Скачать JSON). Add --provider nebius
to let a real model word the answer (the gateway still decides what is allowed).

In hybrid mode all facts go into one engine: sources from your base and from the search add up in K, and the report lists
CONFLICTS between your base and external sources (for example an outdated internal document).
Needs internet for live/hybrid. Tested on recorded fixtures only: see live_gateway.py for the caveats.
"""
import json
from typing import Callable

from contextual_gateway import ContextualGateway, RuleBasedLLM, relate
from live_gateway import SOURCES, http_get, related


class HybridGateway:
    def __init__(self, state_path: str = "", llm=None, mode: str = "hybrid", lang: str = "ru", get: Callable = http_get) -> None:
        assert mode in ("kb", "live", "hybrid")
        self.state, self.llm, self.mode, self.lang, self.get = state_path, llm or RuleBasedLLM(), mode, lang, get

    def _load_kb(self, gw: ContextualGateway) -> None:
        data = json.load(open(self.state, encoding="utf-8"))
        gw.sources = {}
        for f in data["facts"]:
            refs = [(r[1:], True) if r.startswith("!") else (r, False) for r in f.get("refs", [])]
            gw.add_fact(f["id"], f["text"], int(f["K"]), int(f.get("t", 0)), refs)
            gw.sources[f["id"]] = {s["name"] for s in f.get("sources", [])}

    def ask(self, query: str):
        gw = ContextualGateway(self.llm)
        origin, conflicts = {}, []                     # fact id -> "база" | "внешний"
        if self.mode in ("kb", "hybrid"):
            self._load_kb(gw)
            origin = {i: "база" for i in gw.text}
        if self.mode in ("live", "hybrid"):
            docs = []
            for src in SOURCES:
                try:
                    docs += src(query, self.lang, 5, self.get)
                except Exception as e:
                    print(f"[источник недоступен: {e}]")
            for i, (text, d) in enumerate(related(query, docs), 1):
                rel, other = relate(text, gw.text)
                if rel == "conflict" and other in origin and origin[other] == "база":
                    conflicts.append({"kb_fact": other, "kb_text": gw.text[other], "external": text, "source": d["source"], "url": d["url"]})
                fid = gw.ingest(f"live_{i}", text, d["source"], d.get("cls", "external"))
                if rel == "conflict" and other in origin and origin[other] == "база":
                    conflicts[-1]["ext_fact"] = fid
                origin.setdefault(fid, "внешний")
                gw.urls = getattr(gw, "urls", {})
                gw.urls.setdefault(fid, []).append((d["source"], d["url"]))
        v = gw.ask(query)
        # the answer must not contain both sides of a conflict: the side with the lower K is dropped (a tie favours your base)
        losers = set()
        for c in conflicts:
            a, b = c["kb_fact"], c.get("ext_fact")
            if b and b in gw.text:
                losers.add(b if gw.system.K(b) <= gw.system.K(a) else a)
        if losers and v.claims:
            v.claims = [c for c in v.claims if not (set(c["cites"]) & losers)]
            v.answer = " ".join(c["text"] for c in v.claims)
            if v.claims:
                v.K = min(gw.system.K(x) for c in v.claims for x in c["cites"])
            else:
                v.status, v.K = "no_info", 0
        return v, gw, origin, conflicts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["kb", "live", "hybrid"], default="hybrid")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--provider", choices=["none", "nebius"], default="none")
    a = ap.parse_args()
    llm = None
    if a.provider != "none":
        from llm_adapters import make_llm
        llm = make_llm(a.provider)
    v, gw, origin, conflicts = HybridGateway(a.state, llm, a.mode, a.lang).ask(a.query)
    print(f"режим: {a.mode}   статус: {v.status}   K = {v.K}   множество: {v.set}")
    print(v.answer or {"paradox": "Информации нет: запрос парадоксален.", "ungrounded": "Информации нет: запрос ссылается на себя.",
                       "refuted": "Утверждение опровергнуто источниками.", "no_info": "Информации нет."}[v.status])
    shown = [c["cites"] for c in v.claims] if v.claims else []
    for fid in dict.fromkeys(i for cites in shown for i in cites):
        print(f"  [{origin.get(fid, '?')}] K = {gw.system.K(fid):+d}  {gw.text[fid]}")
    if conflicts:
        print("ПРОТИВОРЕЧИЯ между базой и внешними источниками:")
        for c in conflicts:
            print(f"  база ({c['kb_fact']}): {c['kb_text']}\n  внешний ({c['source']}): {c['external']}  {c['url']}")
