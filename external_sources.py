"""Real sources for Kappa Gate: turns Wikidata statements and their references into facts with sources.

Every reference attached to a Wikidata statement counts as one source, so K reflects real, openable references.
Run on your computer (needs internet). Standard library only.

    python external_sources.py "Узбекистан" P36 --out wikidata_state.json
    python external_sources.py "Земля" P2067 P1086 --lang ru --out state.json

The output file is in the same format as the site's export: import it on the site (Интеграция -> Импорт)
or use it with ask.py. Weights: external source +1 each; a statement with no references gets K = 0 (U);
a deprecated statement is skipped.

NOTE: written against the documented Wikidata JSON format and tested on a recorded fixture only.
The author's environment had no access to wikidata.org, so run it once and check the result.
"""
import argparse
import json
import urllib.parse
import urllib.request
from typing import Callable, Dict, List

API = "https://www.wikidata.org/w/api.php"
UA = {"User-Agent": "KappaGate/0.1 (research prototype)"}


def http_get(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def _q(**kw) -> str:
    return API + "?" + urllib.parse.urlencode({**kw, "format": "json"})


def find_entity(name: str, lang: str, get: Callable = http_get) -> str:
    r = get(_q(action="wbsearchentities", search=name, language=lang, uselang=lang, limit=1))
    if not r.get("search"):
        raise SystemExit(f"Wikidata ничего не нашла по запросу «{name}».")
    return r["search"][0]["id"]


def labels(ids: List[str], lang: str, get: Callable = http_get) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        r = get(_q(action="wbgetentities", ids="|".join(chunk), props="labels", languages=f"{lang}|en"))
        for k, e in r.get("entities", {}).items():
            lab = e.get("labels", {})
            out[k] = (lab.get(lang) or lab.get("en") or {}).get("value", k)
    return out


def _value_text(snak: dict, lab: Dict[str, str]):
    dv = snak.get("datavalue")
    if snak.get("snaktype") != "value" or not dv:
        return None
    v, t = dv["value"], dv.get("type")
    if t == "wikibase-entityid":
        return lab.get(v["id"], v["id"])
    if t == "string":
        return v
    if t == "quantity":
        return v["amount"].lstrip("+")
    if t == "time":
        return v["time"].lstrip("+")[:10]
    if t == "monolingualtext":
        return v["text"]
    return None


def _ref_names(ref: dict, lab: Dict[str, str], n: int) -> str:
    snaks = ref.get("snaks", {})
    for s in snaks.get("P854", []):                       # reference URL
        dv = s.get("datavalue")
        if dv:
            return urllib.parse.urlparse(dv["value"]).netloc or dv["value"]
    for s in snaks.get("P248", []):                       # stated in
        dv = s.get("datavalue")
        if dv:
            return lab.get(dv["value"]["id"], dv["value"]["id"])
    return f"Ссылка Wikidata {n}"


def collect(name: str, props: List[str], lang: str = "ru", get: Callable = http_get) -> dict:
    qid = find_entity(name, lang, get)
    claims: Dict[str, list] = {}
    for p in props:
        claims[p] = get(_q(action="wbgetclaims", entity=qid, property=p)).get("claims", {}).get(p, [])
    need = {qid, *props}
    for cl in claims.values():
        for c in cl:
            dv = c["mainsnak"].get("datavalue")
            if dv and dv.get("type") == "wikibase-entityid":
                need.add(dv["value"]["id"])
            for ref in c.get("references", []):
                for sn in ref.get("snaks", {}).get("P248", []):
                    d = sn.get("datavalue")
                    if d:
                        need.add(d["value"]["id"])
    lab = labels(sorted(need), lang, get)
    facts = []
    for p, cl in claims.items():
        for i, c in enumerate(cl, 1):
            if c.get("rank") == "deprecated":
                continue
            val = _value_text(c["mainsnak"], lab)
            if val is None:
                continue
            refs = c.get("references", [])
            names, seen = [], set()
            for n, ref in enumerate(refs, 1):
                nm = _ref_names(ref, lab, n)
                while nm in seen:
                    nm += "'"
                seen.add(nm)
                names.append(nm)
            sources = [{"name": nm, "cls": "external", "w": 1, "against": False} for nm in names] or \
                      [{"name": "Wikidata (без ссылок)", "cls": "unverified", "w": 0, "against": False}]
            facts.append({
                "id": f"wd_{qid}_{p}_{i}",
                "text": f"{lab[qid]} — {lab.get(p, p)}: {val}.",
                "K": sum(s["w"] for s in sources), "t": 0, "refs": [], "sources": sources,
                "history": [{"t": 0, "K": sum(s["w"] for s in sources)}],
            })
    return {"format": "contextual-logic/1", "engine": "external-sources", "facts": facts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="объект, например «Узбекистан»")
    ap.add_argument("props", nargs="+", help="свойства Wikidata, например P36 (столица)")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--out", default="wikidata_state.json")
    a = ap.parse_args()
    data = collect(a.name, a.props, a.lang)
    json.dump(data, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for f in data["facts"]:
        print(f"K = {f['K']:+d}  {f['text']}  источников: {len(f['sources'])} ({', '.join(s['name'] for s in f['sources'])})")
    print(f"Сохранено: {a.out} ({len(data['facts'])} фактов)")
