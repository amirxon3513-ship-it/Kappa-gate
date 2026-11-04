"""Ask the gateway a question using a real model.

    python ask.py state.json "Земля плоская?" --provider nebius
state.json is the file exported from the Kappa Gate site (Интеграция -> Скачать JSON).
"""
import argparse
import json

from contextual_gateway import ContextualGateway
from llm_adapters import make_llm


def load_gateway(path: str, llm) -> ContextualGateway:
    data = json.load(open(path, encoding="utf-8"))
    gw = ContextualGateway(llm)
    for f in data["facts"]:
        refs = [(r[1:], True) if r.startswith("!") else (r, False) for r in f.get("refs", [])]
        gw.add_fact(f["id"], f["text"], int(f["K"]), int(f.get("t", 0)), refs)
    return gw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("query")
    ap.add_argument("--provider", choices=["nebius"], default="nebius")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    v = load_gateway(a.state, make_llm(a.provider, a.model)).ask(a.query)
    print(f"статус: {v.status}   K = {v.K}   множество: {v.set}")
    print(v.answer or {"paradox": "Информации нет: запрос парадоксален.", "ungrounded": "Информации нет: запрос ссылается на себя.",
                       "refuted": "Утверждение опровергнуто базой знаний.", "no_info": "Информации нет."}[v.status])
