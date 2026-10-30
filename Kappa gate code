#!/usr/bin/env python3
"""Kappa Gate on Nebius: the website's gateway as a command-line tool and a small Python API.

NVIDIA Nemotron models on Nebius Token Factory annotate the query and word the answer. The Kappa Gate engine
(context K, sets T/U/F, Axiom 4) decides which facts are allowed into the answer. Standard library only (Python 3.8+).
Needs the `gateway/` folder from this project next to this file.

    export NEBIUS_API_KEY=...                       # Nebius Token Factory key
    export TAVILY_API_KEY=tvly-...                  # only for the `live` command

    python3 kappa_gate_nebius.py models             # which Nemotron models your account offers
    python3 kappa_gate_nebius.py demo               # four queries against the built-in demo knowledge base
    python3 kappa_gate_nebius.py ask "How many months is the warranty on appliances?"
    python3 kappa_gate_nebius.py ask --kb my_state.json "..."   # state exported from the website (Integration -> Download JSON)
    python3 kappa_gate_nebius.py live "What is the capital of Uzbekistan?"   # web evidence through Tavily

As a library:

    from kappa_gate_nebius import load_kb, gate
    gw, evidence = load_kb("examples/demo_kb_en.json")
    verdict = gate(gw, "How many months is the warranty on appliances?")
    print(verdict.status, verdict.K, verdict.answer)

Pick a model with NEBIUS_MODEL, otherwise the largest Nemotron model of your account is used.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gateway"))

import contextual_gateway as cg          # noqa: E402  the K-engine (same axioms as the website)
import llm_adapters as la                # noqa: E402  Nemotron on Nebius Token Factory

BASE = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
DEMO_KB = HERE / "examples" / "demo_kb_en.json"
DEMO_QUERIES = [
    "How many months is the warranty on appliances?",      # confirmed fact, with sources for and against
    "Is the warranty on appliances lifetime and unlimited?",   # a claim the base marks as false
    "This statement is false.",                              # liar paradox: K = 0
    "What is the battery life of the X200 phone?",           # no data: K = 0, "no information"
]


# ---------------------------------------------------------------- Nebius
def list_models():
    key = os.environ.get("NEBIUS_API_KEY", "")
    if not key:
        raise SystemExit("Set NEBIUS_API_KEY.")
    req = urllib.request.Request(BASE + "/models", headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return [m.get("id", "") for m in json.load(r).get("data", [])]
    except urllib.error.URLError as e:
        raise SystemExit("Could not reach Nebius Token Factory: %s" % e)


def _size(model_id):
    m = re.findall(r"(\d+(?:\.\d+)?)b(?![a-z])", model_id.lower())
    return float(m[0]) if m else 0.0


def pick_model():
    """NEBIUS_MODEL if set, else the largest Nemotron text model available to the account."""
    if os.environ.get("NEBIUS_MODEL"):
        return os.environ["NEBIUS_MODEL"]
    skip = ("embed", "rerank", "reward", "guard", "safety", "vl", "omni", "parse", "image")
    nem = sorted((i for i in list_models() if "nemotron" in i.lower() and not any(s in i.lower() for s in skip)), key=_size)
    return nem[-1] if nem else la.DEFAULT_MODELS["nebius"]


class GuardedLLM(la.ModelLLM):
    """Nemotron for analysis and wording, plus a deterministic guard: an obvious self-referential query
    ("this statement is false") is stopped before it reaches the model."""

    def analyze(self, query, facts):
        guard = cg.RuleBasedLLM().analyze(query, [])
        if any(s.get("asserts") for s in guard.get("statements", [])):
            return guard
        return super().analyze(query, facts)


# ---------------------------------------------------------------- knowledge base
def load_kb(path=DEMO_KB, model=None):
    """Load a state file exported from the website. Returns (gateway, evidence) where evidence[id] holds the
    confirming and refuting sources of each fact (the same lists the website shows)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    gw = cg.ContextualGateway(GuardedLLM(la.nebius_call(model or pick_model(), os.environ.get("NEBIUS_API_KEY", ""))))
    evidence = {}
    for f in data["facts"]:
        refs = [(r[1:], True) if r.startswith("!") else (r, False) for r in f.get("refs", [])]
        gw.add_fact(f["id"], f["text"], int(f["K"]), int(f.get("t", 0)), refs)
        src = f.get("sources", [])
        evidence[f["id"]] = {"confirm": [s for s in src if not s.get("against") and s.get("w", 0) > 0],
                             "refute": [s for s in src if s.get("against")],
                             "weak": [s for s in src if not s.get("against") and s.get("w", 0) <= 0]}
    return gw, evidence


def gate(gw, query):
    """The whole gateway in one call: Nemotron annotates, the engine decides, Nemotron words the answer from admitted facts."""
    return gw.ask(query)


# ---------------------------------------------------------------- output
def fmt_k(k):
    return ("+%d" % k) if k > 0 else str(k)


def src_line(label, items):
    if not items:
        return "%s 0" % label
    return "%s %d: %s" % (label, len(items), ", ".join("%s (%s)" % (s["name"], fmt_k(s["w"])) for s in items))


def show(gw, evidence, query, v):
    print("\nQuery   : %s" % query)
    print("Verdict : %s   K = %s   set %s" % (v.status.upper(), fmt_k(v.K), v.set))
    print("Answer  : %s" % (v.answer or {"paradox": "No information: the query is paradoxical (K = 0).",
                                          "ungrounded": "No information: the query refers to itself (K = 0).",
                                          "refuted": "Refuted by the knowledge base. Nothing is passed to the model.",
                                          "no_info": "No information: no confirmed fact (K >= 1) for this query."}[v.status]))
    rows = [(i, "T") for c in v.claims for i in c["cites"]] + [(i, "F") for i in v.blocked] + [(i, "U") for i in v.quarantined]
    seen = set()
    for fid, s in rows:
        if fid in seen or fid not in gw.text:
            continue
        seen.add(fid)
        e = evidence.get(fid, {})
        note = {"T": "admitted to the answer", "F": "blocked, never reaches the model", "U": "quarantine"}[s]
        print("  %s %-4s %s   [%s: %s]" % (s, fmt_k(gw.system.K(fid)), gw.text[fid], fid, note))
        if e:
            print("         %s | %s" % (src_line("confirm", e["confirm"]), src_line("refute", e["refute"])))
            if e["weak"]:
                print("         %s" % src_line("no weight / outdated", e["weak"]))


# ---------------------------------------------------------------- live web (Tavily)
def run_live(query, model):
    import live_gateway as lg
    if not os.environ.get("TAVILY_API_KEY"):
        raise SystemExit("Set TAVILY_API_KEY for the live command.")
    llm = GuardedLLM(la.nebius_call(model, os.environ["NEBIUS_API_KEY"]))
    live = lg.LiveGateway(llm, "en", sources=[lg.tavily])
    v, gw, found = live.ask(query)
    if live.errors and not found:
        raise SystemExit("Web search failed: %s" % live.errors[0])
    import tavily_source as ts
    evidence = {}
    for fid, f in found.items():     # real weights: the class of each page's domain, as on the website
        ws = [{"name": n, "w": cg.SOURCE_WEIGHTS[ts.classify_url(u)]} for n, u in f["urls"]]
        evidence[fid] = {"confirm": [x for x in ws if x["w"] > 0], "refute": [], "weak": [x for x in ws if x["w"] <= 0]}
    show(gw, evidence, query, v)
    for fid, f in found.items():
        for name, url in f["urls"]:
            print("         source: %s  %s" % (name, url))
    print("\nNote: here facts from different pages are matched by word overlap. The website's Live web mode lets Nemotron\n"
          "group the facts that agree and the ones that contradict, which is more reliable.")


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="Kappa Gate on Nebius Token Factory (NVIDIA Nemotron).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list Nemotron models available on your Nebius account")
    sub.add_parser("demo", help="run four demo queries against the built-in knowledge base")
    p = sub.add_parser("ask", help="ask a question against a knowledge base")
    p.add_argument("query")
    p.add_argument("--kb", default=str(DEMO_KB), help="state file exported from the website (default: demo knowledge base)")
    p = sub.add_parser("live", help="answer from the web (Tavily) through the gateway")
    p.add_argument("query")
    a = ap.parse_args()

    if a.cmd == "models":
        ids = list_models()
        nem = [i for i in ids if "nemotron" in i.lower()]
        print("\n".join(nem) if nem else "No Nemotron models found among %d models." % len(ids))
        print("\nSelected by default: %s" % pick_model())
        return
    if not os.environ.get("NEBIUS_API_KEY"):
        raise SystemExit("Set NEBIUS_API_KEY.")
    model = pick_model()
    print("Model: %s  (Nebius Token Factory)" % model)
    if a.cmd == "live":
        return run_live(a.query, model)
    gw, evidence = load_kb(DEMO_KB if a.cmd == "demo" else a.kb, model)
    for q in (DEMO_QUERIES if a.cmd == "demo" else [a.query]):
        show(gw, evidence, q, gate(gw, q))


if __name__ == "__main__":
    main()
