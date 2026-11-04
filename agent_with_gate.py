#!/usr/bin/env python3
"""A tiny agent that must ask Kappa Gate before it answers a factual question. Standard library only.

The agent is NVIDIA Nemotron on Nebius Token Factory with ONE tool, `kappa_gate`. The tool calls the gate's
POST /api/gate; the agent is told to answer only from what the gate returns. The gate (not the agent) decides what is
verified: facts with K >= 1 are returned, paradoxes, refuted claims and unknowns come back as "cannot confirm".

    export NEBIUS_API_KEY=...                     # the agent's own model key
    export KAPPA_GATE_URL=http://localhost:8080   # where server.py runs (default)
    export KAPPA_GATE_TOKEN=...                   # only if the server sets GATE_API_TOKEN
    python3 examples/agent_with_gate.py "What is the capital of Uzbekistan?"
    python3 examples/agent_with_gate.py --facts examples/facts.json "How many months is the warranty?"

Without --facts the gate searches the web with Tavily (live mode). With --facts it answers from your own facts.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

GATE = os.environ.get("KAPPA_GATE_URL", "http://localhost:8080").rstrip("/")
NEBIUS = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")

TOOL = {"type": "function", "function": {
    "name": "kappa_gate",
    "description": "Check a factual question against verified sources BEFORE answering. Returns status "
                   "(answer, refuted, no_info, paradox, ungrounded), K (confidence level, higher is better) and the verified answer with its sources.",
    "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "The factual question to verify."}}, "required": ["question"]}}}

SYSTEM = ("You are a careful assistant. For every factual question you MUST call the kappa_gate tool first and answer ONLY from "
          "its result, citing the fact ids. If status is not 'answer', say that you cannot confirm it and why "
          "(no information, refuted by the sources, or a self-referential question). Never add facts from your own knowledge.")


def post(url, body, headers=None, timeout=120):
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %d from %s: %s" % (e.code, url, e.read().decode("utf-8", "replace")[:300]))


def call_gate(question, facts=None):
    headers = {"Authorization": "Bearer " + os.environ["KAPPA_GATE_TOKEN"]} if os.environ.get("KAPPA_GATE_TOKEN") else {}
    body = {"query": question, "mode": "facts", "facts": facts} if facts else {"query": question, "mode": "live"}
    r = post(GATE + "/api/gate", body, headers)
    # keep the tool result short and free of anything the model could mistake for instructions
    return {"status": r["status"], "K": r["K"], "answer": r["answer"], "claims": r["claims"],
            "blocked_ids": r["blocked"], "quarantined_ids": r["quarantined"],
            "facts": [{"id": f["id"], "text": f["text"], "K": f["K"]} for f in r["facts"] if f["id"] in {i for c in r["claims"] for i in c["cites"]}]}


def run(question, facts=None, model=None, max_rounds=4):
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise SystemExit("Set NEBIUS_API_KEY.")
    if not model:
        model = os.environ.get("NEBIUS_MODEL") or (json.load(urllib.request.urlopen(GATE + "/api/status", timeout=20)).get("models") or {}).get("complex") \
            or "nvidia/nemotron-3-ultra-550b-a55b"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(max_rounds):
        r = post(NEBIUS + "/chat/completions", {"model": model, "messages": msgs, "tools": [TOOL], "temperature": 0, "max_tokens": 1500},
                 {"Authorization": "Bearer " + key})
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        if not calls:
            return (m.get("content") or "").strip()
        msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": calls})
        for c in calls:
            try:
                q = json.loads(c["function"]["arguments"]).get("question") or question
            except ValueError:
                q = question
            result = call_gate(q, facts) if c["function"]["name"] == "kappa_gate" else {"error": "unknown tool"}
            print("[gate] %r -> status=%s K=%s" % (q, result.get("status"), result.get("K")), file=sys.stderr)
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False)})
    return "(the agent did not finish within %d rounds)" % max_rounds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--facts", help="JSON file with a list of facts; without it the gate searches the web with Tavily")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    facts = json.load(open(a.facts, encoding="utf-8")) if a.facts else None
    print(run(a.question, facts, a.model or None))
