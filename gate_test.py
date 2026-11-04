#!/usr/bin/env python3
"""Tests for POST /api/gate and the agent example. Starts the mock upstream and servers itself; needs no keys.
    python3 tests/gate_test.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOCK_PORT, SRV, SRV_AUTH, SRV_DEAD = 9111, 8111, 8112, 8113
procs, failed = [], 0


def start(args, env):
    p = subprocess.Popen(args, env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    return p


def call(port, path, body=None, token=None, method="POST"):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), json.dumps(body).encode() if body is not None else None, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def chat_calls():
    return len([x for x in call(MOCK_PORT, "/__log", method="GET")[1] if x[0] == "POST" and x[1].endswith("/chat/completions")])


def ok(cond, msg):
    global failed
    print(("PASS " if cond else "FAIL ") + msg)
    failed += 0 if cond else 1


try:
    base = {"NEBIUS_API_KEY": "test-nebius", "TAVILY_API_KEY": "tvly-test", "NEBIUS_BASE_URL": "http://127.0.0.1:%d/v1" % MOCK_PORT,
            "TAVILY_URL": "http://127.0.0.1:%d/search" % MOCK_PORT}
    start([sys.executable, str(ROOT / "tests/mock_upstream.py"), str(MOCK_PORT)], {})
    start([sys.executable, str(ROOT / "server.py")], {**base, "PORT": str(SRV)})
    start([sys.executable, str(ROOT / "server.py")], {**base, "PORT": str(SRV_AUTH), "GATE_API_TOKEN": "s3cret"})
    start([sys.executable, str(ROOT / "server.py")], {**base, "PORT": str(SRV_DEAD), "NEBIUS_BASE_URL": "http://127.0.0.1:1/v1"})
    time.sleep(2)

    # 1. facts with explicit K: true fact answers, refuted claim is blocked
    facts = [{"id": "water_boil", "text": "Вода при нормальном давлении кипит при 100 °C.", "K": 120},
             {"id": "flat_earth", "text": "Земля плоская.", "K": -95}]
    st, r = call(SRV, "/api/gate", {"query": "При какой температуре кипит вода?", "facts": facts})
    ok(st == 200 and r["status"] == "answer" and r["K"] == 120 and r["claims"][0]["cites"] == ["water_boil"], "facts mode: verified answer with citation, K=120")
    ok(r["model"] and "nemotron" in r["model"], "answer produced by a Nemotron model (%s)" % r["model"])
    st, r = call(SRV, "/api/gate", {"query": "Земля плоская?", "facts": facts})
    ok(r["status"] == "refuted" and r["K"] < 0 and "flat_earth" in r["blocked"] and r["answer"] == "", "facts mode: refuted claim is blocked, no answer text")

    # 2. automatic K from sources: two sources for the same claim add up, an unverified one stays at 0
    src = [{"id": "w1", "text": "The warranty on appliances is 24 months.", "source": "Policy", "cls": "official"},
           {"id": "w2", "text": "The warranty on appliances is 24 months.", "source": "FAQ", "cls": "internal"},
           {"id": "r1", "text": "Rumour: the shop gives a lifetime guarantee.", "source": "forum", "cls": "unverified"}]
    st, r = call(SRV, "/api/gate", {"query": "How long is the warranty on appliances?", "facts": src})
    kw = {f["id"]: f["K"] for f in r["facts"]}
    ok(st == 200 and r["status"] == "answer" and kw.get("w1") == 5 and "w2" not in kw and kw.get("r1") == 0, "sources add up (3+2=5), duplicate merged, unverified rumour K=0: %s" % kw)
    ok("r1" not in [i for c in r["claims"] for i in c["cites"]], "unverified rumour is not cited in the answer")

    # 3. paradox: answered by the deterministic guard, no model call at all
    before = chat_calls()
    st, r = call(SRV, "/api/gate", {"query": "Это утверждение ложно", "facts": facts})
    ok(r["status"] == "paradox" and r["K"] == 0 and r["set"] == "U" and chat_calls() == before, "paradox -> K=0, zero model calls")
    before = chat_calls()
    st, r = call(SRV, "/api/gate", {"query": "This statement is false", "mode": "live"})
    ok(r["status"] == "paradox" and chat_calls() == before, "paradox in live mode skips the web and the model")

    # 4. live mode through Tavily
    st, r = call(SRV, "/api/gate", {"query": "What is the capital of Uzbekistan?", "mode": "live"})
    ok(st == 200 and r["status"] == "answer" and "Tashkent" in r["answer"] and r["K"] >= 4, "live mode: Tashkent, K=%s" % r.get("K"))
    ok(any(s["name"] == "gov.uz" for f in r["facts"] for s in f.get("sources", [])), "live facts carry their source domains")

    # 5. validation
    for body, label in [({"query": ""}, "empty query"), ({"query": "x", "facts": []}, "empty facts"),
                        ({"query": "x", "facts": [{"id": "a", "text": "t"}]}, "fact without K or source"),
                        ({"query": "x", "facts": [{"id": "a b", "text": "t", "K": 1}]}, "bad id"),
                        ({"query": "x", "facts": [{"id": "a", "text": "t", "K": 1.5}]}, "non-integer K"),
                        ({"query": "x", "facts": [{"id": "a", "text": "t", "source": "s", "cls": "bogus"}]}, "unknown cls"),
                        ({"query": "x", "mode": "other"}, "unknown mode")]:
        st, r = call(SRV, "/api/gate", body)
        ok(st == 400 and r["error"]["code"] == "bad_request", "400 for " + label)

    # 6. bearer token
    st, _ = call(SRV_AUTH, "/api/gate", {"query": "x", "facts": facts})
    ok(st == 401, "401 without token")
    st, _ = call(SRV_AUTH, "/api/gate", {"query": "x", "facts": facts}, token="wrong")
    ok(st == 401, "401 with a wrong token")
    st, r = call(SRV_AUTH, "/api/gate", {"query": "При какой температуре кипит вода?", "facts": facts}, token="s3cret")
    ok(st == 200 and r["status"] == "answer", "200 with the right token")
    st, r = call(SRV_AUTH, "/api/status", method="GET")
    ok(r.get("gate_auth") is True, "/api/status advertises gate_auth")

    # 7. an unreachable model must be an error, never a fake "no information"
    st, r = call(SRV_DEAD, "/api/gate", {"query": "При какой температуре кипит вода?", "facts": facts})
    ok(st == 502 and "error" in r, "model unreachable -> HTTP 502 (got %s)" % st)

    # 8. the agent example: model calls the tool, the tool calls the gate, the model answers from the result
    env = {**os.environ, "NEBIUS_API_KEY": "test-nebius", "NEBIUS_BASE_URL": "http://127.0.0.1:%d/v1" % MOCK_PORT,
           "KAPPA_GATE_URL": "http://127.0.0.1:%d" % SRV, "NEBIUS_MODEL": "nvidia/test"}
    out = subprocess.run([sys.executable, str(ROOT / "examples/agent_with_gate.py"), "What is the capital of Uzbekistan?"], env=env, capture_output=True, text=True, timeout=60)
    ok(out.returncode == 0 and "Tashkent" in out.stdout and "status=answer" in out.stdout, "agent (live): %s" % out.stdout.strip()[:100])
    out = subprocess.run([sys.executable, str(ROOT / "examples/agent_with_gate.py"), "--facts", str(ROOT / "examples/facts.json"), "How long is the warranty on appliances?"],
                         env=env, capture_output=True, text=True, timeout=60)
    ok(out.returncode == 0 and "24 months" in out.stdout and "12 months" not in out.stdout, "agent (own facts): %s" % out.stdout.strip()[:100])
    out = subprocess.run([sys.executable, str(ROOT / "examples/agent_with_gate.py"), "This statement is false"], env=env, capture_output=True, text=True, timeout=60)
    ok(out.returncode == 0 and "status=paradox" in out.stdout, "agent: paradox comes back as paradox: %s" % out.stdout.strip()[:80])
finally:
    for p in procs:
        p.terminate()
print("\n%s" % ("%d FAILED" % failed if failed else "ALL PASSED"))
sys.exit(1 if failed else 0)
