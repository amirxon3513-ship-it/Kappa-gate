#!/usr/bin/env python3
"""Mock of Nebius Token Factory (/v1/models, /v1/chat/completions) and Tavily (/search) for offline tests.

It is NOT a language model: it recognizes the gateway's prompts by their first words and answers with canned,
rule-based JSON so that the whole pipeline (browser -> server.py -> upstream -> browser) can be tested without keys.

    REJECT_EXTRA=1  makes chat/completions answer 400 when chat_template_kwargs/response_format are present,
                    to test the server's retry without optional parameters.
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = []
REJECT_EXTRA = os.environ.get("REJECT_EXTRA") == "1"


def words(s):
    return {w for w in re.findall(r"[a-zа-яё]{4,}", s.lower())}


def facts_in(prompt, header):
    block = prompt.split(header, 1)[1] if header in prompt else ""
    out = []
    for line in block.splitlines():
        m = re.match(r"- ([\w.\-]+): (.+)", line)
        if m:
            out.append((m.group(1), m.group(2)))
        elif out and not line.strip():
            break
    return out


def answer(prompt):
    if prompt.startswith("Ты — модуль разбора запросов"):
        q = prompt.split("Запрос пользователя:", 1)[1].split("<<<", 1)[1].split(">>>", 1)[0].strip()
        facts = facts_in(prompt, "База знаний (id: текст):")
        selfref = re.search(r"this statement is false|это утверждение ложно", q, re.I)
        stm = [{"id": "q1", "text": q, "asserts": [{"ref": "q1", "neg": True}] if selfref else []}]
        rel = [i for i, t in facts if words(t) & words(q)]
        return json.dumps({"statements": stm, "relevant_ids": rel, "claims": []})
    if prompt.startswith("Ответь на вопрос пользователя, используя ТОЛЬКО"):
        facts = facts_in(prompt, "Подтверждённые факты (id: текст):")
        return json.dumps({"claims": [{"text": t, "cites": [i]} for i, t in facts[:2]]})
    if prompt.startswith("Ты — ассистент с доступом"):
        return "MOCK naive answer built from the whole context."
    if prompt.startswith("You extract facts"):
        out = []
        for m in re.finditer(r"^\[(\d+)\] [^:]+: (.+)$", prompt, re.M):
            for sent in re.split(r"(?<=\.)\s+", m.group(2).strip())[:2]:
                if sent:
                    out.append({"src": int(m.group(1)), "text": sent})
        return json.dumps({"facts": out})
    if prompt.startswith("Below are statements"):
        items = re.findall(r"^(\d+): (.+)$", prompt, re.M)
        same, conf = [], []
        seen = {}
        for i, t in items:
            seen.setdefault(t.lower(), []).append(int(i))
        same = [v for v in seen.values() if len(v) > 1]
        caps = [(int(i), t) for i, t in items if "capital" in t.lower()]
        for a in range(len(caps)):
            for b in range(a + 1, len(caps)):
                if caps[a][1].lower() != caps[b][1].lower():
                    conf.append([caps[a][0], caps[b][0]])
        return json.dumps({"same": same, "conflicts": conf})
    if prompt.startswith("Раздели документ"):
        text = prompt.split("<<<", 1)[1].split(">>>", 1)[0]
        return json.dumps({"facts": [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]})
    return "{}"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        LOG.append(("GET", self.path, self.headers.get("Authorization")))
        if self.path.endswith("/models"):
            if self.headers.get("Authorization") != "Bearer test-nebius":
                return self._j(401, {"error": "bad key"})
            ids = ["meta-llama/Llama-3.3-70B-Instruct", "nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-nano-omni",
                   "nvidia/nemotron-3-nano-30b-a3b", "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-embed-1b"]
            return self._j(200, {"data": [{"id": i} for i in ids]})
        if self.path == "/__log":
            return self._j(200, LOG)
        self._j(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        LOG.append(("POST", self.path, self.headers.get("Authorization"), {k: v for k, v in body.items() if k != "messages"}))
        if self.path.endswith("/chat/completions"):
            if self.headers.get("Authorization") != "Bearer test-nebius":
                return self._j(401, {"error": "bad key"})
            if REJECT_EXTRA and ("chat_template_kwargs" in body or "response_format" in body):
                return self._j(400, {"error": "unsupported parameter"})
            if body.get("tools"):  # tool-calling agent: first ask for the tool, then answer from its result
                last = body["messages"][-1]
                if last["role"] == "tool":
                    res = json.loads(last["content"])
                    return self._j(200, {"choices": [{"message": {"content": "GATE SAYS status=%s K=%s: %s" % (res["status"], res["K"], res["answer"])}}]})
                q = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
                return self._j(200, {"choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "kappa_gate", "arguments": json.dumps({"question": q})}}]}}]})
            prompt = body["messages"][-1]["content"]
            return self._j(200, {"choices": [{"message": {"content": "<think>reasoning that must be stripped</think>\n" + answer(prompt)}}]})
        if self.path.endswith("/search"):
            if self.headers.get("Authorization") != "Bearer tvly-test":
                return self._j(401, {"detail": "bad key"})
            q = body.get("query", "")
            res = [
                {"title": "About Uzbekistan", "url": "https://www.gov.uz/en/about", "content": "Tashkent is the capital of Uzbekistan. The country has about 36 million people.", "score": 0.9},
                {"title": "Uzbekistan | Britannica", "url": "https://www.britannica.com/place/Uzbekistan", "content": "Tashkent is the capital of Uzbekistan.", "score": 0.8},
                {"title": "Is it Samarkand? : r/travel", "url": "https://www.reddit.com/r/travel/comments/x", "content": "Samarkand is the capital of Uzbekistan.", "score": 0.4},
            ] if "uzbek" in q.lower() else []
            return self._j(200, {"query": q, "results": res})
        self._j(404, {})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9101
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
