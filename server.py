#!/usr/bin/env python3
"""Kappa Gate server: static site + thin API proxy.

Why a server: API keys must never reach the browser. The page calls /api/chat
(Nebius Token Factory, NVIDIA Nemotron models) and /api/search (Tavily); this
process holds the keys and talks to them.

Standard library only (Python 3.8+), nothing to install.

Environment variables
  NEBIUS_API_KEY        required for AI mode (Nebius Token Factory key)
  TAVILY_API_KEY        required for Live web mode (Tavily key)
  NEBIUS_BASE_URL       default https://api.tokenfactory.nebius.com/v1
  NEBIUS_MODEL_QUICK    optional, Nemotron model id for the "Fast" tier
  NEBIUS_MODEL_DEFAULT  optional, Nemotron model id for the "Standard" tier
  NEBIUS_MODEL_COMPLEX  optional, Nemotron model id for the "Complex" tier
  GATE_API_TOKEN        optional; if set, POST /api/gate requires 'Authorization: Bearer <token>'
  PORT                  default 8080
  HOST                  default 0.0.0.0
  RATE_LIMIT_PER_MIN    default 120 (per client IP, protects your keys)

If the model ids are not set, the server lists the models available to your
Nebius account and picks NVIDIA Nemotron models by size.
"""
import hmac
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent / "web"
NEBIUS_BASE = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
TAVILY_URL = os.environ.get("TAVILY_URL", "https://api.tavily.com/search")
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))
MAX_BODY = 200_000
FALLBACK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
TIERS = ("quick", "default", "complex")

_hits = {}
_hits_lock = threading.Lock()
_models_cache = {"at": 0.0, "map": None}
_models_lock = threading.Lock()


def nebius_key():
    return os.environ.get("NEBIUS_API_KEY", "").strip()


def tavily_key():
    return os.environ.get("TAVILY_API_KEY", "").strip()


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def http_json(url, payload=None, headers=None, method="POST", timeout=90):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "kappa-gate/1.0")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:600]
        raise ApiError(e.code, "upstream_http", body)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ApiError(502, "upstream_error", str(e))


# ---------------------------------------------------------------- models
def _size_of(model_id):
    """Parameter count in billions parsed from ids like ...-550b-a55b (active params ignored)."""
    m = re.findall(r"(\d+(?:\.\d+)?)b(?![a-z])", model_id.lower())
    return float(m[0]) if m else 0.0


def discover_models():
    """Map tier -> Nemotron model id. Env vars win; otherwise ask Nebius which models exist."""
    env = {t: os.environ.get("NEBIUS_MODEL_" + t.upper(), "").strip() for t in TIERS}
    if all(env.values()):
        return env
    now = time.time()
    with _models_lock:
        if _models_cache["map"] and now - _models_cache["at"] < 600:
            cached = dict(_models_cache["map"])
            return {t: env[t] or cached[t] for t in TIERS}
    ids = []
    try:
        res = http_json(NEBIUS_BASE + "/models", headers={"Authorization": "Bearer " + nebius_key()}, method="GET", timeout=20)
        ids = [m.get("id", "") for m in res.get("data", [])]
    except ApiError:
        ids = []
    skip = ("embed", "rerank", "reward", "guard", "safety", "vl", "omni", "parse", "image")
    nem = [i for i in ids if "nemotron" in i.lower() and not any(s in i.lower() for s in skip)]
    nem.sort(key=_size_of)
    if nem:
        picked = {"quick": nem[0], "default": nem[len(nem) // 2], "complex": nem[-1]}
    else:
        picked = {t: FALLBACK_MODEL for t in TIERS}
    with _models_lock:
        _models_cache.update(at=now, map=picked)
    return {t: env[t] or picked[t] for t in TIERS}


# ---------------------------------------------------------------- chat
THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def clean_text(s):
    s = THINK_RE.sub("", s or "")
    if "</think>" in s:  # reasoning without an opening tag
        s = s.split("</think>", 1)[1]
    return s.strip()


def extract_json(text):
    text = clean_text(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        pass
    a = text.find("{")
    while a != -1:
        depth, in_str, esc = 0, False, False
        for i in range(a, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[a:i + 1])
                    except ValueError:
                        break
        a = text.find("{", a + 1)
    raise ApiError(502, "invalid_json", "The model returned a non-JSON answer.")


def chat(prompt, tier, want_json):
    if not nebius_key():
        raise ApiError(503, "not_configured", "NEBIUS_API_KEY is not set on the server.")
    model = discover_models().get(tier) or FALLBACK_MODEL
    system = "You are a precise assistant. Follow the output format exactly. /no_think"
    if want_json:
        system += " Reply with a single valid JSON object and nothing else."
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 3000,
    }
    attempts = [dict(body, chat_template_kwargs={"enable_thinking": False}, **({"response_format": {"type": "json_object"}} if want_json else {}))]
    if want_json:
        attempts.append(dict(body, chat_template_kwargs={"enable_thinking": False}))
    attempts.append(body)
    last = None
    for payload in attempts:
        try:
            res = http_json(NEBIUS_BASE + "/chat/completions", payload, {"Authorization": "Bearer " + nebius_key()}, timeout=120)
            msg = (res.get("choices") or [{}])[0].get("message") or {}
            text = clean_text(msg.get("content") or "")
            if not text:
                raise ApiError(502, "empty_completion", "The model returned no text.")
            return model, text
        except ApiError as e:
            last = e
            if e.status in (400, 422):  # parameter not supported by this model: retry simpler
                continue
            raise
    raise last


# ---------------------------------------------------------------- search
def search(query, max_results):
    if not tavily_key():
        raise ApiError(503, "not_configured", "TAVILY_API_KEY is not set on the server.")
    res = http_json(TAVILY_URL, {
        "query": query,
        "search_depth": "basic",
        "max_results": max(1, min(int(max_results or 6), 10)),
        "include_answer": False,
        "chunks_per_source": 2,
    }, {"Authorization": "Bearer " + tavily_key()}, timeout=40)
    out = []
    for r in res.get("results", []):
        url = str(r.get("url") or "")
        host = urlparse(url).netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        out.append({"title": str(r.get("title") or host)[:200], "url": url, "domain": host,
                    "content": str(r.get("content") or "")[:1800], "score": r.get("score")})
    return out


# ---------------------------------------------------------------- gate API (for agents and RAG backends)
GATEWAY_DIR = Path(__file__).resolve().parent / "gateway"
SAFE_ID = re.compile(r"^[\w.\-]{1,64}$")


def _gw_modules():
    if str(GATEWAY_DIR) not in sys.path:
        sys.path.insert(0, str(GATEWAY_DIR))
    import contextual_gateway
    import live_gateway
    import llm_adapters
    return contextual_gateway, live_gateway, llm_adapters


def _clip(s, n):
    return str(s or "").strip()[:n]


def run_gate(data):
    """POST /api/gate. Two modes:
      {"query": "...", "mode": "facts", "facts": [{"id","text","K"} | {"id","text","source","cls"}]}
      {"query": "...", "mode": "live"}                 (Tavily search, needs TAVILY_API_KEY)
    The model (Nemotron on Nebius) only annotates the query and words the answer; the K-engine decides."""
    cg, lg, la = _gw_modules()
    query = _clip(data.get("query"), 400)
    if not query:
        raise ApiError(400, "bad_request", "query is required (max 400 characters).")
    mode = data.get("mode") or ("facts" if "facts" in data else "live")
    if mode not in ("facts", "live"):
        raise ApiError(400, "bad_request", "mode must be 'facts' or 'live'.")
    tier = data.get("tier") if data.get("tier") in TIERS else "default"

    # deterministic safety net first: an obvious self-referential query never reaches the model or the web
    guard = cg.RuleBasedLLM().analyze(query, [])
    if any(st.get("asserts") for st in guard.get("statements", [])):
        return {"status": "paradox", "set": "U", "K": 0, "answer": "", "claims": [], "removed": [], "blocked": [], "quarantined": [],
                "facts": [], "warnings": [], "model": None}
    if not nebius_key():
        raise ApiError(503, "not_configured", "NEBIUS_API_KEY is not set on the server.")
    if mode == "live" and not tavily_key():
        raise ApiError(503, "not_configured", "TAVILY_API_KEY is not set on the server.")

    errors, models = [], []

    def call(prompt):
        try:
            model, text = chat(prompt, tier, True)
            models.append(model)
            return text
        except ApiError as e:
            errors.append(e)
            raise

    llm = la.ModelLLM(call)
    if mode == "live":
        live = lg.LiveGateway(llm, _clip(data.get("lang"), 5) or "en", sources=[lg.tavily])
        v, gw, found = live.ask(query)
        if live.errors and not found:  # the search itself failed: say so instead of a fake "no information"
            raise ApiError(502, "upstream_error", "Web search failed: " + live.errors[0][:200])
        urls = {fid: [{"name": n, "url": u} for n, u in f["urls"]] for fid, f in found.items()}
    else:
        raw = data.get("facts")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 100:
            raise ApiError(400, "bad_request", "facts must be a list of 1 to 100 items.")
        gw, urls, seen = cg.ContextualGateway(llm), {}, set()
        for i, f in enumerate(raw):
            if not isinstance(f, dict):
                raise ApiError(400, "bad_request", "each fact must be an object.")
            fid, text = str(f.get("id") or "f%d" % (i + 1)), _clip(f.get("text"), 600)
            if not SAFE_ID.match(fid) or fid in seen or not text:
                raise ApiError(400, "bad_request", "fact %d: need a unique id (letters, digits, _ - .) and text." % (i + 1))
            seen.add(fid)
            refs = [(str(r)[1:], True) if str(r).startswith("!") else (str(r), False) for r in (f.get("refs") or [])][:20]
            if "K" in f:
                if not isinstance(f["K"], int) or isinstance(f["K"], bool) or abs(f["K"]) > 1000:
                    raise ApiError(400, "bad_request", "fact %d: K must be an integer between -1000 and 1000." % (i + 1))
                gw.add_fact(fid, text, f["K"], 0, refs)
            elif f.get("source"):
                kind = f.get("cls") or "external"
                if kind not in cg.SOURCE_WEIGHTS:
                    raise ApiError(400, "bad_request", "fact %d: cls must be one of %s." % (i + 1, ", ".join(cg.SOURCE_WEIGHTS)))
                gw.ingest(fid, text, _clip(f["source"], 120), kind)
            else:
                raise ApiError(400, "bad_request", "fact %d: give either K or source (+ optional cls)." % (i + 1))
        v = gw.ask(query)

    if errors and v.status == "no_info":  # the model was unreachable: say so instead of a fake "no information"
        raise errors[0]
    facts_out = [{"id": fid, "text": gw.text[fid], "K": gw.system.K(fid), "set": cg.set_of(gw.system.K(fid)),
                  **({"sources": urls[fid]} if fid in urls else {})} for fid in gw.text]
    return {
        "status": v.status, "set": v.set, "K": v.K, "answer": v.answer,
        "claims": [{"text": _clip(c.get("text"), 600), "cites": [x for x in c.get("cites", []) if isinstance(x, str)]} for c in v.claims],
        "removed": [{"text": _clip(c.get("text"), 600)} for c in v.stripped if isinstance(c, dict)],
        "blocked": v.blocked, "quarantined": v.quarantined, "facts": facts_out,
        "warnings": ["model call failed: %s" % e.code for e in errors][:3], "model": models[-1] if models else None,
    }


# ---------------------------------------------------------------- http
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8"}


def rate_ok(ip):
    now = time.time()
    with _hits_lock:
        q = [t for t in _hits.get(ip, []) if now - t < 60]
        if len(q) >= RATE_LIMIT:
            _hits[ip] = q
            return False
        q.append(now)
        _hits[ip] = q
        if len(_hits) > 5000:
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                _hits.pop(k, None)
    return True


class Handler(BaseHTTPRequestHandler):
    server_version = "KappaGate/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _client_ip(self):
        fwd = self.headers.get("X-Forwarded-For", "")
        return fwd.split(",")[0].strip() if fwd else self.client_address[0]

    def _send(self, status, payload, ctype="application/json; charset=utf-8", extra=None):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _err(self, e):
        self._send(e.status if 400 <= e.status < 600 else 502, {"error": {"code": e.code, "message": e.message}})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            tiers = {}
            if nebius_key():
                try:
                    tiers = discover_models()
                except Exception:
                    tiers = {}
            return self._send(200, {"nebius": bool(nebius_key()), "tavily": bool(tavily_key()), "models": tiers,
                                    "gate": True, "gate_auth": bool(os.environ.get("GATE_API_TOKEN", "").strip()),
                                    "provider": "Nebius Token Factory", "base_url": NEBIUS_BASE})
        if path == "/healthz":
            return self._send(200, {"ok": True})
        if path in ("/", ""):
            path = "/index_en.html"
        target = (ROOT / path.lstrip("/")).resolve()
        if ROOT not in target.parents or not target.is_file():
            return self._send(404, b"Not found", "text/plain; charset=utf-8")
        self._send(200, target.read_bytes(), MIME.get(target.suffix.lower(), "application/octet-stream"))

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/chat", "/api/search", "/api/gate"):
            return self._send(404, {"error": {"code": "not_found", "message": "Not found"}})
        token = os.environ.get("GATE_API_TOKEN", "").strip()
        if path == "/api/gate" and token:
            sent = self.headers.get("Authorization", "")
            if not hmac.compare_digest(sent.encode(), ("Bearer " + token).encode()):
                return self._send(401, {"error": {"code": "unauthorized", "message": "Missing or wrong bearer token."}})
        if not rate_ok(self._client_ip()):
            return self._send(429, {"error": {"code": "rate_limited", "message": "Too many requests. Wait a minute."}})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                raise ApiError(413, "bad_request", "Request body is empty or too large.")
            data = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError
        except ApiError as e:
            return self._err(e)
        except ValueError:
            return self._err(ApiError(400, "bad_request", "Invalid JSON body."))
        try:
            if path == "/api/gate":
                return self._send(200, run_gate(data))
            if path == "/api/chat":
                prompt = str(data.get("prompt") or "")
                if not prompt.strip() or len(prompt) > 60_000:
                    raise ApiError(400, "bad_request", "Prompt is empty or too long.")
                tier = data.get("tier") if data.get("tier") in TIERS else "default"
                want_json = bool(data.get("json"))
                model, text = chat(prompt, tier, want_json)
                if want_json:
                    return self._send(200, {"model": model, "json": extract_json(text)})
                return self._send(200, {"model": model, "text": text})
            query = str(data.get("query") or "").strip()
            if not query or len(query) > 400:
                raise ApiError(400, "bad_request", "Query is empty or longer than 400 characters.")
            return self._send(200, {"results": search(query, data.get("max_results"))})
        except ApiError as e:
            return self._err(e)
        except Exception as e:  # never leak a traceback or a key
            sys.stderr.write("internal error: %r\n" % (e,))
            return self._err(ApiError(500, "internal", "Internal error."))


def main():
    host, port = os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8080"))
    srv = ThreadingHTTPServer((host, port), Handler)
    print("Kappa Gate on http://%s:%d  (Nebius key: %s, Tavily key: %s)" % (host, port, "set" if nebius_key() else "MISSING", "set" if tavily_key() else "MISSING"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
