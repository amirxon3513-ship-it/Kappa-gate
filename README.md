# Kappa Gate: no context, no answer

An anti-hallucination gateway for RAG, built on Contextual Logic. Every statement gets an integer context **K**.
Only **K ≥ 1** reaches the model's answer; paradoxes and unverified statements get **K = 0** and are quarantined;
refuted statements (**K ≤ −1**) are blocked. The admission decision is made by a deterministic engine, not by a language model.

Built for the **Nebius × NVIDIA Global AI Hackathon 2026**.

| Part | What runs it |
|---|---|
| Query analysis, fact extraction, answer wording | **NVIDIA Nemotron** models on **Nebius Token Factory** |
| Live web evidence | **Tavily** Search API |
| Admission decision (K, sets T/U/F, Axiom 4, answer verification) | Deterministic engine (JavaScript in the page, Python in `gateway/`) |

## Run it

Python 3.8+, nothing to install (standard library only).

```bash
export NEBIUS_API_KEY=...        # Nebius Token Factory key
export TAVILY_API_KEY=tvly-...   # Tavily key (only needed for "Live web")
python3 server.py                # http://localhost:8080
```

Open `http://localhost:8080/` (English) or `/index.html` (Russian). The keys stay on the server; the browser never sees them.
Without keys the site still works in "Local engine" mode.

Optional settings (see the top of `server.py`):

| Variable | Default | Meaning |
|---|---|---|
| `NEBIUS_MODEL_QUICK` / `_DEFAULT` / `_COMPLEX` | auto | Nemotron model ids for the three tiers. If unset, the server asks Nebius which models your account has and picks Nemotron models by size (smallest, middle, largest). Open `/api/status` to see the choice. |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1` | OpenAI-compatible endpoint |
| `PORT`, `HOST` | `8080`, `0.0.0.0` | |
| `RATE_LIMIT_PER_MIN` | `120` | Per client IP; protects your keys on a public demo |

Docker: `docker build -t kappa-gate . && docker run -p 8080:8080 -e NEBIUS_API_KEY -e TAVILY_API_KEY kappa-gate`

## Deploy on Nebius

The inference already runs on Nebius Token Factory. To host the site on Nebius AI Cloud too, run the container above
(or `python3 server.py` behind port 80/443) on a Nebius AI Cloud VM or as a Nebius serverless container, and pass the two
keys as environment variables or secrets. Follow the current Nebius docs for the exact console or CLI steps.

## Modes on the "Check" page

* **Nemotron · Nebius**: the knowledge base is yours (demo presets, uploaded documents). Nemotron annotates the query and
  words the answer from admitted facts; the engine decides what is admitted. The left pane shows what an ordinary RAG
  would answer from the same unfiltered context.
* **Live web · Tavily**: for the question, the server searches the web with Tavily. Nemotron extracts short facts from the
  pages and marks which facts say the same thing and which contradict each other. Every domain is **one source**; its
  class sets its weight in K (government, universities and international bodies +3, other sites +1, forums and social
  networks 0). Same fact from another domain raises K, a contradiction from a reliable domain lowers it. Facts with
  K = 0 are quarantined, K ≤ −1 blocked. Self-referential queries skip the search entirely.
* **Local engine**: no AI at all (patterns and a small built-in model). Works without a server.

## Command line: `kappa_gate_nebius.py`

The website's gateway without a browser. Nemotron on Nebius annotates the query and words the answer, the K-engine decides.

```bash
export NEBIUS_API_KEY=...
python3 kappa_gate_nebius.py models                       # Nemotron models on your account
python3 kappa_gate_nebius.py demo                         # 4 demo queries: confirmed, blocked, paradox, no data
python3 kappa_gate_nebius.py ask --kb my_state.json "..." # state exported from the website
TAVILY_API_KEY=tvly-... python3 kappa_gate_nebius.py live "What is the capital of Uzbekistan?"
```

Each verdict prints K, the set (T/U/F), the answer and, for every fact, how many sources confirm and refute it, the
same evidence the website shows. It also works as a library: `from kappa_gate_nebius import load_kb, gate`.

## Use it from your own AI agent or RAG backend: `POST /api/gate`

The same gateway as an HTTP tool. The model on Nebius only annotates the query and words the answer; the K-engine decides.

```bash
# 1) your own facts: either an explicit K, or a source (K is computed: each independent source adds its weight)
curl -s localhost:8080/api/gate -H 'content-type: application/json' -d '{
  "query": "How long is the warranty on appliances?",
  "facts": [
    {"id": "w1", "text": "The warranty on appliances is 24 months.", "source": "Policy 2026", "cls": "official"},
    {"id": "w2", "text": "The warranty on appliances is 24 months.", "source": "Support FAQ", "cls": "internal"},
    {"id": "w0", "text": "The warranty on appliances is 12 months.", "K": -2}
  ]}'

# 2) live web through Tavily
curl -s localhost:8080/api/gate -H 'content-type: application/json' -d '{"query": "What is the capital of Uzbekistan?", "mode": "live"}'
```

Response: `status` (`answer`, `refuted`, `no_info`, `paradox`, `ungrounded`), `set` (T/U/F), `K`, `answer`, `claims` (each with the ids it cites),
`removed` (sentences the gateway dropped), `blocked` and `quarantined` fact ids, and every fact with its K and (live) source domains.
`cls` is one of `official` (+3), `internal` (+2), `external` (+1), `unverified` (0), `outdated` (−2), `false` (−3).

* A self-referential query ("this statement is false") is answered by the deterministic guard with `paradox`, K = 0, without touching the model or the web.
* If the model or the search is unreachable the endpoint returns an HTTP error (502), never a fake "no information".
* Set `GATE_API_TOKEN` to require `Authorization: Bearer <token>` on this endpoint (do it on any public deployment).
* `examples/agent_with_gate.py` is a small agent (Nemotron with one tool, `kappa_gate`) that must call the gate before it answers.
  It works with `--facts examples/facts.json` or, without it, with live web search.

## Files

```
kappa_gate_nebius.py  command-line gateway on Nebius Nemotron (models, demo, ask, live)
server.py            static site + /api/chat (Nebius) + /api/search (Tavily) + /api/status; holds the keys
web/                 index_en.html, index.html (RU), theory_en.html, theory.html
gateway/             the same engine in Python: contextual_gateway.py, live_gateway.py (Tavily), llm_adapters.py (Nebius), ...
examples/            agent_with_gate.py (tool-calling agent behind the gate), facts.json
tests/mock_upstream.py   offline stand-in for Nebius and Tavily
tests/gate_test.py       tests for /api/gate and the agent example (starts its own mock and servers)
tests/e2e.js             browser test (Playwright) against server.py + the mock
Dockerfile, .env.example
DEVPOST.md           text for the hackathon submission
```

Python usage without the site:

```bash
cd gateway
python3 live_gateway.py "Is Tashkent the capital of Uzbekistan?" --provider nebius
python3 ask.py state.json "How many months is the warranty?"        # state.json exported from the site
```

## Demo video

`tests/record_demo.js` records a walkthrough (types a prompt on the test bench, scrolls through the evidence: how many
sources confirm and refute each fact) with a cursor and captions, and writes `demo_out/kappa_gate_demo.mp4`.
It needs node, Playwright and ffmpeg. With no keys on the server it records the Local engine mode; with keys, the
Nemotron or Live web mode:

```bash
python3 server.py &                                   # with NEBIUS_API_KEY and TAVILY_API_KEY set
node tests/record_demo.js                             # auto: Nemotron mode if the server has a Nebius key
DEMO_MODE=live DEMO_KB=0 PROMPT="What is the capital of Uzbekistan?" node tests/record_demo.js
```

The default knowledge base for the recording is `examples/demo_kb_en.json`.

## Tests

```bash
python3 tests/gate_test.py       # /api/gate + agent example, self-contained

# browser test of the site
python3 tests/mock_upstream.py 9101 &
NEBIUS_API_KEY=test-nebius TAVILY_API_KEY=tvly-test NEBIUS_BASE_URL=http://127.0.0.1:9101/v1 \
  TAVILY_URL=http://127.0.0.1:9101/search PORT=8090 python3 server.py &
NODE_PATH=$(npm root -g) node tests/e2e.js
```

The mock is a rule-based stand-in, not a language model: it checks the plumbing (keys stay server-side, model discovery,
retries, Live-mode flow, paradox handling, both languages), not the quality of Nemotron's answers.

## Honest limits

* K is only as good as the source classes and the groupings behind it. Domain heuristics are transparent rules, not a
  truth oracle, and two pages of one site count as one source.
* In Live mode the grouping of "same" and "contradicts" is done by the model. If that call fails, the page falls back to the
  built-in word-overlap relation, which is cruder (the Python `live_gateway.py` always uses it).
* The gateway guarantees that only facts with K ≥ 1 reach the answer and that self-referential queries are blocked. It does
  not guarantee that the K values themselves are right.
* Nemotron model ids differ between Nebius accounts and change over time; set `NEBIUS_MODEL_*` if the automatic choice is not what you want.
