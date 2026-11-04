# Devpost submission text: Kappa Gate

**Track:** Best Apps & Agents (also eligible for Best Use of Tavily)

**Tagline:** No context, no answer: an anti-hallucination gateway for RAG and agents.

## Inspiration

RAG systems put everything they retrieve into the prompt: current facts, outdated versions, rumours, even statements that
contradict themselves. The model then sounds equally sure about all of it. We built Kappa Gate on our own Contextual
Logic framework, where no statement is true or false on its own: its status depends on context. We turned that into a
filter that sits between the knowledge base and the model.

## What it does

Every statement gets an integer context **K**, computed from the number and reliability of independent sources.

* **K ≥ 1**: verified, the only thing allowed into the answer, with a source for every sentence.
* **K = 0**: unknown. Paradoxes ("this statement is false"), self-references and unverified claims are quarantined, and the gateway says plainly "no information".
* **K ≤ −1**: refuted. Blocked, even if it looks like a perfect answer.

The decision is made by a deterministic engine, not a language model. The model only annotates the query and words the
answer from facts the engine already admitted; any sentence without a link to a fact with K ≥ 1 is removed.

The test bench shows two columns for the same query: what an ordinary RAG would answer from the whole, unfiltered
context, and what Kappa Gate lets through.

**Live web mode:** Tavily searches the web for the question, Nemotron extracts short facts from the pages and marks which
say the same thing and which contradict each other, and every domain counts as one source with a reliability weight
(government, universities and international bodies +3, other sites +1, forums and social networks 0). Two independent
sources agreeing raise K; a contradiction from a reliable source lowers it; a post on a forum stays in quarantine.
Self-referential queries never reach the web.

## How we built it

* **NVIDIA Nemotron on Nebius Token Factory**: query analysis, fact extraction, grouping of agreeing and conflicting facts, answer wording, and the "ordinary RAG" comparison. Three model tiers (fast, standard, complex) are picked from the Nemotron models available in the account.
* **Tavily Search API**: live evidence for the web mode.
* **Server** (`server.py`, Python standard library): serves the site and proxies `/api/chat` and `/api/search`, so the API keys never reach the browser. Rate-limited per IP.
* **Engine**: the same axioms in JavaScript (in the page) and Python (`gateway/`), including the check that blocks self-referential statements (Axiom 4) and the theorem that K = 0 exactly when a statement is undefined.
* **Gate API** (`POST /api/gate`): the same gateway as a tool for agents and RAG backends, with an example agent (Nemotron with one tool) that must call the gate before it answers. Self-referential queries are stopped before they reach the model or the web, and an unreachable model or search is an error, never a fake "no information".
* Bilingual site (English and Russian), a theory page with the definitions, axioms and proofs, JSON export and import of the knowledge base.

## Challenges

* Making the model useful without letting it decide. Every model output is treated as untrusted: ids it invents are discarded, claims without a citation are removed, and a deterministic fallback replaces a failed call.
* Deciding what counts as an independent source on the open web. We settled on a transparent rule (one domain, one source, a weight by class) and documented its limits instead of pretending it is a truth oracle.

## What's next

Source-reputation learning, per-domain weights editable by the user, a hybrid mode that reports conflicts between a private
knowledge base and the web, and an agent-tool wrapper so any agent can call the gate before it answers.

## Built with

Nebius Token Factory, NVIDIA Nemotron, Tavily, Python, JavaScript.

## 2-minute demo script

1. Home page: the number line K, three buckets T / U / F. One sentence on the idea.
2. **Check → Nemotron · Nebius**, query "Это утверждение ложно" (liar paradox). Left: ordinary RAG answers something. Right: K = 0, "no information", with the Axiom 4 trace.
3. Query "Земля плоская?". Left: plain RAG gets the false fact in its prompt next to the true ones. Right: refuted by the knowledge base, K ≤ −1, the false fact is blocked and never reaches the model.
4. **Live web · Tavily**, a question with a checkable answer (for example "What is the capital of Uzbekistan?"). Show the pages Tavily found with their class and weight, the K of each fact, a forum post sitting in quarantine, and the final answer with its sources.
5. **Knowledge base**: paste a short policy document, show K computed from sources, then add a contradicting outdated source and show K drop.
6. Close on the stack: Nemotron on Nebius Token Factory + Tavily, with the decision made by the K-engine.
