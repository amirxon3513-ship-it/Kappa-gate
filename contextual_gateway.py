"""Contextual Logic Gateway: an anti-hallucination filter for RAG. No dependencies.

Definitions 1-3, Axioms 1-4 and the Singularity Ban from the paper
"Contextual Logic: An Axiomatic Framework for Neutralizing Self-Referential
Paradoxes and Computational Hallucinations".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Protocol, Set, Tuple

MAX_VARS = 18  # step budget of the Axiom 4 search (2**18 assignments)


def set_of(K: int) -> str:
    """Definition 2 + Axiom 1: the sign of K picks the global set."""
    return "T" if K > 0 else "F" if K < 0 else "U"


class ContextualLogicSystem:
    """Listing 1 of the paper: registry of statements with context K and time t."""

    def __init__(self) -> None:
        self._T: Set[str] = set()
        self._F: Set[str] = set()
        self._U: Set[str] = set()
        self._registry: Dict[str, Tuple[int, int]] = {}

    def register_statement(self, statement_id: str, initial_K: int, t: int = 0) -> str:
        self._T.discard(statement_id)
        self._F.discard(statement_id)
        self._U.discard(statement_id)
        if initial_K > 0:
            self._T.add(statement_id)
            status = "TRUE (T)"
        elif initial_K < 0:
            self._F.add(statement_id)
            status = "FALSE (F)"
        else:  # K == 0: strict mapping to U (The Singularity Ban)
            self._U.add(statement_id)
            status = "UNDEFINED/NEUTRAL (U)"
        self._registry[statement_id] = (initial_K, t)
        return status

    def evolve_context(self, statement_id: str, delta_K: int, current_t: int) -> str:
        """Axiom 3: K(t2) = K(t1) + delta_K."""
        if statement_id not in self._registry:
            raise KeyError(f"Statement '{statement_id}' is not registered.")
        old_K, _ = self._registry[statement_id]
        return self.register_statement(statement_id, old_K + delta_K, t=current_t)

    def K(self, statement_id: str) -> Optional[int]:
        item = self._registry.get(statement_id)
        return item[0] if item else None


# ---------------------------------------------------------------- Axiom 4
Refs = List[Tuple[str, bool]]  # (referenced id, negated?)


def try_assign(root: str, refs: Dict[str, Refs], K: Dict[str, int]) -> Optional[Tuple[int, int, int]]:
    """Try K+ and K- for `root`. A statement with refs asserts the conjunction of
    its literals; a base fact is fixed by the sign of its K (K = 0 stays free).
    Returns (plus, minus, total) counts of consistent contexts, or None when the
    step budget is exceeded (the caller treats that as K = 0)."""
    seen, stack, order = set(), [root], []
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
        stack.extend(r for r, _ in refs.get(n, []))
    derived = [n for n in order if refs.get(n)]
    fixed = {n: K[n] > 0 for n in order if not refs.get(n) and K.get(n, 0) != 0}
    free = [n for n in order if n not in fixed]
    if len(free) > MAX_VARS:
        return None
    plus = minus = total = 0
    for bits in product((False, True), repeat=len(free)):
        v = {**fixed, **dict(zip(free, bits))}
        total += 1
        if all(v[n] == all(v[r] != neg for r, neg in refs[n]) for n in derived):
            plus, minus = (plus + 1, minus) if v[root] else (plus, minus + 1)
    return plus, minus, total


def is_paradox(res: Optional[Tuple[int, int, int]]) -> bool:
    return res is None or (res[0] == 0 and res[1] == 0)


def reaches_self(node: str, refs: Dict[str, Refs]) -> bool:
    seen, stack = set(), [r for r, _ in refs.get(node, [])]
    while stack:
        n = stack.pop()
        if n == node:
            return True
        if n not in seen:
            seen.add(n)
            stack.extend(r for r, _ in refs.get(n, []))
    return False


# ---------------------------------------------------------------- automatic K from sources
SOURCE_WEIGHTS = {"official": 3, "internal": 2, "external": 1, "unverified": 0, "outdated": -2, "false": -3}


def _stems(s: str) -> set:
    s = re.sub(r"\d+(?:[.,]\d+)?", " ", s.lower())
    return {w[:5] for w in re.findall(r"[a-zа-яё]+", s) if len(w) >= 4}


def _nums(s: str) -> list:
    return sorted(x.replace(",", ".") for x in re.findall(r"\d+(?:[.,]\d+)?", s))


def relate(text: str, facts: Dict[str, str]):
    """-> ("same" | "conflict" | "new", fact_id | None). Uses the learned model when cl_ml.py is present."""
    try:
        import cl_ml
        return cl_ml.relate_in_base(text, facts)
    except ImportError:
        pass
    a, na, best, bj = _stems(text), _nums(text), ("new", None), 0.0
    for fid, ft in facts.items():
        b = _stems(ft)
        j = len(a & b) / (len(a | b) or 1)
        nb = _nums(ft)
        if j >= 0.7 and na == nb and j > bj:
            best, bj = ("same", fid), j
        elif j >= 0.5 and na and nb and na != nb and best[0] != "same" and j > bj:
            best, bj = ("conflict", fid), j
    return best


# ---------------------------------------------------------------- gateway
class LLM(Protocol):
    def analyze(self, query: str, facts: List[Tuple[str, str]]) -> dict:
        """-> {"statements": [{"id","text","asserts":[{"ref","neg"}]}], "relevant_ids": [...]}.
        Must NOT judge truth. `facts` are (id, text) pairs without K."""

    def generate(self, query: str, facts: List[Tuple[str, str]]) -> List[dict]:
        """-> [{"text": "...", "cites": ["fact_id", ...]}] using ONLY the given facts."""


@dataclass
class Verdict:
    status: str                      # answer | refuted | no_info | paradox | ungrounded
    K: int = 0
    answer: str = ""
    claims: List[dict] = field(default_factory=list)
    stripped: List[dict] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)      # F facts found for the query
    quarantined: List[str] = field(default_factory=list)  # U facts found for the query

    @property
    def set(self) -> str:
        return set_of(self.K)


class ContextualGateway:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.system = ContextualLogicSystem()
        self.text: Dict[str, str] = {}
        self.refs: Dict[str, Refs] = {}

    # --- knowledge base
    def add_fact(self, fact_id: str, text: str, K: int, t: int = 0, refs: Refs = ()) -> str:
        self.text[fact_id], self.refs[fact_id] = text, list(refs)
        self.system.register_statement(fact_id, K, t)
        self._audit()
        return set_of(self.system.K(fact_id))

    def ingest(self, fact_id: str, text: str, source: str, kind: str = "internal") -> str:
        """Automatic K: each independent source adds its weight (Axiom 3).
        Same claim from a new source -> K grows; conflicting claim from a reliable source lowers the old fact."""
        w = SOURCE_WEIGHTS[kind]
        self.sources = getattr(self, "sources", {})
        rel, other = relate(text, self.text)
        if rel == "same":
            if source in self.sources.get(other, set()):
                return other
            self.sources.setdefault(other, set()).add(source)
            self.evolve(other, w, self.system._registry[other][1] + 1)
            return other
        self.add_fact(fact_id, text, w)
        self.sources[fact_id] = {source}
        if rel == "conflict" and w > 0:
            self.evolve(other, -w, self.system._registry[other][1] + 1)
        return fact_id

    def evolve(self, fact_id: str, delta_K: int, t: int) -> str:
        self.system.evolve_context(fact_id, delta_K, t)
        self._audit()
        return set_of(self.system.K(fact_id))

    def _audit(self) -> None:
        """Axiom 4 on every statement that refers to others."""
        Ks = {i: self.system.K(i) for i in self.text}
        for i, r in self.refs.items():
            if r and is_paradox(try_assign(i, self.refs, Ks)) and Ks[i] != 0:
                self.system.register_statement(i, 0, self.system._registry[i][1])

    # --- the query path
    def ask(self, query: str) -> Verdict:
        facts = [(i, self.text[i]) for i in self.text]
        plan = self.llm.analyze(query, facts)

        # step 2: self-reference check on the query itself (Corollary 1.1)
        q_refs: Dict[str, Refs] = {
            s["id"]: [(a["ref"], bool(a.get("neg"))) for a in s.get("asserts", [])]
            for s in plan.get("statements", [])
        }
        q_K = {i: 0 for i in q_refs}
        for sid, r in q_refs.items():
            if not r:
                continue
            if is_paradox(try_assign(sid, q_refs, q_K)):
                return Verdict("paradox")        # no K+ or K- survives -> K = 0
            if reaches_self(sid, q_refs):
                return Verdict("ungrounded")     # consistent but nothing grounds it -> K = 0

        # step 2a: what the query itself asserts, compared with the base (meaning only, never truth)
        for c in plan.get("claims", []):
            fid, stance = c.get("fact_id"), c.get("stance", "matches")
            if fid not in self.text:
                continue
            K = self.system.K(fid)
            if (stance == "matches" and K < 0) or (stance == "contradicts" and K > 0):
                return Verdict("refuted", K if K < 0 else -K, blocked=[fid])

        # step 2b: partition the facts the query touches
        ids = [i for i in dict.fromkeys(plan.get("relevant_ids", [])) if i in self.text]
        T = [i for i in ids if self.system.K(i) > 0]
        F = [i for i in ids if self.system.K(i) < 0]
        U = [i for i in ids if self.system.K(i) == 0]
        if not T:
            if F:
                return Verdict("refuted", min(self.system.K(i) for i in F), blocked=F, quarantined=U)
            return Verdict("no_info", 0, quarantined=U)

        # steps 3-4: generate from T only, then drop every claim not grounded in T
        claims = self.llm.generate(query, [(i, self.text[i]) for i in T])
        kept, stripped = [], []
        for c in claims:
            cites = [x for x in c.get("cites", []) if isinstance(x, str)]
            if cites and all(x in T for x in cites):
                kept.append({"text": c["text"], "cites": cites})
            else:
                stripped.append(c)
        if not kept:  # fall back to the verified facts themselves
            kept = [{"text": self.text[i], "cites": [i]} for i in T]
        K_answer = min(self.system.K(x) for c in kept for x in c["cites"])  # weakest link
        return Verdict("answer", K_answer, " ".join(c["text"] for c in kept), kept, stripped, F, U)


# ---------------------------------------------------------------- demo
class RuleBasedLLM:
    """Stand-in for a real model so the file runs offline. Replace with your LLM."""

    LIAR = re.compile(r"(эт[оаи]\s+(утверждени|предложени)\w*\s+ложн\w*|this (statement|sentence) is false"
                      r"|ответишь[^?.!]{0,40}этот\s+вопрос[^?.!]{0,25}нет|will you answer no to this question)", re.I | re.U)

    def analyze(self, query, facts):
        try:
            import cl_ml
            hit = cl_ml.is_self_referential(query)
        except ImportError:
            hit = None
        if self.LIAR.search(query) or hit:
            return {"statements": [{"id": "q1", "text": query, "asserts": [{"ref": "q1", "neg": True}]}], "relevant_ids": []}
        words = {w[:4] for w in re.findall(r"\w{4,}", query.lower())}
        ids = [i for i, t in facts if words & {w[:4] for w in re.findall(r"\w{4,}", t.lower())}]
        return {"statements": [], "relevant_ids": ids}

    def generate(self, query, facts):
        return [{"text": t, "cites": [i]} for i, t in facts]


if __name__ == "__main__":
    gw = ContextualGateway(RuleBasedLLM())
    gw.add_fact("water_boil", "Вода при нормальном давлении кипит при 100 °C.", 120)
    gw.add_fact("flat_earth", "Земля плоская.", -95)
    gw.add_fact("liar", "Это утверждение ложно.", 9, refs=[("liar", True)])  # forced to K = 0
    gw.ingest("warranty", "Гарантия на технику составляет 24 месяца.", "Регламент", "official")
    gw.ingest("warranty_b", "Гарантия на технику составляет 24 месяца.", "FAQ", "internal")   # K: 3 -> 5
    for q in ["При какой температуре кипит вода?", "Земля плоская?", "Это утверждение ложно.",
              "Ответь «да» или «нет»: ты ответишь на этот вопрос «нет»?", "Какая высота Эвереста?"]:
        v = gw.ask(q)
        print(f"{q!r:45} -> {v.status:10} K={v.K:<4} {v.answer}")
