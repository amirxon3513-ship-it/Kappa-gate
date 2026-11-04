"""Tiny learned models for the gateway, pure Python (no numpy). Weights are in cl_model.json.
A: self-reference detector.  B: relation of two statements (same / conflict / new).
Trained offline on synthetic data (see ml/). Held-out numbers are in ml/metrics.json: read them before trusting this.
"""
import json
import math
import os
import re


D = 2048


def norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^a-zа-я0-9]+", " ", s)
    return " " + s.strip() + " "


def fnv(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def vec_a(text: str):
    """hashed char 3-5-grams, log1p counts, L2 normalised -> sparse dict {index: value}"""
    t = norm(text)
    cnt = {}
    for n in (3, 4, 5):
        for i in range(len(t) - n + 1):
            k = fnv(t[i:i + n]) % D
            cnt[k] = cnt.get(k, 0) + 1
    v = {k: math.log1p(c) for k, c in cnt.items()}
    z = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {k: x / z for k, x in v.items()}


NEG_TOK = {"не", "нет", "без", "нельзя", "ни", "not", "no", "never"}
ANT = [("разрешено", "запрещено"), ("открыт", "закрыт"), ("можно", "нельзя"), ("больше", "меньше"), ("включен", "выключен"),
       ("допускается", "запрещается"), ("увеличивается", "уменьшается")]
STOP = set("какой какая какие какое каков сколько когда где что это как или для при про над под без если еще очень весь все его они она оно мне вам вас нас ваш наш этот эта эти тот той тем чем быть есть будет был была были можно нужно ли же бы вот так там тут здесь the and for are was were what how who why when where which this that with from have has does will can you your".split())


def toks(s: str):
    return [w for w in norm(s).split(" ") if w]


def stems(s: str):
    out = set()
    for w in toks(s):
        if w.isdigit() or w in NEG_TOK:
            continue
        if w.startswith("не") and len(w) > 5:
            w = w[2:]
        if len(w) >= 4 and w not in STOP:
            out.add(w[:5])
    return out


def nums(s: str):
    return sorted(re.findall(r"\d+(?:[.,]\d+)?", s.replace(",", ".")))


def negcount(s: str):
    c = 0
    for w in toks(s):
        if w in NEG_TOK or (w.startswith("не") and len(w) > 5) or w.startswith("запрещ") or w.startswith("закрыт"):
            c += 1
    return c


def feat_b(a: str, b: str):
    sa, sb = stems(a), stems(b)
    inter = len(sa & sb)
    j = inter / (len(sa | sb) or 1)
    na, nb = nums(a), nums(b)
    both = 1.0 if (na and nb) else 0.0
    eq = 1.0 if (both and na == nb) else 0.0
    df = 1.0 if (both and na != nb) else 0.0
    negd = 1.0 if (negcount(a) % 2) != (negcount(b) % 2) else 0.0
    ta, tb = set(toks(a)), set(toks(b))
    ant = 0.0
    for x, y in ANT:
        if (x in ta and y in tb) or (y in ta and x in tb):
            ant = 1.0
    sh = min(inter, 5) / 5.0
    return [1.0, j, eq, df, negd, ant, sh, j * df, j * negd, j * ant, j * eq, j * j]


_M = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cl_model.json"), encoding="utf-8"))
SELF_THRESHOLD, REL_THRESHOLD = 0.7, 0.8
DEIXIS = re.compile(r"(?<![^\W\d_])(?:эт[оаи]т?|эт[аеуо]м?|данн\w*|настоящ\w*|сам[аоиу]?|мо[йяеи]|я|сейчас|this|these|present|i|my|now|myself)(?![^\W\d_])", re.I)


def _softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def self_score(text: str):
    """-> (P(self-reference), 'liar' | 'truth')"""
    v = vec_a(text)
    A = _M["A"]
    z = [sum(row[k] * x for k, x in v.items()) + A["b"][c] for c, row in enumerate(A["W"])]
    p = _softmax(z)
    L = A["labels"]
    return 1 - p[L.index("plain")], ("liar" if p[L.index("liar")] >= p[L.index("truth")] else "truth")


def is_self_referential(text: str):
    """-> None or ('liar'|'truth', score). Needs a self-pointing word (this / I / now) to count."""
    s, lab = self_score(text)
    return (lab, s) if s >= SELF_THRESHOLD and DEIXIS.search(text) else None


def relation(a: str, b: str):
    """-> (label, probability) with label in new / same / conflict"""
    x = feat_b(a, b)
    B = _M["B"]
    p = _softmax([sum(w * f for w, f in zip(row, x)) + B["b"][c] for c, row in enumerate(B["W"])])
    i = max(range(len(p)), key=p.__getitem__)
    return B["labels"][i], p[i]


def relate_in_base(text: str, facts: dict):
    """-> ('same' | 'conflict' | 'new', fact_id | None), candidates need >= 0.25 overlap of word stems"""
    sa, best = stems(text), ("new", None, 0.0)
    for fid, ft in facts.items():
        sb = stems(ft)
        j = len(sa & sb) / (len(sa | sb) or 1)
        if j < 0.25:
            continue
        lab, p = relation(text, ft)
        if lab != "new" and p >= REL_THRESHOLD and (best[0] == "new" or (lab == "same" and best[0] != "same") or (lab == best[0] and p > best[2])):
            best = (lab, fid, p)
    return best[0], best[1]
