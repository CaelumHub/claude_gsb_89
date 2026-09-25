"""
explain.py
----------
Recommendation *explanation* layer, deliberately separate from the
recommendation computation in :mod:`algorithms`.

Design contract
~~~~~~~~~~~~~~~
1. **理由生成与推荐计算分开** — ``algorithms.hybrid_recommend`` decides *who*
   to recommend; this module independently answers *why*, by re-examining the
   real frozen graph.  It never reads recommendation scores, so a candidate
   cannot be "justified" by the very score that selected it.
2. **依据可复用** — :func:`build_evidence` returns pure structured evidence
   (IDs/counts/metrics derived from the graph, no rendered text);
   :func:`render_text` and :func:`hydrate_names` turn that evidence into
   readable Chinese.  The structured evidence is cached together with the
   recommendation and can be re-rendered at any time without recomputing
   anything against the graph.
3. **每条推荐都有可读理由** — every pair receives at least a fallback summary;
   every declared signal is backed by an entry in ``signals`` with concrete
   data the UI can show as a relation chain or list.
4. **与真实图数据一致、不虚构** — mutual-friend IDs come from neighbour-set
   intersection and are double-checked with ``graph.has_edge``; tags come from
   the user records; structural cosine comes from the same landmark embedding
   used by the embedding recommender; popularity degree/rank come from the
   degree sequence.  Thresholds below only decide *whether to mention* a
   signal — never invent numbers.  A weak structural similarity (below the
   mention threshold) is simply omitted.

Signal priority (primary reason): mutual friends → shared tags → structural
position similarity → popularity.  The readable sentence composes every signal
that genuinely applies, so hybrid recommendations can read
"有 3 位共同好友（…），且共同标签「科技」；同时结构位置高度相似".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

try:
    from . import config
    from .algorithms import landmark_embedding, _cosine
    from .graph import Graph
except ImportError:  # pragma: no cover
    import config
    from algorithms import landmark_embedding, _cosine
    from graph import Graph


# Signal type codes (stable identifiers for the frontend; Chinese labels live
# next to the evidence so no rendering layer has to translate codes).
SIGNAL_MUTUAL = "mutual_friends"
SIGNAL_TAGS = "shared_tags"
SIGNAL_STRUCTURE = "structural_similarity"
SIGNAL_POPULAR = "popularity"

PRIMARY_LABELS = {
    SIGNAL_MUTUAL: "共同好友",
    SIGNAL_TAGS: "标签重合",
    SIGNAL_STRUCTURE: "结构相似",
    SIGNAL_POPULAR: "热门用户",
}
# Order also defines priority when selecting the primary signal.
SIGNAL_PRIORITY = [SIGNAL_MUTUAL, SIGNAL_TAGS, SIGNAL_STRUCTURE, SIGNAL_POPULAR]


# ===========================================================================
# Structured evidence — pure graph facts, JSON-serialisable, reusable
# ===========================================================================
def build_evidence(
    graph: Graph,
    user: int,
    candidate: int,
    user_tags: Optional[Dict[int, Set[str]]] = None,
    embedding: Optional[Dict[int, List[float]]] = None,
    degree_rank: Optional[Dict[int, int]] = None,
    avg_degree: Optional[float] = None,
) -> Optional[dict]:
    """Derive the factual evidence pack for one ``(user, candidate)`` pair.

    Returns ``None`` when either endpoint is absent from the graph.  The pack
    contains only graph-derived facts; readable text is produced separately by
    :func:`render_text`.

    ``embedding`` / ``degree_rank`` / ``avg_degree`` are optional reusable
    caches computed once per request via :func:`get_embedding` /
    :func:`graph_degree_stats`, so explaining many candidates never rescans
    the whole graph per candidate.
    """
    if not graph.has_node(user) or not graph.has_node(candidate) or user == candidate:
        return None

    signals: List[dict] = []

    # --- Signal 1: mutual friends (real neighbour-set intersection) ---------
    user_friends = set(graph.neighbors(user))
    cand_friends = set(graph.neighbors(candidate))
    mutual = user_friends & cand_friends
    # Defensive re-verification: every declared mutual friend must be adjacent
    # to BOTH endpoints in the actual graph (no fabricated relation chains).
    mutual = {f for f in mutual if graph.has_edge(f, user) and graph.has_edge(f, candidate)}
    if mutual:
        # Rank the displayed bridges: lower-degree connectors first (tighter,
        # more meaningful shared circles), tie-broken by id for determinism.
        bridges = sorted(mutual, key=lambda f: (graph.degree(f), f))
        signals.append({
            "type": SIGNAL_MUTUAL,
            "label": PRIMARY_LABELS[SIGNAL_MUTUAL],
            "count": len(mutual),
            # Each chain is the real 2-hop path user -> mutual -> candidate.
            "chains": [[user, f, candidate] for f in bridges[: config.EXPLAIN_MAX_MUTUAL_FRIENDS]],
            "mutual_friend_ids": bridges[: config.EXPLAIN_MAX_MUTUAL_FRIENDS],
            "shown": min(len(mutual), config.EXPLAIN_MAX_MUTUAL_FRIENDS),
        })

    # --- Signal 2: shared tags (real tag-set intersection) ------------------
    if user_tags:
        my_tags = set(user_tags.get(user, set()))
        cand_tags = set(user_tags.get(candidate, set()))
        shared = sorted(my_tags & cand_tags)
        if shared:
            signals.append({
                "type": SIGNAL_TAGS,
                "label": PRIMARY_LABELS[SIGNAL_TAGS],
                "count": len(shared),
                "tags": shared[: config.EXPLAIN_MAX_SHARED_TAGS],
                "shown": min(len(shared), config.EXPLAIN_MAX_SHARED_TAGS),
                "overlap_ratio": round(len(shared) / max(1, len(my_tags)), 4),
            })

    # --- Signal 3: structural position similarity (landmark cosine) ---------
    if embedding is None:
        embedding = landmark_embedding(graph)
    vec_u = embedding.get(user)
    vec_c = embedding.get(candidate)
    if vec_u is not None and vec_c is not None:
        cosine = _cosine(vec_u, vec_c)
        if cosine >= config.EXPLAIN_WEAK_STRUCTURE_COSINE:
            strong = cosine >= config.EXPLAIN_STRONG_STRUCTURE_COSINE
            signals.append({
                "type": SIGNAL_STRUCTURE,
                "label": PRIMARY_LABELS[SIGNAL_STRUCTURE],
                "cosine": round(cosine, 4),
                "strong": strong,
                # Same metric the embedding recommender ranks by — nothing invented.
                "metric": "landmark_distance_cosine",
            })

    # --- Signal 4: popularity (real degree + rank in the degree sequence) ---
    degree = graph.degree(candidate)
    if degree_rank is None:
        degree_rank = degree_ranking(graph)
    if avg_degree is None:
        avg_degree = graph_stats(graph)["avg_degree"]
    rank = degree_rank.get(candidate, 0)
    if (
        degree >= config.EXPLAIN_POPULAR_MIN_DEGREE
        and degree >= avg_degree * config.EXPLAIN_POPULAR_RATIO
    ):
        signals.append({
            "type": SIGNAL_POPULAR,
            "label": PRIMARY_LABELS[SIGNAL_POPULAR],
            "degree": degree,
            "rank": rank,
            "total_users": graph.node_count,
        })

    primary = _primary_signal(signals)
    evidence = {
        "user": user,
        "candidate": candidate,
        "signals": signals,
        "primary_signal": primary,
        "primary_label": PRIMARY_LABELS[primary] if primary else "网络扩展",
        "metrics": {
            "user_degree": graph.degree(user),
            "candidate_degree": degree,
            "mutual_friend_total": len(mutual),
            "distance": _two_hop_distance(graph, user, candidate, user_friends, mutual),
            "avg_degree": round(avg_degree, 3),
        },
    }
    # Rendered text is kept next to the evidence but is trivially regenerable
    # from the structured signals (render_text is deterministic).
    evidence["summary"] = render_text(evidence)
    return evidence


def _primary_signal(signals: List[dict]) -> Optional[str]:
    present = {s["type"] for s in signals}
    for sig in SIGNAL_PRIORITY:
        if sig in present:
            return sig
    return None


def _two_hop_distance(
    graph: Graph,
    user: int,
    candidate: int,
    user_friends: Set[int],
    mutual: Set[int],
) -> int:
    """Cheap, truthful hop descriptor for the relation chain.

    1 = already connected (normally excluded from recommendations),
    2 = reachable through a mutual friend, 3 = connected via a friend's
    friend-without-shared-bridge, -1 = no short path found within the bound.
    """
    if candidate in user_friends:
        return 1
    if mutual:
        return 2
    # Bounded expansion: is the candidate within 3 hops (friend-of-friend)?
    hop2: Set[int] = set()
    for f in user_friends:
        hop2.update(graph.neighbors(f))
    if candidate in hop2:
        return 3
    return -1


# ===========================================================================
# Text rendering — deterministic, runs over evidence alone (reusable)
# ===========================================================================
def render_text(evidence: dict) -> str:
    """Compose the readable Chinese reason from structured evidence only."""
    signals = {s["type"]: s for s in evidence.get("signals", [])}
    fragments: Dict[str, str] = {}

    mutual = signals.get(SIGNAL_MUTUAL)
    if mutual:
        n = mutual["count"]
        shown = mutual.get("shown", n)
        if n > shown:
            fragments[SIGNAL_MUTUAL] = f"与你有 {n} 位共同好友（展示 {shown} 位）"
        elif n == 1:
            fragments[SIGNAL_MUTUAL] = "与你有 1 位共同好友"
        else:
            fragments[SIGNAL_MUTUAL] = f"与你有 {n} 位共同好友"

    tags = signals.get(SIGNAL_TAGS)
    if tags:
        names = "、".join(f"「{t}」" for t in tags["tags"])
        extra = " 等" if tags["count"] > tags.get("shown", tags["count"]) else ""
        fragments[SIGNAL_TAGS] = f"共同标签 {names}{extra}"

    structure = signals.get(SIGNAL_STRUCTURE)
    if structure:
        pct = round(structure["cosine"] * 100, 1)
        # Render without a trailing ".0" for whole percentages.
        pct_text = str(int(pct)) if float(pct).is_integer() else str(pct)
        if structure["strong"]:
            fragments[SIGNAL_STRUCTURE] = f"结构位置高度相似（相似度 {pct_text}%）"
        else:
            fragments[SIGNAL_STRUCTURE] = f"在网络中的结构位置较相似（相似度 {pct_text}%）"

    popular = signals.get(SIGNAL_POPULAR)
    if popular:
        fragments[SIGNAL_POPULAR] = (
            f"是连接广泛的热门用户（{popular['degree']} 位好友，"
            f"全网第 {popular['rank']} 位）"
        )

    if fragments:
        # Primary signal leads; supporting signals follow in priority order.
        ordered = [fragments[sig] for sig in SIGNAL_PRIORITY if sig in fragments]
        head = ordered[0]
        return head + ("；" + "，".join(ordered[1:]) if len(ordered) > 1 else "")

    # Fallback: always a readable reason, but a generic one — never fabricated
    # specifics.  Distance is only stated when positively known.
    distance = evidence.get("metrics", {}).get("distance", -1)
    if distance == 3:
        return "处于你的好友的好友圈中，可能值得认识"
    return "基于当前网络结构的扩展推荐（暂无共同好友或重合标签）"


# ===========================================================================
# Name hydration — evidence keeps IDs; names are resolved at presentation
# ===========================================================================
def hydrate_names(evidence: dict, users: Dict[int, dict]) -> dict:
    """Fill display names into cached evidence without touching the graph.

    Evidence is stored with IDs only (so it never goes stale on rename).  This
    pass adds a ``names`` map and per-chain node names at read time.  Unknown
    ids fall back to ``"用户 <id>"`` — same convention as the rest of the API.
    """
    def name_of(uid: int) -> str:
        rec = users.get(uid)
        if rec and rec.get("name"):
            return rec["name"]
        return f"用户 {uid}"

    out = dict(evidence)
    out["names"] = {}
    ids = {evidence.get("user"), evidence.get("candidate")}
    for sig in evidence.get("signals", []):
        for fid in sig.get("mutual_friend_ids", []):
            ids.add(fid)
    ids.discard(None)
    for uid in ids:
        out["names"][str(uid)] = name_of(uid)

    hydrated_signals = []
    for sig in evidence.get("signals", []):
        s = dict(sig)
        if s.get("chains"):
            s["chains"] = [list(chain) for chain in s["chains"]]
        hydrated_signals.append(s)
    out["signals"] = hydrated_signals
    return out


# ===========================================================================
# Reusable per-request caches
# ===========================================================================
def get_embedding(graph: Graph, cache: Optional[dict] = None) -> Dict[int, List[float]]:
    """Return a (possibly cached) landmark matrix for this graph."""
    if cache is not None and "embedding" in cache:
        return cache["embedding"]
    emb = landmark_embedding(graph)
    if cache is not None:
        cache["embedding"] = emb
    return emb


def degree_ranking(graph: Graph) -> Dict[int, int]:
    """Map ``node -> rank`` (1-based, highest degree first; ties by id)."""
    ordered = sorted(graph.nodes, key=lambda nid: (-graph.degree(nid), nid))
    return {nid: i + 1 for i, nid in enumerate(ordered)}


def graph_stats(graph: Graph) -> dict:
    """Degree-sequence facts computed once and shared across candidates."""
    n = max(1, graph.node_count)
    total = sum(graph.degree(nid) for nid in graph.nodes)
    return {"avg_degree": total / n, "node_count": graph.node_count}


def explain_pair(
    graph: Graph,
    user: int,
    candidate: int,
    users: Optional[Dict[int, dict]] = None,
    user_tags: Optional[Dict[int, Set[str]]] = None,
    embedding: Optional[Dict[int, List[float]]] = None,
    degree_rank: Optional[Dict[int, int]] = None,
    avg_degree: Optional[float] = None,
) -> Optional[dict]:
    """Convenience one-shot: evidence pack + rendered text + resolved names."""
    evidence = build_evidence(
        graph, user, candidate, user_tags, embedding, degree_rank, avg_degree
    )
    if evidence is None:
        return None
    if users:
        evidence = hydrate_names(evidence, users)
    return evidence
