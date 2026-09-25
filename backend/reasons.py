"""
reasons.py
----------
Human-readable, **evidence-backed** recommendation explanations.

This module is deliberately separate from the recommendation *computation* in
``algorithms.py``: the recommender decides *who* to recommend, this module
explains *why*, for any ``(user, candidate)`` pair -- so the same explanation
can be reused by the recommendation list, the batch endpoint or a standalone
"why was this suggested?" query.

Hard rule: **nothing here is invented**.  Every evidence object is derived
from the frozen graph / user profiles passed in, and each one carries the raw
ids/counts behind it so callers can render a relation chain or re-verify the
claim.  When no positive signal exists the explanation says so explicitly
(the "network expansion / popularity" fallback) instead of fabricating one.

Evidence types, in priority order:

1. ``common_friends``      -- shared neighbours (with bridge chains u -> f -> c)
2. ``shared_tags``         -- tag overlap (which tags, overlap ratio)
3. ``same_community``      -- Louvain community id + community size
4. ``structural_similar``  -- landmark-embedding cosine, close degrees, shared
                              2-hop reach (structural role equivalence)
5. ``popularity``          -- degree rank / percentile (always available; also
                              the honest fallback when nothing stronger exists)

Signatures (``graph_signature`` / ``users_signature``) let the service layer
cache explanations and reuse them only while the underlying graph and tag data
are unchanged.
"""

from __future__ import annotations

import random
import zlib
from typing import Dict, List, Optional, Set, Tuple

try:
    from . import algorithms
    from .graph import Graph
except ImportError:  # pragma: no cover
    import algorithms
    from graph import Graph


# ---------------------------------------------------------------------------
# Evidence type metadata (single source of truth for labels)
# ---------------------------------------------------------------------------
TYPE_COMMON_FRIENDS = "common_friends"
TYPE_SHARED_TAGS = "shared_tags"
TYPE_SAME_COMMUNITY = "same_community"
TYPE_STRUCTURAL = "structural_similar"
TYPE_POPULARITY = "popularity"

EVIDENCE_LABELS = {
    TYPE_COMMON_FRIENDS: "共同好友",
    TYPE_SHARED_TAGS: "标签重合",
    TYPE_SAME_COMMUNITY: "同社群",
    TYPE_STRUCTURAL: "结构相似",
    TYPE_POPULARITY: "热门拓展",
}

# How many bridge users / chains to retain in the serialised evidence.
MAX_FRIEND_IDS = 8
MAX_FRIEND_NAMES_IN_TEXT = 5
MAX_CHAINS = 3

# Structural-similarity calibration.  The absolute cosine of landmark-distance
# vectors is not very discriminative (distant nodes all collapse to max-depth
# coordinates), so we require the candidate to sit in the top quartile of a
# deterministic calibration sample *and* have a comparable degree.
STRUCT_MIN_NODES = 10
STRUCT_CALIBRATION_SAMPLE = 400
STRUCT_QUANTILE = 0.75
STRUCT_DEGREE_RATIO = 2.0
STRUCT_HOP2_BUDGET = 20_000          # cap neighbour visits for 2-hop overlap

# Priority order for picking the primary evidence / composing the sentence.
_PRIORITY = [
    TYPE_COMMON_FRIENDS,
    TYPE_SHARED_TAGS,
    TYPE_SAME_COMMUNITY,
    TYPE_STRUCTURAL,
    TYPE_POPULARITY,
]


# ===========================================================================
# Signatures -- cheap fingerprints used for cache validation
# ===========================================================================
def graph_signature(graph: Graph) -> str:
    """Fingerprint the graph topology (node/edge counts + CRC of adjacency)."""
    crc = 0
    for u, v, _w in graph.iter_edges():
        crc = zlib.crc32(f"{u}:{v};".encode("utf-8"), crc)
    degree_sum = 0
    for nid in graph.nodes:
        degree_sum += graph.degree(nid)
    return f"g-{graph.node_count}-{graph.edge_count}-{degree_sum}-{crc & 0xffffffff:08x}"


def users_signature(users: Dict[int, dict]) -> str:
    """Fingerprint user names + tags (the profile fields evidence uses)."""
    crc = 0
    for uid in sorted(users):
        record = users[uid] or {}
        name = record.get("name", str(uid))
        tags = ",".join(sorted(record.get("tags") or []))
        crc = zlib.crc32(f"{uid}:{name}:{tags};".encode("utf-8"), crc)
    return f"u-{len(users)}-{crc & 0xffffffff:08x}"


# ===========================================================================
# Explainer
# ===========================================================================
class RecommendationExplainer:
    """Derives structured evidence + readable text for ``(user, candidate)``.

    Parameters
    ----------
    graph:
        The *same frozen graph snapshot* the recommender used -- this is what
        guarantees explanation/recommendation consistency.
    users:
        ``uid -> {"name", "tags", ...}`` profile records.
    communities:
        Optional ``uid -> community_id`` map (Louvain output).
    embeddings:
        Optional precomputed landmark embeddings (``uid -> vector``); pass a
        shared dict from the service so explanations don't recompute BFS
        passes per call.  Computed lazily when omitted.
    """

    def __init__(
        self,
        graph: Graph,
        users: Dict[int, dict],
        communities: Optional[Dict[int, int]] = None,
        embeddings: Optional[Dict[int, List[float]]] = None,
    ) -> None:
        self.graph = graph
        self.users = users
        self.communities = communities or {}
        self._embeddings = embeddings
        # Per-user, bounded-size caches so a k-item recommendation list pays
        # neighbourhood / calibration costs only once.
        self._hop2_cache: Dict[int, Optional[Set[int]]] = {}
        self._struct_threshold: Dict[int, float] = {}
        n = graph.node_count
        self._avg_degree = (2.0 * graph.edge_count / n) if n else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(self, user: int, candidate: int) -> dict:
        """Return the full explanation record for one pair.

        Shape::

            {"user", "candidate", "primary_type", "reason_text", "evidence": [...]}
        """
        evidences: List[dict] = []

        # A user may exist as a profile record without any edges (cold start);
        # that is a legitimate state, not stale data.  Only a pair unknown to
        # *both* the graph and the user store means the snapshot is stale.
        user_known = self.graph.has_node(user) or user in self.users
        candidate_known = self.graph.has_node(candidate) or candidate in self.users
        if not user_known or not candidate_known:
            return {
                "user": user,
                "candidate": candidate,
                "primary_type": None,
                "primary_label": "数据已过期",
                "reason_text": "图数据已变更，请重新生成推荐。",
                "evidence": [],
            }

        friends_u = set(self.graph.neighbors(user)) if self.graph.has_node(user) else set()
        friends_c = set(self.graph.neighbors(candidate)) if self.graph.has_node(candidate) else set()

        ev_common = self._evidence_common_friends(user, candidate, friends_u, friends_c)
        if ev_common:
            evidences.append(ev_common)

        ev_tags = self._evidence_shared_tags(user, candidate)
        if ev_tags:
            evidences.append(ev_tags)

        ev_community = self._evidence_same_community(user, candidate)
        if ev_community:
            evidences.append(ev_community)

        ev_struct = self._evidence_structural(user, candidate, friends_u, friends_c)
        if ev_struct:
            evidences.append(ev_struct)

        # Popularity is always computable and doubles as the honest fallback.
        evidences.append(self._evidence_popularity(candidate))

        order = {t: i for i, t in enumerate(_PRIORITY)}
        evidences.sort(key=lambda ev: order[ev["type"]])
        primary = evidences[0]["type"]
        text = self._compose_text(user, candidate, evidences)
        return {
            "user": user,
            "candidate": candidate,
            "primary_type": primary,
            "primary_label": EVIDENCE_LABELS[primary],
            "reason_text": text,
            "evidence": evidences,
        }

    def explain_many(self, pairs: List[Tuple[int, int]]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for user, candidate in pairs:
            out[f"{user}:{candidate}"] = self.explain(user, candidate)
        return out

    # ------------------------------------------------------------------
    # Evidence builders -- each returns None when the data does not support it
    # ------------------------------------------------------------------
    def _evidence_common_friends(
        self, user: int, candidate: int, friends_u: Set[int], friends_c: Set[int]
    ) -> Optional[dict]:
        shared = friends_u & friends_c
        if not shared:
            return None
        # Most connected bridges first -- they are the most recognisable ones.
        bridges = sorted(shared, key=lambda f: (-self.graph.degree(f), f))
        bridge_records = [
            {"id": f, "name": self._name(f)} for f in bridges[:MAX_FRIEND_IDS]
        ]
        chains = [[user, f, candidate] for f in bridges[:MAX_CHAINS]]
        return {
            "type": TYPE_COMMON_FRIENDS,
            "label": EVIDENCE_LABELS[TYPE_COMMON_FRIENDS],
            "count": len(shared),
            "shared_friends": bridge_records,
            "chains": chains,
        }

    def _evidence_shared_tags(self, user: int, candidate: int) -> Optional[dict]:
        tags_u = set(self.users.get(user, {}).get("tags") or [])
        tags_c = set(self.users.get(candidate, {}).get("tags") or [])
        shared = sorted(tags_u & tags_c)
        if not shared:
            return None
        return {
            "type": TYPE_SHARED_TAGS,
            "label": EVIDENCE_LABELS[TYPE_SHARED_TAGS],
            "shared_tags": shared,
            "overlap": len(shared),
            "user_tag_count": len(tags_u),
            "candidate_tag_count": len(tags_c),
        }

    def _evidence_same_community(self, user: int, candidate: int) -> Optional[dict]:
        comm_u = self.communities.get(user)
        comm_c = self.communities.get(candidate)
        if comm_u is None or comm_c is None or comm_u != comm_c:
            return None
        size = sum(1 for c in self.communities.values() if c == comm_u)
        return {
            "type": TYPE_SAME_COMMUNITY,
            "label": EVIDENCE_LABELS[TYPE_SAME_COMMUNITY],
            "community_id": int(comm_u),
            "community_size": size,
        }

    def _evidence_structural(
        self, user: int, candidate: int, friends_u: Set[int], friends_c: Set[int]
    ) -> Optional[dict]:
        n = self.graph.node_count
        if n < STRUCT_MIN_NODES:
            return None
        emb = self._get_embeddings()
        vec_u = emb.get(user)
        vec_c = emb.get(candidate)
        if vec_u is None or vec_c is None:
            return None
        cosine = algorithms._cosine(vec_u, vec_c)
        threshold = self._structural_cutoff(user, friends_u)
        du, dc = self.graph.degree(user), self.graph.degree(candidate)
        if du == 0 or dc == 0:
            return None
        ratio = max(du, dc) / min(du, dc)
        if cosine < threshold or ratio > STRUCT_DEGREE_RATIO:
            return None
        hop2_u = self._two_hop(user, friends_u)
        hop2_c = self._two_hop(candidate, friends_c)
        hop2_common = None
        if hop2_u is not None and hop2_c is not None:
            hop2_common = len(hop2_u & hop2_c)
        return {
            "type": TYPE_STRUCTURAL,
            "label": EVIDENCE_LABELS[TYPE_STRUCTURAL],
            "embedding_cosine": round(cosine, 3),
            "user_degree": du,
            "candidate_degree": dc,
            "hop2_common": hop2_common,
        }

    def _evidence_popularity(self, candidate: int) -> dict:
        degree = self.graph.degree(candidate)
        # Rank by degree (1 = most connected); deterministic tie-break by id.
        rank = 1
        for nid in self.graph.nodes:
            if self.graph.degree(nid) > degree:
                rank += 1
        # Isolated (edge-less) users exist in the user store but not the graph.
        total = max(self.graph.node_count, len(self.users))
        percentile = round(rank * 100.0 / total, 1) if total else 100.0
        return {
            "type": TYPE_POPULARITY,
            "label": EVIDENCE_LABELS[TYPE_POPULARITY],
            "degree": degree,
            "rank": rank,
            "total": total,
            "top_percentile": percentile,
            "avg_degree": round(self._avg_degree, 1),
        }

    # ------------------------------------------------------------------
    # Text composition (Chinese, UI-ready; every number comes from evidence)
    # ------------------------------------------------------------------
    def _compose_text(self, user: int, candidate: int, evidences: List[dict]) -> str:
        by_type = {ev["type"]: ev for ev in evidences}
        cand = self._name(candidate)
        parts: List[str] = []

        common = by_type.get(TYPE_COMMON_FRIENDS)
        tags = by_type.get(TYPE_SHARED_TAGS)
        community = by_type.get(TYPE_SAME_COMMUNITY)
        structural = by_type.get(TYPE_STRUCTURAL)
        popularity = by_type[TYPE_POPULARITY]

        if common:
            names = [b["name"] for b in common["shared_friends"][:MAX_FRIEND_NAMES_IN_TEXT]]
            extra = " 等" if common["count"] > len(names) else ""
            parts.append(
                f"你们有 {common['count']} 位共同好友（{('、'.join(names))}{extra}）"
            )
            if community:
                parts.append(self._clause_community(community))
            if tags:
                parts.append(self._clause_tags(tags))
            parts.append("朋友的朋友是最可靠的推荐依据")
        elif tags:
            parts.append(
                f"你们的兴趣标签高度重合：共同标签 {tags['overlap']}/{tags['user_tag_count']} 个"
                f"（{'、'.join(tags['shared_tags'])}）"
            )
            if community:
                parts.append(self._clause_community(community))
            if structural:
                parts.append(self._clause_structural(structural))
        elif community:
            parts.append(self._clause_community(community, full=True))
            if tags:
                parts.append(self._clause_tags(tags))
            if structural:
                parts.append(self._clause_structural(structural))
        elif structural:
            parts.append(self._clause_structural(structural, full=True))
            if community:
                parts.append(self._clause_community(community))
        else:
            # Honest fallback: say what we know, never invent a relationship.
            d = popularity["degree"]
            if popularity["top_percentile"] <= 10:
                parts.append(
                    f"{cand} 是全站热门用户：好友数 {d}，全站排名 "
                    f"{popularity['rank']}/{popularity['total']}（前 {popularity['top_percentile']:g}%）"
                )
                parts.append("关注 TA 能快速拓展你的人脉覆盖")
            elif d >= popularity["avg_degree"]:
                parts.append(
                    f"{cand} 的好友数（{d}）高于全站平均（{popularity['avg_degree']:g}）"
                )
                parts.append("适合作为人脉拓展对象")
            else:
                parts.append(
                    f"目前你们之间暂无共同好友、共同标签或同社群关系；{cand} 作为人脉拓展候选"
                    f"（好友数 {d}，全站排名 {popularity['rank']}/{popularity['total']}）"
                )
            return "；".join(parts) + "。"

        return "，".join(parts) + "。"

    @staticmethod
    def _clause_tags(ev: dict) -> str:
        return f"共享 {ev['overlap']} 个标签（{'、'.join(ev['shared_tags'])}）"

    @staticmethod
    def _clause_community(ev: dict, full: bool = False) -> str:
        base = f"同属社群 {ev['community_id']}（{ev['community_size']} 人）"
        return f"你们{base}，社交圈高度重叠" if full else base

    @staticmethod
    def _clause_structural(ev: dict, full: bool = False) -> str:
        hop = ev.get("hop2_common")
        hop_text = f"，有 {hop} 个共同二度人脉" if hop else ""
        base = (
            f"在图中的结构位置相似（地标距离向量余弦 {ev['embedding_cosine']:g}，"
            f"度数 {ev['user_degree']} vs {ev['candidate_degree']}{hop_text}）"
        )
        return f"你们{base}" if full else f"且{base}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _name(self, uid: int) -> str:
        return self.users.get(uid, {}).get("name", str(uid))

    def _get_embeddings(self) -> Dict[int, List[float]]:
        if self._embeddings is None:
            self._embeddings = algorithms.landmark_embedding(self.graph)
        return self._embeddings

    def _structural_cutoff(self, user: int, friends_u: Set[int]) -> float:
        """Cosine cutoff = upper-quantile over a deterministic node sample."""
        if user in self._struct_threshold:
            return self._struct_threshold[user]
        emb = self._get_embeddings()
        target = emb.get(user)
        pool = [
            nid
            for nid in self.graph.nodes
            if nid != user and nid not in friends_u
        ]
        if len(pool) > STRUCT_CALIBRATION_SAMPLE:
            rng = random.Random(0x5EED + user)
            pool = rng.sample(pool, STRUCT_CALIBRATION_SAMPLE)
        cosines = sorted(algorithms._cosine(target, emb[nid]) for nid in pool if nid in emb)
        if cosines:
            idx = int(STRUCT_QUANTILE * (len(cosines) - 1))
            cutoff = cosines[idx]
        else:
            cutoff = 1.0
        self._struct_threshold[user] = cutoff
        return cutoff

    def _two_hop(self, user: int, friends: Optional[Set[int]] = None) -> Optional[Set[int]]:
        """Nodes reachable from ``user`` in exactly 2 hops (None if too costly)."""
        if user in self._hop2_cache:
            return self._hop2_cache[user]
        if friends is None:
            friends = set(self.graph.neighbors(user))
        budget = STRUCT_HOP2_BUDGET
        result: Set[int] = set()
        aborted = False
        for f in friends:
            budget -= self.graph.degree(f)
            if budget < 0:
                aborted = True
                break
            for x in self.graph.neighbors(f):
                result.add(x)
        if aborted:
            self._hop2_cache[user] = None
            return None
        result.difference_update(friends)
        result.discard(user)
        self._hop2_cache[user] = result
        return result
