"""Near-duplicate detection via MinHash + LSH (SPEC §7.1).

Dependency-free implementation (no ``datasketch``): a band-based LSH over
MinHash signatures of word-shingles. Documents with estimated Jaccard
similarity above ``threshold`` are grouped into clusters. Per SPEC, near-dups
are *flagged*, not auto-removed; the caller records the cluster id in metadata.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from knowledge_index.ingestion.normalize import normalize_for_dedup

# A fixed large prime modulus for the universal-hash family used by MinHash.
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _shingles(text: str, k: int) -> set[int]:
    """Hash the set of k-word shingles of ``text`` to 32-bit ints."""
    words = normalize_for_dedup(text).split()
    if len(words) < k:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    out: set[int] = set()
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "big")
        out.add(h)
    return out


@dataclass
class _HashCoeffs:
    a: list[int]
    b: list[int]


def _make_coeffs(num_perm: int, seed: int = 0) -> _HashCoeffs:
    """Deterministic (a, b) coefficients for ``num_perm`` hash permutations."""
    a: list[int] = []
    b: list[int] = []
    for i in range(num_perm):
        ha = int.from_bytes(
            hashlib.blake2b(f"{seed}:a:{i}".encode(), digest_size=8).digest(), "big"
        )
        hb = int.from_bytes(
            hashlib.blake2b(f"{seed}:b:{i}".encode(), digest_size=8).digest(), "big"
        )
        a.append((ha % (_MERSENNE_PRIME - 1)) + 1)  # non-zero
        b.append(hb % _MERSENNE_PRIME)
    return _HashCoeffs(a, b)


class MinHashDeduplicator:
    """Flag near-duplicate documents above a Jaccard ``threshold`` (default 0.9).

    Usage::

        dedup = MinHashDeduplicator()
        for doc_id, text in docs:
            dedup.add(doc_id, text)
        clusters = dedup.clusters()  # {cluster_id: [doc_id, ...]}
    """

    def __init__(
        self,
        *,
        threshold: float = 0.9,
        num_perm: int = 128,
        shingle_k: int = 5,
        bands: int = 32,
    ) -> None:
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_k = shingle_k
        self.bands = bands
        self.rows = num_perm // bands
        self._coeffs = _make_coeffs(num_perm)
        self._signatures: dict[str, list[int]] = {}
        # band index: (band_no, band_hash) -> set of doc_ids
        self._buckets: dict[tuple[int, int], set[str]] = {}

    def signature(self, text: str) -> list[int]:
        """Compute the MinHash signature (one min per permutation) for ``text``."""
        shingles = _shingles(text, self.shingle_k)
        if not shingles:
            return [0] * self.num_perm
        sig: list[int] = []
        a, b = self._coeffs.a, self._coeffs.b
        for i in range(self.num_perm):
            mn = _MAX_HASH
            ai, bi = a[i], b[i]
            for s in shingles:
                hv = ((ai * s + bi) % _MERSENNE_PRIME) & _MAX_HASH
                if hv < mn:
                    mn = hv
            sig.append(mn)
        return sig

    def add(self, doc_id: str, text: str) -> list[str]:
        """Index a document; return doc_ids of existing near-duplicates."""
        sig = self.signature(text)
        self._signatures[doc_id] = sig
        candidates = self._candidates(sig)
        matches = [
            other
            for other in candidates
            if other != doc_id
            and self._estimated_jaccard(sig, self._signatures[other]) >= self.threshold
        ]
        for band in range(self.bands):
            key = (band, self._band_hash(sig, band))
            self._buckets.setdefault(key, set()).add(doc_id)
        return matches

    def _band_hash(self, sig: list[int], band: int) -> int:
        start = band * self.rows
        chunk = tuple(sig[start : start + self.rows])
        return int.from_bytes(hashlib.blake2b(repr(chunk).encode(), digest_size=8).digest(), "big")

    def _candidates(self, sig: list[int]) -> set[str]:
        out: set[str] = set()
        for band in range(self.bands):
            key = (band, self._band_hash(sig, band))
            out |= self._buckets.get(key, set())
        return out

    def _estimated_jaccard(self, sig_a: list[int], sig_b: list[int]) -> float:
        equal = sum(1 for x, y in zip(sig_a, sig_b, strict=True) if x == y)
        return equal / self.num_perm

    def clusters(self) -> dict[str, list[str]]:
        """Return connected near-dup clusters as ``{cluster_id: [doc_id, ...]}``.

        Only clusters with more than one member are returned. ``cluster_id`` is
        the lexicographically smallest doc_id in the cluster.
        """
        # union-find over near-dup edges
        parent: dict[str, str] = {d: d for d in self._signatures}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

        for doc_id, sig in self._signatures.items():
            for other in self._candidates(sig):
                if (
                    other != doc_id
                    and self._estimated_jaccard(sig, self._signatures[other]) >= self.threshold
                ):
                    union(doc_id, other)

        groups: dict[str, list[str]] = {}
        for doc_id in self._signatures:
            groups.setdefault(find(doc_id), []).append(doc_id)
        return {root: sorted(members) for root, members in groups.items() if len(members) > 1}


def jaccard(text_a: str, text_b: str, k: int = 5) -> float:
    """Exact Jaccard similarity of the k-shingle sets of two texts (for tests)."""
    sa, sb = _shingles(text_a, k), _shingles(text_b, k)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def iter_unique(docs: Iterable[tuple[str, str]], **kwargs: object) -> list[str]:
    """Return doc_ids that are the canonical representative of each cluster."""
    dedup = MinHashDeduplicator(**kwargs)  # type: ignore[arg-type]
    for doc_id, text in docs:
        dedup.add(doc_id, text)
    clusters = dedup.clusters()
    drop: set[str] = set()
    for members in clusters.values():
        drop.update(members[1:])  # keep first, drop rest
    return [d for d in dedup._signatures if d not in drop]


__all__ = ["MinHashDeduplicator", "jaccard", "iter_unique"]
