---
name: corpus-grep
description: Search the indexed corpus for exact lexical matches with a regex.
when_to_use: Use when the query contains quoted phrases, unique identifiers, code-like tokens, or any string you expect to appear verbatim in a doc. Beats vector retrieval on lexical-bridge queries (SPEC §15.2).
---

# corpus_grep

Run an ACL-filtered regex search over raw doc text in the corpus.

- Call `corpus_grep` with `pattern` (the regex), an optional `glob` to scope to
  a subtree, and `max_hits` / `context_lines` to bound the result size.
- Pass `user_principals` so the store only returns docs the caller can read.
- Each `GrepHit` carries the doc id, the matching line + ±N context, and a
  `Source` citation pinned to the underlying chunk — cite from the hit.
- Prefer `corpus_grep` over the vector retriever for quoted exact phrases,
  rare identifiers, and code-like queries. Combine with `corpus_neighbors`
  for multi-hop bridging.
