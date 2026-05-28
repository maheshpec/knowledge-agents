---
name: corpus-neighbors
description: Walk the knowledge graph from a chunk to its multi-hop neighbors.
when_to_use: Use for relational / multi-hop queries — "who founded the company that acquired X". Pair with corpus_grep or vector retrieval as the seed step.
---

# corpus_neighbors

Walk the knowledge graph from a chunk to nearby chunks, ranked by hop distance.

- Call `corpus_neighbors` with `chunk_id` (a seed from a prior retrieval /
  grep step) and `hops` (default 1; bounded by a per-tool max).
- Pass `user_principals` so the walk respects ACLs at every hop.
- Each returned `ChunkRef` carries (chunk_id, doc_id, text, hops, citation);
  lower `hops` means a more direct relation path.
- Use as the *second* step in `dci_then_vector` / `vector_then_dci` chained
  routes (SPEC §15.2): seed with grep or vector, then expand via the KG.
