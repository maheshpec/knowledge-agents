# DCI eval slices (SPEC §15.5)

Three query slices + a citation-audit slice that target the Direct Corpus
Interaction acceptance criteria, plus a 12-document synthetic fixture corpus.
Hermetic, ACL-clean, no PII — committed in full so CI runs are reproducible.

The slices are designed so the routing strategies separate cleanly:

| Slice (`*.jsonl`) | DCI grep wins | Vector hybrid wins | Chained wins | §15.5 criterion |
|---|---|---|---|---|
| `lexical_bridge` | ✅ exact identifiers / quoted strings | — | — | 1 (`dci` beats vector by ≥ 10% recall@5) |
| `paraphrastic` | — | ✅ semantic-only phrasing | — | 2 (vector ties/beats DCI within noise) |
| `multihop` | — | — | ✅ grep anchor + vector expansion | 3 (`dci_then_vector` beats either alone) |
| `citation_audit` | grep citations land on the gold doc | — | — | 5 (citation precision > 0.95) |

## Why the separation is sharp on this fixture

- Gold docs for **lexical_bridge** contain unique identifiers like
  `compute_recall_at_k`, `ERR_PAYLOAD_TOO_LARGE`, `API_KEY_ROTATION_INTERVAL` —
  tokens that a natural-language bag-of-words vector embedding never sees in its
  vocabulary, so vector recall is near-zero. Grep nails them on first try.
- Gold docs for **paraphrastic** describe their topic in prose that shares no
  word with the query, but is semantically close. Grep on the query terms turns
  up nothing; the vector path finds the doc through shared topical vocabulary.
- Gold docs for **multihop** require bridging two named entities (Falcon engine
  ↔ Comet caching layer). One run of grep finds one anchor; vector alone misses
  the bridge; `dci_then_vector` grep-anchors then expands.

`citation_audit.jsonl` is the lexical slice with `relevant_chunk_ids` populated
so we can check that DCI tool citations actually attribute to the correct chunk
(SPEC §15.5 #5).

## Schema

Each line is a `GoldQuery` (SPEC §9.1):

```json
{
  "query_id": "...",
  "query": "...",
  "relevant_chunk_ids": ["doc-X::chunk-0"],
  "relevant_doc_ids": ["doc-X"],
  "expected_answer": "...",
  "intent": "lookup|synthesis|relational",
  "difficulty": "easy|medium|hard",
  "notes": ""
}
```

`corpus/` holds the markdown source. The acceptance test (`tests/integration/
test_phase5_acceptance.py`) ingests these as one chunk per doc with stable
`chunk_id = "<doc-id>::chunk-0"`, so the chunk ids referenced above are stable
across runs.
