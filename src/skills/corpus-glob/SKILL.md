---
name: corpus-glob
description: List docs in the corpus that match a path pattern.
when_to_use: Use to scope a follow-up grep / read to a subtree or to enumerate docs of a given type (e.g. ``*.md``, code). Pair with corpus_grep / corpus_read.
---

# corpus_glob

Enumerate docs by logical path pattern (`/collection/source/doc_id` style).

- Call `corpus_glob` with `pattern` (default `**/*`) and an optional `types`
  filter (e.g. `["md", "code"]`) plus `limit` to bound the result count.
- Pass `user_principals` so the listing only includes docs the caller can read.
- Each `DocRef` is a (doc_id, path, title, source, type, length) pointer;
  pass `doc_id` to `corpus_read` or `corpus_describe`.
- Use as the first step when you know the answer lives under a specific
  collection / source but not the exact doc.
