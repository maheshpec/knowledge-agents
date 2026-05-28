---
name: corpus-describe
description: Get title / source / authors / length / ACL / ingest time for a doc.
when_to_use: Use to qualify a candidate before reading it, or to surface authorship + provenance for a citation.
---

# corpus_describe

Return a `DocMetadata` record for a doc the caller can see.

- Call `corpus_describe` with `doc_id`.
- Pass `user_principals` so an opaque doc returns a metadata-empty record
  rather than leaking its title / source.
- Use the result to decide whether a `corpus_read` is worth the byte budget.
