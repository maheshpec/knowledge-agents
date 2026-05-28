---
name: corpus-read
description: Read a doc (whole or windowed) with a citation pinned to its chunk.
when_to_use: Use after a grep / glob / neighbors call surfaces a candidate doc and you need full context. Window with start_line/end_line + max_bytes.
---

# corpus_read

Return a windowed slice of a doc together with a `Source` citation.

- Call `corpus_read` with `doc_id` and an optional `(start_line, end_line)`
  window plus `max_bytes` to cap the byte budget.
- Pass `user_principals` so a read against a doc the caller cannot see returns
  empty content (not a leak of existence).
- The returned `DocSlice.citation` carries the chunk + char span the slice
  covers — cite the chunk in any claim derived from `content`.
- Long files: iterate by advancing `start_line` per call rather than raising
  `max_bytes` past the policy ceiling.
