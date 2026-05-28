---
name: corpus-ls
description: Browse the logical tree of the corpus directory-style.
when_to_use: Use to explore the corpus shape when you do not know which collection or source holds the answer. Pair with corpus_glob / corpus_read.
---

# corpus_ls

List entries directly under a path in the logical corpus tree.

- Call `corpus_ls` with `path` (default `/`); a `DirectoryListing` returns
  entries that are either directories (collections / sources) or docs.
- Pass `user_principals` so hidden subtrees stay hidden.
- Descend incrementally: ls `/`, then ls a collection, then ls a source —
  cheaper than a recursive glob when you only need one branch.
