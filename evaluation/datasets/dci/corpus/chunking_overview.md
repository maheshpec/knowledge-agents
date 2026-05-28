# Chunking overview

Splitting a document into smaller pieces matters because the embedding model
has a context window and because a tight passage is easier for the reader to
verify. The right size is corpus-dependent: technical prose tolerates larger
pieces than dialog, and figure-heavy pages need different handling than plain
text. The harness keeps several strategies behind one interface so the
self-improvement loop can pick the best per workload.

Overlap between pieces is a hedge against losing a thought at a piece boundary;
the cost is duplicated tokens at query time, so the trade-off is real.
