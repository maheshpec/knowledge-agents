# Reranker design

A cross-encoder reranker takes the top candidates from a fast first-stage
retriever and rescores them by jointly attending to query and candidate text.
The win is precision at the top of the list: the first-stage retriever favours
recall, so its top slots tend to include passages that share vocabulary with the
query but are off-topic, and the reranker culls them.

Operationally a reranker is a latency tax, so the harness only runs one when
the first stage returns more candidates than the prompt can hold, and otherwise
passes results through untouched.
