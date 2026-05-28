# Observability primer

A trace is a tree of spans tied together by a shared trace id. A span is one
unit of work — a function call, a remote request — with a name, a start, an
end, and a parent. When the tree is whole, you can ask questions like "what
fraction of total latency was spent waiting on the index?" and read the answer
straight off the structure.

Metrics roll up the same observations into time series. Logs add a free-text
channel for ad-hoc context. The three together form the standard kit.
