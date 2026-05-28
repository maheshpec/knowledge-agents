# Comet caching layer

Comet is the prompt-and-result cache that sits in front of upstream LLM
providers. Entries are keyed on a content hash of the rendered prompt plus the
model id; the TTL is configured per route and defaults to one hour.

Comet's main consumer is the Falcon engine, which checks Comet before issuing
an upstream call and writes the upstream response back on a miss. A separate
warmer process pre-populates Comet for hot prompts off the critical path.

Cache eviction is LRU bounded by total bytes. Operators can pin specific keys
through the admin API when a known-hot prompt would otherwise churn.
