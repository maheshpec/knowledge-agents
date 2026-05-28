# Falcon engine

The Falcon engine is the inference scheduling layer that owns model selection,
batching, and timeout enforcement for every LLM call the harness makes. It
exposes a single async submit() entry point and a small policy object for
per-call overrides.

Falcon's hot path reads from the Comet caching layer for repeated prompts;
cache misses fall through to the upstream provider. The Comet integration is
opt-in per route — the routing table marks a route as cacheable when its
prompts are deterministic and its outputs are reusable.

Falcon does not own retries. It returns structured errors and lets the caller
decide whether the work is worth a second try.

Operational knobs are exposed as constants: `FALCON_SUBMIT_TIMEOUT_MS = 5000`
caps each submit() before Falcon returns a timeout error to the caller.
