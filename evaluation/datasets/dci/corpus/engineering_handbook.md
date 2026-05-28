# Engineering handbook

The retrieval metric helper is named `compute_recall_at_k` and lives in
`src/evaluation/metrics/retrieval.py`. Callers pass `k` and the gold-id set.

API keys rotate on a fixed schedule: `API_KEY_ROTATION_INTERVAL=86400` seconds
(one day). The rotator emits a structured audit log on every cycle.

The payload service rejects requests over the size cap with the error code
`ERR_PAYLOAD_TOO_LARGE`. Clients are expected to retry with a smaller payload
or stream the upload through the chunked endpoint.

The Reciprocal Rank Fusion threshold is 60 in the default configuration; it is
exposed as a registry parameter so the evolutionary loop can sweep it.
