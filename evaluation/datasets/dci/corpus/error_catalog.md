# Error catalog

| Code | Meaning | Recovery |
|---|---|---|
| `ERR_RATE_LIMIT_EXCEEDED` | Caller is over the per-minute quota. | Back off; respect `Retry-After`. |
| `ERR_SCHEMA_VALIDATION` | Payload failed Pydantic validation. | Fix the payload; do not retry as-is. |
| `ERR_UPSTREAM_TIMEOUT` | A downstream call exceeded its deadline. | Idempotent retry with jitter. |
| `ERR_TENANT_NOT_PROVISIONED` | Tenant is unknown to the routing table. | Call the onboarding endpoint first. |

Each error is reported with a stable `error_code` field so downstream alerting
can route by exact match.
