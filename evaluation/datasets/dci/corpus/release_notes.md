# Release notes — v2.6.1

The retry policy backoff sequence is now `RETRY_BACKOFF_MS=[100, 400, 1600]`,
replacing the linear schedule from v2.5. Operators tracking the schedule via
the metrics pipeline should update their alerts.

Connection draining respects the new `SHUTDOWN_GRACE_SECONDS=30` setting; in-
flight requests are allowed to complete before the process exits.

The deprecated `LEGACY_AUTH_TOKEN_HEADER` flag is removed in this release.
