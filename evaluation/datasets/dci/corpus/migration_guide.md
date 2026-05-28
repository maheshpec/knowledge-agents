# Migration guide

When you move a project from the v1 to the v2 client, the most disruptive
change is that result types are now Pydantic models instead of dictionaries.
Callers that indexed into results with bracket notation need to switch to
attribute access. The compatibility shim covers the common cases but does not
preserve types under serialization round-trips.

The configuration file moves from YAML to TOML to match the rest of the
ecosystem. A converter script ships under `scripts/migrate_config.py`.
