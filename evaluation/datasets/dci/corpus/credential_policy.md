# Credential policy

When access credentials reach the end of their permitted lifetime, the system
treats the holder as unauthorized until a replacement is issued. The renewal
flow walks the holder through proving possession of the prior secret and
optionally a second factor, then mints a fresh one with a new expiry stamp.

Stale credentials are not a fault; they are an expected, normal part of the
lifecycle. Operators should plan rotations rather than treat them as incidents.

Holders who delay renewal past the grace period are quietly de-provisioned —
their downstream privileges fall away even before the upstream identity does.
