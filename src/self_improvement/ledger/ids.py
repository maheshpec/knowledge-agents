"""Time-orderable IDs and content-addressable hashing for the ledger (SPEC §8.2.1).

``uuid7`` gives experiment IDs that sort by creation time (the high bits are a
millisecond timestamp), so a lexical sort of gen-NNN.jsonl lines is also a
chronological sort — handy for replay and audit. Implemented inline per RFC 9562
to avoid an extra runtime dependency.

``config_hash`` is the content address of a :class:`PipelineConfig`: identical
configs hash identically so the loop can dedupe / share evaluation results.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from uuid import UUID

from self_improvement.registry.pipeline_config import PipelineConfig


def uuid7() -> UUID:
    """Generate a UUIDv7 (48-bit ms timestamp + version/variant + randomness)."""
    unix_ms = int(time.time() * 1000)
    rand = os.urandom(10)
    # Layout (128 bits): 48 ts | 4 ver | 12 rand_a | 2 var | 62 rand_b.
    value = unix_ms << 80
    value |= 0x7 << 76  # version 7
    value |= (rand[0] & 0x0F) << 72  # 4 bits of rand_a (top nibble)
    value |= rand[1] << 64  # 8 more bits of rand_a
    value |= 0b10 << 62  # RFC 4122 variant
    value |= int.from_bytes(rand[2:], "big") & ((1 << 62) - 1)  # rand_b
    return UUID(int=value)


def config_hash(config: PipelineConfig) -> str:
    """Stable 16-char content hash of a pipeline config (key order-independent)."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


__all__ = ["uuid7", "config_hash"]
