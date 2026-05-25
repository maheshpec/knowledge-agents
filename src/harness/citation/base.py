"""Citation schemas for structured, grounded generation (SPEC §6.13).

The model is constrained (via Anthropic tool-use) to emit a :class:`CitedDraft`:
an ordered list of segments, each carrying the ``chunk_id``s from the candidate
set that support it. The enforcer then validates and reflows it into the final
:class:`common.schemas.GenerationResult` with ``claim_span`` offsets.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CitedSegment(BaseModel):
    """One claim (or contiguous supported span) plus its backing citations."""

    text: str
    citation_ids: list[str] = Field(default_factory=list)  # chunk_ids; [] = uncited
    confidence: Literal["high", "medium", "low"] = "high"


class CitedDraft(BaseModel):
    """The structured generation output the model is constrained to produce."""

    segments: list[CitedSegment] = Field(default_factory=list)
    refused: bool = False  # true if the model declines (e.g. no supporting evidence)
    refusal_reason: str | None = None


Strictness = Literal["strict", "loose", "off"]

__all__ = ["CitedSegment", "CitedDraft", "Strictness"]
