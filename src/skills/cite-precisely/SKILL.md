---
name: cite-precisely
description: Attach a precise source citation to every factual claim.
when_to_use: Use whenever the answer asserts facts that must be grounded in the retrieved evidence — definitions, figures, dates, quotes, or any claim a reader might challenge.
---

# Cite Precisely

When answering, ground **every** factual claim in the retrieved evidence.

- Break the answer into short segments, each making a single claim.
- For each segment, cite the `chunk_id`(s) from the candidate set that directly
  support it. Prefer the most specific passage over a general one.
- Quote verbatim only short spans; paraphrase longer support but still cite it.
- If no passage supports a claim, **do not make the claim**. Say what the
  evidence does and does not cover.
- Never cite a `chunk_id` that is not in the provided candidate set.

A well-cited answer lets the reader verify each statement against its source in
one hop.
