# ADR-0002: MALT task-family split

## Status

Deferred -- not implemented. MALT is outside the current core study. This ADR
preserves a leakage boundary for any separately reviewed future revival; it does
not authorize implementation.

## Decision

MALT train, calibration, and evaluation partitions must be disjoint at the task-family level.

## Rationale

Transcripts from one task family can share structure, wording, and execution patterns. A random transcript split can leak task-family information and produce an optimistic estimate.

## Consequences

All primary MALT estimates use task-family grouping. A random split may appear only as an explicitly labelled leaky optimistic baseline.
