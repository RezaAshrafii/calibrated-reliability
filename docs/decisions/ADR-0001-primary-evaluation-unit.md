# ADR-0001: Primary C-MAPSS evaluation unit

## Status

Accepted.

## Decision

The primary C-MAPSS evaluation unit is the engine endpoint: one prediction for the final observed cycle of each test engine.

## Rationale

Rows from one engine are temporally dependent and are not independent examples. Engine-level partitioning prevents trajectory information from crossing partitions, while endpoint evaluation reflects the operational RUL task and avoids overweighting long trajectories.

## Consequences

Metrics and bootstrap confidence intervals must be reported at engine level. Cycle-level plots are exploratory, not the primary inferential result.
