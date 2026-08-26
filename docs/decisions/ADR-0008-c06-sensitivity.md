# ADR-0008: C06 cap and calibration sensitivity

## Decision

C06 is a preregistered FD001-to-FD001 sensitivity analysis of the C02 split-conformal design. It evaluates four fixed conditions: `primary` (cap 125, calibration cut points 40–90%), `cap_130` (cap 130, 40–90%), `early_calibration` (cap 125, 40–65%), and `late_calibration` (cap 125, 65–90%). Every condition uses the same engine-level 60/20/20 split, five declared seeds, temporal windows, point-model hyperparameters, alpha values, endpoint evaluation, and 2,000-resample engine-level bootstrap policy as C02.

The conditions are descriptive sensitivities, not candidates selected using validation or official-test performance. C06 produces one immutable artifact per condition and seed. Each run records its cap and cut-point policy, split, cut points, frozen model specification, input hashes, configuration hash, lockfile hash, environment, and artifact hashes.

## Consequences

Changing a cap changes the supervised target and therefore requires a condition-specific base-training fit. Changing only the calibration cut-point policy retains the same declared model and feature procedure, but C06 does not select among conditions after observing outcomes.
