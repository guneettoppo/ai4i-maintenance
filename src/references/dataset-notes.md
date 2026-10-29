# Dataset notes — AI4I 2020

Working notes on what the data can and cannot support. These are the constraints
the modelling decisions in `README.md` are built on.

## What it is

10,000 rows of a synthetic milling process. Columns: `UDI`, `Product ID`, `Type`
(quality variant L/M/H), five continuous sensor channels, the binary
`Machine failure` flag, and five one-hot columns naming the failure mode
(TWF, HDF, PWF, OSF, RNF).

Synthetic, generated to mirror a real milling setup and distributed by the UCI
Machine Learning Repository under CC BY 4.0. Absolute performance on a specific
machine does not transfer; the structure (rare target, heterogeneous modes,
physically meaningful channels) does.

## Constraints that shaped the pipeline

1. **Rare positive class.** 339 failures out of 10,000 (3.39%). Rank metrics and an
   explicit operating point, not accuracy.
2. **Heterogeneous modes.** Each mode has its own physical driver, so aggregate
   recall hides a mode being missed. Per-mode detection is reported separately.
3. **Label noise.** `Machine failure` is not exactly the OR of the five mode flags:
   9 rows are flagged with no mode, 18 carry a mode with a zero flag. This caps
   achievable accuracy and makes the last decimal of AUC meaningless.
4. **No temporal or machine identity.** One row per pass, no timestamps, no machine
   IDs. Sequence models and grouped splits are out of reach with this file; that is
   a data limitation, not a modelling choice.
5. **Collinear channels.** corr(torque, rpm) = −0.876 and corr(air, process temp) =
   0.876. Attribution must be read two ways (SHAP and permutation) because
   collinear features split credit differently under each.

## Deliverable preference

A working, reproducible prediction pipeline — train script, saved model,
predictions, metrics — over a written report. Every number in the README comes out
of `scripts/evaluate.py` writing `reports/metrics.json`.
