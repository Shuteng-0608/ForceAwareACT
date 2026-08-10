# Paired Fibonacci R4 mm Temporal Rollout Plan

## Purpose

This experiment tests whether the Q=1 signed-temporal behavior observed at the
zero-offset hole transfers to spatial perturbations. It compares the paired50
official ACT checkpoint and the paired50 high-rate Contact-CVAE checkpoint on
the same fixed 100-point x/z disk.

This document preregisters the method. It does not contain experiment results.

## Fixed point set

- File: `configs/experiments/fibonacci_disk_100_r4mm.csv`
- SHA256: `a26e2a5a3e215b6cbb7696c82e5154a19b978139b4696840c652ceed0861df94`
- Number of points: 100
- Plane: x/z, with y offset fixed to zero
- Maximum radius: approximately 3.990 mm
- Point order and coordinates must not be changed during the experiment.

The Fibonacci disk is a deterministic, area-balanced spatial perturbation set.
It is not a stochastic repeat protocol. A rollout seed is paired across every
model/k configuration at a point, but changing a seed does not by itself imply
that the deterministic simulator produces a distinct trajectory.

## Preregistered configurations

Both policies use every signed temporal decay below:

- 0.0100: official ACT temporal-decay anchor
- 0.0500: stable middle-range anchor
- 0.0710: fastest zero-offset Contact-CVAE result
- 0.0735: lowest-force zero-offset Contact-CVAE result
- 0.0750: narrow zero-offset recovery-island probe

Policies:

- `official_act`
- `highrate_contact_v3`, deployed with `z_contact=0`

The full design contains `100 points x 5 k values x 2 policies = 1000`
rollouts. Every point uses `seed_base + point_index - 1`, and that seed is
identical across all configurations at that point.

## Locked fairness contract

- action mode: joint position
- action query interval: Q=1
- action selection: signed temporal aggregation
- policy rate: 30 Hz
- maximum rollout length: 600 policy steps
- EMA alpha: 1.0
- maximum per-step joint target delta: 0.02 rad
- hard force stop: 100 N
- safe-force threshold: 40 N
- success distance: 3 mm
- success dwell: 0.1 s
- axial push: disabled
- ordinary and HUD videos: disabled during the full numerical batch

ACT does not receive force as policy input. Contact-CVAE receives only its
causal 500 Hz force-history contract and uses a zero deployment latent. The
shared force stop is an external safety mechanism, not an ACT policy input.

## Execution gates

1. Generate and inspect the immutable experiment plan without executing actions.
2. Verify non-video raw/compensated high-rate force metrics.
3. Run a small functional preflight using inner, middle, and outer points.
4. Audit checkpoint identity, point offsets, Q=1, causal force use, zero latent,
   success semantics, force metrics, and resume behavior.
5. Obtain explicit approval before the full 1000-rollout execution.

The preflight is an engineering check and must not be used to adaptively select
k values. If k values are changed after observing Fibonacci performance, the
100-point set becomes a development set and a separate holdout point set is
required for an unbiased final comparison.
