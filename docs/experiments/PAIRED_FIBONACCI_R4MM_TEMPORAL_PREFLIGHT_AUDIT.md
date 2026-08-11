# Paired Fibonacci R4 mm Temporal Rollout Preflight Audit

## Scope

This audit records the functional GPU preflight performed before the formal
100-point paired rollout. It is an engineering gate, not a model-selection or
performance experiment.

- Code commit before execution: `183f230`
- Output: `runs/paired_fibonacci_r4mm_temporal_preflight_b1`
- Protocol fingerprint:
  `6d1f9572e49cb942dc5f4bb90a78bbcbb823aa874e3e10ad24613151ae24b292`
- Point-set SHA256:
  `a26e2a5a3e215b6cbb7696c82e5154a19b978139b4696840c652ceed0861df94`
- Selected point indices: 1, 50, 100
- Selected signed temporal decays: 0.01 and 0.0735
- Policies: official ACT and high-rate Contact-CVAE
- Planned/completed: 12/12
- Process failures: 0
- Videos: disabled
- Force metrics: enabled without video rendering

The three selected points span approximately 0.283 mm, 2.814 mm, and
3.990 mm radial offsets. Every configuration at a point used the same seed,
defined as `point_index - 1` for this seed-base-zero preflight.

## Contract audit

All 12 rollouts passed the following checks.

### Shared rollout contract

- CUDA was selected and resolved successfully.
- Input images retained checkpoint-native shape `[1, 2, 3, 480, 640]`.
- Joint state shape was `[1, 7]`.
- Action mode was joint position.
- Signed temporal aggregation queried the policy every frame (`Q=1`).
- Policy rate was 30 Hz and the maximum length was 600 policy steps.
- EMA alpha was 1.0 and maximum joint-target delta was 0.02 rad.
- Geometric success required 3 mm distance for a continuous 0.1 s dwell.
- Safe-force threshold was 40 N and hard stop was 100 N.
- Axial push and video recording were disabled.
- Point coordinates and seeds exactly matched the immutable experiment plan.

### Official ACT boundary

- Policy variant was `official_act`.
- Force-history contract was `not_used`.
- Force input shape was `[1, 0, 6]` and valid force samples remained zero.
- No force value entered ACT inference; force was used only by the external
  monitoring and hard-stop mechanism.

### High-rate Contact-CVAE boundary

- Policy variant was `act_aligned_high_rate_contact_cvae`.
- Force tensor shape was `[1, 7, 20, 6]`, the grouped representation of the
  causal high-rate history contract.
- Contract was `causal_raw_500hz_last_100_grouped_state_intervals_v2`.
- Valid history grew from one sample to 100 samples.
- Maximum history span was 0.198 s and latest force age was at most 0.001 s.
- Sampling rate was 500 Hz; skipped and duplicate sample counts were both zero.
- Deployment latent source was `zero`; maximum observed latent magnitude was
  exactly zero.

### Force-metric integrity

- Every policy row had one interval record.
- For every rollout,
  `force_metrics_interval_count == steps_executed`.
- For every rollout,
  `force_metrics_sample_count == physics_steps_total + steps_executed`.
- Raw metric maxima exactly matched the independent physics safety maxima.
- Summary raw/compensated maxima exactly matched maxima recomputed from CSV.
- Raw/compensated duration above 40 N and excess-force exposure exactly matched
  the sums recomputed from interval CSV rows.
- `safe_success_compensated` was exactly reproducible from task success and the
  compensated 40 N threshold.

## Functional outcomes

These outcomes confirm that trajectories reach success, timeout, and hard-stop
branches correctly. They must not be used to alter the preregistered formal k
set.

| Policy | k | Task success | Raw-safe success | Comp-safe success | Hard stops |
|---|---:|---:|---:|---:|---:|
| Official ACT | 0.0100 | 0/3 | 0/3 | 0/3 | 1 |
| Contact-CVAE | 0.0100 | 0/3 | 0/3 | 0/3 | 3 |
| Official ACT | 0.0735 | 0/3 | 0/3 | 0/3 | 0 |
| Contact-CVAE | 0.0735 | 3/3 | 0/3 | 0/3 | 0 |

Contact-CVAE at k=0.0735 reached geometric success at approximately 9.833 s,
11.900 s, and 15.500 s. All three exceeded the 40 N safety threshold. This is
consistent with treating geometric success and safe success as separate primary
outcomes.

The `nan` values printed for official ACT predicted-force trends are deliberate
not-applicable placeholders: official ACT has no force prediction head. No
joint, measured-force, distance, timing, or control value used in the audit was
non-finite.

## Resume audit

The complete preflight was rerun with `--skip-existing`. All 12 summaries were
validated and skipped without invoking a policy. The four aggregate artifacts
were byte-identical before and after resume:

- `per_rollout.csv`:
  `a9f80daff8ed7368a1c9266b62ddff29473e89e2c7f9248ec75f36ad06a2e9dc`
- `per_point.csv`:
  `99b78135d3315707d6669a47b83f38dbe10215b4673923013f1d87a90b37b927`
- `per_configuration.csv`:
  `a2cd954b2ee1f94ea1ab6744934a3e0941ff3e5d2767aac115281aa8cc1890d9`
- `aggregate_summary.json`:
  `7740099b7e531ed6216c6f80ddf1176eadc6b2cc698ee5be2ff916278c9597b6`

## Gate decision

The implementation is functionally ready for the formal 100-point experiment.
The formal run remains gated on explicit user approval. The preregistered five
k values must remain 0.01, 0.05, 0.071, 0.0735, and 0.075 for both policies,
giving 1000 paired rollouts in total.

The preflight averaged approximately 12 wall-clock seconds per rollout. A simple
projection is about 3.3 hours, but 4--5 hours should be reserved because the
full set contains more long timeouts, checkpoint startup, filesystem writes,
and possible operational interruptions.
