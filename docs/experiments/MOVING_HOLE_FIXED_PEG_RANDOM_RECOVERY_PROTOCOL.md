# Random-target and physical-disturbance recovery evaluation, seed 0

## Scope and frozen matrix (2026-09-07, before executing cases)

Four existing best checkpoints are used without retraining or tuning. Architecture,
normalization, temporal aggregation, force compensation, 30 Hz policy / 500 Hz
native force history, 1 ms physics and collector terminal hold are unchanged.
Every rollout has a maximum of 1200 policy steps (40 simulated seconds) and the
existing 40 N physics-step hard stop. Videos record both training camera views.

The suite contains **25 cases × 4 models = 100 paired rollouts**:

| Cases | Count per model | Definition |
| --- | ---: | --- |
| Nominal control | 1 | Original fixed target, no external force |
| Random target | 8 | Four uniform-area XZ points per annulus: 0–1.5 mm, 1.5–3 mm; uniform azimuth; Python RNG seed 20260907 |
| Near-mouth pulse | 8 | Remaining axial distance first reaches 45 mm; peak 2 or 6 N × ±X/±Z |
| Mid-insertion pulse | 8 | Remaining axial distance first reaches 25 mm; peak 2 or 6 N × ±X/±Z |

The eight target points are a **balanced stratified sample**, not uniform over
the whole disk; do not report their unweighted average as a uniform-disk estimate.
All cases use policy seed 0 and deployment latent zero. Pairing means the same
case and trigger rule, not identical physical state: each policy reaches the
trigger on its own trajectory. This is an end-to-end robustness/recovery test,
not a reset-to-identical-contact-state counterfactual comparison.

Random target cases translate only `fixed_peg_fixture` in world XZ using the
legacy `--hole-offset-*` flags. The training dataset's peg target was always
nominal; these cases are local target-location OOD, NOT exact training collision
replays. Collector collision data instead perturb the socket trajectory (initial
radial errors 4/6/8/10 mm and in-hole commanded lateral displacement 3.4/3.8 mm).

Pulse cases leave the target unchanged. A one-shot world-coordinate force at
the `hole_tool` center of mass is written to MuJoCo `xfrc_applied` at each physics
step: `F(t) = direction * peak * sin²(pi * elapsed / 0.5 s)`. The total ideal
impulse is 0.5 or 1.5 N·s. No direct qpos edit, sensor injection, auxiliary
admittance, axial push, or reset/flush of policy history is applied. This is an
external-load stress test, **not** an exact replay of the expert's position-command
disturbance. Original external wrench is restored after release/exit.

Remaining distance is `-dot(hole_goal - fixed_peg_tip, [0,-1,0])`; the trigger
requires remaining distance in `[0, threshold]` and no terminal hold yet. The
near-mouth threshold does not itself guarantee contact. A trigger can happen
with lateral error; record that error and distinguish actual contact explicitly.

## Outcomes and interpretation

- Untriggered: failed to reach the trigger (or finished without exposure); never
  a recovery success. Report all planned cases as well as exposed denominators.
- Released: the entire 0.5 s pulse elapsed and external load returned to baseline.
- Recovered: released, then completed existing insertion AND collector terminal
  hold acceptance after release. Recovery time is measured from release to this
  terminal acceptance, not merely a decrease in measured force.
- Physical contact during pulse: the sum of magnitudes of MuJoCo contact forces
  between the fixed cylindrical peg and socket-body geoms exceeded 0.1 N at
  any sampled physics step. This is distinct from external force and sensor load.
- Contact recovery: both physical contact during pulse and recovery above.
- Hard stops, timeouts and terminal-hold rejections remain failures. Low sensor
  force without peg/socket engagement is not evidence of contact recovery.

`disturbance_trace.csv` captures every physics sample from trigger through exit:
external force, sensor force, physical contact force/count, remaining/lateral
errors and release flag. `summary.json` contains impulse and recovery metadata.
The sample timestamp is the end of the `mj_step`; the external force applies to
the interval just executed. MuJoCo sensors/contacts use its standard step output
semantics (as in the existing rollout); no extra forward call alters the protocol.
The reported contact magnitude sums per-contact magnitudes, not net force.

## Staging and reproducibility

First run the four-model engineering subset: `control`, `random_01`, `rim_2n_xp`,
`mid_6n_xp` (16 rollouts). Confirm clean control behavior, actual applied impulse,
static fixed peg and sensor response; do not change model or success thresholds
based on task failures. Then complete all other preregistered cases.

```bash
conda activate forceact
cd /home/stw/ForceAwareACT_workspace/ForceAwareACT
PYTHONPATH=src python scripts/run_sampling_ablation_recovery_suite.py
PYTHONPATH=src python scripts/run_sampling_ablation_recovery_suite.py --execute --pilot-only
PYTHONPATH=src python scripts/run_sampling_ablation_recovery_suite.py --execute
```

Plan-only writes nothing. Execution saves an immutable plan including case
coordinates, commands, code/checkpoint/XML SHA256 fingerprints. Resumption checks
the full plan, rollout contract, logs and video frame counts; mismatched or
incomplete cases are never silently overwritten. Process logs sit beside each
model directory. Per-case results and counts update after every validated run.
Output: `runs/moving_hole_fixed_peg_random_recovery_v1_seed0/`.

This is a small single-training-seed, deterministic-policy test. Direction cases
are designed conditions, not independent training replicates. Do not infer broad
statistical superiority or tune on the already-evaluated holdout set.
