# E5 final pretraining report: native 500 Hz contact-CVAE

Date: 2026-08-05  
Branch: `refactor/act-aligned-contact-cvae`  
Decision: **implementation ready; formal training is gated by one local-GPU burn-in**

## 1. Scope frozen for the first comparison

The first controlled experiment uses the existing paired-50 manifest:

- data root: `mujoco_data/peg_hole_100`;
- manifest: `configs/experiments/peg_hole_paired50_seed0.json`;
- selected episodes: 50;
- training/validation episodes: 40/10;
- training/validation state windows: 12,427/3,047;
- untouched holdout episodes: 50;
- batch size target: 8;
- action chunk: 100;
- official reference duration: 2,000 epochs;
- matched training budget at batch 8: 10,000 optimizer steps per model.

The official baseline samples one random timestep from each of 40 episodes per
epoch, giving five optimizer steps per epoch and 10,000 total steps. The
high-rate trainer shuffles all state windows and stops after the same 10,000
steps, approximately 6.43 complete passes over 12,427 training windows. The
optimizer-update budget is matched, but the episode/timestep sampling
distribution is intentionally not claimed to be identical.

## 2. Frozen model and data contracts

### Official ACT baseline

- architecture: `official_act_single_arm_v1`;
- canonical parameter count: 83,908,744;
- inputs: two native-resolution RGB images and normalized 7-D qpos;
- posterior during training: qpos + 100-action chunk;
- deployment latent: exact zero `z_motion`;
- outputs: `[B,100,7]` actions;
- no force input or force loss.

### Native-force contact-CVAE

- architecture: `act_aligned_contact_cvae_highrate_force_v2`;
- training: `act_aligned_highrate_conditional_cvae_training_v3`;
- canonical parameter count: 104,038,597;
- images: native 480 x 640, with the same train/rollout preparation;
- online force: causal last 100 native 500 Hz wrench samples;
- online packing: at most seven state intervals, 20 samples per interval;
- posterior future force: every wrench in `(t_j,t_{j+1}]` for each action;
- shared local force encoder: one registered four-layer dilated TCN;
- posterior/online/policy Transformer encoder depth: four;
- policy decoder depth: seven;
- deployment latent: exact zero `z_contact` by default;
- outputs: action `[B,100,7]`, endpoint force `[B,100,6]`, and native-force
  auxiliary prediction `[B,100,20,6]`.

The force normalization mean/std uses every native wrench sample in the 40
training episodes only. Action and qpos statistics remain state-rate. Action,
future-force interval, and native-sample masks are independent.

The objective is:

```text
action L1
+ endpoint-force L1
+ 0.5 * interval-balanced native-force L1
+ 10 * KL(q_contact || N(0,I))
+ KL(stopgrad(q_contact) || p_contact_conditional)
```

The prior's online condition is detached for the prior-match term. That loss
updates the conditional prior itself, but not the posterior, image path, online
force path, or their shared 500 Hz encoder.

## 3. Completed evidence gates

### Dataset/timestamp audit: passed

All 100 local episodes were audited after the causal-prefix correction:

- state rate: 30.3030303029–30.3030303037 Hz;
- force rate: 499.9999999976–500.0000000118 Hz;
- 30,877 adjacent state intervals;
- native samples per interval: 16 (10,309 intervals) or 17 (20,568);
- maximum: 17, below capacity 20;
- full online window span: 0.197–0.199 s;
- 71 anchors genuinely require seven intervals;
- no timestamp assignment, grouping, truncation, or causality failures.

Training with a complete episode and rollout with only the visible state-time
prefix now produce elementwise-identical online force values, relative times,
and masks. Synthetic prefix boundaries never use future state timestamps.

### Model/training preflight: passed

A real-data CPU preflight and one-step train/reload cycle passed:

- outputs: action `[1,6,7]`, endpoint force `[1,6,6]`, native force
  `[1,6,20,6]`, posterior mean `[1,32]`;
- every key module, including the shared encoder and native-force head, received
  finite nonzero total-loss gradients;
- prior-match posterior gradient: exactly 0;
- prior-match shared-force-encoder gradient: exactly 0;
- prior-match conditional-prior gradient: positive;
- changing one valid 2 ms future force point changed posterior mean;
- optimizer covered every trainable parameter exactly once;
- checkpoint model/config/optimizer/progress/DataLoader RNG/runtime RNG strict
  reload: passed.

### MuJoCo rollout preflight: passed

A 10-query dry-run using the newly saved checkpoint passed:

- policy queries: 10;
- physics steps: 333, alternating 33/34 per query;
- achieved policy rate: 30.03003 Hz;
- continuous force samples: 167 including the initial point;
- online valid samples: 1 initially, then saturated at 100;
- final interval counts: `[0,16,17,17,16,17,17]`;
- maximum 100-point span: 0.198 s;
- maximum newest-force age: 1 ms;
- skipped/duplicate native samples: 0/0;
- deployment latent source: zero, maximum absolute value exactly 0;
- temporal aggregation: official executor, `k=0.01`;
- measured-force safety monitoring: every 1 ms physics step;
- safe-success/hard-stop thresholds: 40 N/100 N.

Official-ACT, state-rate contact-CVAE, and motion-control paths remain separate.
The final full suite result is `567 passed, 1 skipped`; the skipped test is an
existing environment-dependent case.

## 4. Mandatory local GPU gate

The Codex execution environment did not expose CUDA, so canonical batch-8 peak
memory on the local RTX 3070 is not yet measured. Do not start the formal run
until these commands pass in the `forceact` environment.

First run canonical batch-1 graph preflight:

```bash
PYTHONPATH=src python scripts/preflight_act_aligned_high_rate_contact_cvae.py \
  mujoco_data/peg_hole_100 \
  --device cuda \
  --batch-size 1 \
  --output runs/e5_highrate_preflight_b1.json
```

Then run a fresh 100-step batch-8 burn-in on the paired split:

```bash
PYTHONPATH=src python scripts/train_act_aligned_high_rate_contact_cvae.py \
  mujoco_data/peg_hole_100 \
  --experiment-manifest configs/experiments/peg_hole_paired50_seed0.json \
  --output-dir runs/e5_highrate_burnin_b8_seed0 \
  --device cuda \
  --batch-size 8 \
  --num-workers 2 \
  --seed 0 \
  --run-mode burn_in \
  --max-train-steps 100 \
  --official-reference-epochs 2000 \
  --checkpoint-interval-steps 100 \
  --log-interval 10
```

Accept the burn-in only if:

- `burn_in_summary.json` has `passed=true` and reload audit passed;
- all losses and gradient norms remain finite;
- `loss_force_highrate` is present and finite;
- CUDA peak allocated/reserved memory leaves a practical margin;
- no DataLoader or interval-capacity exception occurs.

If batch 8 is out of memory, use batch 4 for **both** official ACT and the
high-rate model. At batch 4 both schedules become 20,000 optimizer steps for
2,000 reference epochs. Do not silently use different batch sizes or update
budgets for the two compared models.

## 5. Formal paired training commands

Use new output directories and run the GPU gate first.

Official ACT:

```bash
PYTHONPATH=src python scripts/train_official_act.py \
  mujoco_data/peg_hole_100 \
  --experiment-manifest configs/experiments/peg_hole_paired50_seed0.json \
  --output-dir runs/paired50_official_act_e2000_b8_seed0 \
  --device cuda \
  --batch-size 8 \
  --num-epochs 2000 \
  --num-workers 2 \
  --seed 0 \
  --checkpoint-interval-epochs 100 \
  --log-interval 10
```

Native-force contact-CVAE:

```bash
PYTHONPATH=src python scripts/train_act_aligned_high_rate_contact_cvae.py \
  mujoco_data/peg_hole_100 \
  --experiment-manifest configs/experiments/peg_hole_paired50_seed0.json \
  --output-dir runs/paired50_highrate_contact_v3_e2000_b8_seed0 \
  --device cuda \
  --batch-size 8 \
  --num-workers 2 \
  --seed 0 \
  --run-mode formal \
  --official-reference-epochs 2000 \
  --checkpoint-interval-steps 1000 \
  --log-interval 10
```

Monitor the 10,000-step high-rate run:

```bash
python scripts/monitor_act_aligned_training.py \
  runs/paired50_highrate_contact_v3_e2000_b8_seed0 \
  --target-steps 10000 \
  --checkpoint-interval 1000 \
  --watch \
  --interval 10
```

## 6. Rollout comparison contract

Use temporal mode first. Both models must share:

- identical hole positions and rollout seeds;
- `--action-select-mode temporal --temporal-agg-decay 0.01`;
- 30 Hz policy rate and the same `--max-rollout-steps` (default 600);
- identical `max_delta_q=0.02`, `ema_alpha=1.0`, joint ctrlrange clipping;
- identical success semantics, 40 N safe threshold, and 100 N hard stop;
- exact zero deployment latent;
- no axial push unless it is enabled for both.

The high-rate checkpoint automatically requires 100 force samples over
0.198 s and rejects legacy force-window overrides. Its ring remains continuous
across policy boundaries and its summary records input span, newest-force age,
interval counts, model-input peak, skipped/duplicate samples, and independent
1 kHz safety peak.

## 7. Known research limitations—not implementation blockers

1. Demonstration control evolves at 1 kHz between state samples. Therefore
   `(t_j,t_{j+1}]` is the wrench response over a continuously evolving control
   segment, not a guaranteed zero-order hold of `action[j]`.
2. The native-force auxiliary head provides high-rate supervision but does not
   mathematically force the decoder to use the posterior. The intervention
   audit proves graph connectivity, not that a trained latent will avoid
   collapse.
3. The high-rate model has about 20.1 million more parameters than official
   ACT. The first comparison is algorithm/system matched, not parameter-count
   matched.
4. Offline L1/KL cannot determine task success. Final judgment must use paired
   rollout success, safe-success, force peaks, and failure modes.
5. Non-temporal chunk execution and postprocessing ablations remain later
   experiments. They must not be mixed into the first temporal comparison.

Subject to the local CUDA burn-in above, no known code or data-contract blocker
remains before paired formal training.
