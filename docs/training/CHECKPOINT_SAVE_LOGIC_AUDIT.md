# Checkpoint Save Logic Audit

> Historical snapshot note (2026-07-16): this audit predates epoch validation,
> `checkpoint_best.pt`, early-stopping metadata, and the current reproducibility
> fields. Its analysis of the named 100k runs is preserved, but it is not the
> current trainer contract. Use
> [`MODEL_TRAINING_AND_EARLY_STOPPING_MANUAL.md`](MODEL_TRAINING_AND_EARLY_STOPPING_MANUAL.md)
> and [`ARCHITECTURE.md`](../architecture/ARCHITECTURE.md) for current behavior.

## Scope and Limitations

This is a source-only audit of the current ForceAwareACT checkpoint save, load, and resume semantics. The real Linux checkpoint files from these completed experiments are not present in this macOS workspace and were not searched for or loaded:

- `forceaware_motion_cvae_betam5e4_trajectory100k`
- `act_baseline_motion_cvae_betam5e4_trajectory100k`

The current executable implementation is the primary source of truth. Local docs, tests, and Git history are used only to clarify intent. Because the real files are unavailable, this report does not claim actual tensor equality or inequality for the Linux checkpoints.

## Active Training Entry Point

The two relevant policy families use these current trainers:

- ForceAware Motion CVAE: `scripts/train_minimal.py`, selected with `--policy-variant force_aware_motion_cvae` (`scripts/train_minimal.py:51-56`, `scripts/train_minimal.py:309-314`).
- ACT baseline Motion CVAE: `scripts/train_act_baseline.py`, which constructs `ACTPolicyBaseline` directly (`scripts/train_act_baseline.py:192-193`).

The ACT baseline trainer imports the shared checkpoint helpers from `scripts.train_minimal`: `build_checkpoint_payload`, `checkpoint_step_path`, `resolve_checkpoint_steps`, and `save_checkpoint_atomic` (`scripts/train_act_baseline.py:27-32`). Therefore the two relevant 100k experiments share the same checkpoint path formatting, payload builder, periodic schedule resolver, and atomic save helper.

## Relevant Source Files

- `scripts/train_minimal.py`: active force-aware trainer, checkpoint schedule, payload builder, atomic save, periodic save, final save.
- `scripts/train_act_baseline.py`: active ACT baseline trainer using the same checkpoint helpers.
- `scripts/train_contact_prior_stage2.py`: separate stage-2 trainer; final-only `checkpoint.pt` save, not relevant to the two named 100k experiments.
- `scripts/evaluate_motion_cvae_modes.py`: dedicated Motion-CVAE evaluator checkpoint loading.
- `scripts/evaluate_act_baseline_modes.py`: dedicated ACT baseline evaluator checkpoint loading.
- `scripts/evaluate_inference_modes.py`, `scripts/run_mujoco_policy_rollout.py`, `scripts/run_policy_inference_smoke.py`, `scripts/audit_model_components.py`: generic inference/audit checkpoint loading.
- `tests/test_action_mode_pipeline.py`, `tests/test_act_baseline_checkpointing.py`: tests covering checkpoint scheduling, payloads, and final-vs-periodic equality for current code.

## Checkpoint Save Call Sites

| File/function | Save path | Classification | Notes |
|---|---|---|---|
| `scripts/train_minimal.py:446-455` | `checkpoint_step_XXXXXXXX.pt` | periodic numbered checkpoint | Saved only when one-based `step` is in the resolved schedule. |
| `scripts/train_minimal.py:505-513` | `checkpoint.pt` | final post-training checkpoint | Saved after normal loop completion. |
| `scripts/train_act_baseline.py:256-265` | `checkpoint_step_XXXXXXXX.pt` | periodic numbered checkpoint | Same shared helper and same one-based step semantics. |
| `scripts/train_act_baseline.py:299-307` | `checkpoint.pt` | final post-training checkpoint | Saved after normal loop completion. |
| `scripts/train_contact_prior_stage2.py:283-292` | `checkpoint.pt` | final post-training checkpoint for stage-2 prior distillation | Uses direct `torch.save`; no numbered checkpoints. Not part of the two named experiments. |
| `scripts/compute_normalization_stats.py:68` | user-selected stats `.pt` | normalization stats artifact | Not a model checkpoint. |
| tests under `tests/` | temporary `.pt` files | test fixtures | Not production checkpoint writers. |

No best-metric checkpoint, rolling latest checkpoint, interruption checkpoint, resume checkpoint, or compatibility alias writer is implemented in the active 100k training paths.

## Training-Loop Operation Order

### ForceAware Motion CVAE via `scripts/train_minimal.py`

The loop is `for step in range(1, args.max_steps + 1)` (`scripts/train_minimal.py:358`), so `step` is one-based. One iteration performs this order:

1. Set `last_step = step` (`scripts/train_minimal.py:359`).
2. Load next batch from an infinite dataloader cycle (`scripts/train_minimal.py:360`).
3. Normalize batch if stats are provided (`scripts/train_minimal.py:361-362`).
4. Compute beta warmups and branch flags (`scripts/train_minimal.py:363-378`).
5. `optimizer.zero_grad(set_to_none=True)` (`scripts/train_minimal.py:380`).
6. Forward pass through the selected policy (`scripts/train_minimal.py:381-429`).
7. Compute the policy-specific loss (`scripts/train_minimal.py:390-440`).
8. Backward pass: `losses["loss_total"].backward()` (`scripts/train_minimal.py:443`).
9. Optimizer update: `optimizer.step()` (`scripts/train_minimal.py:444`).
10. If scheduled, build and save `checkpoint_step_XXXXXXXX.pt` (`scripts/train_minimal.py:446-455`).
11. Write CSV log row (`scripts/train_minimal.py:457-479`).
12. Print progress line (`scripts/train_minimal.py:481-503`).
13. After the loop exits normally, build and save `checkpoint.pt` with `step=last_step` (`scripts/train_minimal.py:505-513`).

There is no scheduler, no AMP autocast/scaler, and no explicit global step object.

### ACT Baseline Motion CVAE via `scripts/train_act_baseline.py`

The loop is also `for step in range(1, args.max_steps + 1)` (`scripts/train_act_baseline.py:234`). The order is analogous:

1. `last_step = step` (`scripts/train_act_baseline.py:235`).
2. Batch load and optional normalization (`scripts/train_act_baseline.py:236-238`).
3. Beta warmup (`scripts/train_act_baseline.py:240`).
4. `optimizer.zero_grad(set_to_none=True)` (`scripts/train_act_baseline.py:241`).
5. Forward pass (`scripts/train_act_baseline.py:242-247`).
6. Loss computation (`scripts/train_act_baseline.py:248-252`).
7. Backward pass (`scripts/train_act_baseline.py:253`).
8. Optimizer update (`scripts/train_act_baseline.py:254`).
9. Periodic numbered checkpoint if scheduled (`scripts/train_act_baseline.py:256-265`).
10. CSV log row (`scripts/train_act_baseline.py:267-283`).
11. Print progress (`scripts/train_act_baseline.py:284-297`).
12. Final `checkpoint.pt` after loop (`scripts/train_act_baseline.py:299-307`).

There is no scheduler or AMP scaler here either.

## Step-Count Convention

The current convention is one-based completed optimizer updates:

- The first iteration has `step == 1` and performs the first optimizer update before any checkpoint for step 1 can be written.
- `last_step` is assigned at the top of each iteration and equals `args.max_steps` after normal completion.
- `build_checkpoint_payload(..., step=step)` stores the same one-based step value for periodic saves (`scripts/train_minimal.py:448-453`; `scripts/train_act_baseline.py:258-263`).
- `checkpoint.pt` stores `step=last_step` after normal completion (`scripts/train_minimal.py:506-511`; `scripts/train_act_baseline.py:300-305`).

`resolve_checkpoint_steps` adds periodic save points with `range(save_every, max_steps + 1, save_every)` (`scripts/train_minimal.py:158-159`). Thus `--max-steps 100000 --save-every 10000` includes `100000`.

## Numbered Checkpoint Semantics

A file named `checkpoint_step_00100000.pt` is saved during the iteration where `step == 100000`, immediately after `optimizer.step()` for that iteration and before CSV logging/printing. It therefore represents the model and optimizer state after exactly 100000 completed optimizer updates under the current code.

It is not:

- the state before the 100000th optimizer update;
- a zero-based loop index of 99999 displayed as 100000;
- a loop index 100000 corresponding to 100001 updates.

No scheduler or AMP scaler exists, so there is no scheduler/scaler ordering to apply.

## `checkpoint.pt` Semantics

In the current active trainers, `checkpoint.pt` is a final post-training checkpoint for normal uninterrupted runs.

Evidence:

- It is saved only after the `with args.log_csv.open(...)` training loop block completes (`scripts/train_minimal.py:505-513`; `scripts/train_act_baseline.py:299-307`).
- It is not written after each periodic checkpoint.
- It is not a rolling latest checkpoint.
- It is not selected by a metric and is not a best checkpoint.
- There is no interruption handler that writes it on exception or signal.
- It uses the same `build_checkpoint_payload` helper as numbered checkpoints in both relevant trainers.

For a normal uninterrupted `--max-steps 100000` run, `checkpoint.pt` stores `step == 100000` and represents exactly 100000 completed optimizer updates.

If an old `checkpoint.pt` already exists in an output directory and the process crashes before final save, current code does not remove or update that old file. That is a hygiene risk, not normal-run semantics.

## Checkpoint Payload

The shared checkpoint payload builder is `build_checkpoint_payload` (`scripts/train_minimal.py:168-180`). It writes exactly:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
    "step": step,
}
```

Fields not written by the current active trainers:

- `scheduler_state_dict`
- `scaler_state_dict`
- random number generator states
- epoch
- explicit `global_step`
- metrics
- save reason
- wall-clock timestamp
- normalization tensor contents

Normalization metadata is represented only indirectly through `config["normalization_stats_path"]`; the normalization stats tensors are not embedded in the model checkpoint.

Periodic numbered checkpoints and `checkpoint.pt` use the same payload builder in `train_minimal.py` and `train_act_baseline.py`. The payload structure is the same; the stored `step` differs only when the final step is not itself a periodic checkpoint. In a 100k run with `--save-every 10000`, both the final numbered checkpoint and `checkpoint.pt` store `step == 100000`.

## Resume Semantics

There is no current training resume implementation.

Searches of the active trainers found no `--resume`, `--resume-from`, `resume_from`, or equivalent CLI path. Documentation also states that no training resume CLI is implemented (`README.md:165`, `docs/reference/COMMAND_RECIPES.md:74`, `docs/architecture/ARCHITECTURE.md:233`).

Consequences:

- There is no default resume path.
- `checkpoint.pt` is not used by trainer code as a resume checkpoint.
- Numbered checkpoints are not consumed by trainer code for resume.
- Optimizer state restoration for continuing training is not implemented.
- Scheduler/scaler restoration is not applicable because no scheduler/scaler is used.
- RNG restoration is not implemented.
- The stored `step` has no executable resume interpretation in current code.

Concrete example: if a checkpoint stores `step = 100000`, the current trainers do not define a next optimizer update after resume. Rerunning a trainer starts a fresh loop at `step == 1`; it does not load the checkpoint. A future resume implementation would need to define whether the next update is `100001`, but that behavior is absent today.

## Inference-Loading Semantics

Evaluation and rollout scripts load model weights from a checkpoint selected by an explicit `--checkpoint` argument. They do not default to `checkpoint.pt`; users choose the path.

Relevant paths:

- Motion-CVAE evaluator: `torch.load(checkpoint_path, map_location="cpu")`, validates `policy_variant`, extracts `model_state_dict` if present, and loads strict model weights (`scripts/evaluate_motion_cvae_modes.py:209-230`). It can also accept a raw state dict through `_state_dict_from_checkpoint` (`scripts/evaluate_motion_cvae_modes.py:163-172`).
- ACT baseline evaluator: requires checkpoint config metadata and loads `model_state_dict` strict (`scripts/evaluate_act_baseline_modes.py:154-199`).
- Generic inference evaluator: requires a dict checkpoint and loads `checkpoint["model_state_dict"]` strict (`scripts/evaluate_inference_modes.py:422-428`).
- MuJoCo rollout: requires a dict checkpoint, dispatches model construction by `config.policy_variant`, and loads `checkpoint["model_state_dict"]` strict (`scripts/run_mujoco_policy_rollout.py:922-928`).
- Model audit script reads checkpoint config to infer policy variant/model config (`scripts/audit_model_components.py:74-95`).

Assuming `checkpoint.pt` and `checkpoint_step_00100000.pt` contain identical `model_state_dict` tensors and compatible config, they should be interchangeable for inference under these loaders. This audit did not test actual equality because the Linux files are unavailable.

## ForceAware Versus Baseline Consistency

For the two named experiments:

- ForceAware Motion CVAE uses `scripts/train_minimal.py` with `policy_variant == "force_aware_motion_cvae"`.
- ACT baseline Motion CVAE uses `scripts/train_act_baseline.py`.

Their checkpoint semantics are identical for:

- one-based loop step convention;
- optimizer update before checkpoint save;
- periodic save condition;
- `checkpoint_step_XXXXXXXX.pt` filename formatting;
- final `checkpoint.pt` save after the loop;
- payload builder and top-level payload fields;
- absence of scheduler, AMP scaler, and resume.

Policy-specific differences affect model construction, dataset force inclusion, loss computation, and config contents, but not save timing or payload shape.

## Expected Relationship Between `checkpoint.pt` and the 100k Checkpoint

For a normal uninterrupted current-code run with:

```text
--max-steps 100000
--save-every 10000
```

source code proves:

1. `checkpoint_step_00100000.pt` is saved after optimizer update 100000.
2. `checkpoint.pt` is saved after the loop with `step == 100000`.
3. No optimizer step, scheduler step, scaler update, backward pass, or model mutation occurs between the final periodic save and the final `checkpoint.pt` save.
4. Both saves use the same payload builder in the two relevant trainers.

Source code therefore suggests the logical `model_state_dict` and `optimizer_state_dict` should be equal between the final numbered checkpoint and `checkpoint.pt` for a normal uninterrupted 100k run.

Source code does not prove:

- that the historical Linux files were generated by exactly this local commit;
- that the bytes of two `torch.save` outputs must match;
- that the actual Linux files have equal tensors or optimizer states.

## Possible Reasons for Different SHA-256 Hashes

Different SHA-256 hashes for `checkpoint.pt` and `checkpoint_step_00100000.pt` do not by themselves prove tensor or optimizer differences.

Evidence-supported or plausible causes include:

- They are two separate `torch.save` calls to different paths, so byte-level serialization may differ even for logically identical payloads.
- PyTorch serialization details, zip container metadata, object memoization, or storage ordering may produce different bytes.
- The historical Linux run may have used a different code revision than the current macOS checkout.
- The final `checkpoint.pt` may have been overwritten by a later command or manual process on Linux.
- A non-normal run, crash/restart, or manual copy could have changed one file.

Causes that current source does not support for a normal uninterrupted 100k run:

- An extra optimizer update between `checkpoint_step_00100000.pt` and `checkpoint.pt`.
- A scheduler/scaler update between the two saves.
- Different payload builders for the two saves in the relevant trainers.
- A save-reason/timestamp/metric field that differs; these fields are not present in the current payload.

## Potential Bugs or Ambiguities

- Filename semantics are not documented in-code; tests and docs clarify them, but `step` is not typed as `completed_updates`.
- No resume implementation exists, so `step` cannot currently prevent repeat/skip bugs because it is never consumed. A future resume implementation must define `next_step = checkpoint["step"] + 1` if preserving current semantics.
- No RNG states are saved, so exact continuation would not be reproducible even if resume were added using the current payload.
- No scheduler/scaler states are saved because there are no scheduler/scaler objects. If those are added later, the payload must change.
- `checkpoint.pt` is not written until normal loop completion; interruption leaves only the last numbered checkpoint, and any pre-existing `checkpoint.pt` may remain stale.
- `save_checkpoint_atomic` unlinks/replaces a `.tmp` path (`scripts/train_minimal.py:183-193`); this is appropriate for atomicity but is not a resume manifest.
- Historical documentation before commit `08504a1` said no intermediate checkpoint saving existed. Local Git history shows intermediate checkpoint saving was added in commit `08504a1` (`git log --oneline`), so confirm the Linux experiments used that commit or a descendant.

## Hugging Face Naming Recommendation

Conservative recommendations based only on source semantics:

- Preserve the original final file as `checkpoints/checkpoint.pt` until Linux-side content verification is complete.
- Upload the final numbered checkpoint as `checkpoints/checkpoint_step_00100000.pt`.
- If Linux verification confirms `checkpoint.pt` has `step == 100000` and matches the intended final post-training payload, an additional alias such as `checkpoints/checkpoint_final.pt` is reasonable.
- Do not call either file `best` because the training code has no best-metric checkpoint logic.
- Do not call `checkpoint.pt` `latest` unless you explicitly define “latest” as a release alias outside the source code; source semantics call it final post-training, not rolling latest.

## Verification Still Required on the Linux Machine

Run these commands on the Linux machine that has the real checkpoint files. They load on CPU, use `weights_only=False`, print no full tensors, and make no modifications.

```bash
# Set these on the Linux machine where the real checkpoints exist.
EXP1=/path/to/forceaware_motion_cvae_betam5e4_trajectory100k
EXP2=/path/to/act_baseline_motion_cvae_betam5e4_trajectory100k
export EXP1 EXP2

# File sizes and hashes.
for EXP in "$EXP1" "$EXP2"; do
  echo "== $EXP =="
  ls -lh "$EXP/checkpoint.pt" "$EXP/checkpoint_step_00100000.pt"
  sha256sum "$EXP/checkpoint.pt" "$EXP/checkpoint_step_00100000.pt"
done

# Top-level keys, stored step values, policy metadata, and logical state equality.
python - <<'PY_LINUX_CHECKPOINTS'
from pathlib import Path
import os
import torch

experiments = [Path(os.environ["EXP1"]), Path(os.environ["EXP2"])]

def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def tensor_mapping_equal(a, b):
    if set(a) != set(b):
        return False, {"only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a))}
    mismatches = []
    for key in sorted(a):
        va, vb = a[key], b[key]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape or va.dtype != vb.dtype or not torch.equal(va, vb):
                mismatches.append(key)
        elif va != vb:
            mismatches.append(key)
    return not mismatches, mismatches[:20]

for exp in experiments:
    latest = load(exp / "checkpoint.pt")
    numbered = load(exp / "checkpoint_step_00100000.pt")
    print(f"\n== {exp} ==")
    print("checkpoint.pt keys:", sorted(latest.keys()))
    print("checkpoint_step_00100000.pt keys:", sorted(numbered.keys()))
    print("steps:", latest.get("step"), numbered.get("step"))
    print("policy variants:", latest.get("config", {}).get("policy_variant"), numbered.get("config", {}).get("policy_variant"))
    print("configs_equal:", latest.get("config") == numbered.get("config"))
    ok, detail = tensor_mapping_equal(latest["model_state_dict"], numbered["model_state_dict"])
    print("model_state_dict_equal:", ok, detail)
    print("optimizer_state_dict_keys_equal:", latest["optimizer_state_dict"].keys() == numbered["optimizer_state_dict"].keys())
    print("optimizer_state_dict_equal_repr:", latest["optimizer_state_dict"] == numbered["optimizer_state_dict"])
    print("has_scheduler_state_dict:", "scheduler_state_dict" in latest, "scheduler_state_dict" in numbered)
    print("has_scaler_state_dict:", "scaler_state_dict" in latest, "scaler_state_dict" in numbered)
PY_LINUX_CHECKPOINTS
```
