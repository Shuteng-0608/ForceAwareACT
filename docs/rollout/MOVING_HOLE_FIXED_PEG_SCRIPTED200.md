# Moving-hole / fixed-peg scripted200 rollout contract

This document freezes the MuJoCo contract that must be used to evaluate the
four models trained from `moving_hole_fixed_peg/{clean_100,collision_100}`.
It is intentionally separate from the legacy moving-peg/fixed-hole rollout
reports.

## Collection provenance

The episodes were produced by `arm_teleop/vptele/main_scripted.py` with
`vptele/config/config_arm_right_moving_hole.yaml` and
`model/pangu_moving_hole_fixed_peg.xml`.  Collection is a two-stage process:
an expert controller first creates a 30 Hz trace, then the recorder replays it
while writing the aligned state, images, and native-rate force stream.

The task roles are reversed relative to the older rollout environment:

| role | compiled MuJoCo object |
| --- | --- |
| robot-carried socket body | `hole_tool` |
| moving control/success site | `hole_goal_site` |
| fixed fixture body | `fixed_peg_fixture` |
| fixed seating target | `fixed_peg_tip_site` |
| fixed approach target | `fixed_peg_approach_goal_site` |
| fixed peg geom | `fixed_cylindrical_peg` |
| socket collision geoms | `tool_hole_ring_00` ... `tool_hole_ring_23` |
| FT sensors | `hole_ft_force`, `hole_ft_torque` |

The policy camera order remains `ee_cam`, then `base_top_cam`, at 640 x 480.
The physics timestep is 1 ms, policy/state/image rate is 30 Hz, and native FT
sampling is 500 Hz.

A train-split HDF5 provenance check confirms
`ft_compensation_mode=gravity`, the `hole_ft_*` sensors, and the two success
sites above.  Therefore Contact-CVAE must receive the gravity-compensated
sensor-frame wrench computed for tool body `hole_tool` with MuJoCo sensor sign
-1.  Raw `sensordata` is retained only as a separate safety/diagnostic stream.

## Rollout profile

Every scripted200 rollout must pass:

```text
--task-profile moving_hole_fixed_peg
```

This profile binds all scene-dependent values together:

- XML: `arm_teleop/model/pangu_moving_hole_fixed_peg.xml`;
- audited current XML SHA-256:
  `9c86edb298fa3303b1df18a644b6877ab0bea25ed891cc73b5ba9a4eb436a2f4`;
- success sites: `fixed_peg_tip_site` and `hole_goal_site`;
- FT sensors: `hole_ft_force` and `hole_ft_torque`;
- policy wrench: gravity-compensated sensor-frame wrench;
- randomized target: move `fixed_peg_fixture`, never `hole_tool`;
- offset validation geom: `fixed_cylindrical_peg`;
- insertion axis: world `[0, -1, 0]`, with lateral error in XZ;
- success: site distance at most 1 mm, compensated world force norm at most
  5 N, lateral force at most 3 N, arm joint-speed norm at most 0.10 rad/s,
  continuously for 0.10 s;
- rollout hard stop: 40 N compensated force, an intentionally conservative
  deployment guard shared by all four policies.

After the initial 0.10 s success dwell, the policy runner mirrors the
collector's adaptive terminal hold.  It blends over 0.20 s from the last
applied command to an inverse-dynamics, gravity-compensated static position
target.  Final acceptance requires at least 0.50 s total hold and 0.25 s of
continuous stability under the same 5 N / 3 N / 0.10 rad/s gates.  Continuous
hold force above 8 N or lateral force above 5 N for 0.05 s rejects the rollout;
failure to stabilize within 2.0 s also rejects it.  The summary records the
initial `task_success_site_reached` event separately from final
`terminal_hold_accepted` / `task_success`.

Legacy output names such as `hole_offset_*` are retained for compatibility,
but under this profile they mean the requested **fixed-target/peg** offset.
New summaries also include `offset_target_*` aliases and the explicit object
names so this cannot be mistaken for motion of the robot-carried socket.  They
also record the XML SHA-256 so later model drift is detectable; the source
episode records the model path but does not embed an XML checksum.

## Pre-formal dry-run

The following command loads a real checkpoint, renders the collection model,
constructs the native-rate compensated force history, and executes one dry
policy step without applying the predicted action:

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/run_mujoco_policy_rollout.py \
  --checkpoint outputs/moving_hole_fixed_peg_scripted200/sampling_ablation_b16_earlystop/03_contact_cvae_dense_seed0/best.pt \
  --task-profile moving_hole_fixed_peg \
  --terminal-hold-mode collector \
  --contact-latent-mode zero \
  --max-rollout-steps 1 \
  --device cuda \
  --output-dir /tmp/forceact_moving_hole_contract_dryrun
```

The audited dry-run produced image shape `(1, 2, 3, 480, 640)`, force-history
shape `(1, 7, 20, 6)`, exact zero deployment latent, the expected object
bindings, and no applied policy action.

## Frozen nominal center-point pilot

The first closed-loop stage is one deterministic center-point rollout for each
of the four trained policies.  The launcher freezes the following common
contract:

- order: Official ACT episodic, Official ACT dense, Contact-CVAE dense,
  Contact-CVAE episodic;
- seed 0, fixed-target offset `(0, 0, 0)` m, maximum 1200 policy steps (40 s);
- official temporal aggregation with decay 0.01, a fresh query every 30 Hz
  policy step, absolute joint-position actions, 0.02 rad per-step delta clip,
  and no additional EMA (`alpha=1`);
- zero deployment latent for Contact-CVAE (Official ACT has no deployment
  latent);
- collector terminal-hold acceptance, 40 N hard stop, physics-rate raw and
  compensated force metrics, and both policy-camera videos;
- no axial-push controller and no holdout-dataset evaluation.

Plan-only inspection is the default and creates no rollout directory:

```bash
PYTHONPATH=src python scripts/run_sampling_ablation_nominal_rollout_pilot.py
```

The user starts the closed-loop pilot explicitly with:

```bash
PYTHONPATH=src python scripts/run_sampling_ablation_nominal_rollout_pilot.py --execute
```

The launcher runs the four models sequentially, reuses a pre-existing model
output only when its completed summary matches the frozen contract, and refuses
to overwrite incomplete or mismatched output.  It updates `aggregate.csv` after
every completed model.  Only after reviewing this center-point pilot should an
XZ offset grid and repeat count be frozen for the formal comparison.

The initial 900-step pilot was retained as a horizon audit.  Both Official ACT
variants were right-censored by the 30 s limit: episodic entered the 1 mm region
only at the final two policy steps, while dense completed later.  Under the
revised common 1200-step limit, final terminal-hold acceptance occurred at
30.516 s (Official episodic), 36.552 s (Official dense), 21.773 s (Contact
dense), and 21.860 s (Contact episodic).  Therefore 1200, rather than 900, is
the frozen maximum for subsequent center-point and offset comparisons.

Do not enable `--enable-axial-push`: it adds a controller action not present in
either learned policy and would invalidate the four-arm comparison.  Do not use
the legacy `pangu_all_right.xml`, `peg_ft_*`, `wall_task`, or wall-hole ring
names for this dataset.
