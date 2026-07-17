# ForceAwareACT Documentation Index

Last documentation-structure audit: 2026-07-17.

The implementation and CLI parsers are the final source of truth. This index
assigns one role and category to every maintained document so that current
manuals, focused references, audit snapshots, and historical experiment
evidence are not used interchangeably.

## Directory Layout

The `docs/` root intentionally contains only this index.

| Directory | Contents |
| --- | --- |
| [`architecture/`](architecture/) | Current architecture, policy algorithms, ACT baseline, component audits, and historical architecture/design evidence |
| [`data/`](data/) | Dataset/action contracts, new-dataset processing, HDF5 field semantics, replay audits, and arm_teleop data integration |
| [`training/`](training/) | Model training, validation, early stopping, and checkpoint-save behavior |
| [`rollout/`](rollout/) | Single/batch/multi-seed rollout, hole robustness, contact analysis, and sensor analysis |
| [`reference/`](reference/) | Script inventory, compact command recipes, and test inventory |
| [`research/`](research/) | Experiment design, controlled comparisons, and paper positioning |
| [`experiments/`](experiments/) | Immutable or dataset-specific experiment plans and reports |
| [`huggingface_models/`](huggingface_models/) | Per-model release cards and reconstructed training commands |
| [`model_registry/`](model_registry/) | Human-readable and machine-readable released-model registry |

## Start Here

| Need | Canonical document |
| --- | --- |
| Repository overview and quick start | [`../README.md`](../README.md) |
| Local MuJoCo dataset inventory and selection | [`../mujoco_data/DATASET_README.md`](../mujoco_data/DATASET_README.md) |
| Data/model/rollout architecture | [`ARCHITECTURE.md`](architecture/ARCHITECTURE.md) |
| HDF5 action labels through MuJoCo control | [`ACTION_SEMANTICS.md`](data/ACTION_SEMANTICS.md) |
| Every script and its role | [`SCRIPTS_REFERENCE.md`](reference/SCRIPTS_REFERENCE.md) |
| A new HDF5 dataset through training | [`NEW_DATASET_TRAINING_MANUAL.md`](data/NEW_DATASET_TRAINING_MANUAL.md) |
| Five controlled training configurations and early stopping | [`MODEL_TRAINING_AND_EARLY_STOPPING_MANUAL.md`](training/MODEL_TRAINING_AND_EARLY_STOPPING_MANUAL.md) |
| Single, grid, suite, and multi-seed MuJoCo rollout | [`ROLLOUT_EXPERIMENT_MANUAL.md`](rollout/ROLLOUT_EXPERIMENT_MANUAL.md) |
| Compact copy/paste commands | [`COMMAND_RECIPES.md`](reference/COMMAND_RECIPES.md) |
| Tests and verification status | [`TESTING.md`](reference/TESTING.md) |
| Current repository audit and known risks | [`REPOSITORY_ARCHITECTURE_AUDIT.md`](architecture/REPOSITORY_ARCHITECTURE_AUDIT.md) |

Run commands from the repository root after activating the intended Python environment. The canonical docs use `PYTHONPATH=src python ...`; substitute the environment's explicit interpreter when desired.

## Document Roles and Naming

| Name pattern | Meaning | Authority |
| --- | --- | --- |
| `ARCHITECTURE`, `ACTION_SEMANTICS`, `SCRIPTS_REFERENCE`, `TESTING` | Current repository contracts and inventories | Canonical after source/CLI |
| `*_MANUAL` | End-to-end operating procedure with decisions, checks, and troubleshooting | Canonical workflow |
| `COMMAND_RECIPES` | Short commands for users who already understand the workflow | Convenience; manuals win on conflicts |
| `*_ALGORITHM`, `*_IMPLEMENTATION`, `*_ANALYSIS` | Focused conceptual or component reference | Current within its stated scope |
| `*_AUDIT` | Evidence from a source/data inspection at a point in time | Snapshot, not a permanent API contract |
| `HISTORICAL_*`, `INITIAL_*`, `*_PLAN`, `*_REPORT` | Preserved design intent or experiment evidence | Never the current command/default authority |

The audit date or historical banner inside a document is part of its contract.
An audit can remain valuable even after its operational recommendation is
superseded.

## Current Conceptual and Component References

These documents remain valid for their narrower subject, but the start-here documents above take precedence if commands or defaults conflict.

| Document | Unique purpose | Do not use it as |
| --- | --- | --- |
| [`DUAL_LATENT_ALGORITHM.md`](architecture/DUAL_LATENT_ALGORITHM.md) | Detailed `force_aware_act` dual-latent graph, losses, and inference modes | Four-policy inventory or command manual |
| [`ACT_BASELINE_IMPLEMENTATION.md`](architecture/ACT_BASELINE_IMPLEMENTATION.md) | Force-free ACT Motion-CVAE baseline implementation contract | General ForceAwareACT architecture |
| [`EXPERIMENT_DESIGN_AND_PAPER_POSITIONING.md`](research/EXPERIMENT_DESIGN_AND_PAPER_POSITIONING.md) | Research questions, controlled comparisons, and paper claims | Copy/paste command source |
| [`CONTACT_STAGE_ANALYSIS.md`](rollout/CONTACT_STAGE_ANALYSIS.md) | How to interpret `analyze_contact_stage.py` outputs | General rollout manual |
| [`ROLLOUT_SENSOR_ANALYSIS.md`](rollout/ROLLOUT_SENSOR_ANALYSIS.md) | How to interpret temporal sensor/correction plots | Success-definition authority |
| [`HOLE_POSITION_ROBUSTNESS_EVALUATION.md`](rollout/HOLE_POSITION_ROBUSTNESS_EVALUATION.md) | Geometry rationale and original grid/LHS protocol | Current multi-seed command reference |

## Audit Snapshots

Audits answer a bounded question and preserve evidence. Their conclusions may
be superseded by later code, data, or recorder changes.

| Audit | Bounded question |
| --- | --- |
| [`ACT_ALIGNMENT_AUDIT.md`](architecture/ACT_ALIGNMENT_AUDIT.md) | Which components and experiment controls are ACT-faithful? |
| [`ACT_BACKBONE_FORCE_EXTENSION_AUDIT.md`](architecture/ACT_BACKBONE_FORCE_EXTENSION_AUDIT.md) | Where does the ACT backbone end and the force extension begin? |
| [`ARM_TELEOP_MUJOCO_INTEGRATION_REPORT.md`](data/ARM_TELEOP_MUJOCO_INTEGRATION_REPORT.md) | How did the external `arm_teleop` runtime map into ForceAwareACT? |
| [`ARM_TELEOP_HDF5_RECORDING_FIELD_AUDIT.md`](data/ARM_TELEOP_HDF5_RECORDING_FIELD_AUDIT.md) | What exactly did `ee_pose`, peg-tip, hole, and wrench fields represent? |
| [`HDF5_DYNAMIC_JOINT_REPLAY_AUDIT.md`](data/HDF5_DYNAMIC_JOINT_REPLAY_AUDIT.md) | Why can measured-state dynamic replay diverge from demonstration behavior? |
| [`HDF5_REPLAY_TASK_ERROR_AUDIT.md`](data/HDF5_REPLAY_TASK_ERROR_AUDIT.md) | How trustworthy are replayed task-error and hole-site measurements? |
| [`VISION_BACKBONE_AUDIT.md`](architecture/VISION_BACKBONE_AUDIT.md) | Was ResNet18 pretrained/frozen, and what did inspected checkpoints record? |
| [`CHECKPOINT_SAVE_LOGIC_AUDIT.md`](training/CHECKPOINT_SAVE_LOGIC_AUDIT.md) | How did the earlier trainer save/finalize checkpoints? Its banner points to the current trainer contract. |
| [`REPOSITORY_ARCHITECTURE_AUDIT.md`](architecture/REPOSITORY_ARCHITECTURE_AUDIT.md) | What was implemented, risky, duplicated, or deferred on 2026-07-16? |

## Intentional Topic Overlap

These groups share vocabulary but are retained because they answer different
questions:

| Documents | Boundary |
| --- | --- |
| `ARCHITECTURE` / `REPOSITORY_ARCHITECTURE_AUDIT` / historical codebase audit | Current structure / dated risks and deferred refactors / older evidence |
| `ACT_ALIGNMENT_AUDIT` / `ACT_BACKBONE_FORCE_EXTENSION_AUDIT` / `ACT_BASELINE_IMPLEMENTATION` | ACT-faithful comparison criteria / component boundary and gradients / implemented baseline contract |
| `ACT_ALIGNMENT_AUDIT` / `EXPERIMENT_DESIGN_AND_PAPER_POSITIONING` | What is aligned in code / how to turn that into controlled claims and experiments |
| `ARM_TELEOP_MUJOCO_INTEGRATION_REPORT` / recording-field audit / historical command audit | External runtime integration / geometry and field meaning / obsolete no-command recorder behavior |
| New-dataset manual / model-training manual / command recipes | Prepare one dataset / compare five policies and early stopping / abbreviated commands only |
| Rollout manual / hole-position robustness / sensor analysis | End-to-end current operation / geometry-specific protocol rationale / post-run temporal interpretation |
| `SCRIPTS_REFERENCE` / command recipes | Complete script inventory / selected common commands in workflow order |

## Historical Snapshots and Experiment Records

The following files intentionally preserve conclusions, paths, and commands from a particular experiment or earlier code snapshot. They are evidence, not current API specifications.

- [`INITIAL_DUAL_LATENT_DESIGN.md`](architecture/INITIAL_DUAL_LATENT_DESIGN.md) records the original dual-latent implementation design.
- [`HISTORICAL_CODEBASE_AUDIT_2026-07-07.md`](architecture/HISTORICAL_CODEBASE_AUDIT_2026-07-07.md) predates the current policy inventory and canonical docs.
- [`HISTORICAL_ARM_TELEOP_RECORDING_COMMAND_AUDIT.md`](data/HISTORICAL_ARM_TELEOP_RECORDING_COMMAND_AUDIT.md) describes a recorder path without command labels; current main datasets do have command labels.
- [`HISTORICAL_ROLLOUT_ACTION_SELECTION_NOTES.md`](rollout/HISTORICAL_ROLLOUT_ACTION_SELECTION_NOTES.md) preserves early one-checkpoint action-selection observations, not general recommendations.
- [`PEG100_EXPERIMENT_REPORT.md`](experiments/PEG100_EXPERIMENT_REPORT.md) and [`PEG_FIXED_INSERT_100_EXPERIMENT_PLAN.md`](experiments/PEG_FIXED_INSERT_100_EXPERIMENT_PLAN.md) describe specific datasets and staged experiments.
- Files under [`experiments/`](experiments/) are immutable rollout reports. Recompute metrics from raw outputs when changing a safety threshold.
- Files under [`huggingface_models/`](huggingface_models/) and [`model_registry/`](model_registry/) describe released 100k-step artifacts. Their all-data training protocol is not a held-out generalization protocol.
- `architecture/ACT_BACKBONE_FORCE_EXTENSION_AUDIT.json`, `training/checkpoint_save_logic_audit.json`, and `data/ARM_TELEOP_MUJOCO_INTEGRATION_SUMMARY.yaml` are machine-readable companions, not additional manuals.

## Consolidated or Renamed Documents

Do not recreate the old split documents. Their current destinations are:

| Previous name | Current destination | Reason |
| --- | --- | --- |
| `DATASET_ACTION_MODES.md` | [`ACTION_SEMANTICS.md`](data/ACTION_SEMANTICS.md) | Merged with training and rollout action semantics |
| `COMMAND_ACTION_TRAINING_PIPELINE.md` | [`ACTION_SEMANTICS.md`](data/ACTION_SEMANTICS.md) | Same contract, training section |
| `COMMAND_ACTION_ROLLOUT.md` | [`ACTION_SEMANTICS.md`](data/ACTION_SEMANTICS.md) | Same contract, rollout section |
| `ALGORITHM_FRAMEWORK_README.md` | [`DUAL_LATENT_ALGORITHM.md`](architecture/DUAL_LATENT_ALGORITHM.md) | The file describes one dual-latent policy, not the whole repo |
| `EXPERIMENT_WORKFLOWS.md` | [`COMMAND_RECIPES.md`](reference/COMMAND_RECIPES.md) | It is a concise command collection, not the workflow authority |
| `EXPERIMENTS.md` | [`EXPERIMENT_DESIGN_AND_PAPER_POSITIONING.md`](research/EXPERIMENT_DESIGN_AND_PAPER_POSITIONING.md) and the training manual | Its short conceptual summary duplicated those documents |
| `ROLLOUT_SUCCESS_AND_SUMMARY.md` | [`HISTORICAL_ROLLOUT_ACTION_SELECTION_NOTES.md`](rollout/HISTORICAL_ROLLOUT_ACTION_SELECTION_NOTES.md) | Most content was checkpoint-specific historical observation |
| `forceawareact_codebase_architecture_audit.md` | [`HISTORICAL_CODEBASE_AUDIT_2026-07-07.md`](architecture/HISTORICAL_CODEBASE_AUDIT_2026-07-07.md) | Date and historical status are now explicit |
| `contact_dynamics_force_act_design.md` | [`INITIAL_DUAL_LATENT_DESIGN.md`](architecture/INITIAL_DUAL_LATENT_DESIGN.md) | Design intent is distinct from current implementation |
| `ARM_TELEOP_CURRENT_HDF5_RECORDING_AND_COMMAND_AUDIT.md` | [`HISTORICAL_ARM_TELEOP_RECORDING_COMMAND_AUDIT.md`](data/HISTORICAL_ARM_TELEOP_RECORDING_COMMAND_AUDIT.md) | `current` became false after command-labelled data arrived |

## Precedence and Reproducibility

When documents disagree, use this order:

1. Current source code and `--help` output.
2. The canonical start-here documents.
3. Current conceptual/component references.
4. Dated audits.
5. Historical reports and design proposals.

For a reproducible experiment, archive the Git commit and dirty status, episode lists, normalization stats, checkpoint config, exact command, point CSV, both seed dimensions, thresholds, and generated summaries. Do not infer a training or rollout protocol from an output-directory name alone.
