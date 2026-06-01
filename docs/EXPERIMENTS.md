# Experiments

This project currently uses a two-stage contact-aware ACT pipeline for offline HDF5 experiments.

## Pipeline Stages

Stage 1 trains the joint policy with supervised action and future-force prediction. During training, the policy uses posterior motion and contact latents built from future labels. It also computes a conditional contact prior from online features and can add prior distillation through `--lambda-prior`.

Stage 2 freezes the Stage-1 policy and trains only `ContactPriorEncoder`. The posterior contact encoder remains the teacher, and the prior is optimized to match posterior contact outputs from the Stage-1 checkpoint.

## Evaluation Modes

`zero` is the deployable baseline. It runs inference with `z_contact=0`.

`prior` is deployable inference with the deterministic conditional prior mean, `mu_contact_prior`, as `z_contact`.

`posterior` is an oracle/debug mode. It uses future action and force labels and is not deployable.

## Current Findings

On the current 10-episode train-set evaluation, prior inference improves force prediction over the zero-contact baseline and approaches the posterior oracle more closely on force L1.

On the current 8/2 split held-out evaluation, prior inference still improves force prediction relative to the zero-contact baseline.

## Warnings

The current dataset is small. Results on the held-out `val2` split are useful debugging evidence, but they are not a final generalization claim.

More diverse data, roughly 50-100 episodes or more, is needed before making publishable conclusions about contact-prior generalization.

Generated outputs, checkpoints, CSV logs, plots, and HDF5 files should remain local artifacts and should not be committed.
