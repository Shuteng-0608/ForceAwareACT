# ForceAwareACT 100k Model Registry

This registry freezes the first complete Hugging Face archive of the four 100k-step models.

- Registry generated at: `2026-07-08T11:42:15+00:00`
- Release tag: `v1.0.0`
- Dataset repository: `shuteng0608/forceawareact-peg-hole-mujoco`
- Dataset revision: `e6f60d7351d4992f0083028bee0efaceba64f5f2`
- Canonical checkpoint: `checkpoints/checkpoint_step_00100000.pt`
- Historical training source commit: unresolved

## Registered Models

| Experiment | Policy variant | Hugging Face repository | Release revision | Canonical checkpoint SHA-256 | Status |
|---|---|---|---|---|---|
| ForceAwareACT Motion CVAE | `force_aware_motion_cvae` | `shuteng0608/forceawareact-motion-cvae-peg-hole-100k` | `5a044aa3e30078f5acba69278cfa23b142492d39` | `027caee9d284079ba21f1ca7e1fe2081f278f06176a49eb9da95612f0b45e21b` | complete |
| ACT Baseline Motion CVAE | `act_baseline` | `shuteng0608/act-baseline-motion-cvae-peg-hole-100k` | `755a60a6a214d689ab35540531daf5430f8841d2` | `c2db08590121f798d6fab7eecdea84344780dc8f0899515bbc2d4397695cdd8f` | complete |
| ForceAwareACT DualZero | `force_aware_act` | `shuteng0608/forceawareact-dualzero-peg-hole-100k` | `b03a2f1a37d2a074664b26873efc763cccc2c47f` | `3fed8a3c0ba828c22187b23bad2fa4b516f3a49cdda68d23911e61342000271d` | complete |
| ForceAwareACT Contact CVAE | `force_aware_contact_cvae` | `shuteng0608/forceawareact-contact-cvae-betac5e4-lp01-peg-hole-100k` | `771b5d8289dacf302e23314211042802ca0a40f5` | `1858d3fe8579f91c9f1acb2151df9f789e56b49fa4d5e4bd2614f6fe3d88a9e9` | complete |

## Reproducible Download

Always use the full release revision rather than the mutable `main` branch.

### ForceAwareACT Motion CVAE

```bash
hf download shuteng0608/forceawareact-motion-cvae-peg-hole-100k \
  checkpoints/checkpoint_step_00100000.pt \
  --revision 5a044aa3e30078f5acba69278cfa23b142492d39
```

### ACT Baseline Motion CVAE

```bash
hf download shuteng0608/act-baseline-motion-cvae-peg-hole-100k \
  checkpoints/checkpoint_step_00100000.pt \
  --revision 755a60a6a214d689ab35540531daf5430f8841d2
```

### ForceAwareACT DualZero

```bash
hf download shuteng0608/forceawareact-dualzero-peg-hole-100k \
  checkpoints/checkpoint_step_00100000.pt \
  --revision b03a2f1a37d2a074664b26873efc763cccc2c47f
```

### ForceAwareACT Contact CVAE

```bash
hf download shuteng0608/forceawareact-contact-cvae-betac5e4-lp01-peg-hole-100k \
  checkpoints/checkpoint_step_00100000.pt \
  --revision 771b5d8289dacf302e23314211042802ca0a40f5
```

## Verification

After downloading, verify the checkpoint with:

```bash
sha256sum checkpoint_step_00100000.pt
```

The result must match the SHA-256 value in this registry.

## Maintenance Rule

- Do not move the `v1.0.0` tag.
- Future repository changes may advance `main` but must not change this registry entry.
- Create a new tag and a new registry revision for any material model, configuration, or artifact change.
- Do not replace an existing checkpoint under the same tag.
