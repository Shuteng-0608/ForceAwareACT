# Native 500 Hz contact-CVAE training v3

This training path is intentionally separate from the state-rate contact-CVAE,
the motion-CVAE control, and the official-ACT baseline.

## Inputs and targets

- Online force input: the causal last 100 samples from the native 500 Hz
  wrench stream, packed into at most seven state-time intervals.
- Posterior-only future force: for action query `j`, all native samples in
  `(t_j, t_{j+1}]`, left-aligned in 20 slots. No future force is accepted by
  the deployment `forward` method.
- Action target: 100 state-rate actions with an independent action mask.
- Endpoint force target: the last native force sample in each valid future
  interval, with the future-interval mask.
- High-rate force target: every valid native sample in every valid future
  interval, with separate interval and sample masks.

Joint position and action statistics use the state-rate training split. Force
statistics use every native `observations/ft_wrench` sample from the training
split only. Validation data never contributes to normalization.

## Objective

The high-rate force L1 first averages six wrench dimensions, then averages the
valid native samples inside each state interval, then averages valid intervals.
Thus a 17-sample interval does not receive more weight than a 16-sample
interval. An example with no future force interval contributes a differentiable
zero to both force losses while its valid action target remains trainable.

The full loss is:

`action L1 + endpoint force L1 + 0.5 * native-rate force L1 + 10 * KL(q||N(0,I)) + KL(stopgrad(q)||p_conditional)`

The conditional-prior inputs and posterior targets are detached for the prior
matching term. Consequently that term updates the prior but cannot update the
posterior, visual path, online force path, or their shared local 500 Hz encoder.

## Deployment and validation

Training reconstructs with the posterior. Deployment accepts online inputs
only and defaults to an exact all-zero `z_contact`; the conditional-prior mean
is retained as a separately measured mode. Validation reports posterior,
zero-latent, and conditional-prior metrics for actions, endpoint force, and
native-rate force. Model selection remains based on zero-latent action L1.

Before a formal run, execute
`scripts/preflight_act_aligned_high_rate_contact_cvae.py`. It audits optimizer
coverage, module gradients, prior isolation, a single-2-ms future-force
intervention, deployment input separation, output shapes, and exact zero latent.
