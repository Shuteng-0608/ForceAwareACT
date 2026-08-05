# Native 500 Hz force rollout contract

The high-rate checkpoint is handled by an explicit rollout-adapter branch. It
does not reuse the v1 state-rate force-history path.

At reset, the current wrench is sample zero. During every MuJoCo physics step,
force safety is checked at the physics rate (normally 1 kHz). Independently, a
continuous sampler records one wrench every 2 ms into a 100-sample ring buffer.
The ring is never reset at a 30 Hz policy boundary. Policy-state timestamps are
retained separately for the last seven force intervals.

At each policy query, the adapter calls the same
`build_online_force_intervals` function used by training. It selects only force
timestamps less than or equal to the current state timestamp, groups the last
100 samples into `(left, right]` state intervals, applies the checkpoint's
training-split native-force normalization, and produces:

- force intervals: `[1, 7, 20, 6]`;
- relative timestamps: `[1, 7, 20]`;
- native-sample padding mask: `[1, 7, 20]`;
- interval padding mask: `[1, 7]`.

The deployment model receives only images, normalized qpos, and these causal
online-force tensors. It cannot accept future force or action targets. The
default `z_contact` mode is exact zero; deterministic conditional-prior rollout
remains an explicitly selected diagnostic mode.

Temporal aggregation uses the official ACT executor and defaults to `k=0.01`.
The native force buffer does not change action-chunk execution, command
postprocessing, or safety semantics. Measured force is still checked after
every physics step, hard-stop defaults to 100 N, and safe-success defaults to
40 N. Comparisons must use identical temporal aggregation and postprocessing.

For a high-rate checkpoint, any user-supplied force-window CLI values must
match 100 samples and `(100-1)/500 = 0.198 s`; incompatible legacy values are
rejected instead of silently resampling or overriding the checkpoint contract.
