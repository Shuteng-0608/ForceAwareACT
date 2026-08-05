# High-rate force contract

Version: `causal_raw_500hz_last_100_grouped_state_intervals_v2`

The high-rate contact model consumes the native 500 Hz wrench stream. It must
not first align that stream to the approximately 30 Hz state timestamps.

## Online history

At state timestamp `t_j`, the online input is the last 100 raw wrench samples
whose timestamps are not later than `t_j`. The window is left-padded only near
the beginning of an episode. Padding is represented by a boolean mask and is
never filled by repeating a real sample.

The selected samples are grouped by the actual state timestamp boundaries for
local encoding. The fixed capacities are seven state intervals and 20 force
samples per interval. Every selected raw sample must be represented exactly
once; exceeding either capacity is an error rather than a truncation request.

## Future action-response intervals

For an action sample at state timestamp `t_j`, its high-rate future-force
interval is

```text
(t_j, t_j+1]
```

The open left boundary keeps the future label disjoint from the online history
available at `t_j`. The closed right boundary assigns a sample exactly at the
next state timestamp to the preceding action-response segment. Adjacent
intervals therefore neither overlap nor duplicate force samples.

The last action in an episode has no following state timestamp. It remains a
valid action label, but its high-rate force interval is invalid and must be
excluded from force losses. A nominal future timestamp must not be invented.

## Demonstration-controller limitation

An interval is the high-rate wrench response over the state/control segment
following action sample `j`. It is not the response to holding `action[j]`
constant for an entire state interval. During collection, the demonstrator's
controller can update the actuator command at the 1 kHz MuJoCo physics rate
between the recorded 30 Hz state/action samples. This limitation must be stated
when interpreting the learned contact dynamics.

## Timestamp requirements

State and force timestamps must be finite and strictly increasing within one
time domain. Interval membership is always computed from timestamps, never
from an assumed integer sampling-rate ratio. Wrench values must be finite.

## Current dataset audit

For `mujoco_data/peg_hole_100`, all 30,877 adjacent state intervals contain 16
or 17 native force samples. The maximum is 17, so a capacity of 20 does not
truncate the current collection. A 100-sample online window spans 0.197 to
0.199 seconds and can intersect seven state intervals, so the seven-interval
capacity is required.
