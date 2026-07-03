import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path

import pandas as pd

"""
pgrep -af "python.*scripts/train_minimal.py"

watch -n 10 -d   "python scripts/forceact_eta.py \
  --log outputs/peg_hole_100/forceaware_motion_cvae_betam5e4_pilot5k/train_log.csv \
  --max-steps 5000 \
  --pid 411698"



"""


def hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return str(dt.timedelta(seconds=seconds))


def read_csv_safely(path: Path, retries: int = 3) -> pd.DataFrame:
    """Handle the CSV being updated while it is being read."""
    last_error = None
    for _ in range(retries):
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                return df
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)

    if last_error is not None:
        raise SystemExit(f"failed to read log: {last_error}")
    raise SystemExit("log exists, but has no rows yet")


def process_elapsed_seconds(pid: int) -> float | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "etimes="],
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        return None

    return float(output)


parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True)
parser.add_argument("--max-steps", type=int, required=True)
parser.add_argument("--pid", type=int, required=True)
parser.add_argument(
    "--recent-window",
    type=int,
    default=100,
    help="Number of recent rows used for rolling means.",
)
args = parser.parse_args()

log_path = Path(args.log)
if not log_path.exists():
    raise SystemExit(f"log not found yet: {log_path}")

df = read_csv_safely(log_path)

if "step" in df.columns:
    current_step = int(df["step"].iloc[-1])
else:
    current_step = len(df)

elapsed_s = process_elapsed_seconds(args.pid)
process_running = elapsed_s is not None

if elapsed_s is None:
    # Training may have just finished. Use log modification time as a fallback
    # only for displaying the final state, not for a reliable ETA.
    elapsed_s = float("nan")

progress = 100.0 * current_step / max(args.max_steps, 1)
remaining_steps = max(args.max_steps - current_step, 0)

print("=" * 70)
print("ForceAwareACT Training Monitor")
print("=" * 70)
print(f"log: {log_path}")
print(f"PID: {args.pid}")
print(f"process: {'RUNNING' if process_running else 'NOT RUNNING / FINISHED'}")
print(f"step: {current_step}/{args.max_steps} ({progress:.2f}%)")

if process_running:
    completed_steps = max(current_step, 1)
    steps_per_s = completed_steps / max(elapsed_s, 1e-6)
    s_per_step = elapsed_s / completed_steps
    eta_s = remaining_steps * s_per_step
    estimated_total_s = elapsed_s + eta_s
    estimated_finish = dt.datetime.now() + dt.timedelta(seconds=eta_s)

    print(f"elapsed: {hms(elapsed_s)}")
    print(f"avg step time: {s_per_step:.3f} s/step")
    print(f"throughput: {steps_per_s:.3f} steps/s")
    print(f"remaining steps: {remaining_steps}")
    print(f"ETA: {hms(eta_s)}")
    print(f"estimated total: {hms(estimated_total_s)}")
    print(f"estimated finish: {estimated_finish:%Y-%m-%d %H:%M:%S}")
else:
    if current_step >= args.max_steps:
        print("status: target step count reached")
    else:
        print("status: process ended before reaching the target step count")

print("-" * 70)

recent = df.tail(args.recent_window)

metric_names = [
    "loss_total",
    "loss_action",
    "loss_force",
    "kl_motion",
    "kl_contact",
    "loss_prior",
    "beta_motion",
    "beta_contact",
    "lambda_prior",
]

for column in metric_names:
    if column not in df.columns:
        continue

    numeric = pd.to_numeric(df[column], errors="coerce")
    recent_numeric = numeric.tail(args.recent_window)

    last_value = numeric.iloc[-1]
    mean_value = recent_numeric.mean()

    print(
        f"{column:18s} "
        f"last={last_value:12.6f}  "
        f"last{len(recent_numeric):03d}_mean={mean_value:12.6f}"
    )

# Show weighted contributions using the actual logged weights.
last_row = df.iloc[-1]
recent_numeric_df = recent.apply(pd.to_numeric, errors="coerce")
means = recent_numeric_df.mean(numeric_only=True)

if all(
    name in means
    for name in [
        "loss_action",
        "loss_force",
        "kl_motion",
        "kl_contact",
        "loss_prior",
        "beta_motion",
        "beta_contact",
        "lambda_prior",
    ]
):
    # lambda_force is currently 0.1 in this experiment.
    lambda_force = 0.1

    print("-" * 70)
    print(f"Weighted contributions over last {len(recent)} rows")
    print(f"action:      {means['loss_action']:.6f}")
    print(f"force:       {lambda_force * means['loss_force']:.6f}")
    print(f"motion KL:   {means['beta_motion'] * means['kl_motion']:.6f}")
    print(f"contact KL:  {means['beta_contact'] * means['kl_contact']:.6f}")
    print(f"prior:       {means['lambda_prior'] * means['loss_prior']:.6f}")

print("=" * 70)
