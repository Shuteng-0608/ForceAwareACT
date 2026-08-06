#!/usr/bin/env python3
"""Audit whether a trained Contact-CVAE deployment path responds to force."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.act_aligned_training.schema import (  # noqa: E402
    causal_alignment_indices,
    inspect_episode,
    load_state_aligned_force,
)
from force_aware_act.high_rate_force import (  # noqa: E402
    select_causal_force_window,
)
from force_aware_act.inference import (  # noqa: E402
    ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND,
    ACT_ALIGNED_ROLLOUT_KIND,
    RolloutPolicyAdapter,
)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is unavailable")
    return torch.device(requested)


def build_force_variants(
    causal_history: np.ndarray,
    *,
    window_len: int,
    delay_steps: int,
) -> dict[str, np.ndarray]:
    """Create controlled histories without changing image or robot state."""

    values = np.asarray(causal_history, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] == 0:
        raise ValueError("causal_history must have shape [N>=1, 6]")
    if window_len <= 0 or delay_steps <= 0:
        raise ValueError("window_len and delay_steps must be positive")
    actual = values[-window_len:].copy()
    delayed = (
        values[:-delay_steps][-window_len:].copy()
        if values.shape[0] > delay_steps
        else values[:1].copy()
    )
    return {
        "actual": actual,
        "zero_physical_wrench": np.zeros_like(actual),
        "reversed_time": actual[::-1].copy(),
        f"delayed_{delay_steps}_steps": delayed,
    }


def build_native_rate_force_variants(
    causal_window: np.ndarray,
    *,
    delay_force_samples: int,
) -> dict[str, np.ndarray]:
    """Intervene on one fixed-timestamp native-rate causal force window."""

    values = np.asarray(causal_window, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] == 0:
        raise ValueError("causal_window must have shape [N>=1, 6]")
    if delay_force_samples <= 0:
        raise ValueError("delay_force_samples must be positive")
    delayed = np.empty_like(values)
    delay = min(delay_force_samples, values.shape[0])
    delayed[:delay] = values[0]
    if delay < values.shape[0]:
        delayed[delay:] = values[:-delay]
    return {
        "actual": values.copy(),
        "zero_physical_wrench": np.zeros_like(values),
        "reversed_time": values[::-1].copy(),
        f"delayed_{delay_force_samples}_force_samples": delayed,
    }


def _tensor_difference(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    difference = (candidate - reference).detach().float().reshape(-1)
    return {
        "mean_abs_delta": float(difference.abs().mean().cpu()),
        "max_abs_delta": float(difference.abs().max().cpu()),
        "l2_delta": float(torch.linalg.vector_norm(difference).cpu()),
    }


def _array_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference,
        dtype=np.float64,
    )
    return {
        "mean_abs_delta": float(np.abs(difference).mean()),
        "max_abs_delta": float(np.abs(difference).max()),
        "l2_delta": float(np.linalg.norm(difference.reshape(-1))),
    }


def _attention_summary(attention: torch.Tensor) -> dict[str, float | int]:
    probabilities = attention.detach().float().reshape(-1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    return {
        "entropy": float(entropy.cpu()),
        "max_weight": float(probabilities.max().cpu()),
        "argmax_visual_token": int(probabilities.argmax().cpu()),
    }


def _denormalize_high_rate_force(
    adapter: RolloutPolicyAdapter,
    values: torch.Tensor,
) -> np.ndarray:
    mean = values.new_tensor(adapter.normalization["force_mean"])
    std = values.new_tensor(adapter.normalization["force_std"])
    return (values * std + mean).squeeze(0).detach().cpu().numpy()


def _native_window_summary(
    timestamps: np.ndarray,
    *,
    state_timestamp: float,
) -> dict[str, float | int | bool]:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("native force timestamps must be a non-empty vector")
    periods = np.diff(values)
    median_period = float(np.median(periods)) if periods.size else float("nan")
    return {
        "valid_sample_count": int(values.size),
        "first_timestamp": float(values[0]),
        "last_timestamp": float(values[-1]),
        "window_span_seconds": float(values[-1] - values[0]),
        "latest_sample_age_seconds": float(state_timestamp - values[-1]),
        "median_sample_period_seconds": median_period,
        "observed_sample_rate_hz": (
            float(1.0 / median_period)
            if np.isfinite(median_period) and median_period > 0.0
            else float("nan")
        ),
        "strictly_causal": bool(values[-1] <= state_timestamp),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    adapter = RolloutPolicyAdapter.from_checkpoint(checkpoint, device=device)
    if adapter.kind not in {
        ACT_ALIGNED_ROLLOUT_KIND,
        ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND,
    }:
        raise ValueError(
            "force-usage audit requires an ACT-aligned Contact-CVAE checkpoint"
        )

    schema = inspect_episode(args.episode)
    if len(schema.camera_names) != adapter.config.num_cameras:
        raise ValueError("episode camera count does not match checkpoint")
    timestep = args.timestep
    if timestep is None:
        timestep = min(
            schema.num_steps - 1,
            max(adapter.force_window_len - 1, schema.num_steps // 2),
        )
    if not 0 <= timestep < schema.num_steps:
        raise ValueError(f"--timestep must be in [0, {schema.num_steps - 1}]")

    with h5py.File(args.episode, "r") as handle:
        state_timestamps = handle["timestamps/state"][...]
        image_indices = causal_alignment_indices(
            state_timestamps,
            handle["timestamps/image"][...],
        )
        image_index = int(image_indices[timestep])
        images = np.stack(
            [
                np.asarray(
                    handle[f"observations/images/{camera_name}"][image_index],
                    dtype=np.uint8,
                )
                for camera_name in schema.camera_names
            ]
        )
        qpos = np.asarray(
            handle["observations/joint_pos"][timestep],
            dtype=np.float32,
        )
        if adapter.uses_high_rate_force_history:
            force_timestamps = np.asarray(
                handle["timestamps/force"][...], dtype=np.float64
            )
            force_values = np.asarray(
                handle["observations/ft_wrench"][...], dtype=np.float32
            )
        else:
            aligned_force = load_state_aligned_force(handle)

    image_tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32).div_(255.0)
    prepared_images = adapter.prepare_images(image_tensor)
    prepared_qpos = adapter.prepare_qpos(qpos)
    if adapter.uses_high_rate_force_history:
        raw_window = select_causal_force_window(
            force_timestamps,
            force_values,
            anchor_time=float(state_timestamps[timestep]),
            window_len=int(adapter.force_window_len),
        )
        valid_window = ~raw_window.padding_mask
        audit_force_timestamps = raw_window.timestamps[valid_window]
        variants = build_native_rate_force_variants(
            raw_window.values[valid_window],
            delay_force_samples=args.delay_force_samples,
        )
        native_window_summary = _native_window_summary(
            audit_force_timestamps,
            state_timestamp=float(state_timestamps[timestep]),
        )
    else:
        audit_force_timestamps = None
        native_window_summary = None
        variants = build_force_variants(
            aligned_force[: timestep + 1],
            window_len=int(adapter.force_window_len),
            delay_steps=args.delay_steps,
        )

    outputs: dict[str, dict[str, Any]] = {}
    for name, raw_history in variants.items():
        if adapter.uses_high_rate_force_history:
            if audit_force_timestamps is None:
                raise RuntimeError("native-rate force timestamps were not prepared")
            (
                force_intervals,
                relative_time,
                sample_padding_mask,
                interval_padding_mask,
            ) = adapter.prepare_high_rate_force_history(
                audit_force_timestamps,
                raw_history,
                state_timestamps[: timestep + 1],
            )
            padding_mask = sample_padding_mask.flatten(1)
            output = adapter.forward(
                prepared_images,
                prepared_qpos,
                online_force_intervals=force_intervals,
                online_force_relative_time=relative_time,
                online_force_sample_padding_mask=sample_padding_mask,
                online_force_interval_padding_mask=interval_padding_mask,
                contact_latent_mode="zero",
            )
            model_force_input = force_intervals
            interval_sample_counts = (
                (~sample_padding_mask[0]).sum(dim=1).detach().cpu().numpy()
            )
        else:
            force_history, padding_mask = adapter.prepare_force_history(raw_history)
            output = adapter.forward(
                prepared_images,
                prepared_qpos,
                force_history=force_history,
                force_padding_mask=padding_mask,
                contact_latent_mode="zero",
            )
            model_force_input = force_history
            interval_padding_mask = None
            interval_sample_counts = None
        action, force = adapter.denormalize_predictions(output)
        high_rate_force = (
            _denormalize_high_rate_force(adapter, output["pred_force_highrate"])
            if "pred_force_highrate" in output
            else None
        )
        with torch.inference_mode():
            fused, attention = adapter.model.force_vision_fusion(
                output["z_F_online"],
                output["visual_tokens"],
                visual_position=output["visual_position"],
                return_attention=True,
            )
        torch.testing.assert_close(fused, output["z_VF"], atol=1.0e-5, rtol=1.0e-5)
        diagnostics = adapter.deployment_diagnostics(
            output,
            force_padding_mask=padding_mask,
            requested_latent_mode="zero",
        )
        outputs[name] = {
            "raw_history": raw_history,
            "model_force_input": model_force_input,
            "padding_mask": padding_mask.detach().cpu().numpy(),
            "interval_padding_mask": (
                None
                if interval_padding_mask is None
                else interval_padding_mask.detach().cpu().numpy()
            ),
            "interval_sample_counts": interval_sample_counts,
            "z_F_online": output["z_F_online"],
            "z_VF": output["z_VF"],
            "attention": attention,
            "pred_action": action,
            "pred_force": force,
            "pred_force_highrate": high_rate_force,
            "diagnostics": diagnostics,
        }

    reference = outputs["actual"]
    comparisons: dict[str, Any] = {}
    for name, output in outputs.items():
        comparisons[name] = {
            "force_history_valid_samples": output["diagnostics"][
                "force_history_valid_samples"
            ],
            "force_history_padding_samples": output["diagnostics"][
                "force_history_padding_samples"
            ],
            "contact_latent_source": output["diagnostics"]["latent_source"],
            "contact_latent_max_abs": output["diagnostics"]["latent_max_abs"],
            "raw_latest_wrench": output["raw_history"][-1],
            "model_force_input_max_abs": float(
                output["model_force_input"].detach().abs().max().cpu()
            ),
            "interval_sample_counts": output["interval_sample_counts"],
            "z_F_online_vs_actual": _tensor_difference(
                reference["z_F_online"], output["z_F_online"]
            ),
            "z_VF_vs_actual": _tensor_difference(
                reference["z_VF"], output["z_VF"]
            ),
            "cross_attention_vs_actual": _tensor_difference(
                reference["attention"], output["attention"]
            ),
            "cross_attention": _attention_summary(output["attention"]),
            "pred_action_vs_actual": _array_difference(
                reference["pred_action"], output["pred_action"]
            ),
            "pred_force_vs_actual": _array_difference(
                reference["pred_force"], output["pred_force"]
            ),
        }
        if output["pred_force_highrate"] is not None:
            comparisons[name]["pred_force_highrate_vs_actual"] = (
                _array_difference(
                    reference["pred_force_highrate"],
                    output["pred_force_highrate"],
                )
            )

    return {
        "audit_version": "rollout_force_intervention_v2",
        "checkpoint": args.checkpoint,
        "checkpoint_format": adapter.checkpoint_format,
        "architecture_version": adapter.config.architecture_version,
        "episode": args.episode,
        "timestep": timestep,
        "state_timestamp": float(state_timestamps[timestep]),
        "image_index": image_index,
        "device": str(device),
        "force_history_contract": adapter.force_history_contract,
        "force_window_len": adapter.force_window_len,
        "native_force_window": native_window_summary,
        "delay_steps": args.delay_steps,
        "delay_force_samples": args.delay_force_samples,
        "delay_force_seconds": (
            args.delay_force_samples / float(adapter.config.force_sample_rate_hz)
            if adapter.uses_high_rate_force_history
            else None
        ),
        "input_shapes": {
            "images": tuple(prepared_images.shape),
            "qpos": tuple(prepared_qpos.shape),
            "force_history": (
                None
                if adapter.uses_high_rate_force_history
                else (1, adapter.force_window_len, adapter.config.force_dim)
            ),
            "online_force_intervals": (
                (
                    1,
                    adapter.config.max_online_force_intervals,
                    adapter.config.max_force_samples_per_interval,
                    adapter.config.force_dim,
                )
                if adapter.uses_high_rate_force_history
                else None
            ),
        },
        "comparisons": comparisons,
        "interpretation": (
            "Non-zero deltas prove empirical sensitivity to the selected force "
            "intervention; their magnitude must be interpreted relative to action "
            "normalization and rollout behavior, not as a pass/fail threshold."
        ),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hold image/qpos fixed and intervene on the causal force history of "
            "an ACT-aligned Contact-CVAE checkpoint."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--timestep", type=int)
    parser.add_argument("--delay-steps", type=int, default=3)
    parser.add_argument(
        "--delay-force-samples",
        type=int,
        default=10,
        help="Native force samples of sensor delay for high-rate checkpoints.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.episode = args.episode.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if not args.episode.is_file():
        raise FileNotFoundError(f"episode does not exist: {args.episode}")
    result = run_audit(args)
    serialized = json.dumps(_json_safe(result), indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(f"output={output.resolve()}")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
