"""Strict online adapters for the new official and ACT-aligned checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as functional

from force_aware_act.act_aligned_training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
)
from force_aware_act.force_history import (
    CAUSAL_STATE_RATE_FORCE_HISTORY_V1,
    prepare_causal_state_rate_force_history,
)
from force_aware_act.high_rate_force import (
    HIGH_RATE_FORCE_CONTRACT_VERSION,
    HighRateForceContract,
    build_online_force_intervals,
)
from force_aware_act.models.act_aligned import (
    ACT_ALIGNED_ARCHITECTURE_VERSION,
    ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION,
    ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION,
    ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION,
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateContactCVAEPolicy,
    ACTAlignedHighRateDualZeroPolicy,
    ACTAlignedHighRateMotionCVAEPolicy,
)
from force_aware_act.models.official_act import (
    OFFICIAL_ACT_ARCHITECTURE_VERSION,
    OFFICIAL_ACT_NO_LATENT_ARCHITECTURE_VERSION,
    OfficialACTConfig,
    OfficialACTNoLatentConfig,
    OfficialACTNoLatentPolicy,
    OfficialACTPolicy,
)
from force_aware_act.official_act_training.checkpoint import (
    OFFICIAL_ACT_CHECKPOINT_VERSION,
)
from force_aware_act.official_act_training.no_latent_checkpoint import (
    OFFICIAL_ACT_NO_LATENT_CHECKPOINT_VERSION,
)


OFFICIAL_ACT_ROLLOUT_KIND = "official_act"
OFFICIAL_ACT_NO_LATENT_ROLLOUT_KIND = "official_act_no_latent"
ACT_ALIGNED_ROLLOUT_KIND = "act_aligned_contact_cvae"
ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND = "act_aligned_high_rate_contact_cvae"
ACT_ALIGNED_HIGH_RATE_MOTION_ROLLOUT_KIND = "act_aligned_high_rate_motion_cvae"
ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ROLLOUT_KIND = "act_aligned_high_rate_dual_zero"
NO_FORCE_HISTORY_CONTRACT = "not_used"


def checkpoint_uses_rollout_adapter(checkpoint: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint belongs to either new model family."""

    return (
        checkpoint.get("format_version")
        in {
            OFFICIAL_ACT_CHECKPOINT_VERSION,
            OFFICIAL_ACT_NO_LATENT_CHECKPOINT_VERSION,
            CHECKPOINT_FORMAT_VERSION,
        }
        or checkpoint.get("architecture_version")
        in {
            OFFICIAL_ACT_ARCHITECTURE_VERSION,
            OFFICIAL_ACT_NO_LATENT_ARCHITECTURE_VERSION,
            ACT_ALIGNED_ARCHITECTURE_VERSION,
            ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION,
            ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION,
            ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION,
        }
    )


@dataclass
class RolloutPolicyAdapter:
    """Model, normalization, and causal online-input contract."""

    model: torch.nn.Module
    kind: str
    checkpoint_format: str
    normalization: Mapping[str, Any]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> "RolloutPolicyAdapter":
        if not checkpoint_uses_rollout_adapter(checkpoint):
            raise ValueError("checkpoint is not a new rollout-adapter format")
        architecture = checkpoint.get("architecture_version")
        checkpoint_format = str(checkpoint.get("format_version", "model_only"))
        model_config = checkpoint.get("model_config")
        normalization = checkpoint.get("normalization")
        model_state = checkpoint.get("model_state")
        if not isinstance(model_config, Mapping):
            raise KeyError("checkpoint is missing model_config")
        if not isinstance(normalization, Mapping):
            raise KeyError("checkpoint is missing embedded normalization")
        if not isinstance(model_state, Mapping):
            raise KeyError("checkpoint is missing model_state")

        if architecture == OFFICIAL_ACT_ARCHITECTURE_VERSION:
            model = OfficialACTPolicy(
                OfficialACTConfig(**dict(model_config))
            )
            kind = OFFICIAL_ACT_ROLLOUT_KIND
        elif architecture == OFFICIAL_ACT_NO_LATENT_ARCHITECTURE_VERSION:
            model = OfficialACTNoLatentPolicy(
                OfficialACTNoLatentConfig(**dict(model_config))
            )
            kind = OFFICIAL_ACT_NO_LATENT_ROLLOUT_KIND
        elif architecture == ACT_ALIGNED_ARCHITECTURE_VERSION:
            model = ACTAlignedContactCVAEPolicy(
                ACTAlignedConfig(**dict(model_config))
            )
            kind = ACT_ALIGNED_ROLLOUT_KIND
        elif architecture == ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION:
            model = ACTAlignedHighRateContactCVAEPolicy(
                ACTAlignedHighRateConfig(**dict(model_config))
            )
            kind = ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND
        elif architecture == ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION:
            model = ACTAlignedHighRateMotionCVAEPolicy(
                ACTAlignedHighRateConfig(**dict(model_config))
            )
            kind = ACT_ALIGNED_HIGH_RATE_MOTION_ROLLOUT_KIND
        elif architecture == ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION:
            model = ACTAlignedHighRateDualZeroPolicy(
                ACTAlignedHighRateConfig(**dict(model_config))
            )
            kind = ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ROLLOUT_KIND
        else:
            raise ValueError(
                f"unsupported rollout architecture: {architecture!r}"
            )
        model.load_state_dict(model_state, strict=True)
        model.to(device)
        model.eval()
        adapter = cls(
            model=model,
            kind=kind,
            checkpoint_format=checkpoint_format,
            normalization=dict(normalization),
        )
        adapter._validate_normalization()
        return adapter

    @property
    def config(
        self,
    ) -> (
        OfficialACTConfig
        | OfficialACTNoLatentConfig
        | ACTAlignedConfig
        | ACTAlignedHighRateConfig
    ):
        return self.model.config

    @property
    def chunk_len(self) -> int:
        return int(self.config.chunk_len)

    @property
    def image_height(self) -> int:
        return int(self.config.image_height)

    @property
    def image_width(self) -> int:
        return int(self.config.image_width)

    @property
    def uses_force_history(self) -> bool:
        return self.uses_state_rate_force_history or self.uses_high_rate_force_history

    @property
    def uses_state_rate_force_history(self) -> bool:
        return self.kind == ACT_ALIGNED_ROLLOUT_KIND

    @property
    def uses_high_rate_force_history(self) -> bool:
        return self.kind in {
            ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND,
            ACT_ALIGNED_HIGH_RATE_MOTION_ROLLOUT_KIND,
            ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ROLLOUT_KIND,
        }

    @property
    def force_window_len(self) -> Optional[int]:
        return (
            int(
                self.config.online_force_window_len
                if self.uses_high_rate_force_history
                else self.config.force_window_len
            )
            if self.uses_force_history
            else None
        )

    @property
    def force_history_contract(self) -> str:
        return (
            (
                HIGH_RATE_FORCE_CONTRACT_VERSION
                if self.uses_high_rate_force_history
                else CAUSAL_STATE_RATE_FORCE_HISTORY_V1
            )
            if self.uses_force_history
            else NO_FORCE_HISTORY_CONTRACT
        )

    @property
    def has_contact_prior(self) -> bool:
        return self.kind in {
            ACT_ALIGNED_ROLLOUT_KIND,
            ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND,
        }

    def prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """Prepare native `[K,3,H,W]` float images for one online batch."""

        if images.ndim != 4 or images.shape[0] != self.config.num_cameras:
            raise ValueError("online images must have shape [K, 3, H, W]")
        if images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("online images must be floating-point RGB tensors")
        target = (self.image_height, self.image_width)
        if tuple(images.shape[-2:]) != target:
            images = functional.interpolate(
                images,
                size=target,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if (
            self.kind in {
                ACT_ALIGNED_ROLLOUT_KIND,
                ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND,
                ACT_ALIGNED_HIGH_RATE_MOTION_ROLLOUT_KIND,
                ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ROLLOUT_KIND,
            }
            and self.config.imagenet_normalize
        ):
            mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            images = (images - mean) / std
        return images.unsqueeze(0)

    def prepare_qpos(self, qpos: np.ndarray | torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(
            qpos,
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )
        if values.shape != (self.config.q_dim,):
            raise ValueError(f"qpos must have shape [{self.config.q_dim}]")
        normalized = self._normalize(values, "qpos")
        return normalized.unsqueeze(0)

    def prepare_force_history(
        self,
        state_rate_history: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad and normalize the last L causal state-rate wrenches."""

        if not self.uses_state_rate_force_history:
            raise ValueError("this adapter does not accept state-rate force history")
        values = torch.as_tensor(
            state_rate_history,
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )
        if values.ndim != 2 or values.shape[1] != self.config.force_dim:
            raise ValueError(
                f"force history must have shape [N, {self.config.force_dim}]"
            )
        if values.shape[0] == 0:
            raise ValueError("force history must contain the current wrench")
        history, padding_mask = prepare_causal_state_rate_force_history(
            values,
            window_len=int(self.config.force_window_len),
            force_dim=int(self.config.force_dim),
            mean=self.normalization["force_mean"],
            std=self.normalization["force_std"],
        )
        return history.unsqueeze(0), padding_mask.unsqueeze(0)

    def prepare_high_rate_force_history(
        self,
        force_timestamps: np.ndarray,
        force_values: np.ndarray,
        state_timestamps: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the exact training packer and normalization for online v2 force."""

        if not self.uses_high_rate_force_history:
            raise ValueError("this adapter does not accept native-rate force history")
        contract = HighRateForceContract(
            sample_rate_hz=self.config.force_sample_rate_hz,
            online_window_len=self.config.online_force_window_len,
            max_samples_per_interval=self.config.max_force_samples_per_interval,
            max_online_intervals=self.config.max_online_force_intervals,
        )
        _window, packed = build_online_force_intervals(
            force_timestamps,
            force_values,
            state_timestamps,
            state_index=len(state_timestamps) - 1,
            contract=contract,
        )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        values = torch.as_tensor(packed.values, dtype=dtype, device=device)
        values = self._normalize(values, "force")
        sample_mask = torch.as_tensor(
            packed.sample_padding_mask, dtype=torch.bool, device=device
        )
        values = values.masked_fill(sample_mask.unsqueeze(-1), 0.0)
        return (
            values.unsqueeze(0),
            torch.as_tensor(packed.relative_times, dtype=dtype, device=device).unsqueeze(0),
            sample_mask.unsqueeze(0),
            torch.as_tensor(packed.interval_padding_mask, dtype=torch.bool, device=device).unsqueeze(0),
        )

    def forward(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        *,
        force_history: Optional[torch.Tensor] = None,
        force_padding_mask: Optional[torch.Tensor] = None,
        online_force_intervals: Optional[torch.Tensor] = None,
        online_force_relative_time: Optional[torch.Tensor] = None,
        online_force_sample_padding_mask: Optional[torch.Tensor] = None,
        online_force_interval_padding_mask: Optional[torch.Tensor] = None,
        contact_latent_mode: str = "zero",
    ) -> dict[str, Any]:
        with torch.inference_mode():
            if self.kind in {
                OFFICIAL_ACT_ROLLOUT_KIND,
                OFFICIAL_ACT_NO_LATENT_ROLLOUT_KIND,
            }:
                if contact_latent_mode != "zero":
                    raise ValueError(
                        "force-free ACT deployment latent mode must be zero"
                    )
                output = self.model(images, qpos)
                self.deployment_diagnostics(
                    output,
                    force_padding_mask=None,
                    requested_latent_mode=contact_latent_mode,
                )
                return output
            if self.uses_high_rate_force_history:
                high_rate_inputs = (
                    online_force_intervals,
                    online_force_relative_time,
                    online_force_sample_padding_mask,
                    online_force_interval_padding_mask,
                )
                if any(value is None for value in high_rate_inputs):
                    raise ValueError(
                        "high-rate policy requires all online interval tensors"
                    )
                if self.kind == ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND:
                    output = self.model(
                        images, qpos,
                        online_force_intervals,
                        online_force_relative_time,
                        online_force_sample_padding_mask,
                        online_force_interval_padding_mask,
                        contact_latent_mode=contact_latent_mode,
                        deterministic_prior=True,
                    )
                else:
                    if contact_latent_mode != "zero":
                        raise ValueError(
                            "motion and dual-zero controls require zero latent mode"
                        )
                    output = self.model(
                        images, qpos,
                        online_force_intervals,
                        online_force_relative_time,
                        online_force_sample_padding_mask,
                        online_force_interval_padding_mask,
                    )
                self.deployment_diagnostics(
                    output,
                    force_padding_mask=online_force_sample_padding_mask.flatten(1),
                    requested_latent_mode=contact_latent_mode,
                )
                return output
            if force_history is None or force_padding_mask is None:
                raise ValueError(
                    "ACT-aligned Contact-CVAE requires force history and mask"
                )
            output = self.model(
                images,
                qpos,
                force_history,
                force_padding_mask=force_padding_mask,
                contact_latent_mode=contact_latent_mode,
                deterministic_prior=True,
            )
            self.deployment_diagnostics(
                output,
                force_padding_mask=force_padding_mask,
                requested_latent_mode=contact_latent_mode,
            )
            return output

    def deployment_diagnostics(
        self,
        output: Mapping[str, Any],
        *,
        force_padding_mask: Optional[torch.Tensor],
        requested_latent_mode: str,
    ) -> dict[str, Any]:
        """Validate and summarize the deployment-only latent/input contract."""

        if self.kind == OFFICIAL_ACT_NO_LATENT_ROLLOUT_KIND:
            if output.get("latent_mechanism") != "none":
                raise RuntimeError(
                    "latent-free official ACT must declare no latent mechanism"
                )
            return {
                "latent_name": "none",
                "latent_source": "none",
                "latent_max_abs": 0.0,
                "force_history_valid_samples": 0,
                "force_history_padding_samples": 0,
            }
        if self.kind in {
            OFFICIAL_ACT_ROLLOUT_KIND,
            ACT_ALIGNED_HIGH_RATE_MOTION_ROLLOUT_KIND,
        }:
            latent_name = "z_motion"
            source_name = "motion_latent_source"
            expected_source = "zero"
            if self.kind == OFFICIAL_ACT_ROLLOUT_KIND:
                valid_force_samples = 0
                padding_samples = 0
            else:
                valid_force_samples, padding_samples = self._force_mask_counts(
                    force_padding_mask
                )
        elif self.kind == ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ROLLOUT_KIND:
            if output.get("latent_mechanism") != "none":
                raise RuntimeError(
                    "dual-zero output must declare no latent mechanism"
                )
            valid_force_samples, padding_samples = self._force_mask_counts(
                force_padding_mask
            )
            return {
                "latent_name": "none",
                "latent_source": "none",
                "latent_max_abs": 0.0,
                "force_history_valid_samples": valid_force_samples,
                "force_history_padding_samples": padding_samples,
            }
        else:
            latent_name = "z_contact"
            source_name = "contact_latent_source"
            expected_source = (
                "zero" if requested_latent_mode == "zero" else "prior_mean"
            )
            valid_force_samples, padding_samples = self._force_mask_counts(
                force_padding_mask
            )

        latent = output.get(latent_name)
        source = output.get(source_name)
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError(f"deployment output is missing tensor {latent_name}")
        if source != expected_source:
            raise RuntimeError(
                f"deployment latent source mismatch: expected {expected_source!r}, "
                f"got {source!r}"
            )
        if not torch.isfinite(latent).all():
            raise RuntimeError(f"deployment latent {latent_name} contains non-finite values")
        max_abs = float(latent.detach().abs().max().cpu())
        if requested_latent_mode == "zero" and max_abs != 0.0:
            raise RuntimeError(
                f"deployment zero latent is not exactly zero: max_abs={max_abs:.9g}"
            )
        return {
            "latent_name": latent_name,
            "latent_source": str(source),
            "latent_max_abs": max_abs,
            "force_history_valid_samples": valid_force_samples,
            "force_history_padding_samples": padding_samples,
        }

    @staticmethod
    def _force_mask_counts(
        force_padding_mask: Optional[torch.Tensor],
    ) -> tuple[int, int]:
        if force_padding_mask is None:
            raise RuntimeError("force padding mask is required for diagnostics")
        if force_padding_mask.ndim != 2 or force_padding_mask.shape[0] != 1:
            raise RuntimeError(
                "force padding mask must have flattened shape [1, N]"
            )
        return (
            int((~force_padding_mask[0]).sum().item()),
            int(force_padding_mask[0].sum().item()),
        )

    def denormalize_predictions(
        self,
        output: Mapping[str, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        action = self._denormalize(output["pred_action"], "action")
        if "pred_force" in output:
            force = self._denormalize(output["pred_force"], "force")
        else:
            force = output["pred_action"].new_full(
                (*output["pred_action"].shape[:2], 6),
                float("nan"),
            )
        return (
            action.squeeze(0).detach().cpu().numpy(),
            force.squeeze(0).detach().cpu().numpy(),
        )

    def _normalize(self, values: torch.Tensor, prefix: str) -> torch.Tensor:
        mean = values.new_tensor(self.normalization[f"{prefix}_mean"])
        std = values.new_tensor(self.normalization[f"{prefix}_std"])
        return (values - mean) / std

    def _denormalize(self, values: torch.Tensor, prefix: str) -> torch.Tensor:
        mean = values.new_tensor(self.normalization[f"{prefix}_mean"])
        std = values.new_tensor(self.normalization[f"{prefix}_std"])
        return values * std + mean

    def _validate_normalization(self) -> None:
        dimensions = {
            "qpos": self.config.q_dim,
            "action": self.config.action_dim,
        }
        if self.uses_force_history:
            dimensions["force"] = self.config.force_dim
        for prefix, dimension in dimensions.items():
            for suffix in ("mean", "std"):
                name = f"{prefix}_{suffix}"
                if name not in self.normalization:
                    raise KeyError(f"embedded normalization is missing {name}")
                values = np.asarray(self.normalization[name], dtype=np.float64)
                if values.shape != (dimension,) or not np.isfinite(values).all():
                    raise ValueError(f"embedded normalization {name} is invalid")
                if suffix == "std" and np.any(values <= 0):
                    raise ValueError(f"embedded normalization {name} must be positive")
