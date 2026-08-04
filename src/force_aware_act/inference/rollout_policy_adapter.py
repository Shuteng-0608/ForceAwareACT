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
from force_aware_act.models.act_aligned import (
    ACT_ALIGNED_ARCHITECTURE_VERSION,
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
)
from force_aware_act.models.official_act import (
    OFFICIAL_ACT_ARCHITECTURE_VERSION,
    OfficialACTConfig,
    OfficialACTPolicy,
)
from force_aware_act.official_act_training.checkpoint import (
    OFFICIAL_ACT_CHECKPOINT_VERSION,
)


OFFICIAL_ACT_ROLLOUT_KIND = "official_act"
ACT_ALIGNED_ROLLOUT_KIND = "act_aligned_contact_cvae"


def checkpoint_uses_rollout_adapter(checkpoint: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint belongs to either new model family."""

    return (
        checkpoint.get("format_version")
        in {OFFICIAL_ACT_CHECKPOINT_VERSION, CHECKPOINT_FORMAT_VERSION}
        or checkpoint.get("architecture_version")
        in {
            OFFICIAL_ACT_ARCHITECTURE_VERSION,
            ACT_ALIGNED_ARCHITECTURE_VERSION,
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
        elif architecture == ACT_ALIGNED_ARCHITECTURE_VERSION:
            model = ACTAlignedContactCVAEPolicy(
                ACTAlignedConfig(**dict(model_config))
            )
            kind = ACT_ALIGNED_ROLLOUT_KIND
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
    def config(self) -> OfficialACTConfig | ACTAlignedConfig:
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
        return self.kind == ACT_ALIGNED_ROLLOUT_KIND

    @property
    def force_window_len(self) -> Optional[int]:
        return (
            int(self.config.force_window_len)
            if self.uses_force_history
            else None
        )

    @property
    def has_contact_prior(self) -> bool:
        return self.kind == ACT_ALIGNED_ROLLOUT_KIND

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
            self.kind == ACT_ALIGNED_ROLLOUT_KIND
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

        if not self.uses_force_history:
            raise ValueError("official ACT does not accept force history")
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
        window_len = int(self.config.force_window_len)
        values = values[-window_len:]
        valid_length = int(values.shape[0])
        history = values.new_zeros(window_len, self.config.force_dim)
        padding_mask = torch.ones(
            window_len,
            dtype=torch.bool,
            device=values.device,
        )
        history[-valid_length:] = self._normalize(values, "force")
        padding_mask[-valid_length:] = False
        return history.unsqueeze(0), padding_mask.unsqueeze(0)

    def forward(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        *,
        force_history: Optional[torch.Tensor] = None,
        force_padding_mask: Optional[torch.Tensor] = None,
        contact_latent_mode: str = "zero",
    ) -> dict[str, Any]:
        with torch.inference_mode():
            if self.kind == OFFICIAL_ACT_ROLLOUT_KIND:
                if contact_latent_mode != "zero":
                    raise ValueError("official ACT deployment latent must be zero")
                return self.model(images, qpos)
            if force_history is None or force_padding_mask is None:
                raise ValueError(
                    "ACT-aligned Contact-CVAE requires force history and mask"
                )
            return self.model(
                images,
                qpos,
                force_history,
                force_padding_mask=force_padding_mask,
                contact_latent_mode=contact_latent_mode,
                deterministic_prior=True,
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
