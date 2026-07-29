"""Explicit AdamW parameter partition for the ACT-aligned policy."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from force_aware_act.act_aligned_training.config import ACTAlignedTrainingConfig
from force_aware_act.act_aligned_training.motion_config import (
    ACTAlignedMotionTrainingConfig,
)
from force_aware_act.models.act_aligned.motion_policy import (
    ACTAlignedMotionCVAEControlPolicy,
)
from force_aware_act.models.act_aligned.policy import (
    ACTAlignedContactCVAEPolicy,
)


def partition_trainable_parameters(
    model: ACTAlignedContactCVAEPolicy,
) -> Dict[str, List[nn.Parameter]]:
    """Partition ResNet body and main parameters by module identity."""

    if not isinstance(model, ACTAlignedContactCVAEPolicy):
        raise TypeError("model must be an ACTAlignedContactCVAEPolicy")

    return _partition_policy_parameters(model)


def partition_motion_trainable_parameters(
    model: ACTAlignedMotionCVAEControlPolicy,
) -> Dict[str, List[nn.Parameter]]:
    """Partition motion-control parameters with the same AdamW grouping."""

    if not isinstance(model, ACTAlignedMotionCVAEControlPolicy):
        raise TypeError(
            "model must be an ACTAlignedMotionCVAEControlPolicy"
        )
    return _partition_policy_parameters(model)


def _partition_policy_parameters(
    model: nn.Module,
) -> Dict[str, List[nn.Parameter]]:
    backbone_parameters = [
        parameter
        for parameter in model.vision_backbone.body.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    partition = {
        "main": main_parameters,
        "backbone": backbone_parameters,
    }
    _validate_partition(model, partition)
    return partition


def build_act_aligned_optimizer(
    model: ACTAlignedContactCVAEPolicy,
    config: ACTAlignedTrainingConfig,
) -> torch.optim.AdamW:
    """Build the canonical two-group AdamW optimizer."""

    partition = partition_trainable_parameters(model)
    return torch.optim.AdamW(
        [
            {
                "name": "main",
                "params": partition["main"],
                "lr": config.learning_rate,
            },
            {
                "name": "backbone",
                "params": partition["backbone"],
                "lr": config.backbone_learning_rate,
            },
        ],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def build_act_aligned_motion_optimizer(
    model: ACTAlignedMotionCVAEControlPolicy,
    config: ACTAlignedMotionTrainingConfig,
) -> torch.optim.AdamW:
    """Build the identical two-group AdamW optimizer for the control."""

    partition = partition_motion_trainable_parameters(model)
    return torch.optim.AdamW(
        [
            {
                "name": "main",
                "params": partition["main"],
                "lr": config.learning_rate,
            },
            {
                "name": "backbone",
                "params": partition["backbone"],
                "lr": config.backbone_learning_rate,
            },
        ],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def _validate_partition(
    model: nn.Module,
    partition: Dict[str, List[nn.Parameter]],
) -> None:
    all_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    grouped_parameters = partition["main"] + partition["backbone"]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("optimizer parameter groups contain duplicates")
    if set(grouped_ids) != {id(parameter) for parameter in all_parameters}:
        raise RuntimeError(
            "optimizer parameter groups do not cover every trainable parameter"
        )
    if not partition["main"]:
        raise RuntimeError("main optimizer parameter group must not be empty")
    if not partition["backbone"]:
        raise RuntimeError("backbone optimizer parameter group must not be empty")
