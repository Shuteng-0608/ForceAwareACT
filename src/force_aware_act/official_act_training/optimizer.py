"""Official ACT two-group AdamW optimizer."""

from __future__ import annotations

import torch

from force_aware_act.models.official_act import OfficialACTPolicy
from force_aware_act.official_act_training.config import (
    OfficialACTTrainingConfig,
)


def build_official_act_optimizer(
    model: OfficialACTPolicy,
    config: OfficialACTTrainingConfig,
) -> torch.optim.AdamW:
    if not isinstance(model, OfficialACTPolicy):
        raise TypeError("model must be an OfficialACTPolicy")
    backbone_parameters = [
        parameter
        for parameter in model.backbone.body.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    grouped_ids = {
        id(parameter)
        for parameter in main_parameters + backbone_parameters
    }
    all_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if grouped_ids != all_ids:
        raise RuntimeError("optimizer groups do not cover the model")
    return torch.optim.AdamW(
        [
            {
                "name": "main",
                "params": main_parameters,
                "lr": config.learning_rate,
            },
            {
                "name": "backbone",
                "params": backbone_parameters,
                "lr": config.backbone_learning_rate,
            },
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
