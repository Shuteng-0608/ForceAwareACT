import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedTrainingConfig,
    build_act_aligned_optimizer,
    partition_trainable_parameters,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
)


def _model():
    config = ACTAlignedConfig(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        chunk_len=6,
        force_window_len=5,
        image_height=64,
        image_width=64,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )
    return ACTAlignedContactCVAEPolicy(config)


def test_parameter_partition_is_complete_unique_and_module_based():
    model = _model()
    partition = partition_trainable_parameters(model)
    main_ids = {id(parameter) for parameter in partition["main"]}
    backbone_ids = {id(parameter) for parameter in partition["backbone"]}
    all_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    assert main_ids.isdisjoint(backbone_ids)
    assert main_ids | backbone_ids == all_ids
    assert id(model.vision_backbone.body[0].weight) in backbone_ids
    assert id(model.vision_backbone.input_projection.weight) in main_ids
    assert id(model.contact_posterior.mean_head.weight) in main_ids
    assert id(model.contact_prior.mean_head.weight) in main_ids
    assert id(model.query_decoder.query_embedding.weight) in main_ids
    assert id(model.action_head.weight) in main_ids
    assert id(model.force_head.weight) in main_ids


def test_optimizer_has_canonical_adamw_groups_and_hyperparameters():
    model = _model()
    config = ACTAlignedTrainingConfig()
    optimizer = build_act_aligned_optimizer(model, config)
    groups = {group["name"]: group for group in optimizer.param_groups}

    assert isinstance(optimizer, torch.optim.AdamW)
    assert set(groups) == {"main", "backbone"}
    assert groups["main"]["lr"] == config.learning_rate
    assert groups["backbone"]["lr"] == config.backbone_learning_rate
    assert groups["main"]["weight_decay"] == config.weight_decay
    assert groups["backbone"]["weight_decay"] == config.weight_decay
    assert groups["main"]["betas"] == (config.adam_beta1, config.adam_beta2)


def test_partition_rejects_non_act_aligned_models():
    with pytest.raises(TypeError, match="ACTAlignedContactCVAEPolicy"):
        partition_trainable_parameters(nn.Linear(2, 2))
