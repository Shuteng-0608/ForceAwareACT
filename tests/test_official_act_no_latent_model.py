import inspect

import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models.official_act import (  # noqa: E402
    OFFICIAL_ACT_NO_LATENT_ARCHITECTURE_VERSION,
    OfficialACTNoLatentConfig,
    OfficialACTNoLatentPolicy,
)


def _inputs(config, batch_size=2):
    return {
        "images": torch.rand(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        "qpos": torch.randn(batch_size, config.q_dim),
    }


def test_no_latent_config_has_an_explicit_and_truthful_contract():
    config = OfficialACTNoLatentConfig.canonical()
    metadata = config.checkpoint_metadata()

    assert (
        config.architecture_version
        == OFFICIAL_ACT_NO_LATENT_ARCHITECTURE_VERSION
    )
    assert not hasattr(config, "latent_dim")
    assert not hasattr(config, "posterior_token_count")
    assert config.visual_token_count == 600
    assert config.policy_memory_token_count == 601
    assert metadata["policy_memory_layout"] == "qpos_visual"
    assert metadata["latent_mechanism"] == "none"
    assert metadata["deployment_latent"] == "none"


def test_no_latent_policy_removes_every_latent_module_and_token():
    config = OfficialACTNoLatentConfig.compact_smoke()
    policy = OfficialACTNoLatentPolicy(config).eval()

    with torch.no_grad():
        outputs = policy(**_inputs(config))

    assert outputs["latent_mechanism"] == "none"
    assert outputs["visual_tokens"].shape == (
        2,
        config.visual_token_count,
        config.d_model,
    )
    assert outputs["memory_tokens"].shape == (
        2,
        config.visual_token_count + 1,
        config.d_model,
    )
    assert outputs["pred_action"].shape == (
        2,
        config.chunk_len,
        config.action_dim,
    )
    assert outputs["pred_is_pad"].shape == (2, config.chunk_len, 1)
    forbidden = ("latent", "posterior", "prior")
    assert all(
        not any(token in name for token in forbidden)
        for name, _module in policy.named_modules()
    )
    assert all(
        not any(token in name for token in forbidden)
        for name, _parameter in policy.named_parameters()
    )


def test_no_latent_policy_online_api_rejects_future_labels_and_force():
    parameters = inspect.signature(OfficialACTNoLatentPolicy.forward).parameters

    assert "action_chunk" not in parameters
    assert "padding_mask" not in parameters
    assert all("force" not in name for name in parameters)


def test_no_latent_policy_eval_is_deterministic():
    torch.manual_seed(17)
    config = OfficialACTNoLatentConfig.compact_smoke()
    policy = OfficialACTNoLatentPolicy(config).eval()
    inputs = _inputs(config, batch_size=1)

    with torch.no_grad():
        first = policy(**inputs)["pred_action"]
        second = policy(**inputs)["pred_action"]

    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
