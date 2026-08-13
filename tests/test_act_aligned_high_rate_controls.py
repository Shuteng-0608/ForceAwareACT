import inspect

import torch

from force_aware_act.models.act_aligned import (
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateDualZeroPolicy,
    ACTAlignedHighRateMotionCVAEPolicy,
)


def _config(kind: str):
    factory = {
        "motion": ACTAlignedHighRateConfig.motion_control,
        "dual_zero": ACTAlignedHighRateConfig.dual_zero,
    }[kind]
    return factory(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        chunk_len=3,
        local_force_dim=16,
        image_height=32,
        image_width=32,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


def _inputs(config, batch_size=2):
    sample_count = config.max_force_samples_per_interval
    interval_count = config.max_online_force_intervals
    force = torch.randn(
        batch_size,
        interval_count,
        sample_count,
        config.force_dim,
    )
    relative_time = (
        torch.arange(sample_count, dtype=torch.float32)
        .div(config.force_sample_rate_hz)
        .view(1, 1, sample_count)
        .expand(batch_size, interval_count, sample_count)
        .clone()
    )
    return {
        "images": torch.randn(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        "qpos": torch.randn(batch_size, config.q_dim),
        "online_force_intervals": force,
        "online_force_relative_time": relative_time,
        "online_force_sample_padding_mask": torch.zeros(
            batch_size,
            interval_count,
            sample_count,
            dtype=torch.bool,
        ),
        "online_force_interval_padding_mask": torch.zeros(
            batch_size,
            interval_count,
            dtype=torch.bool,
        ),
    }


def test_high_rate_motion_uses_action_only_posterior_and_zero_deployment():
    config = _config("motion")
    model = ACTAlignedHighRateMotionCVAEPolicy(config).eval()
    inputs = _inputs(config)
    action = torch.randn(2, config.chunk_len, config.action_dim)
    action_mask = torch.zeros(2, config.chunk_len, dtype=torch.bool)

    with torch.no_grad():
        posterior = model.forward_train(
            **inputs,
            action_chunk=action,
            action_padding_mask=action_mask,
            sample_posterior=False,
        )
        deployment = model(**inputs)

    assert posterior["motion_latent_source"] == "posterior"
    torch.testing.assert_close(posterior["z_motion"], posterior["mu_motion"])
    assert deployment["motion_latent_source"] == "zero"
    torch.testing.assert_close(
        deployment["z_motion"],
        torch.zeros_like(deployment["z_motion"]),
        atol=0.0,
        rtol=0.0,
    )
    assert deployment["pred_force_highrate"].shape == (
        2,
        config.chunk_len,
        config.max_force_samples_per_interval,
        config.force_dim,
    )
    assert not hasattr(model, "contact_posterior")
    assert not hasattr(model, "contact_prior")
    assert "future_force_intervals" not in inspect.signature(
        model.forward_train
    ).parameters


def test_high_rate_dual_zero_is_structurally_latent_free():
    config = _config("dual_zero")
    model = ACTAlignedHighRateDualZeroPolicy(config).eval()
    output = model(**_inputs(config))

    assert output["latent_mechanism"] == "none"
    assert output["policy_tokens"].shape[1] == config.visual_token_count + 3
    assert output["pred_action"].shape == (
        2,
        config.chunk_len,
        config.action_dim,
    )
    assert output["pred_force_highrate"].shape == (
        2,
        config.chunk_len,
        config.max_force_samples_per_interval,
        config.force_dim,
    )
    forbidden = ("latent", "posterior", "prior")
    assert all(
        not any(token in name for token in forbidden)
        for name, _module in model.named_modules()
    )
    parameters = inspect.signature(model.forward).parameters
    assert "action_chunk" not in parameters
    assert "future_force_intervals" not in parameters


def test_native_rate_force_spike_reaches_both_control_policies():
    torch.manual_seed(4)
    for kind, policy_type in (
        ("motion", ACTAlignedHighRateMotionCVAEPolicy),
        ("dual_zero", ACTAlignedHighRateDualZeroPolicy),
    ):
        config = _config(kind)
        model = policy_type(config).eval()
        inputs = _inputs(config, batch_size=1)
        inputs["online_force_intervals"].zero_()
        with torch.no_grad():
            baseline = model(**inputs)["pred_action"]
            intervened_inputs = dict(inputs)
            intervened_inputs["online_force_intervals"] = inputs[
                "online_force_intervals"
            ].clone()
            intervened_inputs["online_force_intervals"][:, -1, 7, 0] = 25.0
            intervened = model(**intervened_inputs)["pred_action"]
        assert not torch.allclose(baseline, intervened)


def test_new_config_metadata_does_not_relabel_existing_contact_architecture():
    contact = ACTAlignedHighRateConfig(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        local_force_dim=16,
        image_height=32,
        image_width=32,
        pretrained_backbone=False,
        imagenet_normalize=False,
    ).checkpoint_metadata()
    motion = _config("motion").checkpoint_metadata()
    dual_zero = _config("dual_zero").checkpoint_metadata()

    assert "contact_posterior_inputs" not in motion
    assert contact["deployment_contact_latent"] == "zero"
    assert motion["deployment_motion_latent"] == "zero"
    assert dual_zero["latent_mechanism"] == "none"
    assert dual_zero["policy_memory_token_count"] == (
        dual_zero["visual_token_count"] + 3
    )
