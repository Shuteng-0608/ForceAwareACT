import torch

from force_aware_act.models.act_aligned import (
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateContactCVAEPolicy,
    ACTAlignedHighRateForceEncoder,
)


def _config(**overrides):
    values = {
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "chunk_len": 3,
        "local_force_dim": 16,
        "image_height": 32,
        "image_width": 32,
        "pretrained_backbone": False,
        "imagenet_normalize": False,
    }
    values.update(overrides)
    return ACTAlignedHighRateConfig(**values)


def _interval_inputs(config, batch_size, interval_count):
    force = torch.randn(
        batch_size,
        interval_count,
        config.max_force_samples_per_interval,
        config.force_dim,
    )
    relative_time = (
        torch.arange(config.max_force_samples_per_interval).float()
        .mul(1.0 / config.force_sample_rate_hz)
        .view(1, 1, -1)
        .expand(batch_size, interval_count, -1)
        .clone()
    )
    sample_mask = torch.zeros(
        batch_size,
        interval_count,
        config.max_force_samples_per_interval,
        dtype=torch.bool,
    )
    interval_mask = torch.zeros(batch_size, interval_count, dtype=torch.bool)
    return force, relative_time, sample_mask, interval_mask


def _policy_inputs(config, batch_size=2):
    online = _interval_inputs(
        config,
        batch_size,
        config.max_online_force_intervals,
    )
    future = _interval_inputs(config, batch_size, config.chunk_len)
    return {
        "images": torch.randn(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        "qpos": torch.randn(batch_size, config.q_dim),
        "online_force_intervals": online[0],
        "online_force_relative_time": online[1],
        "online_force_sample_padding_mask": online[2],
        "online_force_interval_padding_mask": online[3],
        "action_chunk": torch.randn(batch_size, config.chunk_len, config.action_dim),
        "future_force_intervals": future[0],
        "future_force_relative_time": future[1],
        "future_force_sample_padding_mask": future[2],
        "future_force_interval_padding_mask": future[3],
        "action_padding_mask": torch.zeros(
            batch_size,
            config.chunk_len,
            dtype=torch.bool,
        ),
    }


def test_local_encoder_is_padding_invariant_and_zero_for_empty_intervals():
    config = _config()
    encoder = ACTAlignedHighRateForceEncoder(config).eval()
    force, relative_time, sample_mask, interval_mask = _interval_inputs(config, 1, 2)
    sample_mask[:, 0, 5:] = True
    interval_mask[:, 1] = True
    sample_mask[:, 1] = True
    force[:, 1] = 1000.0
    changed = force.clone()
    changed[:, 0, 5:] = -1000.0

    first = encoder(force, relative_time, sample_mask, interval_mask)
    second = encoder(changed, relative_time, sample_mask, interval_mask)

    torch.testing.assert_close(first[:, 0], second[:, 0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(first[:, 1], torch.zeros_like(first[:, 1]))


def test_single_2ms_spike_changes_local_force_token():
    config = _config()
    encoder = ACTAlignedHighRateForceEncoder(config).eval()
    force, relative_time, sample_mask, interval_mask = _interval_inputs(config, 1, 1)
    force.zero_()
    baseline = encoder(force, relative_time, sample_mask, interval_mask)
    force[:, :, 7, 0] = 20.0
    intervened = encoder(force, relative_time, sample_mask, interval_mask)

    assert not torch.allclose(baseline, intervened)


def test_v2_policy_forward_train_shapes_and_shared_encoder_registration():
    config = _config()
    model = ACTAlignedHighRateContactCVAEPolicy(config).eval()
    inputs = _policy_inputs(config)

    output = model.forward_train(**inputs, sample_posterior=False)

    assert output["pred_action"].shape == (2, config.chunk_len, config.action_dim)
    assert output["pred_force"].shape == (2, config.chunk_len, config.force_dim)
    assert output["pred_force_highrate"].shape == (
        2,
        config.chunk_len,
        config.max_force_samples_per_interval,
        config.force_dim,
    )
    assert output["z_F_online"].shape == (2, config.d_model)
    assert output["future_force_interval_tokens"].shape == (
        2,
        config.chunk_len,
        config.d_model,
    )
    parameter_names = [name for name, _parameter in model.named_parameters()]
    assert any(name.startswith("high_rate_force_encoder.") for name in parameter_names)
    assert not any(
        ".high_rate_force_encoder." in name for name in parameter_names
    )
    assert len({id(parameter) for parameter in model.parameters()}) == len(
        list(model.parameters())
    )


def test_v2_deployment_has_no_future_inputs_and_uses_exact_zero_latent():
    config = _config()
    model = ACTAlignedHighRateContactCVAEPolicy(config).eval()
    inputs = _policy_inputs(config, batch_size=1)

    output = model(
        images=inputs["images"],
        qpos=inputs["qpos"],
        online_force_intervals=inputs["online_force_intervals"],
        online_force_relative_time=inputs["online_force_relative_time"],
        online_force_sample_padding_mask=(
            inputs["online_force_sample_padding_mask"]
        ),
        online_force_interval_padding_mask=(
            inputs["online_force_interval_padding_mask"]
        ),
        contact_latent_mode="zero",
    )

    assert output["contact_latent_source"] == "zero"
    torch.testing.assert_close(
        output["z_contact"],
        torch.zeros_like(output["z_contact"]),
        atol=0.0,
        rtol=0.0,
    )


def test_future_high_rate_spike_changes_contact_posterior_mean():
    torch.manual_seed(0)
    config = _config()
    model = ACTAlignedHighRateContactCVAEPolicy(config).eval()
    inputs = _policy_inputs(config, batch_size=1)
    baseline = model.forward_train(**inputs, sample_posterior=False)["mu_contact"]
    inputs["future_force_intervals"] = inputs["future_force_intervals"].clone()
    inputs["future_force_intervals"][:, 1, 8, 0] += 100.0
    intervened = model.forward_train(**inputs, sample_posterior=False)["mu_contact"]

    assert not torch.allclose(baseline, intervened)


def test_prior_path_cannot_update_shared_or_posterior_encoders():
    config = _config()
    model = ACTAlignedHighRateContactCVAEPolicy(config).train()
    inputs = _policy_inputs(config, batch_size=1)
    output = model.forward_train(**inputs, sample_posterior=False)

    (output["mu_contact_prior"].sum() + output["logvar_contact_prior"].sum()).backward()

    assert any(parameter.grad is not None for parameter in model.contact_prior.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.high_rate_force_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.contact_posterior.parameters()
    )
    assert all(parameter.grad is None for parameter in model.vision_backbone.parameters())
