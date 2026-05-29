import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models import ForceAwareACTPolicy


def _make_policy(chunk_len=50):
    return ForceAwareACTPolicy(
        d_model=128,
        z_dim=16,
        action_dim=7,
        force_dim=6,
        chunk_len=chunk_len,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=256,
        dropout=0.0,
        pretrained_resnet18=False,
    )


def _make_inputs(batch_size=2, num_cameras=2, chunk_len=50):
    return {
        "images": torch.randn(batch_size, num_cameras, 3, 224, 224),
        "qpos": torch.randn(batch_size, 7),
        "force_window": torch.randn(batch_size, 50, 6),
        "action_chunk": torch.randn(batch_size, chunk_len, 7),
        "future_force_chunk": torch.randn(batch_size, chunk_len, 6),
    }


def test_force_aware_act_policy_training_forward_shapes():
    model = _make_policy(chunk_len=50)
    model.eval()
    inputs = _make_inputs(chunk_len=50)

    with torch.no_grad():
        outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=inputs["future_force_chunk"],
            is_training=True,
        )

    assert outputs["pred_action"].shape == (2, 50, 7)
    assert outputs["pred_force"].shape == (2, 50, 6)
    assert outputs["visual_tokens"].shape == (2, 98, 128)
    assert outputs["z_q"].shape == (2, 128)
    assert outputs["z_F_online"].shape == (2, 128)
    assert outputs["z_VF"].shape == (2, 128)
    assert outputs["z_motion"].shape == (2, 16)
    assert outputs["z_contact"].shape == (2, 16)
    assert outputs["decoder_hidden"].shape == (2, 50, 128)
    assert outputs["mu_motion"].shape == (2, 16)
    assert outputs["logvar_motion"].shape == (2, 16)
    assert outputs["mu_contact"].shape == (2, 16)
    assert outputs["logvar_contact"].shape == (2, 16)


def test_force_aware_act_policy_inference_forward_shapes_and_no_posteriors():
    model = _make_policy(chunk_len=50)
    model.eval()
    inputs = _make_inputs(chunk_len=50)

    with torch.no_grad():
        outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
        )

    assert outputs["pred_action"].shape == (2, 50, 7)
    assert outputs["pred_force"].shape == (2, 50, 6)
    assert "mu_motion" not in outputs
    assert "logvar_motion" not in outputs
    assert "mu_contact" not in outputs
    assert "logvar_contact" not in outputs
    assert torch.count_nonzero(outputs["z_motion"]) == 0
    assert torch.count_nonzero(outputs["z_contact"]) == 0


def test_force_aware_act_policy_inference_rejects_future_labels():
    model = _make_policy(chunk_len=4)
    inputs = _make_inputs(chunk_len=4)

    with pytest.raises(ValueError, match="action_chunk must be None"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=None,
            is_training=False,
        )

    with pytest.raises(ValueError, match="future_force_chunk must be None"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=inputs["future_force_chunk"],
            is_training=False,
        )


def test_force_aware_act_policy_training_requires_future_labels():
    model = _make_policy(chunk_len=4)
    inputs = _make_inputs(chunk_len=4)

    with pytest.raises(ValueError, match="action_chunk is required"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=inputs["future_force_chunk"],
            is_training=True,
        )

    with pytest.raises(ValueError, match="future_force_chunk is required"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=None,
            is_training=True,
        )


def test_force_aware_act_policy_small_config_forward():
    model = _make_policy(chunk_len=4)
    model.eval()
    inputs = _make_inputs(batch_size=1, num_cameras=1, chunk_len=4)

    with torch.no_grad():
        outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=inputs["future_force_chunk"],
            is_training=True,
        )

    assert outputs["pred_action"].shape == (1, 4, 7)
    assert outputs["pred_force"].shape == (1, 4, 6)


def test_force_aware_act_policy_backward_pass():
    model = _make_policy(chunk_len=4)
    inputs = _make_inputs(batch_size=1, num_cameras=1, chunk_len=4)

    outputs = model(
        images=inputs["images"],
        qpos=inputs["qpos"],
        force_window=inputs["force_window"],
        action_chunk=inputs["action_chunk"],
        future_force_chunk=inputs["future_force_chunk"],
        is_training=True,
    )
    loss = outputs["pred_action"].pow(2).mean() + outputs["pred_force"].pow(2).mean()
    loss.backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )
