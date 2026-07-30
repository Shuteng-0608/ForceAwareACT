import pytest
import torch

from force_aware_act.models.act_aligned import (
    CONTACT_POSTERIOR_TOKEN_GROUPS,
    POLICY_SPECIAL_TOKEN_NAMES,
    ACTAlignedConfig,
    ACTAlignedShapeContract,
    require_padding_mask,
    require_token_tensor,
)


def test_default_sequence_lengths_and_token_order_are_frozen():
    config = ACTAlignedConfig()

    assert CONTACT_POSTERIOR_TOKEN_GROUPS == ("cls", "qpos", "contact_steps")
    assert POLICY_SPECIAL_TOKEN_NAMES == ("z_contact", "qpos", "z_F_online", "z_VF")
    assert config.visual_grid_height == 15
    assert config.visual_grid_width == 20
    assert config.visual_token_count == 600
    assert config.contact_posterior_token_count == 102
    assert config.force_encoder_token_count == 21
    assert config.policy_memory_token_count == 604


def test_default_batch_first_shape_contract():
    contract = ACTAlignedShapeContract(ACTAlignedConfig())

    assert contract.visual_tokens(3) == (3, 600, 512)
    assert contract.qpos_token(3) == (3, 1, 512)
    assert contract.action_tokens(3) == (3, 100, 512)
    assert contract.force_window_tokens(3) == (3, 20, 512)
    assert contract.contact_posterior_tokens(3) == (3, 102, 512)
    assert contract.force_encoder_tokens(3) == (3, 21, 512)
    assert contract.policy_memory(3) == (3, 604, 512)
    assert contract.decoder_hidden(3) == (3, 100, 512)
    assert contract.action_output(3) == (3, 100, 7)
    assert contract.force_output(3) == (3, 100, 6)


def test_contract_uses_configured_task_dimensions():
    config = ACTAlignedConfig.compact_smoke(
        q_dim=14,
        action_dim=14,
        chunk_len=10,
        force_window_len=50,
        num_cameras=1,
        image_height=480,
        image_width=640,
    )
    contract = ACTAlignedShapeContract(config)

    assert config.visual_token_count == 300
    assert contract.contact_posterior_tokens(2) == (2, 12, 128)
    assert contract.force_encoder_tokens(2) == (2, 51, 128)
    assert contract.policy_memory(2) == (2, 304, 128)
    assert contract.action_output(2) == (2, 10, 14)


def test_token_tensor_validator_accepts_exact_batch_first_float_tensor():
    tensor = torch.zeros(2, 12, 512)

    require_token_tensor(
        tensor,
        name="posterior_tokens",
        expected_shape=(2, 12, 512),
    )


def test_token_tensor_validator_rejects_sequence_first_layout():
    tensor = torch.zeros(12, 2, 512)

    with pytest.raises(ValueError, match="must have shape"):
        require_token_tensor(
            tensor,
            name="posterior_tokens",
            expected_shape=(2, 12, 512),
        )


def test_padding_mask_contract_uses_true_for_padding_and_bool_dtype():
    mask = torch.tensor([[False, False, True], [False, True, True]])

    require_padding_mask(
        mask,
        name="posterior_padding_mask",
        batch_size=2,
        sequence_length=3,
    )

    with pytest.raises(ValueError, match="torch.bool"):
        require_padding_mask(
            mask.float(),
            name="posterior_padding_mask",
            batch_size=2,
            sequence_length=3,
        )


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_shape_contract_rejects_invalid_batch_size(batch_size):
    contract = ACTAlignedShapeContract(ACTAlignedConfig())

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        contract.decoder_hidden(batch_size)
