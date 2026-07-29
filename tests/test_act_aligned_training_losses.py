import pytest
import torch

from force_aware_act.act_aligned_training import (
    ACTAlignedCriterion,
    ACTAlignedTrainingConfig,
    detached_posterior_prior_kl,
    diagonal_gaussian_kl,
    masked_l1_loss,
    standard_normal_kl,
)


def test_masked_l1_divides_by_valid_scalar_elements_only():
    prediction = torch.zeros(1, 3, 2)
    target = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [1000.0, 1000.0]]])
    padding_mask = torch.tensor([[False, False, True]])

    loss = masked_l1_loss(
        prediction,
        target,
        padding_mask,
        name="test",
    )

    torch.testing.assert_close(loss, torch.tensor(4.0))


def test_masked_l1_is_invariant_to_padded_values_and_rejects_all_padding():
    prediction = torch.zeros(2, 3, 1)
    first_target = torch.ones_like(prediction)
    second_target = first_target.clone()
    second_target[:, -1] = 10000.0
    padding_mask = torch.tensor(
        [[False, False, True], [False, False, True]]
    )

    first = masked_l1_loss(
        prediction,
        first_target,
        padding_mask,
        name="test",
    )
    second = masked_l1_loss(
        prediction,
        second_target,
        padding_mask,
        name="test",
    )

    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="at least one valid"):
        masked_l1_loss(
            prediction,
            first_target,
            torch.ones(2, 3, dtype=torch.bool),
            name="test",
        )


def test_diagonal_gaussian_kl_matches_known_values():
    zeros = torch.zeros(2, 3)

    identical = diagonal_gaussian_kl(zeros, zeros, zeros, zeros)
    shifted = diagonal_gaussian_kl(
        torch.ones(2, 3),
        zeros,
        zeros,
        zeros,
    )

    torch.testing.assert_close(identical, torch.tensor(0.0))
    torch.testing.assert_close(shifted, torch.tensor(1.5))


def test_conditional_kl_backpropagates_to_posterior_and_prior():
    posterior_mean = torch.randn(2, 4, requires_grad=True)
    posterior_log_variance = torch.randn(2, 4, requires_grad=True)
    prior_mean = torch.randn(2, 4, requires_grad=True)
    prior_log_variance = torch.randn(2, 4, requires_grad=True)

    diagonal_gaussian_kl(
        posterior_mean,
        posterior_log_variance,
        prior_mean,
        prior_log_variance,
    ).backward()

    for tensor in (
        posterior_mean,
        posterior_log_variance,
        prior_mean,
        prior_log_variance,
    ):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) > 0


def test_standard_normal_kl_and_detached_prior_matching_route_gradients():
    posterior_mean = torch.randn(2, 4, requires_grad=True)
    posterior_log_variance = torch.randn(2, 4, requires_grad=True)
    prior_mean = torch.randn(2, 4, requires_grad=True)
    prior_log_variance = torch.randn(2, 4, requires_grad=True)

    posterior_regularization = standard_normal_kl(
        posterior_mean,
        posterior_log_variance,
    )
    prior_matching = detached_posterior_prior_kl(
        posterior_mean,
        posterior_log_variance,
        prior_mean,
        prior_log_variance,
    )
    prior_matching.backward(retain_graph=True)

    assert posterior_mean.grad is None
    assert posterior_log_variance.grad is None
    assert prior_mean.grad is not None
    assert prior_log_variance.grad is not None

    posterior_regularization.backward()

    assert posterior_mean.grad is not None
    assert posterior_log_variance.grad is not None


def test_criterion_uses_reconstruction_posterior_kl_and_detached_prior_match():
    config = ACTAlignedTrainingConfig(
        force_loss_weight=2.0,
        posterior_kl_weight=3.0,
        prior_match_weight=4.0,
    )
    criterion = ACTAlignedCriterion(config)
    outputs = {
        "pred_action": torch.zeros(1, 2, 1),
        "pred_force": torch.zeros(1, 2, 1),
        "mu_contact": torch.ones(1, 2),
        "logvar_contact": torch.zeros(1, 2),
        "mu_contact_prior": torch.zeros(1, 2),
        "logvar_contact_prior": torch.zeros(1, 2),
    }
    padding_mask = torch.zeros(1, 2, dtype=torch.bool)

    losses = criterion(
        outputs,
        torch.ones(1, 2, 1),
        torch.full((1, 2, 1), 2.0),
        padding_mask,
    )

    torch.testing.assert_close(losses["loss_action"], torch.tensor(1.0))
    torch.testing.assert_close(losses["loss_force"], torch.tensor(2.0))
    torch.testing.assert_close(losses["loss_posterior_kl"], torch.tensor(1.0))
    torch.testing.assert_close(losses["loss_prior_match"], torch.tensor(1.0))
    torch.testing.assert_close(losses["loss_total"], torch.tensor(12.0))
    assert set(losses) == {
        "loss_total",
        "loss_action",
        "loss_force",
        "loss_posterior_kl",
        "loss_prior_match",
    }


def test_criterion_rejects_missing_new_policy_outputs():
    with pytest.raises(KeyError, match="pred_action"):
        ACTAlignedCriterion(ACTAlignedTrainingConfig())(
            {},
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, dtype=torch.bool),
        )
