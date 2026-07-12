import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_minimal import (
    DATALOADER_SEED_OFFSET,
    build_checkpoint_payload,
    compute_initial_model_sha256,
    configure_reproducibility,
    parse_args,
    seed_dataloader_worker,
)


def _fingerprint_for(seed: int, dataset_size: int) -> str:
    configure_reproducibility(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + DATALOADER_SEED_OFFSET)
    DataLoader(
        TensorDataset(torch.arange(dataset_size)),
        batch_size=2,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_dataloader_worker,
    )
    configure_reproducibility(seed)
    return compute_initial_model_sha256(torch.nn.Linear(4, 3))


def _sample_order(dataset_size: int, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + DATALOADER_SEED_OFFSET)
    loader = DataLoader(
        TensorDataset(torch.arange(dataset_size)),
        batch_size=3,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_dataloader_worker,
    )
    return [int(value) for (batch,) in loader for value in batch]


def test_help_exposes_seed_and_deterministic(capsys):
    try:
        parse_args(["--help"])
    except SystemExit as error:
        assert error.code == 0
    help_text = capsys.readouterr().out
    assert "--seed" in help_text
    assert "--deterministic" in help_text


def test_deterministic_mode_configures_required_pytorch_backends(monkeypatch):
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        configure_reproducibility(0, deterministic=True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark


def test_same_seed_and_model_config_have_same_fingerprint():
    assert _fingerprint_for(7, 10) == _fingerprint_for(7, 10)


def test_different_seeds_have_different_fingerprints():
    assert _fingerprint_for(7, 10) != _fingerprint_for(8, 10)


def test_dataset_size_does_not_change_initial_fingerprint():
    assert _fingerprint_for(7, 10) == _fingerprint_for(7, 100)


def test_dataloader_order_is_repeatable_for_same_seed():
    assert _sample_order(20, 11) == _sample_order(20, 11)


def test_checkpoint_payload_contains_reproducibility_metadata():
    configure_reproducibility(5)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    fingerprint = compute_initial_model_sha256(model)

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config={},
        step=3,
        training_seed=5,
        dataloader_seed=6,
        deterministic_enabled=False,
        initial_model_sha256=fingerprint,
    )

    assert payload["training_seed"] == 5
    assert payload["dataloader_seed"] == 6
    assert payload["deterministic_enabled"] is False
    assert payload["initial_model_sha256"] == fingerprint
