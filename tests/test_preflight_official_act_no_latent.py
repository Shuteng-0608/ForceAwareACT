import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models.official_act import (  # noqa: E402
    OfficialACTNoLatentConfig,
    OfficialACTNoLatentPolicy,
)
from scripts.preflight_official_act_no_latent import (  # noqa: E402
    _audit_structure,
)


def test_preflight_structure_audit_accepts_only_the_latent_free_policy():
    model = OfficialACTNoLatentPolicy(
        OfficialACTNoLatentConfig.compact_smoke()
    )

    _audit_structure(model)


def test_preflight_structure_audit_reports_injected_latent_state():
    model = OfficialACTNoLatentPolicy(
        OfficialACTNoLatentConfig.compact_smoke()
    )
    model.register_parameter("latent_probe", torch.nn.Parameter(torch.zeros(1)))

    with pytest.raises(RuntimeError, match="latent_probe"):
        _audit_structure(model)
