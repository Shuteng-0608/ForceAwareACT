# Vision Backbone Audit

This audit checks whether the current ForceAwareACT ResNet18 vision backbone is initialized from ImageNet pretrained weights or trained from scratch.

## Short Answer

Current `scripts/train_minimal.py` training uses ResNet18 from scratch.

The underlying `ResNet18VisionEncoder` supports ImageNet pretrained initialization, but the active minimal training path hard-codes `pretrained_resnet18=False`. The backbone is fully trainable in that path because `freeze_resnet18` is not set and defaults to `False`.

## Exact ResNet18 Initialization Code Path

The vision encoder is implemented in `src/force_aware_act/models/vision.py`:

```python
def _make_resnet18(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    except ImportError as error:
        ...
    except (AttributeError, TypeError):
        from torchvision.models import resnet18

        return resnet18(pretrained=pretrained)
```

Therefore:

- `pretrained=True` uses `ResNet18_Weights.DEFAULT` on modern torchvision.
- `pretrained=False` uses `weights=None`, which means random initialization.
- On older torchvision, the fallback uses `resnet18(pretrained=pretrained)`.

`ResNet18VisionEncoder.__init__` defaults to `pretrained=True` when instantiated directly:

```python
class ResNet18VisionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        ...
        resnet = _make_resnet18(pretrained=pretrained)
```

The policy also defaults to pretrained when constructed directly:

```python
class ForceAwareACTPolicy(nn.Module):
    def __init__(
        ...
        pretrained_resnet18: bool = True,
        freeze_resnet18: bool = False,
        ...
    ) -> None:
        ...
        self.vision_encoder = ResNet18VisionEncoder(
            d_model=d_model,
            pretrained=pretrained_resnet18,
            freeze_backbone=freeze_resnet18,
        )
```

However, the active minimal training script overrides that default.

## Current Training Path

`scripts/train_minimal.py` constructs the model with:

```python
model = ForceAwareACTPolicy(
    pretrained_resnet18=False,
    d_model=128,
    z_dim=16,
    action_dim=7,
    force_dim=6,
    chunk_len=args.chunk_len,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=256,
    dropout=0.0,
    max_force_window_len=max(args.force_window_len, 20),
).to(device)
```

It also saves checkpoint metadata with:

```python
"model": {
    "pretrained_resnet18": False,
    "d_model": 128,
    "z_dim": 16,
    ...
}
```

Conclusion: models trained by `scripts/train_minimal.py` initialize ResNet18 with `weights=None`, so the ResNet18 backbone is trained from random weights.

## Freeze Behavior

Freeze support exists in code:

- `ResNet18VisionEncoder(..., freeze_backbone=True)` sets all backbone parameters to `requires_grad=False`.
- `ForceAwareACTPolicy(..., freeze_resnet18=True)` passes that setting through to the vision encoder.

The current training CLI does not expose a freeze flag, and `scripts/train_minimal.py` does not pass `freeze_resnet18=True`. The default is `False`, so the backbone is fully trainable.

The visual projection layer remains trainable unless separately frozen. There is no current partial-freeze option such as "freeze early layers and fine-tune layer4".

## CLI And Config Support

`scripts/train_minimal.py` exposes no vision-pretraining or vision-freezing CLI flags. In particular, there is no current flag equivalent to:

- `--vision-pretrained`
- `--freeze-vision`
- `--vision-freeze-backbone`
- `--pretrained-backbone`
- `--unfreeze-vision-last-block`

The checkpoint config records `pretrained_resnet18`, but not `freeze_resnet18`.

Checkpoint loading utilities in rollout/evaluation scripts read the saved model config, migrate a legacy `pretrained_vision` key if present, and default missing `pretrained_resnet18` to `False`:

```python
if "pretrained_vision" in model_config and "pretrained_resnet18" not in model_config:
    model_config["pretrained_resnet18"] = model_config.pop("pretrained_vision")
model_config.setdefault("pretrained_resnet18", False)
```

They do not add a saved freeze setting. If `freeze_resnet18` is absent, `ForceAwareACTPolicy` defaults it to `False`.

Note: when loading a checkpoint, the initial ResNet18 construction choice is mostly overwritten by `model.load_state_dict(checkpoint["model_state_dict"])`. The config still matters for reproducibility and for constructing a matching module before loading.

## Checked Checkpoints

The following checkpoint configs were inspected:

```text
outputs/peg_hole_playback_test/overfit_action_trainzero_all10_5k/checkpoint.pt
outputs/peg_hole_playback_test/overfit_action_all10_5k/checkpoint.pt
outputs/peg_hole_playback_test/overfit_delta_joint_cmd_all10_5k/checkpoint.pt
```

All three saved:

```text
model.pretrained_resnet18: False
```

None of the three saved `freeze_resnet18`.

Observed model config summaries:

```text
overfit_action_trainzero_all10_5k:
  action_mode: action
  train_latent_mode: zero
  pretrained_resnet18: False

overfit_action_all10_5k:
  action_mode: action
  train_latent_mode: not recorded in this older checkpoint
  pretrained_resnet18: False

overfit_delta_joint_cmd_all10_5k:
  action_mode: delta_joint_cmd
  train_latent_mode: not recorded in this older checkpoint
  pretrained_resnet18: False
```

## Test Evidence

`tests/test_resnet18_vision_encoder.py` explicitly constructs the encoder with `pretrained=False` for shape tests and verifies `freeze_backbone=True` freezes backbone parameters.

`tests/test_force_aware_act_policy.py` and `tests/test_training_losses.py` construct policy test models with `pretrained_resnet18=False`. This avoids network/download dependency and confirms the from-scratch path is regularly exercised by tests.

## Implications For Small Demonstration Datasets

With 10 to 100 demonstrations, training a ResNet18 backbone from random initialization is likely data-inefficient. The policy must learn low-level visual filters, camera-specific features, task geometry, and visuomotor mappings from a small robotics dataset at the same time.

This can be acceptable for overfit/debug runs, especially when the task is narrow and camera views are stable, but it is a weak default for robust visual generalization. ImageNet initialization is usually a stronger starting point for RGB feature extraction, even when the final robotics domain differs from ImageNet.

The current implementation also supports ImageNet normalization in the dataset via `--imagenet-normalize`, but the current minimal training path does not automatically pair that with pretrained ResNet18. If pretrained weights are enabled later, preprocessing should be audited at the same time.

## Recommended Next Step

Do not change behavior as part of this audit. For a follow-up implementation task, add explicit training options and record them in checkpoints:

```text
--vision-pretrained imagenet|none
--freeze-vision-backbone
--unfreeze-vision-last-block
```

Recommended comparisons for small robotics datasets:

1. From-scratch ResNet18, fully trainable.
2. ImageNet pretrained ResNet18, fully trainable.
3. ImageNet pretrained ResNet18, frozen backbone plus trainable projection.
4. ImageNet pretrained ResNet18, freeze early layers and fine-tune layer4/projection.

For the current 10 to 100 demonstration regime, the most useful first comparison is likely:

```text
current from-scratch baseline
vs.
ImageNet pretrained backbone with either full fine-tuning or frozen early layers
```

## Bottom Line

Pretrained-weight support exists in the model code, but the current minimal training pipeline and the inspected checkpoints use random ResNet18 initialization. The backbone is not frozen; it is fully trainable. There is currently no CLI flag to switch this behavior.
