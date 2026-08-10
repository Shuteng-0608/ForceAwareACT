# Vendored force-feedback HUD provenance

The files below are byte-for-byte copies from the sibling `arm_teleop`
repository. They are kept unchanged so their provenance and future drift can
be audited exactly.

```text
source repository: /home/stw/ForceAwareACT_workspace/arm_teleop
source commit:     18f4566f47b094f40fa69357b8c07cffb89286f5
```

| Local file | Source file | SHA256 |
| --- | --- | --- |
| `force_feedback_overlay.py` | `vptele/utils/force_feedback_overlay.py` | `43dbdffe35bb89a8487fe4af00cbfb2902b0c3a63386be6e1d7edc511802523d` |
| `ft_wrench_utils.py` | `vptele/utils/ft_wrench_utils.py` | `8f8861d6c487c8d8ddcf4006e6dbec5f028ec6cbaa791006f526740b1f7cfa7d` |

ForceAwareACT-specific timing, video, and rollout adaptation belongs in
separate modules. Do not add project-specific behavior to these two vendored
files.
