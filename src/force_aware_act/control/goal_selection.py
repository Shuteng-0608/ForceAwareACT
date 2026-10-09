"""Resolve one explicit hole before a read-only run; no device access."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re

from .contracts import ShadowConfig


DEFAULT_PROFILE = Path(__file__).resolve().parents[3] / "configs/deployment/profiles/contact_right_step_v1"


@dataclass(frozen=True)
class GoalSelection:
    config: ShadowConfig
    profile_id: str
    manifest_path: Path
    config_path: Path
    manifest_sha256: str
    config_sha256: str
    base_sha256: str
    controller_baseline_json: str
    controller_baseline_sha256: str

    def to_dict(self):
        """Self-contained run snapshot, detached from later edits to the profile."""
        goal = self.config.goal
        return {
            "schema_version": "explicit_goal_selection_v1",
            "selection_method": "operator_goal_id",
            "goal_id": goal.goal_id,
            "locked_for_run": True,
            "model_conditioned_on_goal_id": False,
            "profile_id": self.profile_id,
            "manifest_path": str(self.manifest_path),
            "config_path": str(self.config_path),
            "manifest_sha256": self.manifest_sha256,
            "config_sha256": self.config_sha256,
            "base_sha256": self.base_sha256,
            "controller_baseline": json.loads(self.controller_baseline_json),
            "controller_baseline_sha256": self.controller_baseline_sha256,
            "target_position_m": list(goal.target_position_m),
            "frame_id": goal.frame_id,
            "pose_source": goal.pose_source,
            "radius_m": goal.radius_m,
            "dwell_ms": goal.dwell_ms,
            "config": self.config.to_dict(),
        }


def select_goal(goal_id: str, profile: Path = DEFAULT_PROFILE) -> GoalSelection:
    """Load a manifest-listed sphere; never fall back to another hole."""
    if not isinstance(goal_id, str) or re.fullmatch(r"x[0-9]{2}y[0-9]{2}", goal_id) is None:
        raise ValueError("goal-id must be an exact hole label such as x14y08")
    root = Path(profile).resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["schema_version"] != "readonly_profile_manifest_v1"
            or manifest["allow_motion"] is not False):
        raise ValueError("goal selection requires a read-only profile manifest")
    entries = manifest["goal_configs"]
    if not isinstance(entries, list) or not entries or any(
        not isinstance(p, str) or re.fullmatch(r"goals/x[0-9]{2}y[0-9]{2}\.json", p) is None
        for p in entries
    ) or len(entries) != len(set(entries)):
        raise ValueError("invalid goal_configs in profile manifest")
    relative = f"goals/{goal_id}.json"
    if relative not in entries:
        available = ", ".join(Path(p).stem for p in entries)
        raise ValueError(f"unknown goal-id {goal_id}; available: {available}")

    def read_verified(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("configuration path must stay within the selected profile")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if manifest["files_sha256"].get(name) != digest:
            raise ValueError(f"configuration SHA256 differs from manifest: {name}")
        return path, json.loads(raw), digest

    config_path, payload, config_hash = read_verified(relative)
    config = ShadowConfig.from_dict(payload)
    _, payload, base_hash = read_verified(manifest["base_config"])
    base = ShadowConfig.from_dict(payload)
    _, baseline, baseline_hash = read_verified(manifest["controller_baseline"])
    if baseline["apply_on_load"] is not False or baseline["allow_motion"] is not False:
        raise ValueError("controller baseline must be a read-only reference")
    if config.goal.goal_id != goal_id:
        raise ValueError("requested goal-id differs from the configuration goal_id")
    if (config.mode != "shadow" or config.goal.kind != "sphere"
            or config.goal.pose_source != "m6_ccs40_right_flange_fk"
            or not config.goal.coarse_completion_accepted):
        raise ValueError("selected hole requires an accepted shadow FK sphere")
    if replace(config, goal=base.goal) != base:
        raise ValueError("selected hole changes the common controller/guard configuration")
    return GoalSelection(config, manifest["profile_id"], manifest_path, config_path,
                         hashlib.sha256(manifest_bytes).hexdigest(), config_hash, base_hash,
                         json.dumps(baseline,sort_keys=True), baseline_hash)
