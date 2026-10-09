"""One-pass native-stream statistics from explicitly supplied training episodes."""

import numpy as np

from force_aware_act.act_aligned_training.normalization import NormalizationStats, _MomentAccumulator
from .reader import RawEpisode
from .config import RealRobotDataConfig


def compute_real_robot_normalization_stats(data_root, episodes, config=None, *, minimum_std=1e-2):
    if not np.isfinite(minimum_std) or minimum_std <= 0:
        raise ValueError("minimum_std must be finite and positive")
    config = config or RealRobotDataConfig()
    accumulators = [_MomentAccumulator(7), _MomentAccumulator(7), _MomentAccumulator(config.force_dim)]
    count = 0
    for record in episodes:
        episode = RawEpisode(record.resolve(data_root), config)
        for accumulator, values in zip(accumulators[:2], (episode.qpos, episode.action)):
            accumulator.update(values)
        if config.force_dim == 3:
            for start in range(0, len(episode.torque), 2048):
                values, valid = episode.force.read(slice(start, start + 2048))
                if valid.any():
                    accumulators[2].update(values[valid])
        else:
            accumulators[2].update(episode.torque)
        count += 1
    if not count:
        raise ValueError("normalization requires explicit training episodes")
    pairs = [accumulator.finalize(minimum_std) for accumulator in accumulators]
    return NormalizationStats(
        qpos_mean=tuple(pairs[0][0].tolist()), qpos_std=tuple(pairs[0][1].tolist()),
        action_mean=tuple(pairs[1][0].tolist()), action_std=tuple(pairs[1][1].tolist()),
        force_mean=tuple(pairs[2][0].tolist()), force_std=tuple(pairs[2][1].tolist()),
        minimum_std=float(minimum_std))
