# Dataset Split Files

Split files contain one HDF5 episode path per line. Paths are relative to the project root, for example:

```text
mujoco_data/peg_hole_100/20260701_131034_teleop_006/episode.hdf5
```

The HDF5 files themselves are local data artifacts and are not committed to git.

The shared resolver accepts absolute paths and resolves relative entries against the repository root first, then the directory containing the list file. Blank lines and `#` comments are ignored. Prefer repository-root-relative entries so the same list has one unambiguous canonical path across commands.

To create larger dataset splits, add a new text file in this directory and list one episode path per line. Empty lines and lines beginning with `#` are ignored by the scripts. Keep train, validation, and all-data lists separate so normalization, training, and evaluation can be reproduced exactly.

The current `peg_hole_100_{train80,val10,test10}.txt` files were generated from
`outputs/peg_hole_100/all100.txt` with seed `20260701`. Recreate an equivalent
episode-level split with:

```bash
PYTHONPATH=src python scripts/split_episode_list.py \
  --input outputs/peg_hole_100/all100.txt \
  --train-output configs/splits/peg_hole_100_train80.txt \
  --val-output configs/splits/peg_hole_100_val10.txt \
  --test-output configs/splits/peg_hole_100_test10.txt \
  --train-count 80 --val-count 10 --seed 20260701
```

Normalization statistics must be computed from the train split only. When a
validation list is supplied, the trainers reject train/validation overlap and
reject stats whose recorded `episode_paths` do not exactly match the train
split.

`split_episode_list.py` sorts the unique input entries before applying its seeded shuffle, then sorts each output split and writes a provenance header. It rejects duplicates and requires at least one held-out test episode. It does not stratify by operator, collection batch, target, or contact regime; perform a group-aware split separately when those correlations matter.
