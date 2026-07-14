# Dataset Split Files

Split files contain one HDF5 episode path per line. Paths are relative to the project root, for example:

```text
mujoco_data/peg_hole_100/20260701_131034_teleop_006/episode.hdf5
```

The HDF5 files themselves are local data artifacts and are not committed to git.

To create larger dataset splits, add a new text file in this directory and list one episode path per line. Empty lines and lines beginning with `#` are ignored by the scripts. Keep train, validation, and all-data lists separate so normalization, training, and evaluation can be reproduced exactly.

The current `peg_hole_100_{train80,val10,test10}.txt` files were generated from
`outputs/peg_hole_100/all100.txt` with seed `20260701`. Recreate an equivalent
episode-level split with:

```bash
python scripts/split_episode_list.py \
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
