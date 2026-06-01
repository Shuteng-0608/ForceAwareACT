# Dataset Split Files

Split files contain one HDF5 episode path per line. Paths are relative to the project root, for example:

```text
peg_in_hole_hdf5/20260601_153253_teleop_001/episode.hdf5
```

The HDF5 files themselves are local data artifacts and are not committed to git.

To create larger dataset splits, add a new text file in this directory and list one episode path per line. Empty lines and lines beginning with `#` are ignored by the scripts. Keep train, validation, and all-data lists separate so normalization, training, and evaluation can be reproduced exactly.
