# FluidiTy numerical reference

This package provides independent float64 reference calculations on a uniform,
complete-support 2D MAC grid. It is not the production simulator.

Use Python 3.13 and install the pinned, hash-checked runtime, test, and build
dependencies into an isolated environment before installing the local package:

```sh
python3.13 -m venv reference/.venv
reference/.venv/bin/python -m pip install --require-hashes -r reference/requirements.lock
reference/.venv/bin/python -m pip install --no-build-isolation --no-deps -e reference
reference/.venv/bin/python -m pytest reference/tests -q
```

The lock contains hashes for published artifacts on multiple platforms. The
editable installation uses the already pinned build backend. Face arrays are
ordered x faces first (row-major `(ny, nx + 1)`), then y faces (row-major
`(ny + 1, nx)`). Stencil positions are relative to the grid origin.
