# FluidiTy numerical reference

This package provides independent float64 reference calculations on a uniform,
complete-support 2D MAC grid, with selected independent 3D algebraic fixtures. It is not the production simulator; production/CUDA differential gates and full physical scenes remain outstanding.

Use **uv 0.13.7 or newer** and Python 3.13. From the repository root:

```sh
uv sync --project reference --locked
uv run --project reference --locked pytest reference/tests tools/tests -q
uv run --project reference --locked fluidity-reference export reference/artifacts/fixtures
```

`reference/.python-version` selects Python 3.13. `pyproject.toml` declares the
runtime and development dependencies; `uv.lock` is the dependency lock and
contains published artifact hashes for multiple platforms. The default `dev`
group includes pytest and the pinned editable-build tools. Update dependencies
with uv and commit the resulting lockfile; use `--locked` for verification so
stale metadata fails rather than silently changing the lock.

Face arrays are ordered x faces first (row-major `(ny, nx + 1)`), then y faces (row-major
`(ny + 1, nx)`). Stencil positions are relative to the grid origin.

`fluidity-reference export DIRECTORY [--overwrite]` writes four dimensionless
manufactured examples and a manifest containing SHA-256 hashes of the exact
canonical UTF-8 fixture bytes. Existing export files require `--overwrite`.
Each schema-version 1 array records its dtype, shape, nested data and units;
`write_fixture(path, name, arrays, metadata)` requires `metadata['units']` to
map every array name to a nonempty unit string. Nonfinite numbers are rejected.
The two-cell pressure fixture records the default infinite bulk modulus as a
policy string rather than as a JSON number. Fixtures are independent hand
calculations for the reference operators, not a full simulation or CUDA proof.

`ReferenceState` copies arrays into immutable snapshots. `trial_state` calls
an update with mutable array and counter copies; `accept_trial` invokes the
caller-supplied validator and returns a fresh snapshot or raises `TrialRejected`.
This is a reference transaction model, not GPU rollback implementation.
