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
manufactured examples, two SI 3D fixtures, and a manifest containing SHA-256 hashes of the exact
canonical UTF-8 fixture bytes. Existing export files require `--overwrite`.
Each schema-version 1 array records its dtype, shape, nested data and units;
`write_fixture(path, name, arrays, metadata)` requires `metadata['units']` to
map every array name to a nonempty unit string. Nonfinite numbers are rejected.
The two-cell pressure fixture records the default infinite bulk modulus as a
policy string. The fixtures provide independent calculations for the reference
operators.

## Integrated 3D MAC pressure fixture

`uniform_mac_pressure_3d.json` uses SI quantities and unit-width cells.
Metadata lists the cases in `case_names`. For N cases, `step_durations` has
shape `(N,)` in seconds, and `case_cell_offsets`, `case_face_offsets`, and
`case_gauge_offsets` have shape `(N + 1,)`. Consecutive offsets give the
start and exclusive end of each case's rows in the corresponding array.

- For C cell rows, `cell_indices` has shape `(C, 3)` and contains signed
  `(x, y, z)` indices sorted by `(z, y, x)` within each case. All other
  cell arrays have shape `(C,)`. Fluid volumes use m³, target divergences and
  constraint defects use s⁻¹, and integrated RHS and true residuals use m³/s.
  Bulk moduli and previous and expected pressures use Pa. Decode a zero
  bulk modulus as positive infinity to select the incompressible limit.
- For F face rows, `face_indices` has shape `(F, 4)` and contains
  `(normal, x, y, z)`, with normals 0, 1, and 2 for x, y, and z. A face
  points from its low-index cell to the head at the recorded lattice point.
  Rows follow owned slots, ordered by block z/y/x, normal, then local z/y/x.
  All other face arrays have shape `(F,)`. Open areas use m², inverse
  densities use m³/kg, pressure gradient distances use m, and predicted,
  wall, and corrected velocities use m/s. Link kinds 0, 1, and 2 denote
  sealed, internal, and Dirichlet links. The pressure lift in Pa holds
  zero, the oriented jump, or the exterior pressure, respectively.
- `expected_gauge_cell_ordinals` has shape `(G,)` for G gauge rows.
  Each value is a zero-based cell ordinal within its case. Offsets,
  indices, link kinds, and gauge ordinals are dimensionless.

The reference solves a dense integrated system in float64. It evaluates
the physical constraint defect separately by summing corrected fluid and
prescribed wall flux through each cell's faces, adding the compliance
term and subtracting the source before dividing by fluid volume.

## State transactions

`ReferenceState` copies arrays into immutable snapshots. `trial_state` calls
an update with mutable array and counter copies; `accept_trial` invokes the
caller-supplied validator and returns a fresh snapshot or raises `TrialRejected`.
This is a reference transaction model, not GPU rollback implementation.
