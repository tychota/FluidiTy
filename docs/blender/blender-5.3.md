# Blender 5.3 compatibility and grid nodes

Blender 5.3.0 was released on November 10, 2026. The Windows x64 portable executable reports build `02f0899d78e3`. Existing 5.2.3 LTS receipts identify the build used for those measurements. Record the package builder and the runtime build independently. [Official release](https://www.blender.org/releases/5-3/)

| Upstream checkout | Revision inspected |
| --- | --- |
| [Blender source](https://projects.blender.org/blender/blender.git), tag `v5.3.0` | `02f0899d78e374b63e7181d9061e1f7ab03419cc` |
| [Blender manual](https://projects.blender.org/blender/blender-manual.git) | `3481be3f6163a7c34e317b685f09fe67ca1057f0` |
| [Blender developer docs](https://projects.blender.org/blender/blender-developer-docs.git) | `bc46fdf42f1389f8f4c13d2c1d589cf98706e798` |

The portable build registered all six grid node types listed below. The same binary lacked `UILayout.label_multiline` and `label_markdown` in its Python API. Keep help rendering compatible with the runtime API. [Geometry Nodes release notes](https://developer.blender.org/docs/release_notes/5.3/geometry_nodes/#grids) · [UI release notes](https://developer.blender.org/docs/release_notes/5.3/user_interface/)

| Node | FluidiTy application and qualification |
| --- | --- |
| Rasterize Points | Build optional visualization grids from committed points. Rasterize weights and weighted velocity separately, then divide where weight is nonzero. An accumulated point weight needs an explicit normalization and voxel-volume convention before it can be called density. Check point coordinate space, voxel transform, kernel, empty inputs and peak memory. |
| Sample Grid | Sample diagnostic grids back onto points or meshes. Compare interpolation modes on a known field under a nonidentity transform. Include background and sparse boundaries. |
| Get Grid Names | Discover channels in imported volumes. Validate the expected channel name, grid type and transform before binding an asset. |
| Grid Topology Boolean and Deactivate Voxels | Combine or trim active voxel support for display. Compare support and background values after masking. Budget memory across refinement transitions and keep cache data authoritative. |
| Poisson Solver | Evaluate as a node-side experiment with a manufactured solution and measured residual. FluidiTy pressure uses its own discrete operators, boundary conditions and convergence criteria. A substitution would need independent numerical qualification. |

The Blender grid path is an optional derived view of committed output. For M3, check stable physical units and channel meaning across coarse/fine transitions, empty frames and a relocated reopen. Measure time and peak memory for the reference scene before enabling an asset by default. For final release, repeat this on every advertised Blender build and verify the project still renders from its saved native carriers with the extension disabled. See the [milestone specification](../superpowers/specs/2026-11-07-fluidity-design.md#152-milestone-capabilities-not-an-implementation-plan).

## Windows x64 evaluation receipt

The 5.3.0 build created and validated an offline ZIP with a 5.3.0 minimum version. The existing packager also accepts the pinned 5.2.3 LTS builder and writes that build's minimum version for its own package.

- The installed 5.3 package passed CUDA preview, bake, cancellation, resume, incompatible resume rejection, inspection and relinking. A fresh process with the extension disabled rendered two distinct frames after relocation.
- The installed Water Tank example passed CPU tank, free-space, plane and slab bakes, initial velocity checks and changed-material resume rejection.
- The committed Alembic/USD round trip passed point-value, no-overwrite, scene-state and external Geometry Nodes field checks.
- English and French setup and recovery panels rendered at 424 pixels with the preset selector showing the applied material.
- A two-point fixture used Rasterize Points with one scalar weight of 2 per point and a 1-unit voxel size. Sample Grid returned `[2.0, 2.0]` at the input points. This covers scalar wiring at the default transform and kernel. Sparse boundaries, vector channels, nonidentity transforms and solver equivalence remain qualification work.

Local package and receipts are in `out/blender-53-polish`, `out/blender-53-polish-check`, `out/blender-53-polish-domain` and `out/blender-53-polish-ui`. The upstream checkouts and grid fixture are under `out/blender-upstream`.
