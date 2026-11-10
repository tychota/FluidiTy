# M2 integrated implementation evidence

This is a progress receipt, not an M2 completion certificate. The canonical
[M2 gates](../superpowers/plans/2026-11-10-m2-surfaces-and-wave-effects.md)
remain binding. Fixed-cache research results are recorded separately in
[geometric evidence](m2-fixed-cache-evidence.md),
[neural evidence](m2-zhao-cpu-evidence.md), and the MotionInterval study.

## Verified native foundations

On 2026-11-11, the Windows Release CPU build passed all 514 CTests. Native
formatting passed with clang-format 19.1.5. A fresh CUDA build targeting
sm_120 with CUDA 13.4.59 passed 27 focused M2 tests, including actual CUDA
source-cache publication, read-only replay, extension by resume, unchanged
earlier publications, and corrupted-source rejection. Device: RTX 5060 Ti,
16,311 MiB reported by nvidia-smi, driver 610.88.

- `EffectsCache3D` authenticates separate source, Mesh, Whitewater and Volumes
  identities and contiguous payload lineage, using no-replace publication.
- `MotionCapture3D` stores bounded accepted full-state carrier samples and
  discards provisional capture on rollback. Geometry and endpoint-state
  substitution regressions were observed failing and then passing.
- `FrozenMotionCarrier3D` uses the original quadratic stencil or declared
  same-side continuation. Missing support fails; it does not become zero.
- The native density reference conservatively splats material volume and
  applies a normalized Gaussian. Conservation, an independent single-Gaussian
  oracle, analytic trilinear gradients and clipped-support rejection passed.
- The PL extractor and mesh checker cover closed vertex links, edge winding,
  disconnected crossings and overlap beyond shared vertices/edges. A sphere
  satisfies the unit-test topology and measured vertex-correction controls.
  Infeasible pinned correction preserves the input geometry.
- Same-component RK2 tracking preserves IDs/connectivity in a uniform-motion
  fixture and rejects missing support, component changes and failed bounded
  projection. This is not the drop-to-pool or breaking-wave correspondence gate.
- The `FLMS` checkpoint codec preserves persistent attributes and rejects
  inconsistent measured per-component volumes, mixed-component triangles,
  corruption and record budgets. It enforces the 0.5% material-volume threshold
  on encoded component geometry.
- Whitewater host foundations cover exact frozen drag at zero/tiny/stiff rates,
  subdivision, half-radius classification hysteresis, distinct opposing birth
  events, fractional emission state, and the declared fixed-grid source formula.

The geometric reconstruction, tracking, source statistics and whitewater
dynamics described here execute on the CPU. Linking and testing them in a
CUDA-enabled build does not qualify a device implementation.

## Open integrated gates

The accepted observer does not yet publish whitewater source events. Mesh stage
orchestration, occupancy-qualified projection, local repair, droplet handling
and production replay are unfinished. Whitewater transport/publication,
conservative volume exports, Ocean Boundary and the Blender foam workflow are
unfinished. No integrated breaking-wave bake or Blender M2 inspection has
qualified them. New M2-specific Lean contracts and full CPU/device numerical
qualification remain open.

The uniform M1 execution plan is still bounded to 1,024 particles, 128 blocks,
4,096 cells and 24,576 faces. Its small Blender bake does not demonstrate the
canonical full breaking-wave workload. The M1 piston receiving-volume finding
also remains open. No neural dependency is promoted; M3 adaptivity and
simulation narrow-band changes remain excluded.
