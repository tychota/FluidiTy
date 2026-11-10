# M2 integrated implementation evidence

This is a progress receipt, not an M2 completion certificate. The canonical
[M2 gates](../superpowers/plans/2026-11-10-m2-surfaces-and-wave-effects.md)
remain binding. Fixed-cache research results are recorded separately in
[geometric evidence](m2-fixed-cache-evidence.md),
[neural evidence](m2-zhao-cpu-evidence.md), and the MotionInterval study.

## Verified native foundations

On 2026-11-11, the Windows Release CPU build passed all 521 CTests, with no skips. Native
formatting passed with clang-format 19.1.5. A fresh CUDA build targeting
sm_120 with CUDA 13.4.59 passed 27 focused M2 tests, including actual CUDA
source-cache publication, read-only replay, extension by resume, unchanged
earlier publications, and corrupted-source rejection. Device: RTX 5060 Ti,
16,311 MiB reported by nvidia-smi, driver 610.88.

Subsequent verification passed all 565 native-validator Python tests, 23 M2
Python experiment/receipt tests, three effects-schema conformance tests and both
CPU/CUDA accepted-source process checks. The first cross-platform CI attempt
rejected the old exact CLI roster after the new process checks ran. The roster
and CUDA effects receipt validation were corrected through failing/passing tests.
The pre-existing Lean proof/checker job passed on `d28a7d3`; this does not qualify
new M2-specific mathematical contracts.

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
- The conservative volume rasterizer preserves foam/bubble represented counts
  and gas volume, preserves caller-supplied validity in empty frames, and rejects
  clipped or unknown positive-weight support.

## Fixed-cache native experiment

All twelve combinations of frames 0, 48, 96, 144, 192 and 216 and spacings
12.5/6.25 mm passed the isolated-drop controls. Every source points digest matched
the pinned archive. The independent Python density reference matched native raw
vertex/triangle counts and volume within 1e-14 m³. Raw geometry still has the
previously reported roughly 11% volume deficit. Correction moves actual vertices,
uses frozen interpolated field normals, backtracks unsafe geometry, and bounds
**total** displacement from the raw mesh by 0.25h = 6.25 mm.

| Spacing | Corrected volume error range | Frame-0 radial RMS / max | Median / max reconstruction, checks and correction | Peak native process RSS |
| --- | --- | --- | --- | --- |
| 12.5 mm | +0.447% to +0.490% | 0.862 / 2.100 mm | 0.590 / 0.624 s | 12,054,528 bytes |
| 6.25 mm | +0.425% to +0.440% | 0.817 / 1.781 mm | 3.538 / 3.559 s | 32,567,296 bytes |

All corrected samples retained one component, Euler characteristic 2, closed
manifold/orientation controls and zero detected intersections. The maximum actual
offsets were 5.235 and 4.874 mm respectively. Radial discrepancy is measured only
at frame zero against the authored analytic ellipsoid. Later frames have no
asserted analytic surface oracle. Timing excludes cache authentication/startup;
RSS includes the entire fresh native experiment process. These are CPU results.

The initial area-normal experiment failed at frame 144. Interpolated field normals
resolved that failure. A separate regression exposed cumulative per-iteration
offsets exceeding the spec's total bound; that test now fails correction without
modifying its input. Invalid-index validation precedes all stored-normal access.
These failures remain part of the evidence and do not justify weaker gates.

[Machine-readable receipt](m2-native-fixed-cache-data.json) pins the executable,
archive and individual source contents. Reproduce from the repository root with
the authenticated M1 archive and a built `fluidity_surface_fixed_cache_tests`:

```powershell
python -m tools.native.m2_native_surface_baseline `
  --executable out/build/m2-release-cpu/fluidity_surface_fixed_cache_tests.exe `
  --run <pinned-rtx-drop-216-directory> `
  --logs out/m2/new-fixed-cache-logs --output out/m2/new-fixed-cache-report.json
```

The driver requires new output/log paths, checks that executable bytes remain
fixed throughout the run, and rejects missing/skipped native measurements.
This qualifies the fixed-cache correction experiment, not persistent tracking,
wall contact, thin sheets, cavity behavior or the integrated breaking wave.

The geometric reconstruction, tracking, source statistics and whitewater
dynamics described here execute on the CPU. Linking and testing them in a
CUDA-enabled build does not qualify a device implementation.

## Open integrated gates

The accepted observer does not yet publish whitewater source events. Mesh stage
orchestration, occupancy-qualified projection, local repair, droplet handling
and production replay are unfinished. Whitewater transport/publication,
volume export/publication, Ocean Boundary and the Blender foam workflow are
unfinished. No integrated breaking-wave bake or Blender M2 inspection has
qualified them. New M2-specific Lean contracts and full CPU/device numerical
qualification remain open.

The uniform M1 execution plan is still bounded to 1,024 particles, 128 blocks,
4,096 cells and 24,576 faces. Its small Blender bake does not demonstrate the
canonical full breaking-wave workload. The M1 piston receiving-volume finding
also remains open. No neural dependency is promoted; M3 adaptivity and
simulation narrow-band changes remain excluded.
