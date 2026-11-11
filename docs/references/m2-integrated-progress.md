# M2 integrated implementation evidence

This is a progress receipt, not an M2 completion certificate. The canonical
[M2 gates](../superpowers/plans/2026-11-10-m2-surfaces-and-wave-effects.md)
remain binding. Fixed-cache research results are recorded separately in
[geometric evidence](m2-fixed-cache-evidence.md),
[neural evidence](m2-zhao-cpu-evidence.md), and the MotionInterval study.

## Verified native foundations

On 2026-11-11, the Windows Release CPU build passed all 529 CTests, with no skips. Native
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
The pinned Lean build and independent checker passed on `c0ec263` in
[run 42933921543](https://github.com/tychota/FluidiTy/actions/runs/42933921543),
including the four new exact trilinear weight/deposition contracts. These proofs
assume complete unmasked support; they do not certify Gaussian boundaries,
surface topology, gas physics or integrated M2 acceptance.

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

The latest clean CUDA-enabled build passed all 597 registered tests with no
skips, including the pre-existing real-device and interruption suites. The
manufactured static source fixture also compares all six CPU/CUDA accepted
carrier payloads byte for byte. This is a producer control, not a dynamic-wave
carrier error qualification. Windows Debug/Release CI passed `f723e94`; the
Linux/macOS jobs exposed strict missing-field-initializer warnings, now corrected
with explicit record defaults. A supplementary local GCC 16 MinGW syntax/warning
audit passed; qualified cross-platform CI remains required.

## Persistent frame control

`ReconstructSurfaceFrame3D` now connects full accepted source snapshots, the
geometric density reference, authoritative deep occupancy, RK2 transport,
actual-volume correction and the persistent checkpoint codec for one closed
component without contact. A deterministic triangle BVH supplies closest points
on the **already volume-qualified** scaffold. Projection distance is geometric;
Simulate's complete field remains the separate occupancy authority.

An initial raw-field reprojection implementation folded a frozen source at frame
15 because it repeatedly undid and reapplied the volume correction. The scaffold
control passes 24 calm accepted frames with identical vertex coordinates,
unchanged IDs/connectivity, increasing ages and checkpoint round trips. It checks
geometric velocity after the final correction and independently samples fluid
velocity. Small-triangle projection tolerances respect incident altitudes.
Clock substitution, source-record/sample budgets and material-volume underflow
regressions were observed failing before their fixes.

This is a manufactured one-particle control. Contact, component births/deaths,
droplets and topology events explicitly fail pending qualified adapters; this
control does not close drop-to-pool, wet-wall, cavity, thin-sheet or wave gates.

## Mesh publication and replay control

`run --stage mesh` consumes only authenticated Simulate source frames whose
parent point commits exist. The Mesh execution binds source identity, canonical
physical settings and executable bytes. Its frame manifests bind the source
payload and the prior Mesh payload. `replay --stage mesh` reconstructs the
requested prefix and compares exact bytes without changing either cache.
`run --resume` authenticates the existing prefix, restores the latest checked
Mesh checkpoint and reconstructs the target frame. A missing dependency of a
committed frame is corruption; only an absent binary manifest ends the prefix.

The three-frame stationary single-particle control was measured once at two
grid spacings with a fixed 0.25 m Gaussian scale. Wall time includes source
authentication, reconstruction, encoding and publication. Peak RSS is the
process high-water mark. Payload bytes cover all three Mesh frame blobs.

| Spacing | Vertices/frame | Wall time/frame | Peak RSS | Payload bytes |
| --- | --- | --- | --- | --- |
| 0.125 m | 866 | 0.293–0.450 s | 7,733,248 bytes | 457,608 |
| 0.0625 m | 3,830 | 0.827–1.763 s | 12,972,032 bytes | 2,022,600 |

[Measurement receipt](m2-mesh-stage-control-data.json) records executable,
fixture and resolved-scene hashes, frame timings and exact payload digests.
The copied binaries and run caches remain under `out/m2/mesh-stage-control-*`.
Both replay runs reproduced the final payload digest. The process and native
tests also check unchanged source data, settings/build isolation, cancellation
before and after manifest commit and reuse of existing bytes and timestamps.
The full local CPU and CUDA-enabled runs passed 545 and 613 tests respectively
with no skips. The validator passed 565 Python tests. These are host Mesh
controls with manufactured geometry, separate from
the historical fixed-cache experiment and the integrated breaking-wave gate.

At 0.0625 m spacing, resuming from frame 1 to frame 2 took **1.702 s** median
process wall time with checkpoint restore versus **3.981 s** rebuilding the
prefix (three trials per executable). All six runs reproduced the same target
payload digest and preserved the earlier files and timestamps. This measures
one small control; it is not a throughput forecast for the wave workload.
[Resume receipt](m2-mesh-resume-control-data.json) records individual trials
and executable hashes. The resumed CPU suite passed 547 tests with no skips.

For Simulate's particle sphere union, material partitioning now uses positive
sphere overlap before density smoothing. Mesh refuses multiple material
components and refuses to reuse vertex IDs for a coincident replacement
component with no shared particle ancestry. The component matcher retains IDs
deterministically by shared reference volume while preserving each source's
physical target volume. Focused split, merge, birth/death and 2h/3h droplet
hysteresis controls passed; these do not yet publish repaired components or
droplet points. The full CPU suite passed 557 tests with no skips.

## Open integrated gates

The accepted motion codec now stores optional whitewater births independently
of carrier decimation. Tests retain opposing events between stored carrier
samples, clear them on rollback, reject bad time/order/budgets, and verify
separate cell remainders with global point IDs. Source weights use occupied
volume at the supplied quadrature time times potential and accepted duration.
The source statistics no longer allocate a set for each cell's 26 neighbors.
Finishing the motion interval reuses retained buffers instead of copying all
carrier snapshots before serialization. A [compatibility receipt](m2-motion-birth-compat-data.json)
records six identical no-birth source payloads and point payloads across builds.
The latest full CPU suite passed 562 tests. The preceding birth-codec/source
checkpoint passed the clean CUDA-enabled suite of 629 tests.

The Simulate observer still needs the production physical source sampler to
publish nonempty whitewater events. Mesh
local repair, contact, component lineage and droplet handling remain open.
Whitewater transport/publication,
volume export/publication, Ocean Boundary and the Blender foam workflow are
unfinished. No integrated breaking-wave bake or Blender M2 inspection has
qualified them. Full CPU/device numerical qualification remains open.

The uniform M1 execution plan is still bounded to 1,024 particles, 128 blocks,
4,096 cells and 24,576 faces. Its small Blender bake does not demonstrate the
canonical full breaking-wave workload. The M1 piston receiving-volume finding
also remains open. No neural dependency is promoted; M3 adaptivity and
simulation narrow-band changes remain excluded.
