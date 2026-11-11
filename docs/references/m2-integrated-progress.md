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

## Accepted whitewater controls

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

The resolved analytic-sphere control now publishes births from actual accepted
CPU and CUDA Simulate states. It samples occupied volume on a fixed physical
grid at each accepted endpoint, uses analytic curvature and the checked carrier,
and journals events independently of the carrier sample rate. The
[source receipt](m2-whitewater-source-control-data.json) records 24 births from
each backend with two retained carrier samples and earlier births at step 1 of
2. Both process tests verify exact read-only replay. The reported process wall
times include startup and publication; they are not isolated sampler timings
or a CPU/device numerical parity claim.

The host frame control reconstructs nonempty persistent foam from those
accepted events. Focused controls cover birth-time spray, bounded RK2 foam,
explicit bubble fallback and foam expiry before later carrier reads. `FLWW`
checkpoints and `FLWC` physical settings round-trip with bounded codecs.
`run --stage whitewater` now publishes those points under exact source lineage,
with read-only replay and verified checkpoint resume. Real process tests cover
CPU and CUDA accepted sources, unchanged upstream bytes/timestamps, and corrupt
point rejection. Whitewater transport itself runs on the host. The CPU analytic
control had 40 foam and 40 bubble points at frame 2; that frame took 0.033 s
end to end and the process peak RSS was 6,676,480 bytes. This is one small
control, not a breaking-wave throughput measurement. The full suites passed
572 CPU and 640 CUDA-enabled tests with no skips.

## Blender foam projection control

The analytic control's Mesh and Whitewater caches share the exact Simulate
source identity. The locked Alembic bridge exports the existing surface and
foam points to separate native carriers. The signed-volume test caught
Blender's import winding reversal; the bridge now compensates at export.
The native M1 point export remained byte-identical in a three-frame control.

Blender 5.2.3 LTS (`e13b728c6880`) read 0, 16 and 40 baked foam points across
the three frames. The named `fluidity_foam_mask` was exactly zero on the empty
frame and matched an independent nearest-point calculation within `1.5e-7`
on the moving frames. Radius, strength and material edits changed the mask
and left each frame's vertex/index bytes, published cache files/timestamps
and execution IDs unchanged. Two 64×64 CPU Cycles renders of the same surface
with mask off/on differed by up to 0.424 in a pixel channel. The relocated
`.blend` and its two relative Alembic paths reopened in a fresh process with
the extension and solver absent and no frame handlers.

[Blender receipt](m2-blender-foam-control-data.json) records geometry and
archive hashes, foam counts, mask comparisons and render response. Run
`tools/blender/verify_m2_foam_projection.py` in the pinned Blender with
`--bridge`, `--mesh-cache`, `--whitewater-cache` and a new `--output` directory;
the control uses the CPU `whitewater-effects` fixture through frame 2 and
Mesh spacing 0.125 m / sigma 0.25 m. The same script's `--reopen` mode checks
the saved blend without importing the extension. This qualifies the mask
contract for the analytic control. The integrated breaking-wave bake and
artist workflow inspection are separate and remain open.

## Conservative volume control

`run --stage volumes` reads authenticated Whitewater points and publishes
conservative foam and bubble densities in a declared box. Each represented
bubble contributes its physical gas volume. Spray contributes neither density.
The payload stores the validity mask separately from sparse nonzero values,
so a known empty frame stays distinct from unknown space. Deposition rejects
positive support outside the box.

The CPU analytic control used a 24³ box with 0.25 m spacing and a 0.05 m
physical bubble radius. Frame 0 had 13,824 known voxels and no nonzero values.
Frame 2 had 24 nonzero voxels, inventories of 40 foam and 40 bubble equivalents,
and 0.020944 m³ of gas. Its 14,752-byte payload took 0.026 s to produce, with
5.81 MB peak process RSS. Read-only replay reproduced its exact digest in
0.0031 s. These are small control measurements. Blender volume export and
rendering are still open.

## Ocean Boundary analytical controls

The host swell model resolves the finite-depth dispersion root once, then
samples second-order Stokes elevation and velocity in world coordinates.
The uniform return current cancels the depth-integrated Stokes drift to
second order. The model rejects settings where the second elevation harmonic
exceeds a quarter of the first. The separate absorption primitive uses an
exponential update with a cubic spatial ramp.

The tests compare against direct finite-depth formulas at the bed and in the
water column, exercise deep water without overflowing the hyperbolic profiles,
and check that two half steps match a full step for a frozen relaxation target.
The reference inlet control integrates the mean return flow and independent
Stokes drift over a 3 m water column for a 1 m wave and 4 s period.

The solver consumes the resolved settings in its Waves stage after the old
velocity snapshot and acceleration. The low-x zone relaxes toward the swell
and the high-x zone toward rest. Both backends use the same allocation-free
host relaxation on the existing predicted velocity staging. Open faces
requiring continuation receive relaxation after continuation is computed,
before pressure assembly. The optional scene table is bound to the existing
cache identity.

A ten-frame analytic solver fixture exercised 20 accepted steps on CPU and
CUDA. Final particle positions and velocities matched across backends at
17-digit output precision. Relative to the unforced CPU control, forcing
changed position by 0.099989 m and velocity by 0.415088 m/s. All 7 cell keys
and 768 face keys matched and their pressure and velocity matched at the
FlatBuffers JSON export precision. Both caches replayed frame 9 read-only.
This fixture has one particle and exercises the inlet velocity path.

Optional inlet level control fills vacant wet sites on a bounded world
lattice, retires particles above the prescribed surface in the inlet strip,
and seeds new particles at the analytical velocity. The site volume is the
cube of the authored spacing. Trial particle changes and IDs are discarded
on rejection. CPU and CUDA produced identical calm inlet checkpoints through
frame 10, with two new particles and no duplicate births. A moving-swell
CPU test checked dry particle retirement and exact rollback after the inlet
edit. The reference wave still needs source volume and level measurements.

The production outlet relaxation was also tested in a staggered linear
shallow-water flume. At a gauge 4 m before the start of the 6 m absorption
zone, the first returning pulse carried 19.49% of incident energy at 2/s
relaxation, compared with 100% against a closed wall. The 0.05 m grid used a
0.00625 s timestep and a 1 m still-water depth. This isolates outlet
reflection from breaking, pressure projection and particle transport.

## Stationary beach contact control

Mesh clips the density field against an inclined stationary halfspace,
protects contact vertices during bounded volume correction, and removes
inward continued-carrier velocity near the plane during tracking. Whitewater
samples the accepted particle sphere union and clips source probes against
the same halfspace. Transport clamps beach contact and removes inward
velocity. CPU and CUDA process tests cover Simulate, Mesh, Whitewater and
Volumes for the control, including exact read-only Whitewater replay.

The three-frame CPU bake retained 813 vertex IDs on both tracked frames,
with maximum relative volume error 0.259% and frame time 0.228–0.416 s.
Whitewater frame 2 held 16 foam points and 48 bubbles and took 0.021 s.
Blender 5.2.3 imported the resulting 0, 10 and 16 foam points and matched the
nearest-point mask oracle within `7.9e-7`. Projection and material changes
preserved vertex and index hashes and cache identity. Rendered mask changes
reached 0.416 in one pixel channel. Relocated native caches reopened without
the extension. The imported signed volumes agreed with the published Mesh
volumes within `2e-8` m³. See the [Blender beach control receipt](m2-blender-beach-control-data.json).

## Open integrated gates

The three-frame stationary beach control uses a single liquid particle.
Wave-scale source sampling, wetting and topology transitions, droplet
publication and outlet reflection in the particle solver still need
qualification. Blender volume export and the full breaking-wave workflow
remain open. The breaking-wave acceptance bake and inspection have not run.

The uniform M1 execution plan is still bounded to 1,024 particles, 128 blocks,
4,096 cells and 24,576 faces. Its small Blender bake does not demonstrate the
canonical full breaking-wave workload. The M1 piston receiving-volume finding
also remains open. No neural dependency is promoted; M3 adaptivity and
simulation narrow-band changes remain excluded.
