# M3 production integration and wave qualification

## Entry evidence

The bounded 3D qualification covers aligned cubic-normal MAC transfer, the native pressure projection, signed subface ownership, masked PIC, and the first refine/coarsen family remap. The manufactured band tests cover rollback and cavity preservation. Use the measured edge/corner cases and the separately checked F3/F4 contracts as regression controls as these pieces enter the production path.

The bounded M2 column-seed control published 49 Simulate frames on CPU and
SM120. The fixed-cache Mesh continuation through frame 48 reproduced the native
payloads in independent read-only runs. Blender reopened that 49-frame sequence
with foam and volumes attached. The fresh integrated bake, its replay, and the
piston receiving-volume gate remain open. The fixed-cache result supplies a
moving-surface control for M3 without changing the active M2 work.

## Native adaptive Simulate

The [graded Simulate receipt](graded-simulate-runtime.md) covers the authored
stationary tank, graded transfer, physical projection, and accepted device and
checkpoint publication. Explicit family refinement and coarsening now prepare
the new topology and fluxes from the previous accepted epoch and freeze them
for the entire attempt. The family adapter carries open exterior apertures and
signed outward flux into the sibling divergence balance. CPU and real SM120
tests reject a late trial and reproduce the same checkpoint and final state in
a fresh process. The mixed-owner band uses the same acceptance boundary.
[Family transaction](https://github.com/tychota/FluidiTy/pull/108) ·
[Mixed-owner runtime](native-band-runtime.md) ·
[Exact pressure graph and ownership](adaptive-pressure-graph-inputs.md)

1. Compute the actual support and movement halo from accepted positions and
   carrier data. Feed the physical, interface, support and camera upper bounds
   to the width policy. Persist the eligible seconds with the accepted state,
   reset them on veto, and apply deterministic 2:1 closure at faces, edges and
   corners before preparing a family candidate. The frozen topology now
   computes the minimum refinement closure in canonical leaf order and checks
   the proposed cell count before remap. The bounded scheduler can apply the
   complete refinement closure in one trial when each family can be exactly
   rebinned.
   Qualify interrupted versus
   uninterrupted policy decisions and ensure a rejected trial cannot earn age.
2. Extend remap from exact particle rebinning to certified split and merge.
   Preserve mass, reference and current volumes, momentum, constituent inventory
   and enthalpy. Preserve the signed coarse/fine face flux, pressure warm start
   and source or wall contributions over the remapped control volume. Use the
   previous accepted topology and reserve both owners before publication.
3. Run the exact F3 split and F4 merge modules and their axiom audit separately
   from native tests. The F3 covariance and cubic-normal deficit and the F4
   per-row covariance and energy gates are stated for their declared transfer
   support. Require complete unmasked support with the theorem's actual moments
   before using the uniform bridge. The current graded radial weights require
   a separate moment witness and measured angular/energy budget on a transition.
   Refuse a split or merge when either witness is absent. Test that refusal
   preserves IDs, pressure history, source/RNG counters and the accepted clock.
4. Size the leaf, canonical subface and projection reservations from actual
   scene occupancy. The current wire and solver path caps at 4,096 leaves and
   24,576 faces. The indexed transfer improved manufactured gather from
   52.36 ms to 21.08 ms and deposit from 80.01 ms to 32.34 ms on 2,592 leaves,
   8,400 faces and 1,024 particles with identical output. It adds 50,352 bytes
   to the frozen owner in that fixture. [Transfer index](https://github.com/tychota/FluidiTy/pull/114)

### Actual graded support moments

`GradedMacTransfer3D::MeasureSupportMoments` measures partition, weighted mean and centered covariance from the canonical radial gather support. The same affine pivot test reports rank for each velocity component. The witness also records positive geometric support removed by masks and support at other face widths.

At a cell center away from boundaries and width transitions, the measured radial stencil has 28 positive samples per velocity row. In leaf-width-squared units, the binary64 normal and transverse covariances agree with `167/396` and `76/165` within `1e-15`. The affine field reproduces despite the mismatch with the quadratic variance `1/4` and cubic-normal variance `1/3`. A signed coarse/fine fixture also reproduces its affine field and reports the width transition.

`CheckUniformResamplingSupport` refuses the radial stencil, including an accidental binary64 moment match, because uniform split and merge require the complete unmasked tensor support. The checked witness leaves split and merge disabled.

### Checked same-level quadratic transfer

`GradedQuadraticMacTransfer3D` binds the existing FP64 quadratic geometry directly to canonical adaptive face slots. Every tensor node, including a zero-weight tie, requires an interior face with both incident leaves at the particle level, the exact represented anchor displacement, full aperture, and valid same-side carriers. Deposit and gather use the bound rows with the uniform operator's arithmetic. The signed island control matches the uniform sparse operator bit-for-bit for mass, momentum, PIC velocity, FLIP increment, affine slope and gradient. It refuses a mixed-level face and mask, side, low-mass and aperture changes.

The paired quadratic resampling bridge checks the old and proposed canonical
topologies and aperture/carrier banks at each particle's physical level. F3
checks the parent on the old coarse bank and each of eight children on the
new fine bank. F4 checks the inverse support direction and requires a strictly
positive, outward-rounded covariance gap. Geometry and material admissibility
are also required from the caller. The bridge publishes detached proposals.

A signed manufactured case refines a central 3 × 3 × 3 patch inside a
6 × 6 × 6 coarse island (216 old leaves and 405 proposed leaves). It passes
F3 with zero quadratic energy deficit and absolute momentum error within
1e-15 kg·m/s, then passes F4 with separately placed fine children. Wrong-level
support and a missing accepted carrier bank are refused. The production
transaction still needs to stage the entire halo before changing particle IDs
and freeze the same transfer selection through P2G and both G2P histories.

```powershell
cmake --build out/build/resampling-pair-cpu --target fluidity_native_tests --parallel 2
ctest --test-dir out/build/resampling-pair-cpu -R 'GradedQuadraticMacTransfer3D|LiquidUniform(Split|Merge)3D|GradedMacTransfer3D' --output-on-failure
```

### Detached conservative family resampling

The conforming tank candidate stages a full refinement or coarsening halo
before applying a quadratic F3 split or F4 merge. Pressure and signed subface
flux use the same remap as exact particle rebinning. The replacement roster
must reproduce every remapped leaf inventory exactly in the stable particle
order. Failed support and covariance checks leave the accepted roster,
topology and next ID byte-identical.

The signed native case stages a 125-to-314-leaf refinement and replaces one
particle by eight. With dyadic child positions and a positive F4 covariance
gap, the coarsening halo restores the original 125-leaf graph and face
history, with one fresh ID for the merged particle. A lone family without the
required halo is refused in either direction. Duplicate child IDs fail
preflight. The proposal is detached from accepted scheduler publication.

### Certified family trials

The explicit split and merge scheduler requests bind a committed epoch,
the complete parent closure and the source particle IDs. The detached
candidate owner freezes its replacement IDs and actual quadratic face rows
before the physical stages. The split stages eight replacement particles
and the merge stages one. A missing or ineligible row refuses the entire
request. The old and projected grid snapshots use the same rows.

The interrupted replay fixtures start from independent coarse and authored
fine dyadic seeds. Each saves an accepted checkpoint before the certified
operation, kills the process, and reapplies the request after restore. A late
fault checks that the committed host and device versions retain the previous
layout and IDs. The resumed request also reverses its input order to check
that the accepted result follows canonical particle order.

Certified replacement IDs publish with the quadratic basis. An ordinary
attempt on the replacement roster then uses the accepted policy described
below. A resampling source with nonzero active affine slopes must already
carry an accepted quadratic basis. This keeps the measured support
covariance aligned with the source affine data.

### Accepted transfer basis in FLST

An accepted particle may carry the quadratic adaptive basis flag. FLST v5
stores only marked particle IDs in stable roster order inside the adaptive
state table. The reader requires a nonempty, duplicate-free subsequence of
the decoded particle roster. The field is absent in v1–v4, where the flag
defaults to false. Ownership and width ages remain independently valid v5
fields. Uniform states cannot carry the adaptive flag.

The CUDA arena stores the flag beside the affine and PIC bits in its particle
bank. Preparing and discarding a trial preserves the old flag and epoch.
Publishing the trial switches the flag with the particle and topology state.
Ordinary attempts rebuild the marked support from the accepted roster and
freeze it through deposit and both gather histories. A changed topology
clears incompatible existing marks before a replacement trial. If the
stored roster loses complete accepted support, the trial resets its
marked slopes and PIC history before radial deposit. Rejection retains the
previous accepted marks. A single explicit promotion stages zero old
affine slopes when changing from radial to quadratic. Scratch is reserved
from the admitted owner extent. The eight-ID bound applies to one explicit
split or merge request.

The persistent fixture publishes a v5 checkpoint after a real promotion,
terminates the process, and advances the restored state with ordinary
attempts. It compares the completed state against uninterrupted execution
and checks the real CUDA device bank. The CPU controls also exercise
pretrial support loss, late rejection of the reset, and a 16-particle stored
roster.
The CPU and SM120 runs reproduced checkpoint BLAKE3
`a4e42dd663461a1bfbf67d117e7d518bbaf52d03abc59ee9681661d69cb2d79c`
and completed state BLAKE3
`3467cd43610645e6bcf48f8630a0e2cead97e9e19a3e1a88d0f385e887d8b039`.


```powershell
ctest --test-dir out/build/persistent-green-cpu -R 'GradedPersistentQuadraticBasis|quadratic-persistent' --output-on-failure
ctest --test-dir out/build/persistent-green-cuda -R 'GradedExplicitQuadraticRejectionPreservesDeviceEpoch|quadratic-persistent' --output-on-failure
```

```powershell
ctest --test-dir out/build/basis-green-cpu -R 'LiquidCache3D|LiquidState3D|Schema|Adaptive(Family|Band|Width|Closure|Physical)Replay.cpu' --output-on-failure
ctest --test-dir out/build/basis-green-cuda -R 'Adaptive(Family|Band|Width|Closure|Physical)Replay.cuda' --output-on-failure
& ./out/build/basis-green-cuda/fluidity_cuda_arena_tests.exe
```

## Production band and M2 boundary

Build the surface band from Simulate occupancy and its signed physical interface, with support, accepted movement and reseeding margins. Measure the 6h depth and 2h overlap on manufactured and wave fixtures. Blend valid particle and Eulerian carriers before the old-grid snapshot and enforce single ownership of extensive material. Reseed fully liquid cells with the checked density and EOS contract. Refuse unavailable carrier or occupancy.

The v3 motion codec carries accepted Eulerian ownership and inventory with the
canonical subfaces. The native occupancy fixture reconstructs a manufactured
false-inner-shell and real-cavity control from a decoded completed motion
endpoint. The mapper requires supported particle samples at all eight nodes
of every crossing cube. Particle influence alone cannot certify complete
support. Mesh still needs to compute this support and obstacle closure on its
extraction lattice and use the component inventory/flux ledger from the same
accepted epoch for volume targets. Then qualify moving correspondence,
component and cavity lineage, exact replay and rollback on the M2 accepted
source. Verify Whitewater and Volumes use the same authenticated stage
identities. [Occupancy and motion fixture](native-band-occupancy.md)

The bounded handoff accepts an epoch-bound placement request for one leaf
using the 6h depth, 2h carrier overlap, and measured sweep margins. The
native planar Mesh path consumes accepted mixed-owner volume. The explicit
owner-exchange attempt qualifies nonzero projected flux with a same-leaf
stationary contact witness and exact represented inventories. Moving
overlap blending, nonplanar authoritative extraction, and mixed-owner
family remap still require qualification.

## Matched-error wave campaign

The manufactured sealed 8 m³ pressure control passed a common 1e-4 Pa pressure
and 2e-5 s⁻¹ corrected divergence limit at 512 uniform cells and 176 graded
cells. Measured maximum divergence was 5.9605e-6 s⁻¹ uniform and 8.9010e-6 s⁻¹
graded. Reproduce this bounded control with the native
`fluidity_m3_matched_pressure` executable. The 49-frame uniform M2 Simulate
control reached 412 particles, 57 active blocks, 380 cells and 10,944 packed
face slots. It reported 172.133 s CPU and 1,011.106 s SM120 Simulate elapsed
time and 113,546,240 B CUDA arena peak. Those figures cover the bounded
Simulate run and its arena. The full-wave time and simultaneous GPU memory
remain unmeasured.


### Manufactured CUDA pressure control

The same sealed 8 m³ graph and error thresholds now run on SM120. The nine
samples per layout alternate execution order after one warm-up each. Both
backends produced the same measured pressure and divergence maxima for each
layout. The Release wall and CUDA event times below were observed on the
qualification machine while other builds shared the host.

| Backend | Layout | Cells | Pressure error (Pa) | Divergence error (s⁻¹) | Median call (ms) | Median CUDA solver (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | Finest uniform | 512 | 1.1921e-6 | 5.9605e-6 | 0.754 | |
| CPU | Graded | 176 | 1.0729e-6 | 8.9010e-6 | 0.216 | |
| SM120 | Finest uniform | 512 | 1.1921e-6 | 5.9605e-6 | 53.760 | 28.985 |
| SM120 | Graded | 176 | 1.0729e-6 | 8.9010e-6 | 30.762 | 7.550 |

The CUDA event interval covers the solver stream between warm-start evaluation
and final status synchronization. The call interval also covers hierarchy
preparation, device allocation, upload, download and cleanup. This bounded
control records CPU workspace bytes and executed GPU architecture. The job-wide
GPU memory check requires the simultaneous arena, solver, and pipeline peak.

```powershell
cmake --build out/build/matched-pressure-cuda --target fluidity_m3_matched_pressure
ctest --test-dir out/build/matched-pressure-cuda -V -R '^M3MatchedPressure' --no-tests=error
```

### Persistent CUDA pressure workspace

The production adaptive pressure stage uploads the frozen graph into reserved
arena ranges and uses the bounded pressure hierarchy and result bank. A failed
solve leaves the last published result intact. The sealed-cube control can
exercise this same path with `fluidity_m3_matched_pressure --cuda-arena`.

On SM120, median stage times were 15.7926 ms for the 512-cell uniform graph and
4.6444 ms for the 176-cell graded graph at the error limits above. The allocating
CUDA control in the same run took 24.6398 ms and 8.7352 ms. Peak allocated arena
bytes were 9,360,128 and 3,396,608 for the two capacity tiers. Peak active
pressure scratch was 7,411,928 and 2,540,864 bytes. These measurements cover
the sealed-cube arena. Use the full-wave procedure below for the job-wide peak.
### Current capacity boundary

The adaptive topology and state validators admit 4,096 leaves and 24,576
canonical subfaces. At 0.035 m, 4,096 finest leaves cover at most 0.175616 m³
before coarser leaves or the face limit consume capacity. The largest
qualified arena pressure control used 4,096 rows, 13,056 faces, and four
hierarchy levels. Its maximum physical defect was 1.01548e-5 s⁻¹. The
public adaptive CLI starts with the authenticated seed's exact extents.
Native trial fixtures reserve additional particle, leaf and face capacity
explicitly. The public uniform path defaults to 1,024 particles, 128 blocks,
4,096 cells and 24,576 slots.

The 3-frame matched public tank exercises Simulate and Mesh at identical
initial particle inventories and error limits. Its short-run receipts
report zero drift, 458 retained Mesh IDs, and relative volume error
1.11022e-15. [Matched public control](matched-error-pressure.md#public-simulate-and-mesh-control)
### Full wave

- Pin the executable, scene and initial-state digests, device and executed architecture, driver, seed, cache/checkpoint policy, accepted frame range and comparison reference. Establish fixed surface, volume, cavity, trajectory, divergence and source-balance error thresholds before measuring performance. Run uniform and adaptive at the same error target, record achieved errors and qualify the short trajectories before promoting either timing.
- Record unprofiled launch-to-exit time including every requested stage and pipeline drain, published-frame intervals, retries, active levels, remap work and simultaneous peak GPU bytes. Include old and new owners, graph/solver scratch, stage buffers and in-flight snapshots in the memory measurement. Use a matched short capture for profiling, separate from the acceptance run. [CUDA optimization procedure](../native/cuda-optimization.md)
- Run all 240 frames of the reference wave. Require at most 10,800 s and GPU peak at most `min(0.70 × total, 0.80 × free_at_job_start)`. The temporary uniform-sparse narrow-band path may qualify at most 21,600 s under the same cap and leaves adaptivity open. Qualify cage/native Multires when the authored reference output requests it.


## Accepted width policy

The native scheduler can apply an epoch-bound width request to one selected
leaf per physical trial. It derives all eight siblings from the committed
signed keys. An immediately required refinement or a coarsening with enough
previously accepted eligible time prepares a replacement before the trial.
If the family remains eligible, the inactive version receives the accepted
interval credit. A veto or absence of fresh evidence clears that age on
acceptance. Rejection leaves the age and the existing topology unchanged.

The width fixture rejects late, retries at the required smaller timestep,
checkpoints eight eligible siblings in a v4 state, and coarsens on the same
next step after hard interruption. CPU and SM120 produce the same checkpoint
and final BLAKE3 values.

- Checkpoint `f9b23fa77c51d215ca7de7c40f6d47549c025aeec09a55c74e1130ab818fb4ba`
- Final state `efdee2fa2e750765d4b0c7e3a1c24e4e51e533b195ee117f89b22739987bcab5`

```powershell
ctest --test-dir out/build/width-scheduler-cpu -V -R 'GradedWidthHysteresis|AdaptiveWidthReplay.cpu' --no-tests=error
ctest --test-dir out/build/width-scheduler-cuda -V -R '^AdaptiveWidthReplay.cuda$' --no-tests=error
```

The caller still supplies the physical, camera, particle support and
movement evidence. Automatic scene evaluation, multi-family coarsening, and
resampling for families that cannot be exactly rebinned remain open.

## Atomic refinement closure

A width request can refine the selected parent and all coarser neighbors
required by 2:1 grading. The scheduler remaps the closure coarsest first into
temporary state, and publishes the resulting topology, inventory, pressure
warm starts and signed subfaces in one physical acceptance. If any family
requires resampling, the complete preparation refuses before consuming a
trial. This bounded implementation constructs temporary family candidates
and remains subject to the existing leaf and subface caps.

A five-leaf planar control requests the smaller parent next to a coarser
leaf. Both families publish together as 19 leaves. The CUDA fixture checks
the device bank immediately after a late rejection and after acceptance.
CPU and SM120 resumed the same checkpoint BLAKE3
`bae5857e532ca09ad59e773f3d23d081e1354c66f7820c0f16be24a612c989ed`
and produced final state
`14865d96ba661a0ad99e0babe3860932d4637871e2d025fd452bd4ae302b4913`.

```powershell
ctest --test-dir out/build/batch-cpu -V -R 'GradedRefinementFamilies|AdaptiveClosureReplay.cpu' --no-tests=error
ctest --test-dir out/build/batch-cuda -V -R '^AdaptiveClosureReplay.cuda$' --no-tests=error
```
