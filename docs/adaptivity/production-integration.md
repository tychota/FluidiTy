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
   corners before preparing a family candidate. Qualify interrupted versus
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

The current bounded handoff accepts explicit one-leaf retirement and reseed
requests. Automatic placement, overlap blending, conservative nonzero
cross-owner exchange, and remap of mixed owners each require an independent
qualification before the wave.

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

### Full wave

- Pin the executable, scene and initial-state digests, device and executed architecture, driver, seed, cache/checkpoint policy, accepted frame range and comparison reference. Establish fixed surface, volume, cavity, trajectory, divergence and source-balance error thresholds before measuring performance. Run uniform and adaptive at the same error target, record achieved errors and qualify the short trajectories before promoting either timing.
- Record unprofiled launch-to-exit time including every requested stage and pipeline drain, published-frame intervals, retries, active levels, remap work and simultaneous peak GPU bytes. Include old and new owners, graph/solver scratch, stage buffers and in-flight snapshots in the memory measurement. Use a matched short capture for profiling, separate from the acceptance run. [CUDA optimization procedure](../native/cuda-optimization.md)
- Run all 240 frames of the reference wave. Require at most 10,800 s and GPU peak at most `min(0.70 × total, 0.80 × free_at_job_start)`. The temporary uniform-sparse narrow-band path may qualify at most 21,600 s under the same cap and leaves adaptivity open. Qualify cage/native Multires when the authored reference output requests it.
