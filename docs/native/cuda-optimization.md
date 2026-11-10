# CUDA optimization in FluidiTy

Use this procedure for a production workload on a qualified device. The [native build and validation guide](README.md#cuda-hardware-and-sanitizer-gates) defines the toolchain and hardware gates. The [design milestones](../superpowers/specs/2026-11-07-fluidity-design.md#15-validation-milestones-and-review-boundary) define the scene and accuracy requirements. Keep the numerical comparison independent of the profiler.

## Establish the baseline

1. Pin the source revision, executable and fixture SHA-256, device and executed architecture, driver, runtime, toolchain, scene and initial-state digests. Record particle, cell and face counts, active levels, substeps, accepted timestep, precision and cache policy. Author the resolved scene for each exact executable and backend. Match the physical seed and scene settings across the comparison.
2. Choose a short, named workload for the first profile and a full scene for qualification. Include boundary work, variable support, or sparse topology when they are part of the target. Compare CPU and CUDA at equal error. For a numerical change, publish cost and error together.
3. Measure unprofiled end-to-end time from process launch to exit. Report median, p95 and maximum published-frame interval separately from solver work. Use `--trace` to sum `attempt.work_wall_seconds` per frame, including discarded attempts when measuring total attempt work. Use `--measure-job` for the host phase partition, CUDA event intervals and arena ownership. Preserve raw logs, failures, retries and replays.
4. Estimate useful operations and bytes per stage, including active face-history bytes, particle results, stencil reads, and publication. Multiply bytes per dispatch by actual dispatch count. Divide bytes by measured sustained bandwidth for a lower-bound estimate. Count fences and launches separately. Compare these estimates with Nsight before changing precision or optimizing a kernel.

## Profile and qualify an overnight batch

Run the hardware probe, verify the selected physical device is idle of competing solver work, and reserve it for the campaign. Keep baseline and candidate artifacts, caches and capture directories separate. Record the executable, arguments, start/exit, measurements and ownership for each run. Treat a skipped run, fallback, a failed exclusivity check or an incomplete receipt as missing evidence. On Windows, process telemetry may miss short dispatches. Check competing processes separately. Require the measured run to report CUDA and the executed SM, and corroborate attribution with the qualified device probe and the kernel trace.

1. Run a short smoke case and an unprofiled baseline before launching long cases. Keep one short Nsight Systems capture per variant on the same scene and frame window. Inspect kernel totals, transfer directions and bytes, launch count, and host API waits. Nsight Compute is useful once the timeline identifies a kernel worth examining.
2. Use one change per A/B measurement and batch related verified changes in one PR. Repeat a short unprofiled run when timings are close. Gate the full campaign on a matching short trajectory, then run the production acceptance scenes without profiling. Store checkpoint replay, scalar/grouped parity, retry and capacity-growth outcomes, and the exact GPU and named-pipe Compute Sanitizer receipts. Keep failed experiments with their measurements.
3. Start long runs only after the smoke and provenance checks pass. Record time limits and expected terminal receipts in the run ledger. On timeout, identify the owned job by executable, arguments and start time. Prefer the CLI's cooperative `cancel` protocol with redirected stdin, and retain published evidence. Resume only with a matching checkpoint and execution identity. Verify the complete run before merging the tested head.

For a short profile, create `$out` as a fresh directory. Set `$solver`, `$scene`, and `$seed` to the qualified artifact and its authored inputs. Use the installed Nsight Systems executable.

```powershell
& $solver device-info --backend cuda
if ($LASTEXITCODE) { throw 'CUDA device qualification failed.' }
nsys profile --trace=cuda,nvtx --sample=none --show-output=true --force-overwrite=false `
  --output "$out/timeline" $solver run --stage simulate --scene $scene --initial-state $seed `
  --backend cuda --cache "$out/profile-cache" --frame-end 2 --trace --measure-job
if ($LASTEXITCODE) { throw 'CUDA profile failed.' }
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum `
  --format csv --output "$out/timeline-stats" "$out/timeline.nsys-rep"
if ($LASTEXITCODE) { throw 'CUDA profile export failed.' }
```

For CUDA Graph workloads, add `--cuda-graph-trace=node` to a separate short capture when the comparison needs per-node activity. The default graph granularity reports whole graph activity. Node tracing adds overhead, so compare matched captures and use the unprofiled run for throughput.

API wait totals can include device work, copies and queue gaps. Compare them with GPU kernel and copy durations as separate observations. The job's host wall partition is the elapsed-time accounting. CUDA events on different streams can overlap each other and host work. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## Apply the measured M1 changes

| Symptom | Change and invariant |
| --- | --- |
| Scalar gather restages both face histories and fences per particle | Group consecutive particles that share wall side and donor policy. Run one gather dispatch per group and keep original per-particle arithmetic order. Check results in particle order. Replay a reusable grouped rejection through the scalar path so the first numerical error and committed state match. |
| Isolated interface loses projected donors | Compare new and old velocities on the same projected donor mask. Add that change to the full old reconstruction. Batch restricted-old, full-old and projected passes separately. Copy retained results into bounded scratch before a subsequent call reuses the borrowed result bank. Correct and check each particle before its signed-distance query. |
| Transfers copy dormant fields or repeat full face data | Measure bytes and transfer counts in both directions. Move only active fields and extents, reuse immutable data over its valid epoch, and reserve scratch outside physical attempts. Verify padding, zero extents, aliasing, host pinning and completion fences. Compact normalization improved transfer volume. The compact-force experiment saved bytes and was slower in wall time. |
| Wet-wall work is serialized inside one CUDA block | Run multiple independent patch lanes per block and reuse compact wet-support records. Preserve exact wall pose, signed inward flux, receiver ordering, and the shared pressure datum used for assembly and correction. Compare actual patch costs and the full frame. |
| Cache changes appear to improve solver time | Separate publication and checkpoint work from solver attempts. Keep a recoverable checkpoint policy, fence borrowed device sources until D2H completion, and verify the published cache and replay before comparing policies. |

The RTX 5060 Ti SM120 isolated-drop A/B covered ten substeps. The scalar and grouped captures recorded **16,080 → 830 gather launches**, **15.586 → 0.372 s of gather kernels**, **91.664 → 5.023 GB H2D**, and **16.842 → 0.389 GB D2H**. The first 35 accepted traces matched exactly. Across the 216-frame run, median solver work was **2.283 s CPU and 0.327 s CUDA** (6.98×). CUDA p95 and maximum were 0.772 and 0.956 s. The profiled variant's separate measured production run took 98.663 s including setup and publication. These numbers belong to this workload. [M1 transport evidence](../references/m1-transport-evidence.md) · [Merged gather changes](https://github.com/tychota/FluidiTy/pull/44)

The matched 24-frame primary piston experiment reduced median publication interval from 0.865 to 0.746 s through patch lanes and compact wet support. The final 48-frame build measured 0.763 s primary and 1.327 s refined. The refined run spent 36.23 s of 64.21 s in forces, geometry and pressure, and 14.67 s in activation and field work. Its snapshot/cache phase was 1.60 s. Prioritize resident support and bounded geometry batches in that profile. [Merged wet-wall optimization](https://github.com/tychota/FluidiTy/pull/40) · [Projected donor restoration](https://github.com/tychota/FluidiTy/pull/41)

## Reuse at the next milestones

| Milestone | First measurement | Checks to carry forward |
| --- | --- | --- |
| M2 surface tracking and whitewater | Trace field sampling, surface and whitewater construction, and publication separately. Count repeated support-list builds and device-to-host payload per product. | Reuse support only for the matching topology and epoch. Check surface and whitewater cache identities, geometric accuracy, source lineage and replay. |
| M3 adaptive grid and narrow band | Compare uniform and adaptive runs at equal error. Record active levels, stencil classes, coarse/fine transitions, remap work and peak arena bytes including simultaneous old and new owners. | Group by level and donor policy where the access pattern benefits. Preserve coarse/fine signed flux, deterministic remap and rollback. Account for graph rebuilds on topology or capacity change. |
| M4 viscosity, compliance and multiple liquids | Partition pressure, viscosity and interfacial work by phase, operator and iteration. Separate bytes for shared geometry from phase-specific fields and coefficients. | Preserve coupled fluxes, pressure history, constitutive bounds, constituent conservation and interface orientation. Qualify mixed precision against condition-sensitive residuals, physical defects and scene gates. M1 already uses FP32 fine pressure work with higher-precision checks. |

The tuning order is to measure end to end, remove avoidable dispatch and transfer work, then optimize the measured device bottleneck. Tiling, coalescing and occupancy matter after the timeline shows where they can help. [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) · [Böhm's measured CUDA matrix multiplication progression](https://siboehm.com/articles/22/CUDA-MMM) · [BBuf CUDA optimization examples](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)
