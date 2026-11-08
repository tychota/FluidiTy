# Bounded CUDA Graph Replay Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Qualify fixed-capacity CUDA Graph capture and replay of the existing deposition and tiny pressure kernels on the reference Windows GPU. Establish direct-stream parity, bounded failure behavior, and measured submission and stream intervals.

**Architecture:** Two independent fixture workloads reuse the private Step 3 launch bridges. A test-only graph owner captures an identical enqueue sequence on a nonblocking stream, with preallocated pinned host and device buffers that remain alive through the last fence and graph destruction. These are separate numerical fixtures. The minimum complete-support transfer grid exceeds the tiny pressure cell limit.

**Tech Stack:** C++20, CUDA Runtime Graphs, GoogleTest, locked uv/Conan 2/CMake/Ninja, Windows MSVC and sm_120.

**Spec:** [FluidiTy design](../specs/2026-11-07-fluidity-design.md) §§11.2–11.3, 13.1 and 15.2. Existing bounded operators are specified in the [Step 3 plan](2026-11-08-cuda-uniform-operators.md).

## Global constraints

- Use the isolated graph worktree from current main. Keep the spike under native tests and the existing test/validation wiring. Reuse private CUDA launch bridges, preserve existing operators, numerical conventions, fixtures, and public interfaces.
- Same sequence, stream and buffers for direct and graph cases. Use host stream capture in global mode, instantiate outside capture, and keep captured pointers fixed. CUDA allocations, host pinning, synchronous copies, synchronization, elapsed-time queries and graph destruction occur outside capture.
- Fence before rewriting pinned input or reading pinned output. Destroy executable and graph before freeing captured buffers. On capture failure always end capture and destroy any returned partial graph. A failed operation never publishes a trial. Capacity overflow is explicit ResourceExhausted.
- Use actual RTX 5060 Ti CC 12.0 and explicit CMake 120-real and 120-virtual. No unsupported-compiler bypass, fallback, skip, or firewall change. Require existing probe plus graph hardware evidence. Named-pipe Compute Sanitizer in Windows Release.
- C++20 and AGENTS.md. Device code stays unchanged. One task commit per independently verified deliverable. Preserve raw logs in ignored out/ paths and a progress ledger.
- This spike tests one stream and fixed storage. Production graphs, stream topology, conditional nodes, CCCL capture, graph update, GPU allocation, 3D simulation, Blender, AMR and end-to-end speedup claims are later gates.

## Design choices

Stream capture reuses existing immediate launch functions and derives dependencies from the same enqueue path used as the direct comparator. Explicit graph-node assembly would duplicate launch arguments and dependency definitions for this spike. Host-pinned staging gives an observable replay update without recapturing host pointers. Pressure and deposition have different bounded fixture grids and use separate graphs. Timing measures host submission and elapsed stream interval separately and preserves capture/instantiation cost. Report observed times without a speed gate.

## Review focus

- Changed active count and reordered deposits use the same executable, and zero live particles cannot retain a previous sum.
- Excess capacity is rejected before touching staged or accepted output. A device range failure leaves the last accepted outputs intact and the next valid replay succeeds.
- A failed capture ends capture before returning, discards the partial graph and permits a subsequent successful capture.
- Pressure replay updates input contents through pinned addresses, preserves original system and gauge behavior, and matches direct output at both input states.
- Validator rejects absent, extra, skipped, failed, misplaced and wrong-architecture graph receipts. Sanitizer requires every graph test, clean process exit and zero errors.

### Task 1: Fixed-capacity deposition replay

**Files:** create `native/tests/cuda_graph_replay_support.h` and `native/tests/test_cuda_graph_deposit.cc`; modify `CMakeLists.txt`.

**Interfaces:** Add CUDA-only `fluidity_cuda_graph_tests` (GTest main, fluidity_cuda_operators, CUDA::cudart, private native/cuda includes, label gpu). In namespace fluidity::graph_spike provide test-only `CudaGraphReplay` owning graph/executable with `Capture(cudaStream_t stream, EnqueueWork enqueue_work) -> absl::Status` where the callable takes no args and returns absl::Status; `Launch(cudaStream_t stream) const -> absl::Status` and `capture_us() const -> double`. Use shared typed pinned/device buffers and checked runtime helpers as appropriate, with lifetimes visible at call sites. Task 2 may add a test source to the same executable and use this helper.

- [ ] Add hardware cases `CudaGraphReplay.DepositMatchesDirectAndClearsInactiveInputs`, `CudaGraphReplay.RejectedDepositPreservesAcceptedOutputs`, and `CudaGraphReplay.FailedCaptureCanBeRetried`. RED through compilable Unimplemented stubs, run the exact CTest filter on Debug CUDA and retain failure output.
- [ ] Capture identical direct sequence of pinned H2D input, device zeroing of totals and state, existing LaunchDeposit, and pinned D2H results. Capacity is four particle slots, MAC grid 4×4 h=1 origin zero, scale 2^-20 each channel. A valid active particle at (1.5,1.5) has mass 1, velocity (2,-3), V0=2, stress xx=4 and other entries zero. At x-normal face index 6 require mass 393216, pure momentum 786432, stress force 6291456 integer quanta. Neutral padded slots use a valid stencil with zero mass, velocity, affine fit, reference volume and stress. Check direct and replay full totals exactly, 1→2→0 live slots and repeated replay after reordering. Assert actual kernel architecture 1200.
- [ ] Check five live slots return ResourceExhausted without changing accepted totals, one active particle with stress xx=2^60 triggers device range rejection and leaves accepted totals, and later valid replay succeeds. No CUDA API fault injection in memcheck cases. Force capture callback to return FailedPrecondition after enqueuing a valid memset; verify capture ends, no graph is available and a subsequent valid capture works.
- [ ] GREEN with Debug CUDA build and `ctest --preset windows-debug-cuda -R '^CudaGraphReplay\.' --output-on-failure --no-tests=error`. Require all three on hardware. Run existing native and Python tests with locked tools, using the existing validator only for CPU until Task 3 updates exact GPU names.
- [ ] Commit `test(cuda): qualify fixed-capacity deposit graph replay`.

### Task 2: Tiny pressure replay and timing

**Files:** create `native/tests/test_cuda_graph_pressure.cc`; modify `CMakeLists.txt` to add the source to fluidity_cuda_graph_tests.

**Interfaces:** Consume Task 1 graph owner and CUDA-only target. Use the private PreparedPressureInputs, PressureKernelResult and LaunchPressure bridge. Add one case `CudaGraphReplay.PressureMatchesDirectAndMeasuresReplay`.

- [ ] Write the case before implementing the pressure graph fixture. Run the specific Debug CUDA CTest and observe expected RED.
- [ ] Use a 2×1 unit MAC grid with seven faces, one active interior face, sealed exterior, unit mobility/distance/volume, zero sources and infinite bulk moduli. Start with internal predicted velocity 2 and oriented jump 3. Check pressure [0,5], gauge index 0, interior corrected velocity 0, original A/b, residuals and defect. Repeat on the same captured executable with predicted velocity -1 and jump 2, requiring pressure [0,1]. Compare every meaningful output and architecture against direct execution on the same input. Explicitly zero result staging between sequences.
- [ ] Warm both paths four times. Measure 64 sequences per path with unchanged buffers and no fences in the timed submission loop; fence before any pinned reuse. Record event intervals around the sequence and steady-clock submission duration. Report per-sequence microseconds and one-time capture/instantiate microseconds, with finite nonnegative values and no speed threshold. Emit exactly one line `graph_replay compiled_architecture=1200 iterations=64 capture_us=<decimal> direct_submit_us=<decimal> replay_submit_us=<decimal> direct_stream_us=<decimal> replay_stream_us=<decimal>` using actual values. End the test on a valid accepted pressure result after timing.
- [ ] Build and run all four CudaGraphReplay cases on Debug CUDA, existing native labels and locked Python. Preserve RED and GREEN.
- [ ] Commit `test(cuda): qualify tiny pressure graph replay and measure submission`.

### Task 3: Required validation and measured evidence

**Files:** `tools/native/validate.py`, `tools/native/tests/test_validate.py`, `docs/native/README.md`, optionally `docs/native/toolchains.md` for measured data.

- [ ] Test first. Require the four exact graph cases in the existing gpu CTest set, increasing the total to 18. Require one graph timing line in the named pressure case and reject missing, duplicate, foreign, malformed, nonfinite and wrong-architecture receipts. Keep existing 14 CTest names and four operator measurements. Verify test RED then GREEN.
- [ ] In Windows Release, run a third named-pipe memcheck invocation on fluidity_cuda_graph_tests, no filter. Require all four exact graph cases, proper RUN/OK association, one zero-error summary and correct timing receipt. Preserve the existing one probe and eleven operator memcheck cases and their receipt constraints. Release aggregate memcheck count 16. Timeout and launch timeout remain 90 and 30 seconds. CPU-only behavior stays intact.
- [ ] Run all four locked Windows presets, exact device/architecture checks, sanitizer and formatter/lint. Record actual graph timings and methodology, remaining M0 gates, and raw evidence under ignored out/. Record distinct Debug/Release timing values as fixture measurements.
- [ ] Commit `test(cuda): require graph replay hardware and memcheck evidence`.

## Final acceptance

Fresh independent whole-branch review and fix findings. Verify the four-preset matrix at the final exact commit. Report actual measured submission/stream intervals, CUDA version/device, check counts, no skipped hardware cases, and branch/evidence locations. Preserve the local graph branch and worktree.

## Source notes

- [CUDA Graphs, CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA Graph capture constraints and memory lifetime](https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/constraints.html)
