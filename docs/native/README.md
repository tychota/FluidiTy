# Native build and validation

The native foundation contains a C++20 CPU reference implementation, a CUDA hardware probe,
CUDA operators for uniform MAC transfers, pressure and liquid boundary stages, a persistent
3D CUDA liquid substep backend, and bounded CUDA Graph replay tests. CMake presets use Ninja and separate Conan,
build and fixture directories for every configuration. The
[toolchain matrix](toolchains.md) records the measured Windows configuration and its recipe revisions.

## Windows prerequisites

- An x64 Visual Studio 2022 Native Tools PowerShell with the MSVC C++ workload and Windows SDK.
  The measured setup used MSVC 19.44.35229 and SDK 10.0.26100.0. Use the compiler installed on
  your machine and qualify it from that shell.
- Git, uv 0.13.7 or later and Python 3.13. The measured versions were uv 0.13.7 and Python
  3.13.18. The native uv group installs the locked Conan 2.34.0, CMake 4.4.4 and Ninja 1.13.2
  into the reference environment.
- Access to Conan Center or a local cache with the pinned recipes and their dependencies.
  Keep separate host and build profiles in Conan home.
- For the CUDA runs, the installed CUDA Toolkit, a compatible NVIDIA driver and a compute
  capability 12.0 device. The measured setup used CUDA Toolkit 13.4.1, nvcc 13.4.59, driver
  610.88 and a GeForce RTX 5060 Ti. Install Compute Sanitizer with the toolkit for the
  Release memcheck gate.

## CPU-only build on Windows

Run these commands from the repository root. The validator installs from both committed locks,
generates the fixture headers, configures and builds the preset, and runs CTest and Python.

```powershell
chcp.com 65001
$env:PYTHONUTF8 = '1'
uv sync --project reference --locked --group native
if ($LASTEXITCODE) { throw 'uv sync failed.' }
$uvArgs = @('run', '--project', 'reference', '--locked', '--group', 'native')
uv @uvArgs python tools/native/qualify_toolchain.py --mode cpu --output out/qualification/windows-cpu.json
if ($LASTEXITCODE) { throw 'CPU qualification failed.' }

foreach ($profile in @('fluidity-host', 'fluidity-build')) {
    uv @uvArgs conan profile detect --name $profile --force
    if ($LASTEXITCODE) { throw "Profile detection failed for $profile." }
}
uv @uvArgs conan profile show -pr:h fluidity-host -pr:b fluidity-build
if ($LASTEXITCODE) { throw 'Profile inspection failed.' }

foreach ($preset in @('windows-debug-cpu', 'windows-release-cpu')) {
    uv @uvArgs python tools/native/validate.py --preset $preset --host-profile fluidity-host `
        --build-profile fluidity-build --output "out/validation/$preset"
    if ($LASTEXITCODE) { throw "Validation failed for $preset." }
}
```

Inspect the profile output for Windows, x86_64 and the detected MSVC toolset. Profile detection
overwrites these two profile names. The validator supplies C++20, the dynamic CRT and the
selected configuration in the host context. The build context stays on Release and its
dynamic Release CRT. Repeating the command uses the same build tree for an incremental run.
Keep concurrent invocations in separate preset trees with distinct evidence directories.

For manual Conan install and CMake commands, including the explicit ABI flags, see the
[toolchain matrix reproduction commands](toolchains.md#reproduce-on-windows). The lock pins
recipe revisions and `--build=missing` permits building a binary from a pinned recipe.

## VS Code IntelliSense

Open the repository folder with the Microsoft C/C++ extension installed. Run the native
setup and validation for your platform once to generate the preset's
`out/build/<preset>/compile_commands.json` and fixture headers. IntelliSense reads
the compiler, flags and Conan include directories from that database.

The default configuration uses your platform's Debug CPU preset. For CUDA sources,
run the Windows Debug CUDA preset and select **Windows CUDA** using
**C/C++: Select a Configuration**. Each checkout generates its own databases.

## CUDA hardware and sanitizer gates

Use the same shell and profiles after the CPU setup. The CUDA qualification receipt is
separate. Both presets require exactly 35 `gpu` cases across two probe cases, 29 operator
cases and four graph replay cases. The operator group includes three integrated 3D liquid
scheduler cases and two asynchronous snapshot cases. Both presets also run the separate `cuda-mock` API-double suite.

```powershell
uv @uvArgs python tools/native/qualify_toolchain.py --mode cuda --output out/qualification/windows.json
if ($LASTEXITCODE) { throw 'CUDA qualification failed.' }

foreach ($preset in @('windows-debug-cuda', 'windows-release-cuda')) {
    uv @uvArgs python tools/native/validate.py --preset $preset --host-profile fluidity-host `
        --build-profile fluidity-build --output "out/validation/$preset"
    if ($LASTEXITCODE) { throw "Validation failed for $preset." }
}
```

Both CUDA presets require the qualified device and launch receipt from `CudaProbe.SharedVec2Math`.
They also require exactly one timing receipt between the GoogleTest start and completion
markers of each case below, with the listed operator name and compiled architecture 1200.
Standalone operator receipts use finite nonnegative CUDA event times in `kernel_ms`.

| GoogleTest case | Receipt operator |
| --- | --- |
| `CudaTransfer.DepositAnalyticAndCpuParity` | `p2g` |
| `CudaTransfer.GatherAffineMatchesFixtureAndCpu` | `g2p` |
| `CudaPressure.NeumannJumpMatchesFixture` | `pressure_neumann` |
| `CudaPressure.ComplianceRetainsMeanPressure` | `pressure_compliance` |
| `CudaMacTransfer3D.DepositCornerMatchesOracleAndCpu` | `p2g_3d` |
| `CudaMacTransfer3D.GatherCornerMatchesOracleAndCpu` | `g2p_3d` |
| `CudaMacPressure3D.MixedLinksMatchOracle` | `pressure_3d_mixed` |
| `CudaMacPressure3D.MultilevelProjectionMatchesCpu` | `pressure_3d_multilevel` |
| `CudaLiquidBoundary3D.GeometryAndCapillaryMatchOracle` | `liquid_boundary_geometry_3d` |
| `CudaLiquidBoundary3D.SweptContactMatchesOracle` | `liquid_wall_contact_3d` |
| `CudaLiquidBoundary3D.OpenTankGeometryAndPressureMatchOracle` | `liquid_open_tank_geometry_3d` |
| `CudaLiquidBoundary3D.OpenTankSweptContactMatchesOracle` | `liquid_open_tank_contact_3d` |
| `CudaLiquidBoundary3D.CompatibleTransferAndGhostHistoryMatchCpu` | `liquid_boundary_transfer_3d` |
| `CudaLiquidBoundary3D.CoupledBoundaryOperatorsPreserveFailure` | `liquid_boundary_coupled_3d` |
| `CudaLiquidSubstep3D.PersistentSubstepMatchesCpu` | `liquid_substep_3d` |
| `CudaLiquidSubstep3D.ReplaysAfterCapacityGrowth` | `liquid_substep_3d_growth` |
| `CudaLiquidSubstep3D.OpenTankSchedulerMatchesCpu` | `liquid_open_tank_substep_3d` |
| `CudaLiquidSnapshot3D.DelayedCopyFencesSourceEpoch` | `liquid_snapshot_3d` |
| `CudaLiquidSnapshot3D.DeviceFailureUsesLastValidHostSnapshot` | `liquid_snapshot_3d_failure` |

The three scheduler receipts require a finite, strictly positive `substep_device_timeline_ms`
and `timing_scope=topology_through_publication`. The CUDA events bracket the latest successful
substep from the topology callback through publication staging. The interval includes host
preparation and dispatch gaps and starts after the initial field staging. For the growth case,
it describes the latest successful replayed substep. The separate `scheduler_frame_wall_us`
diagnostic measures the full host `AdvanceFrame` call, including growth and replay in the
growth fixture. These fixture measurements do not establish production throughput.

The snapshot receipts require finite, strictly positive `snapshot_copy_ms` and
`timing_scope=d2h_stream_events`. They measure two real CUDA events around the D2H
submission, including stream dependencies and host submission gaps. The delayed fixture
records its gate event after a queued host-controlled latch, then makes both copy streams
wait for that recorded event. Both source epochs are prepared before closing the gate;
this avoids requiring a synchronous arena upload to progress behind a blocked WDDM host
callback. The cases verify distinct source/writer lifetimes, two live epochs, rejected
reuse, canceled and stale completion, complete checked host decode, shared fatal poison
and last-valid host diagnostics. Error controls return a simulated failure after actual
copy/fence/drain work; they do not claim recovery from a genuinely lost device. There is
no snapshot kernel: the architecture receipt comes from the actual producing arena image.

The liquid boundary cases exercise the shared CPU analytic fixtures on the device,
including scaled tiny cuts, interface-local capillarity, the CUDA static pressure bridge,
and compensated continuous thin-wall contact. Geometry uses checked host topology and
sequential POD queries with two reusable allocations. Its event time sums the device
queries; separate diagnostic lines report aggregate successful device time and wall time
for all calls. These bounded fixture timings do not qualify production throughput.

Both CUDA presets require these four graph replay cases.

- `CudaGraphReplay.DepositMatchesDirectAndClearsInactiveInputs`
- `CudaGraphReplay.RejectedDepositPreservesAcceptedOutputs`
- `CudaGraphReplay.FailedCaptureCanBeRetried`
- `CudaGraphReplay.PressureMatchesDirectAndMeasuresReplay`

Only the pressure case emits a graph timing receipt. The validator requires exactly one
receipt between that case's GoogleTest start and completion markers, with compiled
architecture 1200, 64 iterations and finite nonnegative times in microseconds.

Windows CUDA Release additionally requires Compute Sanitizer from the qualified toolkit.
The validator runs three memcheck invocations covering `CudaProbe.SharedVec2Math`,
28 operator cases (including all three scheduler cases, both snapshot cases and the open-tank
geometry and contact cases), and all four graph replay cases for 33 mandatory memcheck cases.
The two `LaunchIgnoresPriorRuntimeError` cases intentionally trigger CUDA API errors and run
only in the 35-case CTest inventory. Each memcheck invocation must exit successfully and
report one zero-error summary. The validator checks the exact GoogleTest start and completion
names, the probe launch receipt, 19 operator timing receipts and one graph timing receipt in
their matching cases. It sets the named-pipe connection override only in the sanitizer process
environment, with a 30-second launch timeout and a 90-second command timeout per invocation.
A missing sanitizer, skipped case or incomplete run prevents publication of a Release
success receipt.

For direct memcheck runs, scope the connection setting to the commands. Use the validator
for the full evidence checks.

```powershell
$sanitizer = Join-Path $env:CUDA_PATH 'compute-sanitizer/compute-sanitizer.exe'
$previousConnection = $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE
try {
    $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = 'named-pipes'
    & $sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 `
        out/build/windows-release-cuda/fluidity_cuda_probe_tests.exe --gtest_filter=CudaProbe.SharedVec2Math
    if ($LASTEXITCODE) { throw 'Release probe memcheck failed.' }
    & $sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 `
        out/build/windows-release-cuda/fluidity_cuda_operator_tests.exe `
        --gtest_filter=*-CudaPressure.LaunchIgnoresPriorRuntimeError
    if ($LASTEXITCODE) { throw 'Release operator memcheck failed.' }
    & $sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 `
        out/build/windows-release-cuda/fluidity_cuda_graph_tests.exe
    if ($LASTEXITCODE) { throw 'Release graph memcheck failed.' }
} finally {
    $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = $previousConnection
}
```

### Focused CUDA arena and stage harness

Run `fluidity_cuda_arena_tests.exe` directly after building each CUDA preset. It covers persistent
arena ownership, graph sorting, field sampling, boundary transfer, contact and pressure stages.
Its inventory is separate from the 35 mandatory CTest cases and 33 mandatory memcheck cases.
Debug runs 43 cases (42 on hardware and one host-only generation exhaustion case). Release
runs 39 (38 on hardware and the same host-only case). Require every case to pass with zero skips.

```powershell
foreach ($configuration in @('debug', 'release')) {
    & "out/build/windows-$configuration-cuda/fluidity_cuda_arena_tests.exe" `
        "--gtest_output=xml:out/arena-$configuration.xml"
    if ($LASTEXITCODE) { throw "CUDA arena $configuration tests failed." }
}

$sanitizer = Join-Path $env:CUDA_PATH 'compute-sanitizer/compute-sanitizer.exe'
$previousConnection = $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE
try {
    $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = 'named-pipes'
    & $sanitizer --tool memcheck --require-cuda-init yes --error-exitcode 97 --launch-timeout 30 `
        out/build/windows-release-cuda/fluidity_cuda_arena_tests.exe
    if ($LASTEXITCODE) { throw 'Release CUDA arena memcheck failed.' }
} finally {
    $env:NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE = $previousConnection
}
```

Require all 39 cases and one zero-error summary for the focused Release memcheck run.

### Device images

Inspect the Release operator executable for the `sm_120` cubins and embedded PTX
from the `compute_120` virtual architecture. The PTX dump reports `.target sm_120`.

```powershell
$cuobjdump = Join-Path $env:CUDA_PATH 'bin/cuobjdump.exe'
$operators = 'out/build/windows-release-cuda/fluidity_cuda_operator_tests.exe'
& $cuobjdump --list-elf $operators
& $cuobjdump --list-ptx $operators
& $cuobjdump --dump-ptx $operators | Select-String '^\.target|LiquidBoundaryGeometry3DKernel|LiquidWallContact3DKernel'
& $cuobjdump --dump-sass $operators | Select-String 'arch = sm_120|LiquidBoundaryGeometry3DKernel|LiquidWallContact3DKernel'
```

## Fixtures and evidence

CMake exports `affine_apic.json`, `two_cell_pressure.json`, `compliant_pressure.json` and
`stiff_viscosity.json` into `out/fixtures/<preset>/` with a SHA256 manifest. It validates the JSON
on each build and regenerates the test header at `out/build/<preset>/generated/fixture_data.h`
when the exported values, tolerances or provenance change. The transfer, pressure and viscosity
tests use the exported expected values and compare against the fixture's absolute plus
relative tolerance. To restore a missing or modified export, run this command for the preset.

```powershell
uv @uvArgs python -m fluidity_reference.cli export "out/fixtures/$preset" --overwrite
if ($LASTEXITCODE) { throw 'Fixture export failed.' }
```

Run the validator again after restoring the files. A successful run writes `summary.json`
and path-free CTest reports under `--output`. The receipt records the source revision and
working source hash, both lock hashes, profile and effective ABI hashes, tool versions,
fixture and header hashes, commands and separate results for Python, native CPU, GPU,
CUDA doubles, sanitizer and proofs. The proof gate remains `not_run` in this validator.
The validator clears inherited GoogleTest and pytest selectors. It checks each named native
case's execution output and compares the Python results with the complete pytest collection.
The CUDA receipt records the measured operator, scheduler and snapshot timings in
`evidence.gpu.operators` and the pressure graph measurement in `evidence.gpu.graph_replay`. The graph receipt
contains one capture and instantiation time, the iteration count and separate
per-sequence host submission and stream intervals for direct and graph execution.
See the fixture and timing method in the
[toolchain matrix](toolchains.md#windows-cuda-graph-replay-gate-measured-on-2026-11-08).

Full command arguments, tool output, Python and CTest XML, the live requalification
receipt and all three Release memcheck logs stay under `out/native-validation/<preset>/run-*/`.
These local files can contain machine paths. An invocation with one valid `--output` destination clears its previous reports,
including when another command-line argument is invalid.
On failure, read the named stage's log and CTest report if the suite ran. Requalify after
compiler, SDK, toolkit, driver or device changes. If CMake cached a different compiler,
start with a fresh preset build tree. A localized MSVC include scan requires code page
65001, and Debug requires a working MSVC program database service.

## Hosted CPU checks and remaining M0 work

The [Native CPU workflow](../../.github/workflows/native-cpu.yml) targets Windows 2022 x64,
Ubuntu 24.04 with GCC 13, and macOS 15 ARM64 with Xcode 16.4. It runs Debug and Release
Ninja presets with the same committed locks. The matrix has separate compiler- and
configuration-keyed Conan caches and uploads the path-free receipt and CTest report.
The measured Windows hardware runs are documented in the [toolchain matrix](toolchains.md).
Hosted CPU checks leave the CUDA, sanitizer and proof results at `not_run`.

The uniform CUDA spike covers complete-support quadratic MAC transfers with host-prepared
geometry and a dense pressure solve for up to eight cells. The graph spike covers fixed
buffers on one stream for separate deposition and two-cell pressure fixtures. Production
graph integration belongs to later solver work. The
[Blender reader qualification](../blender/README.md) records measured point-cache playback.
The [2D adaptive-basis qualification](../adaptivity/README.md) records measured M0 transfer, flux ownership,
and masked support fixtures. Production adaptivity still requires 3D solver and CUDA throughput gates.
Proof pilots and later formal contracts use the separate [Lean workflow](../../proofs/README.md).
The project owner must select a license before public release. See the
[design milestones and evidence
requirements](../superpowers/specs/2026-11-07-fluidity-design.md#15-validation-milestones-and-review-boundary).

The compatible boundary transfer gate executes masked basis normalization, quotient
weight gradients, recentered affine momentum, guarded nearest-even quanta reduction,
face velocity normalization, independent old/new ghost QR and full gather on CUDA.
The QR uses the same device-safe arithmetic as the CPU numerical seam. The bounded
path uses one sequential kernel and at most six allocations per call, with no CPU
numerical fallback. It is an operator qualification path, not a throughput claim.
The coupled case feeds CUDA-frozen geometry through CUDA pressure, old/new ghost
histories and gather, then independently corrects stored and advecting velocity in
CUDA contact. Informational diagnostics use `boundary_device=` and
`CTEST_FULL_OUTPUT`; only the exact operator lines are validation receipts.

## Simulate command and inspection protocol

`fluidity_simulate` emits protocol version 1 JSON Lines on stdout. Successful
inspection, schema and device queries emit one `result`. Execution may emit
progress before its terminal record. Failures emit a terminal `error` to stdout
and the same structured record to stderr, with `E-<DOMAIN>-<NAME>`, domain, name, field,
object and cause. Exit 0 means success, 2 means invalid arguments, and 1 means an
operational failure or unavailable command. Successful defaults produce no stderr.
Arguments and protocol strings are UTF-8; Windows uses the Unicode process entry
point and filesystem APIs. Scene inputs are the shared binary `FLRS` records.

```text
fluidity_simulate schema
fluidity_simulate check --scene resolved.flrs
fluidity_simulate check --cache CACHE --frame 0
fluidity_simulate check --cache CACHE --scene resolved.flrs
fluidity_simulate device-info --backend cpu
fluidity_simulate device-info --backend cuda --device 0
```

`schema` reports the generated FlatBuffers identifiers, the embedded shared
resolved schema, and actual encoded runtime defaults with every scalar present.
Positive infinity is the JSON string `"Infinity"`; finite doubles preserve round
trip precision, including the exact mass and momentum quanta. `build_digest` is
BLAKE3 over the actual executable bytes, independent of its filename or argv.
These values describe this binary; they do not authorize an older cache to resume.

`check` uses `OpenForInspection` and bounded verified reads. With `--cache`, it
checks resolved cache identity; `--frame` additionally verifies that one committed
manifest, its deterministic inspection mirror, and its point content. It does not
scan or claim verification of unspecified frames or retained checkpoints. Adding
`--scene` compares the explicit canonical resolved fingerprint. Inspection creates
no directory or lock, changes no cache content, and always reports
`resume_authorized:false`. Missing or corrupt physical records and out-of-bounds
data fail closed. Cache checks also report operational `job_status`. A held
whole-job lease reports `baking`. A baking marker without a lease reports
`crashed`. With no active writer, `complete`, `cancelled` and `failed` describe the
last terminal run and `idle` means no marker. Unrecognized or unreadable
operational metadata reports `unknown`. Native access failures also include a
`job_status_diagnostic` object with the status name and cause. The result
still includes any verified frame data.
Default decoding limits are those in `LiquidCacheLimits3D`.

`device-info --backend cpu` reports CPU availability and whether CUDA was compiled;
`cuda_available:null` means CUDA was not probed. An explicit CUDA request uses the
qualified native arithmetic probe and reports device name, compute capability,
executed image architecture, runtime/driver versions and successful operation
receipts. A CPU-only binary fails explicit CUDA requests; it never substitutes CPU.

The checked `simulate::Command`, `ParseCommand`, `ExecutableDigest` and
`ProtocolWriter` interfaces are in `native/include/fluidity/host/simulate_cli.h`.
`run --stage simulate --scene scene.flrs --cache cache --backend cpu --frame-end 20`
executes the real scheduler. CUDA builds accept `--backend cuda --device 0` and use
only the real persistent CUDA backend. Omitting the backend inherits the resolved
scene; an explicit mismatch fails. Execution requires canonical checked FLRS bytes
whose build digest matches this actual executable and whose backend matches the
selected implementation. A moved identical binary remains compatible. Changed
executable bytes or resolved numerical inputs fail before publication.

Model version 1 fixes output cadence at **24 frames per second**. Each target is
computed from its absolute ordinal divided by 24; time is never accumulated from
rounded frame durations. Unrepresentable/nonadvancing frame ordinals fail, and
retained checkpoints must have that exact clock. No local timing override is
accepted on resume. The bounded qualification capacity plan (1024 particles,
128 blocks, 4096 cells, 24576 faces, 256 sources) is persisted in separate resume
metadata; it is not physical state or a fingerprint setting. Capacity exhaustion
is explicit; this CLI does not request automatic growth.

`--initial-state initial.flst` supplies a canonical raw frame-zero state, **not a
checkpoint envelope**. Its exact bytes must be the resolved initial-particles
upstream content. Geometry, seed, source identities, zero source counters and clock
must match. Without such upstream content, the CLI builds an empty state with the
resolved source identities. Unsupported external source/obstacle artifact roles
fail explicitly. Already verified retained checkpoints need no initial file; if one
is supplied, it is always verified. Source emission preserves binary timestep
arithmetic: a real-valued integer rate/time product is not a promise of an integer
birth at its rounded binary boundary.

A new run publishes frame zero and its checkpoint, then the inclusive
`--frame-start`/`--frame-end` range and any prerequisite checkpoint frames (every ten).
It simulates omitted earlier frames without publishing their points. Existing
caches require `--resume`; resume starts from the latest verified retained checkpoint
at or before the explicit frame start, or the end when no start was specified.
Previously published intermediate output is reproduced and compared by exact bytes,
then reused by the idempotent store without modification. Retention keeps two full
checkpoints. An older unavailable range fails `NOT_FOUND` unless a verified initial
state is supplied for reconstruction; present intermediate frames must still match.

`step --scene scene.flrs --cache cache` executes **one accepted physical substep**
from the latest retained checkpoint (or a new verified initial state). Permitted dt
retries remain scheduler-owned. It reports accepted state and never publishes a
partial point frame. Repeated step requests start again from that retained state.
`--preview` on run likewise executes read-only, creates no cache and verifies any
existing matching frames. Cache-producing M1-B preview will require separate
resolved settings and identity; it is not implemented by this flag.

`replay --cache cache --frame N --substep 0` selects completed frame N.
`--frame N --substep S` for S>0 selects accepted substep S during N to N+1, matching
the persisted clock. Interior replay completes the final frame schedule and verifies
its published points before returning the selected state. Capacity rollback discards
observations from the abandoned pass. Replay requires that selected completed frame
(or N+1 for an interior selection) to be published. It is read-only, including
failures caused by a missing lock. `OpenForReplay(expected_fingerprint)` permits
verified checkpoints but denies every publication; unauthenticated inspection still
cannot authorize resume.

CUDA numerical reuse requires the same resolved execution and selected device,
compute capability, exact NVIDIA driver release, and CUDA driver and runtime API
versions. A new CUDA writer durably publishes a versioned digest of these values
in `.simulate-cuda-compatibility` before frame zero. Resume, replay, step and
cache-backed preview check that record before solver work. A CUDA cache with a
missing, malformed or incompatible record still supports portable `check` of its
committed frames. Create a fresh cache to run it under another CUDA environment.

`--trace` emits accepted-state records (clock, exact state digest, inventories,
source counters/remainders and attempt/retry counts) from the final schedule of each
frame. `published:false` distinguishes these from `frame` publication records.
The `accepted_measurement` object gives the exact accepted duration and the
step-start capillary bound in seconds, with JSON null when no bound applies.
It reports the pressure iteration count and the original integrated RHS and true
residual L2 norms in m³/s, including gauge rows. The maximum volume-scaled residual
and physical constraint defect use s⁻¹. An empty pressure domain reports zero
for those pressure measurements and for its frozen center-selected liquid volume
in m³. The signed step-start piston flux uses the original pressure RHS incidence
and blocked-face data in m³/s, positive inward. Non-tank scenes report zero piston
flux and null poses. Tank poses are the low-x wall coordinates in world meters
used by the start-of-step geometry and end-of-step contact.

A traced `run` emits an independent `attempt` record for each actual physical
attempt before the corresponding frame's accepted traces. The ledger reports the
target frame, one-based frame-local attempt number, zero-based capacity pass,
start clock, attempted duration and optional retry duration in seconds, monotonic
host work time in seconds, disposition and failure code. The work interval includes
physical attempt fences and excludes observation, replay growth and cache
publication. A capacity record identifies whether a restart occurred. An
accepted attempt is marked `retained` only if it survives in the settled frame
schedule. Retry, fatal and capacity work remains in the ledger. Fatal or cancelled
scheduling emits completed attempt records before its error. Accepted traces
require a settled schedule. Successful rollback keeps the prior pass in the
ledger and discards its accepted witnesses. The cumulative physical
`accepted_step` clock can repeat across passes in attempt start clocks and is
separate from frame-local `accepted_attempts`, which includes discarded work.

The run-only `--trace-geometry` option implies `--trace` and adds each accepted
particle ID and world position in meters, in committed particle storage order.
Use the previous settled trace or the verified initial FLST to recover step-start
particle geometry. Both trace options produce the same execution identity and
cache bytes as an untraced run.

Add `--measure-job` to `run`, `replay` or `step` for one terminal
`job_measurement` record immediately before the result or error. This option
is independent of tracing. The record carries the backend when resolved,
terminal status, total actual physical attempts and the number retained by
complete frame schedules. Accepted substeps in an interrupted or replayed-away
schedule contribute actual work. A single `step` stops inside a schedule.

The `host_wall` phases partition elapsed monotonic seconds from job creation
through teardown and terminal status persistence. Cold setup includes cache
and backend admission and initial reservations. Emit, activation and field
sampling include trial reset, topology and initial face tagging. The next
phases cover deposit and normalization, forces, geometry and pressure, and
gather, contact and acceptance. Scheduling covers work between attempts,
including retries, capacity growth and optional trace output. Snapshot and
cache time includes D2H polling, host verification or publication and frame
notifications. Drain includes owner teardown and the heartbeat join.

`cuda_events` sums measured dispatches through their existing completion
fences across all attempts. Event intervals include transfers and can include
submission gaps on their stream. The pre-topology field intervals are a subset
of emit and activation, and snapshot D2H intervals use the independent copy
streams, including completed canceled copies. Event durations overlap host wall
time, and copy streams can overlap each other. Setup and teardown have host wall
intervals. CPU jobs emit null for both CUDA sections.

`cuda_memory` uses the shared arena and snapshot ledger. Current bytes are
sampled immediately before CUDA teardown. Peaks include simultaneous current,
prospective and leased retired owners and all snapshot queues. The receipt also
reports remaining owned bytes after drain, quarantined owners, and the integer
floor of the smaller of 70% of total device bytes and 80% of free device bytes at
job start. Growth checks the fixed effective ceiling before attempting
allocation. Physical state bytes are the populated state-bank payload,
including the final occupied particle bin. Device scratch covers the owned
scratch slices, with a separate peak of checked active extents. Snapshot output
covers pending, ready and writer-borrowed logical D2H payload. Snapshot queues
reserve pinned host bytes and borrow their device source from the arena.

Free-device observations are system-wide and can include concurrent activity.
CUDA context, module and graph bookkeeping are opaque. The application
reservation figures attribute only requested device and pinned payload.

Result records include `frames_per_second`, resolved/execution identity and backend;
CUDA results additionally carry actual compiled architecture, graph replay count,
reserved device bytes and D2H event milliseconds. Point radius is the spherical
radius of accepted volume V0*J in meters. Snapshots use the real bounded asynchronous
queue: all source copies drain before the next whole-frame advance, while the host
writer lease remains live through manifest publication. `LiquidProgressObserver3D`
runs outside physical attempts after acceptance or rollback. It checks for
cooperative cancellation without taking ownership of a device source lease.

During `run`, the independent worker targets a heartbeat every 500 ms. It
reports monotonic `elapsed_milliseconds`, the known `backend`, the
`last_settled_clock` (frame, substep, epoch and accepted step), and the last
confirmed `last_published_frame`. Unknown fields are `null`. The settled clock
advances only after a complete frame schedule because capacity replay can
discard its provisional substeps. Published progress advances only after a
verified checkpoint read or successful frame publication. A live reader can
inspect committed frames while a job runs. The short publication lock can
return `ABORTED` if the reader races with a publication.

Send `cancel` followed by LF or CRLF over redirected stdin. EOF and unrelated
lines leave the job running. The solver checks at safe boundaries outside
physical attempts, drains any active snapshot before releasing its resources,
and stops without waiting for stdin to close. Clients must keep reading stdout
during a run so protocol writes can complete. Previously committed frames
remain readable. After creating a fresh cache, the writer finishes publishing
its frame-zero recovery point before honoring a pending cancellation.

The exclusive OS lease in `.simulate-job.lock` prevents a second CLI writer for
the same cache. It is separate from the short publication lock and releases
on process exit. The atomically replaced `.simulate-job-status` marker records
run status. Neither file contributes to fingerprints, immutable frame data,
or replay identity. Preview has a heartbeat and cancellation without writing
operational files.

Nine mandatory CTest `cli` cases launch real executables for arguments, schema,
identity, inspection, CPU selection, CPU execution, CPU replay, validation and
process lifecycle. Four separate `cuda-cli` cases require the real CUDA probe,
execution, replay and process lifecycle.
The locked validator requires exact inventories, process receipts, qualified image
identity, positive actual graph/device reservations, finite positive D2H
intervals and measured heartbeat and cancellation bounds.
The GPU and sanitizer inventories contain 35 and 33 cases, respectively. The Python driver
is invoked with required binary paths, rejects optimized Python, and never silently
skips an unavailable executable or device.

### Committed-state reliability campaign

After building either locked Windows CUDA preset, run the longer F9 campaign
separately from the quick CLI and operator suites. It uses the production CLI
for 17 cooperative cancellations and 17 process kills. A test-only executable
reports 16 injected failures after a successful real CUDA snapshot event.
The campaign verifies every surviving manifest and point payload, then resumes
each cache through frame 10 and compares the checkpoint and frame bytes against
an uninterrupted execution of the same artifact. It also delays the initial
CUDA snapshot and checks that heartbeat output continues during the wait.

```powershell
uv run --no-sync --project reference --locked --group native ctest --preset windows-release-cuda -L '^cuda-reliability$' --no-tests=error -V
```

The test prints a JSON receipt with the producing device, each stopped and
recovered frame inventory, the recovered state digests, and the CUDA event
timing. The registered campaign accepts a real CC12.0 device with a matching
executed image. Qualification of the reference machine requires the receipt's
device name to equal `NVIDIA GeForce RTX 5060 Ti`. Direct qualification runs can
enforce this by passing `--expected-device-name "NVIDIA GeForce RTX 5060 Ti"` to
`tools/native/tests/run_liquid_fault_campaign.py`. Keep the verbose log under the
ignored output directory. The test fixture injects device errors after actual
CUDA completions.
