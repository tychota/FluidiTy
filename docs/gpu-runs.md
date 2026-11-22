# GPU runs

The hardware run happens once, after all modules are written. Until then every GPU result in this repository is
"compiled, not run": the code builds, and no test has run on a GPU.

GPU targets: the CUDA presets build for RTX 30, 40 and 50 series cards (`sm_86`, `sm_89`, `sm_120`). None of them is
tested yet. The RTX 5060 Ti is the card of the planned run.

## How to run it

On the Windows machine with the RTX 5060 Ti, in PowerShell 7 (`pwsh`), from the repository root:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/run_gpu_tests.ps1
```

It needs `git`, `uv`, `nvcc` 12.8 or newer (older versions cannot build for `sm_120`), `nvidia-smi`, Visual Studio with
the C++ tools, and the folder of the CUDA runtime DLL (`cudart64_*.dll`) on `PATH`; each missing piece prints its own
`FAIL` line before any test runs. It builds the preset `windows-release-cuda`, runs the `cpu` tests, runs the `gpu`
tests twice, and runs each GPU test program under `compute-sanitizer`. The tests use the GPU that `nvidia-smi` lists as
index 0 (the script sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`), or the first index of `CUDA_VISIBLE_DEVICES`; the `gpu:` line
prints it. Test labels: `cpu` tests run everywhere; `no_gpu` tests run only in the CUDA container (a machine without a
GPU, because they assert that no GPU is usable), so this script never runs them; `gpu` tests run only on hardware. The
first run builds the Conan packages, so it takes a while.

Paste back the block between the two lines of three backticks that the script prints last. It looks like this
(placeholders in angle brackets):

```text
FluidiTy GPU run
commit:     <full hash> (<clean|modified>)
gpu:        <name>, driver <version>, index <n>
toolkit:    <nvcc release line>
build:      PASS|FAIL
cpu tests:  <passed>/<total> PASS|FAIL
gpu tests:  <passed>/<total> PASS|FAIL
repeat run: PASS <n> identical digests|FAIL <reason>
notes:      <n> lines in notes-first.txt, <m> discrete mismatches[ (send notes-first.txt)]|none written|NOT RUN
sanitizer:  see the forms below
result:     PASS|FAIL
```

The `sanitizer:` line takes one of these forms:

- `PASS 0 errors`
- `FAIL <n> errors in <k> executables`
- `FAIL 0 errors counted, <n> executables exited with failure`
- `FAIL not installed`
- `FAIL no gpu test programs found`
- `FAIL not run`

A `gpu tests:` line reads `<passed>/<total> FAIL <n> skipped or disabled` when tests did not run: a skipped or
disabled test is not a pass.

A `cpu tests:` line reads `FAIL cudart64_*.dll is in no folder of PATH` when the CUDA runtime DLL check failed: the
build passed but no test could start, so no test ran and the later test lines read `FAIL not run`.

The repeat run means the GPU wrote the same bytes twice. For each of the two runs the script sets
`FLUIDITY_RESULT_DIGEST_FILE` to its own file in the log folder (`digests-first.txt`, `digests-second.txt`); the
`gpu` tests of the buffers, the sums and the stable order append a named digest of every GPU result they check.
`repeat run: PASS` requires the same tests with the same outcomes and two identical, non-empty digest files. The
contact tests add theirs too.

Some gpu tests also write notes: a time per item, or a finding that does not fail the test but that must be read.
Each of the two runs appends them to its own file in the log folder, `notes-first.txt` and `notes-second.txt` (the
script sets `FLUIDITY_RESULT_NOTES_FILE`; the tests call `WriteResultNote` and `WriteResultTiming`). Notes are never
compared and never decide PASS or FAIL. The `notes:` line of the block counts the lines of `notes-first.txt` and,
among them, the discrete mismatches (below). Send `notes-first.txt` with the block, always: it holds the timings.

## What to look for: contact (`move_particles_gpu_test`, `shared_formulas_gpu_test`)

The GPU half of `MoveParticles` (`src/fluidity/contact/move_particles_gpu.cu`) and the shared contact formulas run
for the first time here. A pass looks like this:

- `gpu tests:` counts the tests of both programs among the passed ones: 12 in `move_particles_gpu_test` and 1 in
  `shared_formulas_gpu_test`. A failure of `MoveParticles.SweepAndMoveCasesOnGpu_MatchCpu`,
  `ObliqueCasesOnGpu_MatchCpu`, `RoundCasesOnGpu_MatchCpu` or `MillionParticlesOnGpu_MatchCpuAndTimed` means a GPU
  result differs from the CPU's: by bits for a scene with no turning or spinning solid, beyond the GPU bounds of the
  method paper's contact section otherwise. The log (`ctest-gpu`) lists up to 20 failing scenes, each with its set, case
  index, scene name where it has one, field, CPU value, GPU value (17 digits and hexadecimal, so they read back
  exactly), the difference and the bound, and says how many more were cut.
- `repeat run: PASS`: the digests of the contact tests (`MoveParticles on the GPU, ...`,
  `MoveParticlesOnGpuInLaunchSize, ...`, `Shared contact formulas on the GPU, 32 inputs`) are the same in both runs.
- `notes:` with 0 discrete mismatches, and in `notes-first.txt`:
  - one line per set, `MoveParticles, catalogue: 0 discrete mismatches in 25 scenes with a turning or spinning solid
    (allowed 2), 0 failing scenes`, and the same for the oblique and round cases (no turning solid there);
  - `MoveParticles visit stack: <bytes> bytes x <threads> resident threads`;
  - the timings `MoveParticles, 1,000,000 particles, <n> solids, GPU` and `..., CPU one thread` for 1, 8 and 32
    solids. For reference, one thread of an Apple M3 Pro took about 100, 700 and 2,800 ns per particle on a similar
    scene. The GPU's own ratio of 32 to 1 solids shows whether the far-solid rule works there as on the CPU.
- `sanitizer: PASS 0 errors`. Under memcheck `move_particles_gpu_test` makes about 20,500 calls (each a dozen
  allocations, copies and four launches): expect 10 to 40 minutes for it alone, against 2 to 4 minutes per normal
  run. Let it finish on the first run: it is the only check of the GPU half's memory accesses (the reuse of the
  failure buffer, the atomics). If time is short, a run whose other lines pass is still worth sending, with the
  sanitizer line as it ends.
- In the log of the first gpu run (`ctest-gpu`), the lines `largest difference ... m, ... m/s, ...` per set: 0 for
  every `no turn` line (anything else is a failure already), and the size of the last-bit differences of the turning
  and spinning catalogue scenes and of the shared formulas, which calibrates the bounds.

### Discrete mismatches on turning and spinning scenes

The catalogue has 25 scenes with a turning or spinning solid. Their poses go through a sine, whose last bits differ
between the CPU's library and the GPU's, so their positions and velocities are compared within the GPU bounds of the
method paper's contact section. A scene built on a threshold can then end with another discrete result on the GPU:
another face, kind, solid index, hit count or error. Such a scene is a discrete mismatch: the test does not fail on it,
but writes one line per differing field to `notes-first.txt`, `discrete mismatch, catalogue, case <i> <scene>: <field>:
cpu <value>, gpu <value>`, and counts it. The test fails when more of these scenes have a mismatch than one in ten,
rounded down (more than 2 of 25): a mistake in the GPU's turning code would change most of them. A reading of the scenes
expects 0 to 2 (an estimate: no GPU has run them), most likely in `SettleNearSpinningCylinder`, `TurningStackedBoxes`,
`TurningBoxNearFloor` or `TurningCylinderRimHit`.

A nonzero count is not a failure. It means the rule "discrete fields equal on both processors" of the method paper's
contact section does not hold for that scene on this card, which is a decision for the owner. Send `notes-first.txt`:
its lines name each scene and field with both values, enough to rebuild the case on the CPU. Scenes without angular
velocity stay exact: any difference there fails the test.

### First exercised on the owner's GPU

The CPU tests run the per-particle and per-thread functions over every thread block and thread, but these parts run
for the first time on the card. What a failure of each would look like:

- The 64-bit `atomicMin`, the 32-bit `atomicOr` and the 64-bit `atomicAdd`, through the cast to `unsigned long
  long` (`move_particles_gpu.cu`, `ToAtomicInteger`): `MoveParticlesOnGpuInLaunchSize.*` fail with a wrong particle id
  or wrong counts, or the counts of the set tests differ (`counts.hit_count: cpu ..., gpu ...`).
- The reuse of the first-failure buffer for the minima, read whole and then written over by one thread
  (`StartReduction`, `WriteFirstFailure`): the failure tests report a failure with nonsense fields (an error 255, a
  particle no test used), or the sanitizer reports an access in `WriteFirstFailure`.
- The early return of `FindSmallestFailingIndex` on the flag: a run without failure that reports one, or a failure
  reported as none (`result: cpu failure ..., gpu success`).
- The four launches, their status chain and the wait (`MoveParticlesOnGpuInLaunchSize`, `CheckGpuWork`), and the CPU
  half's GPU branch (`move_particles_cpu.cc`: reading the counts and the failure back): every `MoveParticles` test fails
  with `failure.error` `kRunOnCallFailed` while `SharedFormulas` passes.
- The launch bound and the stack: the visit function uses 255 registers and 12,784 bytes of stack per thread on
  `sm_120` (11,264 on `sm_86` and `sm_89`). `BuildContactBuffers` now sets the GPU's per-thread stack to that size
  (`cudaDeviceSetLimit`), and the driver takes it at once, about 0.7 GB on a 36-multiprocessor card, for the rest of the
  process: a card too full fails there with `kOutOfMemory` and the stack's bytes as magnitude, not at the first launch.
  A launch that still fails (registers, a stack the driver did not give) shows as `kRunOnCallFailed` on every
  `MoveParticles` test while `SharedFormulas` passes. Watch the GPU memory in Task Manager: about 0.7 GB stays taken.
- `MoveParticle` as GPU code: the `no turn` largest differences must be 0, and the catalogue, oblique and round sets
  pass bit for bit. A difference there is a broken same-bits promise: send the log lines.
- The shared formulas as GPU code (`testing/shared_formulas_runner_gpu.cu`): `SharedFormulas.OnGpu_MatchCpu` names the input and
  the value that differs.
- `run_gpu_tests.ps1` with its notes file and `notes:` line: if the block lacks the `notes:` line or reads `NOT RUN`
  after the gpu tests ran, send `notes-first.txt` and the script's output.

What a first run may show instead, and what it means:

- `MoveParticles` tests fail with `kRunOnCallFailed` and the shared formulas pass: the visit function did not launch
  or failed while it ran (above). Keep the logs.
- Only the turning or spinning scenes fail, by a few rounding units: the GPU bound of the method paper's contact section
  is too tight for this card's sine; the printed difference and bound say by how much.

## What to look for: grid (the GPU tests of the shared grid functions)

The grid tests build every grid on the GPU and again on the CPU, ask the same questions of both, and demand the same
bits: places, counts and every double (the weights and face weights included, since their formulas use only `+ - * /`,
`sqrt` and `floor` and the build forbids fused multiply-add). The programs are `build_grid_on_gpu_functional_test`,
`cell_size_rule_on_gpu_functional_test`, `find_cell_at_on_gpu_functional_test`,
`nearby_face_weights_on_gpu_functional_test`, `cell_surroundings_on_gpu_functional_test` and
`shared_grid_functions_on_gpu_functional_test` and `border_blocks_on_gpu_functional_test`. The GPU runners they share are compiled once, in the library
`fluidity_grid_testing`. What to look at first, in this order:

1. **Does it build on Windows?** The runners start a GPU function that takes a function pointer as a template argument,
   from an unnamed namespace, and the test `.cu` files get `-Xcompiler=/EHsc`. Only GCC has compiled both. If MSVC
   refuses, the fix is a small function type in place of the pointer, or one plain GPU function per runner.
2. **Run `build_grid_on_gpu_functional_test` before the others.** Every other grid test builds its grids on the GPU. If
   that build is wrong, they fail with confusing messages. Look at it first.
3. **`GridRunners.OnGpu_FunctionsLoadWithTheirBlockSize`.** A failure means the binary has no code for this card, or the
   driver is too old. It also writes one note per runner to `notes-first.txt`: the local memory and the registers the
   compiler gave that GPU function, and the most threads per block it allows. The weights runner and the face search
   runner hold the most local memory.
4. **Any test that stops on `kCallFailed` or `kOutOfMemory`.** The message now ends with the failed CUDA call, its error
   name and its text, for example `copy of the results to the CPU, which also waits for the GPU function:
   cudaErrorIllegalAddress (an illegal memory access was encountered)`. Each name needs another fix. An illegal address:
   run that program under `compute-sanitizer --tool memcheck`, which names the line. `cudaErrorLaunchTimeout`: the
   display driver stopped a long call; the face search walks are the longest, so try them first with the card not
   driving the display. `cudaErrorNoKernelImageForDevice`: no code for this card. `cudaErrorMemoryAllocation` on the
   copy or the start: the driver could not reserve the local memory of the threads; that memory is about 55,000
   threads times the local bytes of the notes of point 3. `CUDA_LAUNCH_BLOCKING=1` pins down which launch failed.
5. **A difference in bits.** The log names the query index, the field and both values in hexadecimal, for the first ten,
   then the count. If the doubles of the weights differ in the last bits for many queries, check the build log: that
   `--fmad=false` reached `nvcc`, and that MSVC was not given `/arch:AVX2` with `/fp:contract`. With those flags the
   bits must be equal. A difference in a place, a count or a mask is a table difference: compare with
   `build_grid_on_gpu_functional_test` first, since a wrong GPU table shows there before it shows here.

Two things about `border_blocks_on_gpu_functional_test` (`ComputeIsBorderBlock` on every block of 5 random grids, 3 block
sets each, both widths). Its GPU function (`ComputeBorderBlockResultsOnGpu`) has the biggest stack frame of the runners
after the weights runner: 3,664 bytes per thread and 255 registers on `sm_86` and `sm_89`, 2,680 bytes and 255 registers
on `sm_120` (`cuobjdump --dump-resource-usage`, no local memory beyond the stack). That is above the driver's default
1,024 bytes; the driver grows the stack at the first launch, as for the weights runner. The test may be slow on the GPU:
a block next to the set can cost up to about 2 million table reads (the `Work:` of `ComputeIsBorderBlock`), one thread
does all of them, and the threads of a warp that meet different blocks wait for the slowest. If `cudaErrorLaunchTimeout`
appears, run the card without a display attached.

To read what the compiler gave each GPU function of the test programs, without a GPU, run in the CUDA container, on the
programs of the `linux-release-cuda` build:

```sh
cuobjdump --dump-resource-usage <test program>
```

It prints, for each GPU function and each architecture, `REG` (registers), `STACK` (bytes of stack frame), `SHARED`,
`LOCAL` (bytes of local memory) and `CONSTANT`. The build log of the gate shows the same through `ptxas`: the runners'
`.cu` files are compiled with `-Xptxas=-v`.

## What the single run cannot show

The one planned run uses one RTX 5060 Ti (`sm_120`) on one Windows machine with one driver. It cannot show:

- that the `sm_86` (RTX 30) and `sm_89` (RTX 40) code paths run: they are compiled into the program but only an
  `sm_120` card executes code there;
- anything about a second GPU in the same machine, or about choosing a GPU other than index 0;
- a run on Linux with a GPU (the container compiles for Linux but has no GPU), or with another driver or toolkit;
- that results agree across GPU models: the promise is the same output on the same GPU, and only one GPU is used.

The full output of every command is in `out/gpu-run/<UTC time>/`; keep that folder if any line says FAIL. A line that
reads `FAIL not run` (or `NOT RUN`) means an earlier step failed or a tool is missing, and the script says which.

## Record

| module | commit | date | GPU and driver | result | notes |
| --- | --- | --- | --- | --- | --- |
| Foundation | not run | not run | none | compiled, not run | GPU test programs that exist now: `check_can_run_on_gpu_test`, `shared_math_gpu_test`, `buffer_gpu_test`, `ordered_sums_gpu_test`, `stable_order_gpu_test`, `processor_description_gpu_test`. `tools/run_gpu_tests.ps1` was parsed with PowerShell 7.4 in a Linux container and exercised there with stand-in tools; never run on Windows, never run against a GPU. |
| contact | not run | not run | none | compiled, not run | GPU test programs: `move_particles_gpu_test` (12 tests: `MoveParticles.SweepAndMoveCasesOnGpu_MatchCpu`, `MoveParticles.ObliqueCasesOnGpu_MatchCpu`, `MoveParticles.RoundCasesOnGpu_MatchCpu`, `MoveParticles.SameInputTwiceOnGpu_SameBytes`, `MoveParticles.TwoFailuresOnGpu_ReportsLowestParticleId`, `MoveParticlesOnGpuInLaunchSize.FailingParticlesAnywhere_ReportLowestIdInEveryLaunchSize`, `MoveParticlesOnGpuInLaunchSize.EqualIdsFailing_ReportsLowerIndexInEveryLaunchSize`, `MoveParticlesOnGpuInLaunchSize.NoFailure_CountsInEveryLaunchSize`, `MoveParticles.FailedCallCheckOnGpu_WritesNothing`, `BuildContactStackEstimate.OnGpu_IsTheVisitStackForEveryResidentThread`, `MoveParticles.EmptyRangeOnGpu_CountsNothing`, `MoveParticles.MillionParticlesOnGpu_MatchCpuAndTimed`) and `shared_formulas_gpu_test` (1 test: `SharedFormulas.OnGpu_MatchCpu`). `move_particles_gpu.cu` and `testing/shared_formulas_runner_gpu.cu` compile and link in the CUDA container; `move_particles_no_gpu_test` (label `no_gpu`, 2 tests) runs there. No contact device code has run on a GPU. What to look for in the first run: the contact part above. |
