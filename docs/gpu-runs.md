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

## The grid's GPU tests: how to run them and what to send back

### What may fail first: the build

No CUDA file of the grid, and no header it shares with the CPU, has ever been
compiled by `nvcc` with MSVC as the host compiler. CI compiles CUDA on Linux only, and Windows without CUDA. So the
first run may stop at `build: FAIL cmake build`, before any test. That is not a failed test: nothing ran. Send the
build log (the `cmake-build` file of `out/gpu-run/<UTC time>/`) whole, and the block. The first error in it is the one
to fix; the rest often follow from it.

All commands below are for PowerShell 7 (`pwsh`), from the repository root, on the machine with the RTX 5060 Ti.

1. Before anything, record the card's state and close every program that uses the GPU (browsers, games, video):

   ```powershell
   New-Item -ItemType Directory -Force out | Out-Null
   nvidia-smi > out\nvidia-smi-before.txt
   ```

2. The whole run: build, `cpu` tests, `gpu` tests twice, sanitizer. The grid's GPU tests are among them.

   ```powershell
   pwsh -NoProfile -ExecutionPolicy Bypass -File tools/run_gpu_tests.ps1
   nvidia-smi > out\nvidia-smi-after.txt
   ```

3. After the script has built once, the grid's GPU tests alone (two `-L` options mean "both labels"):

   ```powershell
   uv run --frozen ctest --preset windows-release-cuda -L '^gpu$' -L '^grid$'
   ```

   The script's shell had loaded the Visual Studio environment and found the CUDA runtime DLL; a new shell needs the
   folder of `cudart64_*.dll` on `PATH` again (`$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"`, or `bin\x64` in newer
   toolkits).

4. One program alone, through ctest (prints its output when it fails):

   ```powershell
   uv run --frozen ctest --preset windows-release-cuda -R '^build_grid_on_gpu_functional_test$' --output-on-failure
   ```

   Or the program itself, with one TEST, and with the notes and digests written where you can find them:

   ```powershell
   $env:FLUIDITY_RESULT_NOTES_FILE = "$PWD\out\notes-alone.txt"
   $env:FLUIDITY_RESULT_DIGEST_FILE = "$PWD\out\digests-alone.txt"
   .\out\build\windows-release-cuda\tests\grid\build_grid_on_gpu_functional_test.exe --gtest_filter='BuildGrid.OnGpu_EqualsCpuTablesBitForBit'
   ```

   The programs of `tests/grid/` are in `out\build\windows-release-cuda\tests\grid\`; `grid_values_gpu_test.exe` is in
   `out\build\windows-release-cuda\src\fluidity\grid\`. Without `--gtest_filter` the program runs all its TESTs.

5. To find which launch failed, run one program with every launch waited for:

   ```powershell
   $env:CUDA_LAUNCH_BLOCKING = '1'
   .\out\build\windows-release-cuda\tests\grid\stacked_borders_on_gpu_functional_test.exe
   Remove-Item Env:CUDA_LAUNCH_BLOCKING
   ```

6. One program under the memory checker, as the script runs it:

   ```powershell
   compute-sanitizer --tool memcheck --error-exitcode 1 .\out\build\windows-release-cuda\tests\grid\grid_pieces_on_gpu_functional_test.exe
   ```

   `compute-sanitizer.exe` is in `$env:CUDA_PATH\compute-sanitizer\` when it is not on `PATH`.

### Run times

No GPU program has run, so their times are unknown. The table gives, as a hint of the CPU side's share,
the time of the CPU program that checks the same functions, measured by the release gate on an Apple M3 Pro with every
core busy (a GPU program builds its grids on the CPU too). ctest stops a program after 1,800 s and counts it failed.
Under the memory checker a program runs 10 to 100 times slower (unknown here); the script has no time limit for that
step. If one program has run under the checker for more than two hours, end the `compute-sanitizer.exe` process in Task
Manager: the script counts that program as failed and goes on, and still prints the block. Pressing Ctrl+C in the
script's window stops the script and loses the block.

| program | TESTs | GPU time | CPU program of the same functions, release, M3 Pro |
| --- | --- | --- | --- |
| `build_grid_on_gpu_functional_test` | 11 | unknown | `build_grid_functional_test`, 7 s |
| `grid_cell_checks_on_gpu_functional_test` | 9 | unknown | `grid_cell_checks_on_random_grids_functional_test`, 5 s |
| `cell_size_rule_on_gpu_functional_test` | 2 | unknown | `cell_size_factor_on_many_inputs_functional_test`, under 0.1 s |
| `find_cell_at_on_gpu_functional_test` | 2 | unknown | `find_cell_at_near_form_on_positions_functional_test`, 3 s |
| `new_cell_neighbors_on_gpu_functional_test` | 1 | unknown | `new_cell_neighbors_on_random_grids_functional_test`, under 1 s |
| `nearby_face_weights_on_gpu_functional_test` | 4 | unknown | `nearby_face_weights_on_random_grids_functional_test`, 5 s |
| `cell_surroundings_on_gpu_functional_test` | 7 | unknown | touching cells 5 s, corners 4 s, face search 3 s |
| `shared_grid_functions_on_gpu_functional_test` | 4 | unknown | none |
| `border_blocks_on_gpu_functional_test` | 2 | unknown | `border_blocks_against_geometry_functional_test`, 23 s |
| `stacked_borders_on_gpu_functional_test` | 15 | unknown | `stacked_borders_in_grid_pieces_functional_test`, 118 s |
| `grid_pieces_on_gpu_functional_test` | 9 | unknown | `grid_pieces_on_random_grids_functional_test`, 53 s |
| `grid_piece_steps_on_gpu_functional_test` | 1 | unknown | `grid_piece_steps_in_every_thread_order_functional_test`, 16 s |
| `apply_grid_version_on_gpu_functional_test` | 1 | unknown | `apply_grid_version_on_random_grids_functional_test`, 2 s |
| `sloped_surface_scene_on_gpu_functional_test` | 1 | unknown | `sloped_surface_scene_functional_test`, 14 s (about 6 s alone) |
| `grid_values_gpu_test` | 1 | unknown | none (a unit test) |

### A card that also drives the display

Three things differ from a card that only computes:

- *Free memory.* Five tests compare the GPU's free bytes before and after a piece of work and demand that they be
  equal, because their point is that the work takes no GPU memory at all: `BuildGrid.IntoGpuBuffers_AllocatesNothing`
  and `ComputeMemoryBytes.MatchesAllocatedGpuBytes` (`build_grid_on_gpu_functional_test`),
  `CheckSizeStepRule.OnGpu_AllocatesNothing` (`grid_cell_checks_on_gpu_functional_test`),
  `ReserveBorderSearchStackOnGpu.Twice_TakesNothingTheSecondTime` and `BuildBorderSearchMemoryEstimate.MatchesAllocatedGpuBytes`
  (`stacked_borders_on_gpu_functional_test`). They stay exact: any tolerance large enough for another program would
  hide an allocation of the grid's own, which is what they look for. The free bytes are the card's, not the program's,
  so another program that allocates or frees GPU memory at that moment (the desktop, a browser, a video) changes them
  too. Run with no other program using the GPU. A difference in these lines alone, on a card that drives the display,
  is not a defect by itself: run that program alone (point 4) once or twice. A difference that comes back with the
  same number of bytes is the grid's. Each failure message gives both numbers and the difference.
- *The display timeout.* Windows stops a GPU call that keeps a display card busy for more than about 2 seconds
  (Microsoft calls it TDR; the registry value `TdrDelay` under
  `HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers`, in seconds, default 2, read at start-up).
  The error is `cudaErrorLaunchTimeout`, and from then on the program's GPU context is dead: every later TEST of that
  program fails too, with errors that say nothing. Only the first failing TEST of a program can be trusted; read it
  first. The launches most at risk are not cut into short pieces: the border runner of
  `border_blocks_on_gpu_functional_test`, the face search walks of `cell_surroundings_on_gpu_functional_test`, and a
  dense marking launch of 2,048 blocks in `stacked_borders_on_gpu_functional_test`. Best: drive the display from
  another GPU (the processor's own graphics) during the run. Else raise `TdrDelay` (for example to 60) and restart
  Windows, run, and set it back. Say which was done.
- *A skipped line.* A program whose output contains `[  SKIPPED ]` is reported by ctest as skipped as a whole, even if
  other TESTs of it failed, and its output is not printed. The script then reads `gpu tests: <passed>/<total> FAIL <n>
  skipped or disabled`: that program's outcome is unknown, not passed. Run it alone (point 4) and send its full output.
  No grid GPU test skips itself on purpose now: the one that did
  (`ReserveBorderSearchStackOnGpu.FromTheDefault_TakesAboutTheEstimatedBytes`, when the driver gave up no bytes) passes
  instead and writes the note `border search stack: NOT CHECKED, ...` (below).

### Stack sizes

The tests that pin the bytes of stack per thread a GPU function needs
(`BuildGrid.OnGpuFunctions_KeepUnderTheDefaultStack`, `CheckSizeStepRule.OnGpuFunctions_KeepUnderTheDefaultStack`,
`BuildGridPiece.OnGpuFunctions_KeepUnderTheDefaultStack`, `FindBorderBlocks.OnGpuFunctions_KeepTheStatedStack`,
`BuildBorderSearchStackEstimate.OnGpu_IsTheStackForEveryResidentThread`) compare the driver's number with limits read
from code built by `nvcc` 13.4.59 (the CUDA container) for `sm_120`, `sm_86` and `sm_89`. The script allows `nvcc` 12.8
or newer, and another version may give other numbers. A failure message gives the measured bytes, the GPU's name and
the CUDA runtime version; the RTX 5060 Ti is `sm_120`. A value a little above the limit with another toolkit is a
finding to send back, not yet a defect; a value of 0 for the border search fails, because the estimate would then be 0.

### What to send back

After any run, PASS or FAIL:

1. The block the script prints last, between the two lines of three backticks.
2. The whole folder `out/gpu-run/<UTC time>/`, zipped: the logs of every command, `summary.txt`, `notes-first.txt`
   (the timings and findings, including the free-memory and stack notes), `digests-first.txt` and `digests-second.txt`.
3. `out\build\windows-release-cuda\Testing\Temporary\LastTest.log` (the full output of every test of the last ctest run)
   and the folder `out\build\windows-release-cuda\gtest_reports\` (one XML file per program, with each TEST's time and
   the recorded stack bytes, `largest_stack_bytes`).
4. For a program that failed or was skipped: its full output from running it alone (point 4); for a GPU call error,
   also its output with `CUDA_LAUNCH_BLOCKING=1` (point 5).
5. `out\nvidia-smi-before.txt` and `out\nvidia-smi-after.txt`.
6. If the build failed: the `cmake-build` log of the run folder, whole (it is in the zip of point 2, but say so).
7. The versions: Windows (`winver`), Visual Studio and MSVC (`cl` prints its version), and whether the card drove the
   display during the run.

## What to look for: grid (the GPU tests of the shared grid functions)

The grid tests build every grid on the GPU and again on the CPU, ask the same questions of both, and demand the same
bits: places, counts and every double (the weights and face weights included, since their formulas use only `+ - * /`,
`sqrt` and `floor` and the build forbids fused multiply-add). The programs are `build_grid_on_gpu_functional_test`,
`grid_cell_checks_on_gpu_functional_test`, `cell_size_rule_on_gpu_functional_test`,
`find_cell_at_on_gpu_functional_test`, `nearby_face_weights_on_gpu_functional_test`,
`cell_surroundings_on_gpu_functional_test`, `shared_grid_functions_on_gpu_functional_test`,
`border_blocks_on_gpu_functional_test`, `stacked_borders_on_gpu_functional_test`,
`grid_pieces_on_gpu_functional_test`, `grid_piece_steps_on_gpu_functional_test`,
`apply_grid_version_on_gpu_functional_test` and `sloped_surface_scene_on_gpu_functional_test` of `tests/grid/`, and
`grid_values_gpu_test` of `src/fluidity/grid/`. The GPU runners they share are compiled once, in the library
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

Two things about `stacked_borders_on_gpu_functional_test` (the border search of a range on the GPU, `FindBorderBlocks` on
a GPU grid, compared with the CPU's list and count by bits). The GPU function that marks, one thread per place of the set
(`WriteBorderReadMarks` in `cell_block_range_gpu.cu`), keeps the results of the readers as locals: 3,600 bytes of stack and
255 registers per thread on `sm_120`, 3,048 bytes and 255 registers on `sm_86` and `sm_89` (`cuobjdump
--dump-resource-usage`, release build, no local memory beyond the stack); the five others keep no stack. That is above the
driver's default 1,024 bytes. `BuildBorderSearchBuffers` raises the stack when the buffers are built (`ReserveBorderSearchStackOnGpu`,
as contact does), so a card too full for it fails there with `kOutOfMemory`. The memory plan counts it with
`BuildBorderSearchStackEstimate`, a driver stack entry of that many bytes for every resident thread (about 55,296 threads
on a 36-multiprocessor card, 199 MB). The driver keeps one stack for the largest need of any GPU function, so summing this
entry with contact's counts more than the driver takes. The marking runs in launches of at most 2,048 blocks of the set
each (`kSetBlocksPerMarkingLaunch`), each waited for, and a set of more than 14,128,181 blocks fails with
`kTooManyCellBlocks` before any launch.

What to look at first, in this order:

1. **A launch timeout.** Run the whole program, release preset, with no display on the card if you can. A
   `cudaErrorLaunchTimeout` means one marking launch of 2,048 dense blocks took about 2 seconds; the test grids should
   not, so tell us, and `kSetBlocksPerMarkingLaunch` should come down.
2. **Empty lists, or a failure at the first marking call.** If every list is empty and `FindLayerOrFail` prints an
   error name, the first marking launch failed. Rerun with `CUDA_LAUNCH_BLOCKING=1` to see which launch.
3. **The stack digest.** `FindBorderBlocks.OnGpuFunctions_KeepTheStatedStack` writes the local bytes per thread. It should
   be 3,600. That number was read from code built by `nvcc` 13.4.59; another toolkit may give another one
   ("Stack sizes" above). If it is 0 or far below the `STACK` of `cuobjdump`, the estimate is wrong.
4. **Resident threads and the bytes taken.** `ReserveBorderSearchStackOnGpu.FromTheDefault_TakesAboutTheEstimatedBytes`
   writes a note with the bytes the GPU gave up and the estimate. Resident threads should be 55,296 on the 5060 Ti and the
   bytes about 199 MB (a little less: the driver may already hold the default 1,024 bytes). If the GPU gave up no bytes
   at all, the test does not skip (a skip would hide the whole program): it passes and writes the note `border search
   stack: NOT CHECKED, ...`. Send that note: it means the driver takes the stack later than the reserve, or an earlier
   call raised it. The tolerance is 25 percent; if it fails, the note says by how much.
5. **A list that differs from the CPU's.** A block missing from the GPU list points to the marking (the byte store, or
   the number of places). An extra block points to the clear of the marks or a stale word of the scan. A right set of
   blocks in the wrong places points to the prefix sum, which runs in place on `block_count + 1` words.
6. **`OnGpuTwice_GivesIdenticalBytes`.** A difference means something depends on timing. Nothing should. Suspect the clear,
   or the work areas of the prefix sum.
7. **`OnGpuFailures_EqualCpuFailures`.** All four bad lists and the same buffer must give `kBadInput` on both processors,
   the output buffer must be as it was, and the call after each failure must give the CPU's list. `kRunOnCallFailed` here
   means the atomic minimum on the key or the reset.
8. **`OnGpu_AllocatesNothing`.** It makes one call before counting. If it fails now, something in the search allocates on the
   CPU.
9. **`MatchesAllocatedGpuBytes` and `HandsOutExactlyTheEstimate`.** The free-byte equality after the build. On a card
   that drives the display, read "A card that also drives the display" above before taking a difference as a defect.
10. **`OnGpuMoreBlocksThanOneLaunch_SameBitsAsCpu`.** It runs a set larger than one marking launch. A difference here and
    nowhere else points to the split into launches.

On another card (`sm_86`, `sm_89`) only compiled code exists: reread points 3 and 4 there.

Two things about `grid_pieces_on_gpu_functional_test` (the buffer form of `BuildGridPiece` on a GPU grid, compared with the
CPU's piece table for table by bits: version, counts, start of the range, piece map, block ids, the 36 neighbour entries,
cell states, face piece kinds). Each of its six GPU functions (`grid_piece_gpu.cu`) is a thin loop of one thread per
border entry, loaded item, piece block, cell place or face place. They keep no stack: the gate's `ptxas` lines give 0
bytes of stack, no spill and at most 40 registers per thread on `sm_86`, `sm_89` and `sm_120`, so there is no driver
stack entry and no reserve step. The face launch is not cut into chunks: a thread does a bounded amount of work, but a
piece of the largest room (17,895,697 blocks, 4.3 billion face places) is one launch like the grid build's own; if
`cudaErrorLaunchTimeout` ever shows there, cut it as the border marking is.

What to look at first, in this order:

1. **`BuildGridPiece.OnGpuFunctions_KeepUnderTheDefaultStack`.** The stack digest should be 0. A value above 1,024 means
   the compiler changed and the piece build needs a stack reserve and an estimate entry as the border search has.
2. **A failure of `OnGpuRandomGrids_EqualCpuPiece` or `OnGpuNestedBallRanges_EqualCpuPiece`.** The log names the piece,
   the table, the place and both values. The piece number counts two pieces for each tested range, the one with the
   near border and then the one with the stacked border, in the order the tests build their ranges
   (`BuildTestedRanges`). Read the failure by table:
   - `range_first_block` wrong: the count of border entries below the range, made by function 2.
   - `map` wrong: function 3. A border list that was not ascending would also stop it.
   - `neighbours` wrong, and nothing else: function 4.
   - `active_cells_per_size_level` wrong while `cell_states` are right: the 64-bit adds of function 5.
   - `face_pieces` wrong while `face_piece_kinds` are right: the sum of function 6.
   - Every table wrong at once: suspect the read back of the GPU's piece, or the one-cell room, before the six GPU
     functions.
3. **`OnGpuFailures_EqualCpuFailures` and `OnGpuAfterFailure_BuildsTheCpuPiece`.** Each bad border list must give
   `kBadInput`, too little room `kTooManyCellBlocks`, with cells zero, as on the CPU; the second test builds a good
   piece into the same buffers after a failure. `kRunOnCallFailed` here means a GPU function did not start or the
   GPU work failed. `kInternalFault` means the code that reads the status record rejected it; the CPU runs that code on
   the replayed record (`grid_piece_steps_in_every_thread_order_functional_test`), so a difference points at the GPU's
   record, not at the reading.
4. **`OnGpuTwice_GivesIdenticalBytes` and `OnGpu_AllocatesNothing`.** A difference between the two passes means a result
   depends on thread timing; nothing should. The allocation test makes one call before counting.
5. **`OnGpuMoreFacePlacesThanThreads_EqualCpuPiece`.** 72,000 blocks, more face places than threads of the largest
   launch, so each thread of function 6 has several places. A difference only here points to the grid-stride loop.
   For each table that differs the log gives: the table, the range tested (first block and block count), how many
   entries differ, and for the first and the last differing entry its index, its piece block, its place in the
   block, its thread (the index modulo 16,776,960) and both values in hexadecimal. Indexes at or above 16,776,960 in
   `face piece kinds` (17,280,000 places) point to the second pass of the grid-stride loop; a scatter of entries
   points to a different defect. Write down the run time too. The test needs several gigabytes of GPU memory for the two
   grids, their pieces and the read back copies; the largest room a grid may hold (17,895,697 blocks) needs more
   than 16 GB, so no test builds it and it cannot happen on the RTX 5060 Ti with 16 GB.

One thing about `sloped_surface_scene_on_gpu_functional_test` (one test, `StaggeredGrid.SlopedSurfaceInBoxOnGpu_EqualsCpu`):
it runs the pieces of the grid module together on one scene, a box of liquid with a sloped surface (9,584 cells in 218
blocks, 28 ranges of 8 blocks, 1,000 positions). It builds the scene's grid on both processors, then compares by bits the
whole grid read back, each layer of each range's stacked touching border, each range's grid piece read back (counts,
piece map and a digest of every table), and the weights of the three axes at the 1,000 positions, asked on the GPU's whole
grid and in the GPU's piece of the range that holds each position. What to look at first, in this order:

1. **Run the single-purpose programs first** (`build_grid_on_gpu_functional_test`, `stacked_borders_on_gpu_functional_test`,
   `grid_pieces_on_gpu_functional_test`, `nearby_face_weights_on_gpu_functional_test`). A failure there names the one
   function at fault with a smaller input; this test fails the same way on a bigger one.
2. **Which comparison failed.** The message names the comparison: `weights of the 1,000 positions in the whole grid`
   (the GPU weights function on the whole grid), `layers of the stacked borders` (the border search, one call per layer,
   each on the list of the layer before), `grid pieces` (the piece build and its read back) or `weights of the positions
   in the piece of their range` (the same weights function, on a piece's tables). Whole grid weights wrong while the
   build test passes: the weights function and the sheet's mix of three size levels, which random grids reach less. A
   piece wrong in `map_digest` or a table digest: see the points of `grid_pieces_on_gpu_functional_test`. A layer wrong
   only in a late call: the list of the layer before was not read as it was written.
3. **The whole grid read back first.** `ReadGrid` of the GPU grid is compared with the CPU's tables before any range; a
   failure there is in the build, not in the borders or pieces.
4. **Time.** The CPU takes about 6 s in release. The GPU runs 28 pieces and 28 weight launches of about a hundred
   queries each (three per position), so a launch of the weights function that takes seconds points to the stack, as in
   `nearby_face_weights_on_gpu_functional_test`.

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
| grid | not run | not run | none | compiled, not run | GPU test programs, each one ctest entry, with the `gpu` tests of each (69 tests in all): `grid_values_gpu_test` (1 test: `WriteValues.OnGpuThenReadValues_ReturnsTheSameValues`); `apply_grid_version_on_gpu_functional_test` (1 test: `ApplyGridVersion.OnGpuGrids_KeepEveryValueAtItsPlace`); `border_blocks_on_gpu_functional_test` (2 tests: `ComputeIsBorderBlock.OnGpu_SameBitsAsCpu`, `ApplyCellPlaceReadMarks.OnGpu_SameMarksAsCpu`); `build_grid_on_gpu_functional_test` (11 tests: `BuildGrid.OnGpu_EqualsCpuTablesBitForBit`, `BuildGrid.OnGpuFailures_EqualCpuFailures`, `BuildGrid.OnGpuFailedBuildThenRebuild_EqualsFreshBuffers`, `BuildGrid.RebuildIntoUsedGpuBuffers_EqualsFreshBuffers`, `BuildGrid.IntoGpuBuffers_AllocatesNothing`, `ComputeMemoryBytes.MatchesAllocatedGpuBytes`, `BuildGridBuffers.OnGpuIntoRunMemory_HandsOutExactlyTheEstimate`, `ComputeGridCounts.OnGpu_EqualsCpu`, `BuildGrid.OnGpuTwice_GivesIdenticalBytes`, `BuildGrid.OnGpuSharedBuilderBuffers_OtherLiveGridUnchanged`, `BuildGrid.OnGpuFunctions_KeepUnderTheDefaultStack`); `cell_size_rule_on_gpu_functional_test` (2 tests: `ComputeWantedCellWidth.OnGpu_SameBytesAsCpu`, `ComputeNewCellSizeLevel.OnGpu_SameBytesAsCpu`); `cell_surroundings_on_gpu_functional_test` (7 tests: `FindCornerPlaces.OnGpu_EqualsCpu`, `ComputeCornerPosition.OnGpu_SameBitsAsCpu`, `FindCellsAroundEdge.OnGpu_EqualsCpu`, `FindTouchingCells.OnGpu_EqualsCpu`, `FindFacePieces.OnGpu_EqualsCpu`, `FindCellsNearFace.OnGpu_EqualsCpu`, `BuildFaceSearch.OnGpu_WalksLikeCpu`); `find_cell_at_on_gpu_functional_test` (2 tests: `FindCellAt.OnGpu_EqualsCpu`, `FindCellAt.OnGpuWithNoQuery_IsDone`); `grid_cell_checks_on_gpu_functional_test` (9 tests: `CheckSizeStepRule.OnGpu_EqualsCpu`, `CheckSizeStepRule.OnGpuIdWithBitSixtyOneSet_IsBadInputNamingIt`, `CheckSizeStepRule.OnGpuTooLongListWithBadId_SameFailureAsCpu`, `CheckSizeStepRule.OnGpuSeveralFailures_SameFirstFailureAsCpu`, `CheckSizeStepRule.OnGpuTwice_GivesIdenticalFailure`, `CheckSizeStepRule.OnGpu_AllocatesNothing`, `CheckSizeStepRule.OnGpuFunctions_KeepUnderTheDefaultStack`, `BuildSortedCellList.OnGpu_SortedListEqualsCpu`, `CheckSizeStepRule.OnGpuLargeListFailingAtLastPosition_SameFailureAsCpu`); `grid_piece_steps_on_gpu_functional_test` (1 test: `BuildGridPiece.StepsOnGpu_SameBitsAsCpu`); `grid_pieces_on_gpu_functional_test` (9 tests: `BuildGridPiece.OnGpuRandomGrids_EqualCpuPiece`, `BuildGridPiece.OnGpuNestedBallRanges_EqualCpuPiece`, `BuildGridPiece.OnGpuRandomGrids_ComparePiecesWithAllKindsOfEntries`, `BuildGridPiece.OnGpuFailures_EqualCpuFailures`, `BuildGridPiece.OnGpuAfterFailure_BuildsTheCpuPiece`, `BuildGridPiece.OnGpu_AllocatesNothing`, `BuildGridPiece.OnGpuTwice_GivesIdenticalBytes`, `BuildGridPiece.OnGpuMoreFacePlacesThanThreads_EqualCpuPiece`, `BuildGridPiece.OnGpuFunctions_KeepUnderTheDefaultStack`); `nearby_face_weights_on_gpu_functional_test` (4 tests: `ComputeNearbyFaceWeights.OnGpu_SameBitsAsCpu`, `ComputeNearbyFaceWeights.OnGpuTwice_GivesIdenticalBytes`, `ComputeFaceWeight.OnGpu_SameBitsAsCpu`, `ComputeFaceWeight.OnGpuOnSplitNeighbours_SameBitsAsCpu`); `shared_grid_functions_on_gpu_functional_test` (4 tests: `SharedGridFormulas.OnGpu_SameBitsAsCpu`, `ComputeCellPlaceFacts.OnGpu_SameBitsAsCpu`, `ComputeFacePlaceFacts.OnGpu_SameBitsAsCpu`, `GridRunners.OnGpu_FunctionsLoadWithTheirBlockSize`); `sloped_surface_scene_on_gpu_functional_test` (1 test: `StaggeredGrid.SlopedSurfaceInBoxOnGpu_EqualsCpu`); `stacked_borders_on_gpu_functional_test` (15 tests: `ReserveBorderSearchStackOnGpu.FromTheDefault_TakesAboutTheEstimatedBytes`, `ReserveBorderSearchStackOnGpu.Twice_TakesNothingTheSecondTime`, `FindBorderBlocks.OnGpuRandomGrids_SameBitsAsCpu`, `FindBorderBlocks.OnGpuOneBlockAtTheGridsEnds_SameBitsAsCpu`, `FindBorderBlocks.OnGpuNestedBallRanges_SameBitsAsCpu`, `FindBorderBlocks.OnGpuMoreBlocksThanOneLaunch_SameBitsAsCpu`, `FindBorderBlocks.OnGpuShortOutput_SameCountAsCpu`, `FindBorderBlocks.OnGpuFailures_EqualCpuFailures`, `FindBorderBlocks.OnGpuEmptySet_FindsNoBlock`, `FindBorderBlocks.OnGpu_AllocatesNothing`, `FindBorderBlocks.OnGpuTwice_GivesIdenticalBytes`, `BuildBorderSearchMemoryEstimate.MatchesAllocatedGpuBytes`, `BuildBorderSearchBuffers.OnGpuIntoRunMemory_HandsOutExactlyTheEstimate`, `FindBorderBlocks.OnGpuFunctions_KeepTheStatedStack`, `BuildBorderSearchStackEstimate.OnGpu_IsTheStackForEveryResidentThread`). Never run against a GPU. What to look for in the first run: the grid part above. |
| contact | not run | not run | none | compiled, not run | GPU test programs: `move_particles_gpu_test` (12 tests: `MoveParticles.SweepAndMoveCasesOnGpu_MatchCpu`, `MoveParticles.ObliqueCasesOnGpu_MatchCpu`, `MoveParticles.RoundCasesOnGpu_MatchCpu`, `MoveParticles.SameInputTwiceOnGpu_SameBytes`, `MoveParticles.TwoFailuresOnGpu_ReportsLowestParticleId`, `MoveParticlesOnGpuInLaunchSize.FailingParticlesAnywhere_ReportLowestIdInEveryLaunchSize`, `MoveParticlesOnGpuInLaunchSize.EqualIdsFailing_ReportsLowerIndexInEveryLaunchSize`, `MoveParticlesOnGpuInLaunchSize.NoFailure_CountsInEveryLaunchSize`, `MoveParticles.FailedCallCheckOnGpu_WritesNothing`, `BuildContactStackEstimate.OnGpu_IsTheVisitStackForEveryResidentThread`, `MoveParticles.EmptyRangeOnGpu_CountsNothing`, `MoveParticles.MillionParticlesOnGpu_MatchCpuAndTimed`) and `shared_formulas_gpu_test` (1 test: `SharedFormulas.OnGpu_MatchCpu`). `move_particles_gpu.cu` and `testing/shared_formulas_runner_gpu.cu` compile and link in the CUDA container; `move_particles_no_gpu_test` (label `no_gpu`, 2 tests) runs there. No contact device code has run on a GPU. What to look for in the first run: the contact part above. |
