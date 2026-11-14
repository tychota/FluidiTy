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
`repeat run: PASS` requires the same tests with the same outcomes and two identical, non-empty digest files.

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
