# FluidiTy coding rules

These rules and `docs/glossary.md` are the authority for how code is written and how it is checked. The design
of the whole programme is `docs/design/programme.md`. The reader of every file is a good C++ programmer who has
never read a fluid simulation paper.

Some files and folders named here do not exist yet: they arrive with later commits and modules, such as
`contact/solid_contact.h` of the worked example and the folders `reference/`, `proofs/` and `blender/`.

## The model in seven sentences

1. **Plain words beat field jargon.** If a good programmer who has never read a fluid paper cannot guess the
   meaning, the word is replaced by a plain description, even a longer one. The literature word is given once in
   a comment so papers stay searchable.
2. **One word, one meaning.** A word that could carry two meanings (`wall`, `slot`, `row`, `component`,
   `residual`, `kernel`) is split or banned.
3. **Context belongs in a type or a namespace, not in every name.** Long names that repeat qualifiers are
   shortened by a small struct, an enum or a namespace.
4. **Headers teach.** Each header explains the physical idea in two or three plain sentences before stating the
   contract.
5. **Comments are structured.** File headers and function docstrings follow fixed labelled sections and end with
   a `Read:` line naming the method paper section and the source paper.
6. **Simplicity can change behaviour.** Machinery that only buys bit-identical results or rounding guarantees is
   left out, and conventions follow the simplest choice (Z-up, like Blender).
7. **The vocabulary is guarded.** The glossary is the source of truth, this file carries the rules, and every
   commit that adds a public name lists it for human review.

## Naming rules

1. **Plain words.** Before adding a domain word, try a plain description. A longer name is fine. Put the
   literature word in the teaching header once: "papers call this the interpolation stencil".
2. **Glossary first.** A domain word may appear in an identifier or a comment only if it is in
   `docs/glossary.md`. Add the entry in the same commit.
3. **No abbreviations.** Spell out MAC, PIC, APIC, P2G, G2P, EOS. Ordinary programming abbreviations (`id`,
   `gpu`, `cpu`, `min`, `max`) are fine.
4. **Words everywhere, no symbols.** No single-letter or paper-symbol locals (`h`, `J`, `C`, `L`, `a`, `b`).
   Loop counters and `x`, `y`, `z` are fine. Put the formula in a comment above the code:

   ```cpp
   // Method paper <eq:pressure>:  A = h^2,  V = h^3
   const double cell_width = grid_geometry.cell_width;
   const double face_area = cell_width * cell_width;
   const double cell_volume = face_area * cell_width;
   ```

5. **Let a type carry shared context.** When several names repeat the same qualifiers, introduce a small type.

   ```cpp
   // One value for each of the particle's two velocities.
   template <typename T>
   struct PerParticleVelocity {
     T for_next_step;  // Kept on the particle for the next step.
     T for_this_step;  // Moves the particle in this step.
   };

   // Speed toward the solid at the hit, m/s. Negative means approaching.
   PerParticleVelocity<double> approach_speed;
   ```

6. **Namespaces replace affixes.** No `3D` suffix, no `Liquid` or `Adaptive` prefix. Use nested namespaces that
   match the folders: `fluidity::contact`, `fluidity::grid`, `fluidity::step`.
7. **Units.** The name says the kind of quantity in words (`_rate`, `_area`, `_fraction`, `_volume`, `_speed`,
   `_duration`). The unit goes in the field's comment, first. Add a unit suffix only when the word could mean two
   units in this code base (`tilt_angle_rad`). Every physical field in a header has a unit comment.
8. **Verbs.** Functions start with one of the verbs in "Allowed verbs" below. The step stages and the particle
   verbs of the glossary are the allowed physical verbs. Any other verb is listed in the commit message for the
   owner's review.
9. **Role suffixes.** Only three: `Inputs` (what one call needs), `Settings` (chosen once and kept), `Result`
   (what one call returns). Other types are named after their content with no role word (`StepTimings`). The
   error pair of a module is `<Module>Error` and `<Module>Failure` ("Structure" below).
10. **Flags.** A flag that changes behaviour means two functions (`SplitCell`, `MergeCells`). Where that does not
    fit, use a two-value enum. No bare `bool` parameters. Boolean fields read as yes/no questions
    (`has_liquid`, `same_side`).
11. **Limits and tolerances.** Each number is declared once in the module that owns it, with a comment saying
    where the value comes from. If a user may reasonably override it (for example the maximum cell count), it is a
    field of `StepSettings` with that default, not a constant. True mathematical tolerances are `constexpr`.
12. **No project history in code.** No milestone or gate codes (`m5_`, `F3`, `M1`, "qualification") in file
    names, identifiers or comments. Name the physics: `snow_material.h`, "splits one particle into eight".
13. **Tests.** `TEST(FunctionOrType, Situation_Outcome)`, one behaviour per test. A name that needs "And" is two
    tests. Example: `TEST(MoveParticle, StartsInsideSolid_IsRejected)`.
14. **CPU and GPU.** No `Cuda`, `Gpu`, `Device` or `host` in public names. Callers see one function per
    operation. Where it runs is a setting: `enum class RunOn : std::uint8_t { kCpu, kGpu }`, chosen by the caller.
    The two implementations are private files: `particles_to_grid_cpu.cc`, `particles_to_grid_gpu.cu`.

## Allowed verbs

A function name starts with one of these verbs:

- `Compute`: returns a value, no side effect.
- `Build`: returns a new data structure.
- `Check`: returns ok or an error, changes nothing.
- `Find`: searches, may return nothing.
- `Apply`: changes its target in place.
- `Solve`: linear or iterative solve.
- `Read`: copies from disk or from the GPU.
- `Write`: copies to disk or to the GPU.
- `Copy`: copies values between two arrays on the same processor.
- `Free`: frees memory (`FreeBufferBytes`).
- `Take`: gets preallocated buffers back out of an object that is no longer needed (`TakeBuffers`).
- `To<Type>`: converts a value to that type (`ToMessage`, `ToName`).

Inside `namespace math`, a function that only returns a value is named with the plain math word for it, with no
verb: `math::Dot`, `math::Cross`, `math::Transpose`, `math::Determinant`.

These physical names are allowed exactly as written. They are the step entry point, the step stages and the
particle and cell verbs of the glossary:

- Step entry point: `TryStep` (one try of a step, which may be rejected).
- Step stages: `AddNewParticles`, `ChooseActiveCells`, `ParticlesToGrid`, `SaveGridBeforeForces`,
  `ApplyForces`, `DriveWaves`, `CollideSolids`, `FindLiquidSurface`, `SolveViscosityAndPressure`,
  `SolvePressure`, `SaveGridAfterPressure`, `GridToParticles`, `MoveParticles`, `MakeFoamAndSpray`,
  `CheckStep`, `CommitStep`, `ResizeCells`.
- Particle verbs: `AddParticles`, `SplitParticle`, `MergeParticles`, `RemoveParticles`, `AdjustParticleCount`,
  `MixParticleVelocity`, `GroupParticlesByBlock`.
- Cell verbs: `SplitCell`, `MergeCells`.

## Comment rules

1. **Reader**: a good C++ programmer who has never read a fluid simulation paper.
2. **Glossary words only.** Define a word in the file header or do not use it.
3. **Short sentences.** One idea each, about 20 words at most, subject first, active voice. No noun chains.
   - Before: "Any old-stage ghost required rank fallback."
   - After: "At least one filled face had too few neighbours, so the particle followed the grid velocity only."
4. **Say why, not what.** Body comments give the reason, the invariant or the trap. Delete comments that repeat
   the code.
5. **Show a small example** with numbers wherever a contract is hard to picture.
6. **Format.** 120 columns, clang-format 23.1, `//` comments.

### File header

Every `.h` and `.cuh` file opens with this header.

```cpp
// Solid contact for liquid particles.
//
// Purpose: move one particle for one step without letting it enter a solid.
//
// Idea: a particle travels in a straight line during one step. If the line meets a solid, the particle
//   stops moving into it, loses some sideways speed to friction, and spends the rest of the step sliding
//   along the surface.
//
// Words:
//   sweep  the straight path of one step
//   hit    the first time the path meets a solid's surface
// Owns:    hit detection, friction response
// Uses:    solid shapes (contact/solid_shapes.h)
// Used by: the MoveParticles stage (step/)
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>
//          <first author> <year>, <title> (method paper ref <n>)
```

`Words` is present only when the file introduces domain words, and lists only those; the rules check does not require
it. `Uses` names what the header itself includes from the project and from other libraries, never a header of the C++
standard library; every CPU-only header names its guard there, "CPU-only guard (base/cpu_only.h)". What only the
implementation (the `.cc` or `.cu` file) uses goes on an `Implementation uses:` line after `Uses`. `Read` follows
"Citations" below.

### Function docstring

Every public function has a docstring in its header.

```cpp
// Moves one particle for one step and stops it at a solid surface.
//
// Method: sweeps the straight path against the solid. On a hit, removes the speed into the solid,
//   applies friction, then moves the remaining time along the surface.
// Units: meters, seconds, world frame.
// Ties: when two faces are hit at the same time, the first in x, y, z order is reported.
// Example: particle at x = 0.9 moving at +1 m/s, solid face at x = 1.0, step 0.2 s.
//   Hit at 0.1 s. End position x = 1.0, speed 0. result.hit_fraction = 0.5.
// Fails: kBadInput           a value is not finite or is out of range
//        kStartsInsideSolid  the start is deeper inside than the tolerance allows
//        kTooCloseToCall     rounding prevents a safe answer; shorten the step
//        On failure the result holds a ContactFailure and no position.
// Read: docs/method-paper/fluidity-method.typ <sec-contact>
std::expected<SolidContactResult, ContactFailure> MoveParticle(const SolidContactInputs& inputs);
```

Sections, in this order: summary line, `Method`, `Units`, `Ties` or `Order` (only if relevant), `Example`,
`Requires`, `Fails`, `Read`. Omit a section that has nothing to say. Small helpers need only the summary line.

### Function bodies

- No fixed line limit. **Extract a step into its own function only when it stands alone**: it needs four or
  fewer inputs, has a name that needs no comment, and could be called or unit-tested by itself.
- A step that touches many locals of the enclosing function stays inline.
- A body with more than three blocks numbers them. The docstring or the comment at the top of the body lists the
  steps; each block starts with the matching number.

  ```cpp
  // Steps: 1 check inputs  2 find the hit  3 respond  4 slide
  ...
  // Step 2: find the first hit along the path.
  // Measured from the moving solid, the distance to its surface changes linearly with time.
  ```

- Every branch that returns an error says why the case can occur or what the caller should do.

### Citations

```cpp
// Read: docs/method-paper/fluidity-method.typ <eq:pressure>
//       Batty 2007, A Fast Variational Framework for Accurate Solid-Fluid Coupling (method paper ref 21)
//       Gibou 2002, doi:10.1006/jcph.2001.6977
```

- First line: the label in our method paper. It explains our variant. The cited section must exist and match
  the code; add the label to the paper in the same commit when it is missing.
- Following lines: the source paper as first author, year, then DOI or URL.
- **Copy every DOI and URL from the method paper's bibliography. Never write one from memory.** If the
  bibliography has no DOI for an entry, cite it as "(method paper ref n)".
- Cite only papers the method paper's section actually relies on. If you cannot tell which reference a function
  implements, write only the method paper label and list the function under "citation needed" in the commit
  message.

## One worked example

`src/fluidity/contact/solid_contact.h`, its types only. The file header and the docstrings follow the
templates above.

```cpp
namespace fluidity::contact {

enum class ContactKind {
  kNone,                // The particle met no solid.
  kHit,                 // The path met a solid during the step.
  kStartedInsideSolid,  // The start was slightly inside and was moved to the surface.
};

struct SolidContactInputs {
  Vec3 start_position{};                 // m, world, at the start of the step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world, before the response.
  Solid solid{};                         // Shape, pose and velocity during the step.
  double step_duration = 0;              // s, positive.
  double friction = 0;                   // No unit, zero or more. Papers: Coulomb coefficient.
  double start_overlap_tolerance = 0;    // m, zero or more.
};

struct SolidContactResult {
  Vec3 position{};                       // m, world, at the end of the step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world, after the response.
  std::optional<double> hit_fraction;    // When the hit happens, 0..1 of the step. Empty without a hit.
  Vec3 normal_out_of_solid{};            // Unit vector at the hit. Zero without a hit.
  // Speed toward the solid just before the response, m/s. Negative means approaching.
  PerParticleVelocity<double> approach_speed{};
  ContactKind kind = ContactKind::kNone;
};

}  // namespace fluidity::contact
```

## Structure

**Folders by concept.** Namespaces mirror folders: `src/fluidity/grid/` is `fluidity::grid`.

```text
src/fluidity/
  base/             vectors, small matrices, error and result types (shared by CPU and GPU)
  contact/  grid/  particles/  transfer/  pressure/  viscosity/
  materials/  solids/  gas/  step/  cell_resize/
  io/  surface/  foam_and_spray/  cli/
reference/          Python reference, same folder names
proofs/             Lean proofs that describe current code
blender/            the add-on
docs/               glossary, method paper, GPU run log, one design per module
```

A module design may add a folder for a concept it owns (for example `batches/` beside `cell_resize/`). A new
folder name is a public name and is listed for review. The code is three-dimensional only; there is no 2D path.

**One operation, two private implementations.** Each operation has one public header, for example
`transfer/particles_to_grid.h`, and two private files beside it, `particles_to_grid_cpu.cc` and
`particles_to_grid_gpu.cu`. The formulas both sides need (weight curves, contact response, material response)
live in shared headers. Loops, memory layout and scheduling are not shared.

**Private GPU headers.** A private header `<name>_gpu.h` declares only what `<name>_gpu.cu` defines; only files of
the same folder include it. Its names may say `Gpu` and use any verb. It includes no CUDA header, so the CPU file
that calls it compiles without the CUDA compiler. The rules check skips the backend-name and verb rules for files
whose stem ends in `_gpu` (`.h`, `.cu`), and fails an `#include` of a `*_gpu.h` header from another folder or from
a header that is not itself `_gpu`. The one exception is a test (a file with `.test.` in its name), which may
include a `<name>_runner_gpu` header of the `testing/` folder of its own folder (the test is a `.cc` or `.cu` file).

**Names of test files.** A file with `.test.` in its name is a test, and nothing else carries the word `test` in
its name, apart from the `testing/` folders, the `tests/` folder, `tools/run_gpu_tests.ps1` and the Python tests under
`tools/`. Each module folder may hold a `testing/` folder for code that its tests share and that is not itself a test.
The stem of a unit test is the stem of a header of the same folder, so that an editor folds the test under that
header; when the file tests one concern of the header, a topic follows the stem, as in
`staggered_grid.cell_count.test.cc`, and the topic leaves out the words the header's name already says. This departs
from Google's `<name>_test.cc`. A functional test is named for what it exercises, not for a header
(`tests/<module>/<what it exercises>.test.cc`, see "Unit tests and functional tests" below).

| What | Name |
|---|---|
| a test | `<header>.test.cc` |
| a test that needs a GPU | `<header>.gpu.test.cc` |
| a test of the behaviour without a GPU | `<header>.no_gpu.test.cc` |
| a test that only compiles GPU code | `<header>.gpu_compile.test.cu` |
| a test of one concern of a header | `<header>.<topic>.test.cc` (also `.gpu.test.cc`, `.no_gpu.test.cc`) |
| the code that runs a GPU check for a test | `testing/<name>_runner_gpu.h`, `testing/<name>_runner_gpu.cu` |
| cases, sets and runs that tests share | `testing/<name>_cases.h`, `testing/<name>_sets.h`, `testing/<name>_runs.h` |
| other code that tests share | `testing/<name>.h`, `testing/<name>.cc` (for example `base/testing/base_support.h`) |
| a test of the code in `testing/` | `testing/<name>.test.cc` |
| a functional test | `tests/<module>/<what it exercises>.test.cc` (also `_on_gpu`, for one that needs a GPU) |

The Python tests under `tools/` keep `<name>_test.py`, which is what pytest finds. The CTest label that names the
module (`base`, `grid`, `contact`) is the folder of the `CMakeLists.txt`, so a test in `testing/` carries it too.
The name of a CMake target does not have to follow the file name.

**Unit tests and functional tests.** Two kinds of test, in two places.

- A **unit test** checks one function on input written by hand in the test, and takes at most 50 ms in the debug
  build. It lives beside its header: `src/fluidity/<module>/<header>.test.cc`. Its CTest label is `unit`.
- A **functional test** runs several steps together or works on generated input: it builds a whole grid, loops over
  random grids, compares the CPU against the thread or GPU form over many cases, or runs a scene. It lives in
  `tests/<module>/<what it exercises>.test.cc`, one folder per module, each with its own `CMakeLists.txt`. Its CTest
  label is `functional`. The name of a functional test says what it exercises (`grid_tables_on_random_grids`), and it
  need not start with the stem of a header.
- The kind follows the definition, not the clock. A test that loops over generated or random input, over a list of
  cases that code computes, or over many steps (a hundred or more) is functional even when it takes 10 ms. A test
  that checks one call, or a short hand-written list of calls, on input written by hand is a unit test, even when
  the function under test builds a whole grid (`BuildGrid`).
- `e2e/` is for runs through the public entry of the program. It holds only a `README.md` that says so for now.

A test file that holds both kinds is split: the unit tests stay where they are, the functional tests move to
`tests/<module>/`, and a helper that both parts call goes to `src/fluidity/<module>/testing/<name>_cases.h`, written
once. `fluidity_add_test` takes `TIER unit|functional` (unit when it is left out), and gives each test three labels: its
backend (`cpu`, `gpu`, `no_gpu`), its module and its tier. A unit test is
one ctest entry per TEST (one process each, so isolation is kept). A functional test program is ONE ctest entry,
named like the program (`grid_tables_on_random_grids_functional_test`), that runs every TEST of the program in one
process, so the fixtures its tests share (the random grids and their checkers) are built once. The entry carries
the same three labels and a timeout (`FLUIDITY_FUNCTIONAL_TEST_TIMEOUT_SECONDS`). Because the TESTs of a functional
program share a process, none may rely on static state, a counter or a file left by an earlier TEST, or on running
alone: build such state in the TEST or reset it in a fixture. The module label of a test in `tests/<module>/` is
`<module>`. The tools that cover the test files under `src/` (format, rules, tidy, the compiled-everything check of the
cuda step) cover `tests/` in the same way; a test in `tests/<module>/` may include the private GPU headers of
`fluidity/<module>/`. In the debug run of the gate, a `unit` test that takes more than 200 ms fails the step (the
rule is 50 ms for a test run alone, and 200 ms allows for one test per core at once); the gate prints a `NOTE` for each
one between 50 and 200 ms.

**The two halves of an operation.** The public function lives in `<name>_cpu.cc`: it checks its inputs, chooses the half
by `RunOn`, runs the CPU loop itself, and calls the GPU half under `#if defined(FLUIDITY_HAS_GPU)`. `Buffer` is laid out
differently: its public functions are templates in `base/buffer.h` that call the byte functions of
`base/buffer_bytes.h`, which `base/buffer_cpu.cc` defines and which call the GPU half. The GPU half is `<name>_gpu.cu`,
declared in the private `<name>_gpu.h`. It returns a `RunOnCallStatus` (`base/run_on_call.h`), which the CPU half maps
into its own error with one mapping function. A `.cu` file starts GPU functions, waits for the GPU and holds temporary
GPU memory of its own only through the helpers of `base/gpu_launch.cuh` (`ApplyGpuFunction`, `CheckGpuWork`,
`GpuArray`). It may call the CUDA runtime directly for a copy or a fill (`cudaMemcpy`, `cudaMemset`), for a device query
(such as `cudaGetDeviceCount`), and in `base/buffer_gpu.cu` to allocate and free the memory of a `Buffer`; the status
of every direct copy, fill or free goes through `ToRunOnCallStatus`, which gives `kCallFailed` for any failure, and
only a failed allocation gives `kOutOfMemory`. A GPU function is a thin loop
around shared per-item or per-thread functions marked `FLUIDITY_CPU_AND_GPU`, which live in a shared header; CPU
tests call them over every item, or every thread block and thread in launch order and in reverse, so the GPU
arithmetic is tested without a GPU.

**Working buffers.** An operation that needs temporary arrays takes them as working buffers: a move-only `<Name>Buffers`
built once, whose exact bytes `Compute<Name>BuffersSize(run_on)` gives for memory plans (`SumBuffers` in
`base/ordered_sums.h` is the first). A run builds them with `Build<Name>Buffers(run_on, max_count, RunMemory&
run_memory)`, as one piece of the run's reserved block (`base/run_memory.h`): the block is always the last parameter,
and `run_on == run_memory.run_on()` is a `Requires:`. The plain form `Build<Name>Buffers(run_on, max_count)`, which
allocates memory of its own, is for tests and tools. Working buffers hold their memory in one `Buffer` of bytes, so
their GPU memory is held through `Buffer`, not through `GpuArray`, and the GPU half receives plain pointers into it.
Their contents mean nothing between calls. Each function has one form, which takes its working buffers and allocates
nothing, so `kOutOfMemory` comes only from `BuildRunMemory` and the plain forms of `Build<Name>Buffers` and
`BuildBuffer`; a form that takes a `RunMemory` returns `kRunMemoryTooSmall` when its piece does not fit. A count above
`max_count()` is `kSizeMismatch`, and a failed GPU call is `kRunOnCallFailed`. The working buffers live on the processor
of the other buffers: a precondition ("Language").

**`RunOn` is the caller's choice.** `enum class RunOn : std::uint8_t { kCpu, kGpu }` lives in the shared header
`base/run_on.h`, so any shared type can carry it. The caller chooses it when the simulation is built.
`CheckCanRunOn(RunOn)` answers whether that choice can run here. Nothing ever changes it silently: when the GPU
was chosen and cannot run, the call fails; the code never moves the work to the CPU on its own.

**One step entry point.** Every scene runs through one `TryStep(StepInputs)`. There is no scene-specific
public step function and no mode flag threaded through a module.

**One error enum per module.** Each module has `enum class <Module>Error`, whose first value is `kNone`.
An error with its context is `struct <Module>Failure { <Module>Error error; ... }`, where the context says which
particle, which face or what size. A function that can fail returns `std::expected<T, <Module>Failure>`.
`std::expected` is used directly, with no alias. Beside every error enum sit two functions:

- `std::string_view ToMessage(<Module>Error error)`: the plain-language message.
- `std::string_view ToName(<Module>Error error)`: the stable code used by the CLI and the add-on. It is the
  value's name in snake_case without the `k`: `kSolidTooThin` becomes `solid_too_thin`.

`base/` is the one exception: its functions return the bare enums `RunOnError` and `BufferError`
(`std::expected<T, BufferError>`), because they have no context worth adding. A module maps them into its own
`<Module>Failure`, adding the context it knows.

Inner code returns the enum and context and never allocates. Text is added by `ToMessage` at the outer
boundary. When several items fail on the GPU, the one reported is the first in a fixed order (by stable id), so
the error is the same on every run. A failed check writes nothing. After a failed GPU call (`kRunOnCallFailed`) the
outputs are unspecified and the run must stop.

**Material sets.** A material set is the group of materials that share one grid velocity. Materials that do not
mix are separate sets, each with its own face sums; `CollideSolids` applies the contact rule at the faces they
share. Every liquid is in one set, and gas markers are in one set. Gas and liquid can run in one scene.
The materials of the main sequence are liquid, gas, snow and springy solid. Sand is a last, separate module;
nothing else depends on it.

**Exactness.**

- Same GPU and same input give identical output on every run.
- Discrete outputs (integers, orders, counts, flags) are equal on the CPU and the GPU. Floating-point results agree
  within a tolerance, except a function whose header promises the same bits on both processors (unless the result
  is NaN). Today: the functions so marked in `base/scalar_math.h`, `base/vec3.h` and `base/mat3.h`, and every
  function of `base/tree_sum.h` and `base/ordered_sums.h`.

Sums are plain doubles in a fixed order, never floating-point atomics. Each module states its own tolerance.

**Z-up.** Coordinates are right-handed with Z up, like Blender. Gravity points along `{0, 0, -1}`.

**Output files.** Everything a person or another program plays back (particles, surface, foam and spray,
volumes, solid motion) is written in an open format: Alembic, USD or OpenVDB, readable with the add-on disabled.
FluidiTy's own array file holds only the saved state and internal run records.

## Language

- **Standards.** Host code is C++23 and uses `std::expected`. Device code and shared headers are written in the
  C++20 subset that the CUDA compiler can build.
- **Header kinds.**
  - `.h`: shared by CPU and GPU, or CPU only.
  - `.cuh`: needs the CUDA compiler.
  - `.cc`: host source.
  - `.cu`: device source.
- **Shared headers.** A function in a shared header is marked `FLUIDITY_CPU_AND_GPU` (from
  `base/cpu_and_gpu.h`). A shared header never includes `<expected>`, `<string>` or anything that allocates.
  If a `.cu` file would pull such a header in, the header is split, not guarded with `#if`.
  A shared CPU-and-GPU function takes `Vec3`, `Mat3` and other small trivially copyable structs by value: GPU code
  cannot bind a reference to a namespace-scope `constexpr` constant such as `kIdentity`. The rules check fails
  `const Vec3&` and `const Mat3&` in any file that contains `FLUIDITY_CPU_AND_GPU`.
- **CPU-only headers.** Every CPU-only header includes `base/cpu_only.h` first. That header stops the build
  with a clear message if GPU code includes it.
- **No exceptions and no RTTI** in the library. Never call `std::expected::value()`: it throws. Test
  `has_value()` and use `*`.
- **Preconditions.** A condition that only a bug in the caller can break, such as two buffers on different
  processors, is a precondition, not an error value. The docstring states it on a `Requires:` line, and the code
  checks it with `assert` from `<cassert>`: in a debug build a broken precondition stops the program with the file,
  the line and the condition; in a release build, where CMake defines `NDEBUG`, the check compiles to nothing.
  There is no project macro, because `assert` already does this and needs no new name.
- **No Abseil.**
- **Numerical flags.** Device code is compiled with `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`.
  For doubles only `--fmad=false` matters: an NVIDIA GPU always keeps subnormal doubles and rounds double division
  and square root correctly, so the last three flags matter for single precision only. Host code uses
  `-ffp-contract=off` (GCC, Clang) and `/fp:precise` (MSVC). No fast-math anywhere.
- **Format.** Google C++ style through the repository's clang-format 23.1 settings: 120 columns, two-space
  indent, `//` comments, braces on every control-flow body.
- **Targets.** Every C++ and CUDA target is created through the helpers of `cmake/fluidity_targets.cmake`
  (`fluidity_add_test`, `fluidity_add_test_library`, or `fluidity_apply_compile_flags` for the library), because the
  numerical flags are set per target. Configuring fails, naming the target, for one created any other way.

## Checklist for each file

- [ ] File header present with Purpose, Idea, Owns, Uses, Used by, Read, and Words when the file introduces domain
      words.
- [ ] Every public function has the sectioned docstring; `Fails` lists each error and its cause, `Requires` each
      precondition.
- [ ] Every physical field has a unit comment.
- [ ] No banned word from the glossary in identifiers or comments.
- [ ] No abbreviation, paper symbol, milestone code, `3D` suffix, bare `bool` parameter or repeated magic number.
- [ ] Only verbs from "Allowed verbs" and role suffixes from naming rule 9.
- [ ] Bodies with more than three blocks have numbered steps; error branches say why.
- [ ] At least one docstring example with numbers per non-trivial public function.
- [ ] `Read` lines copied from the method paper's bibliography, never from memory.
- [ ] New public names listed in the commit message.

## Checks, notes and review

`tools/check_rules.py` reports two levels. A FAIL is a rule with no false positive: it fails the gate. A NOTE is a
detector that can be wrong: it never fails the gate, and it is printed only for the lines a change adds. The gate
prints `NOTE <n> doubtful lines: uv run python tools/check_rules.py --notes-since <base> --json out/check-commit/notes.json`
when there are any, and that command writes the list: path, line, rule, the note and the source line. The lines are
those of the change being checked: the staged lines (`<base>` is `HEAD`) when the tree differs from `HEAD`, else the
lines of `HEAD` itself (`<base>` is `HEAD~1`).

- Whoever hands a change in fixes every NOTE of it, or answers it in one line (why the line stays).
- A rule is one entry in `RULE_TABLE` (id, level, files, reason) and one pair of tests: it fires, and it stays
  quiet on the nearest innocent text.
- A fault that a review finds three times becomes a NOTE.
- A NOTE that has no wrong hit over a whole module becomes a FAIL.
- A rule that never fires over two modules, or whose notes are left unfixed more than one time in three, is removed.

## Commit rule

A commit lands only when all of these hold:

1. It does one thing and its message says why.
2. The strict CPU build passes locally (warnings are errors) and every test passes.
3. The CUDA target compiles in the Linux container.
4. `clang-format` 23.1 reports no change and `clang-tidy` 23.1 reports nothing on host sources.
5. Every file it touches passes "Checklist for each file" above.
6. New public names are listed in the commit message with a one-line reason. New domain words are added to the
   glossary in the same commit.
7. The method paper section the code cites exists and matches the code.
8. Tests arrive with the code they test, not after.
9. The owner is the author. A commit message carries no AI trailer (no `Co-Authored-By`, no `Assisted-by`, no
   session link). Each commit has a git note in `refs/notes/claude-session` with two lines, `session: <link>` and
   `models: <model ids, comma separated>`; `tools/add_session_note.py` writes it ("Commands").

A commit passes every part of this gate that exists at that commit, at the `commit` speed (see "Commands"):
`quick` while working, `commit` before each commit, `full` before a push and in CI.

**GPU honesty.** Device code is "compiled, not run" until the owner's hardware run, which happens once, after
all modules are written. Until then every module's status reads "GPU: compiled, not run". `docs/gpu-runs.md`
records each run: the commit, the device and the result per module. Nothing is described as working on the GPU
before its record exists.

**Stop and ask.** When a name meets a word that the glossary does not define and `docs/design/programme.md`
section 9 lists as open for a module, work on that name stops and the owner is asked. Do not guess.

## Commands

- Gate for every commit. The gate checks what will be committed, so the order is: stage (`git add`), run the gate
  from the repository root, `uv run python tools/check_commit.py`, then commit. Add `--only <step>` to run one
  step: `format`, `rules`, `cpu`, `tidy`, `paper` or `cuda`. It prints one line per check, in this order.
  `--speed quick|commit|full` chooses how much runs:
  - `quick`, while working: format; rules; the test programs of the modules that changed built in the debug preset;
    their debug unit tests; tidy (the cache makes it check only what changed). A changed file differs from `HEAD` (staged or not) or is
    untracked. A change under `src/fluidity/<module>/` selects that module's unit tests; a change under
    `tests/<module>/` or under `src/fluidity/<module>/testing/` selects the module too and also runs its functional
    tests (`NOTE functional tests: <names>`), because they use the shared test code of that folder; a
    change elsewhere under `src/` or `tests/` or under `cmake/`, or to a top-level build file, selects the unit tests
    of every module; a change to docs or tools selects none. Only the test programs of the selected modules are built
    (`NOTE built: <n> test programs`; with every module selected the whole preset is built, with none nothing is; the
    list `test_programs.txt` is read after the configure that rewrites it, and a `fluidity_add_test` program of a
    selected module's `CMakeLists.txt` that the list lacks also builds the whole preset), so a quick run does not see
    a break in a module that is not selected. Conan and the configure are left out
    (`NOTE conan and configure: not run (their files are unchanged)`) while the build folder holds the hash of the
    contents of `conanfile.py`, `conan.lock`, `CMakePresets.json`, `CMakeUserPresets.json` (when there is one),
    `pyproject.toml`, `uv.lock`, every `CMakeLists.txt` and every file under `cmake/`. The hash stored after a pass is
    that of the files as they were before Conan and the configure began. A compiler or SDK upgrade, and a configure
    made by another tool (an IDE, a hand-run `cmake -D...`), are not in it, so run `--speed commit` after one. The gate
    prints `NOTE modules: <names>` (or `all`, or `none`) and `NOT RUN <step> (quick)` for the release build, the
    paper and cuda, so a quick pass is never read as a full one. It never satisfies the commit rule: a quick run
    whose checks all passed exits with code 3, never 0, and prints `QUICK ONLY: not the commit gate` as its last
    line. A top-level build file is `CMakeLists.txt`, `CMakePresets.json`, `conanfile.py`, `conan.lock`,
    `pyproject.toml` or `uv.lock`. An untracked file counts as changed, but only with `--allow-dirty`, because the
    gate stops at `FAIL tree` for it otherwise.
  - `commit`, the default: every step below, with the parallel tests and the tidy cache. Run it before each commit.
  - `full`: `commit` with `--no-cache`, so tidy checks every file. Run it before a push; CI uses it.
  `--only <step>` works at every speed. At `quick` it runs the named step in full (every module's tests, both
  presets), and the exit code is still 3.
  - `FAIL tree: unstaged or untracked: <paths>`: printed first, only when a file is untracked or has unstaged
    changes outside the ignored paths; every step is then `NOT RUN`. Staged changes are fine. With `--allow-dirty`
    the gate prints `NOTE tree: unstaged or untracked files are being tested` instead and goes on; use it for work
    in progress, never for a commit.
  - `FAIL tree: changed while the gate ran (<what changed>)`: printed last, at every speed and with `--only`, when
    the index tree (`git write-tree`) or the set of unstaged or untracked files differs from the start, checked after
    the last step and right before and after the cuda container run, which reads the live working tree. It exits 1
    whatever the steps printed, so do not edit files while the gate runs.
  - `PASS format`: the clang-format on `PATH` is the version `pyproject.toml` pins, and it reports no change in any
    tracked C++ or CUDA file. Another version fails the step and names both versions.
  - `rules note: <path>:<line>: NOTE <rule> <text>`: any NOTE rule of `tools/check_rules.py` (a word banned in one
    sense only, `short-local`, `many-inputs`, `process-word`, `test-name-and`, `test-without-check`), on a line the
    staged changes add. Read it, fix it or answer it; a note never fails the gate.
  - `PASS rules`: `tools/check_rules.py` reports no FAIL under `src/`, `tests/` and `docs/method-paper/`, the pytest
    suite of `tools/` passes, and `actionlint` accepts the files of `.github/workflows/`. The files are checked
    several at once, and the same scan lists the doubtful lines. The pytest suite runs one process per test file, all
    at once. At `quick` speed the suite is left out (`NOT RUN tools tests (quick, tools unchanged)`) while no changed
    file is under `tools/`, `cmake/` or `.github/` or is one of the files the tests read: a top-level `CMakeLists.txt`,
    `pyproject.toml` or `uv.lock`, `AGENTS.md` or `docs/glossary.md` (`TOOLS_TEST_FOLDERS` and `TOOLS_TEST_INPUTS` in
    `tools/check_commit.py` list them in one place); at `commit` and `full` speed it always runs.
  - `PASS message`: `tools/check_message.py` accepts the message of every commit the push would send, that is, of
    `@{upstream}..HEAD`, or of `HEAD` alone when the branch has no upstream or nothing is unpushed. A message passes
    with a subject of at most 72 columns, a blank line after it, body lines of at most 72, no trailer line (the
    literal last line `GPU: compiled, not run` is not one), no tool or model name (not counted inside a backtick span
    or in a token with a `/`, such as a branch name), no process word (brief, review, reviewer, fixup, subagent,
    agent, each also in its regular inflections: `reviews`, `reviewed`), no `fixup!`, `squash!` or `amend!` subject,
    and no word the glossary bans. The step runs when the working tree has no change, so run the gate again after the
    commit. With changes it prints `NOT RUN message (uncommitted changes)` and the run can still exit 0; with
    `--only message` the same line exits 1, because nothing was checked. `--speed quick` leaves it out. It also reads
    one commit (`tools/check_message.py <commit>`), a range (`--range A..B`) or a file (`--file PATH`, which drops the
    `#` lines and the scissors part, as git does). CI checks the pushed range in the same way.
  - The message of the commit being made is checked before the commit exists by a `commit-msg` hook, tracked as
    `tools/hooks/commit-msg`. Enable it once per clone (it needs `uv` on `PATH` and fails the commit without it):

    ```sh
    git config core.hooksPath tools/hooks
    ```

  - `PASS cpu macos-debug-cpu` and `PASS cpu macos-release-cpu`: Conan fetches the packages of `conan.lock` (at every
    run but a quick one with unchanged files, so a changed lockfile always reaches the build), and the strict CPU build of that preset configures,
    builds with warnings as errors, and passes every test. On Linux or Windows the gate uses the presets of that
    platform. On the debug preset the step also fails, naming the test, when a `unit` test took more than 200 ms
    (`FAIL cpu macos-debug-cpu: unit tests over 200 ms: <name> (<ms> ms)`). It fails too when a test has neither the
    `unit` nor the `functional` label, or when the JUnit file of ctest is missing. It prints
    `NOTE unit test over 50 ms: <name> (<ms> ms)` for each unit test between 50 and 200 ms. A functional entry (a whole program) has no
    time limit but its ctest timeout; the step prints `NOTE functional tests run: <n> gtest tests in <m> programs`.
  - `PASS tidy`: the clang-tidy on `PATH` is the pinned version, and it reports nothing, with the debug preset's
    compile commands, on every tracked `.cc` file under `src/` and `tests/` that the CPU build compiles, and on each header
    compiled alone (`header_check/`). A `.cc` file that only the CUDA build compiles, such as a GPU-only test, is
    tidied by the `cuda` step. `tools/run_tidy.py` runs one clang-tidy per file, one per core at once, and prints
    the failures sorted by path. A file that passed is not checked again while nothing it depends on has changed
    (its text, the headers it includes, its compile command, the `.clang-tidy` files, the clang-tidy version); the
    cache is `out/tidy-cache/`, and `--no-cache` ignores it and stores nothing. After a pass the cache also keeps the
    list of the file's headers with the size and modification time of each, of the file, of the `.clang-tidy` files and
    of the compiler, the values of `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH` and `SDKROOT`, and the names in the
    folders where headers are looked for (the include folders, the folders of those variables and the folders the
    compiler searches by itself, which it is asked for once per run). No record is stored for a file that lists a
    generated header under `out/` (the Conan packages in `out/conan-home` excepted). While all of these are unchanged the
    file counts as passed without running the compiler to list its headers. Any difference, or a file modified in the
    last two seconds, falls back to the contents key above.
  - `PASS paper`: the method paper compiles with typst, and every label on a `Read:` line under `src/` and `tests/`,
    including its continuation lines, is defined in the paper. A label starts with its kind: `<sec-...>`,
    `<fig-...>`, `<proc-...>`, `<eq-...>` or `<tab-...>` (`:` may replace `-`); other text in angle brackets, such as
    `<first author>`, is not a label.
  - `PASS cuda`: in the CUDA container (`tools/cuda-container/Dockerfile`), everything of the preset
    `linux-release-cuda` compiled and linked, every `.cu` file and the `gpu` tests included; clang-tidy reported
    nothing on the `.cc` files that only this build compiles (CMake lists them in `gpu_only_sources.txt`); the tests
    labelled `cpu` and `no_gpu` passed (`ctest -L '^(cpu|no_gpu)$'`); every tracked `.cc` and `.cu` file under
    `src/` and `tests/` is in this build's compile commands; and every tracked `.h` and `.cuh` file under `src/` is compiled
    alone there, through its header-alone file (`header_check/`), except a test header (a header directly in
    `src/fluidity/<module>/testing/`), which the test programs that include it compile (`FAIL cuda: not compiled by the CUDA build: <paths>`
    otherwise).
    Nothing ran on a GPU: the container has none, so GPU code stays "compiled, not run". The container keeps its
    tools, packages and build tree in a Docker volume named `fluidity-cuda-out-<hash>`, one per checkout path; list
    them with `docker volume ls --filter name=fluidity-cuda-out` and remove one of a deleted checkout with
    `docker volume rm <name>`.
  - `FAIL <check>: <reason>`: that check failed. The reason names the command; the full output of the step's
    commands is in `out/check-commit/<step>.log`. `FAIL cuda: Docker is not running (start it with: colima start)`
    means exactly that: the cuda step is never skipped and never passes without running.
  - `NOT RUN <check>`: an earlier check failed, so this one did not run.

  Exit codes (`tools/check_commit.py --help` lists them): 0 means that no check failed at `commit` or `full` speed
  (the message step may still read `NOT RUN message (uncommitted changes)`; the `commit-msg` hook covers it); a commit
  lands only then. 1 means that a check failed, or that a step named by `--only` could not run. 3 means a `quick` run whose checks all passed; it is
  not the commit gate and never satisfies the commit rule. `quick` never exits 0.
- CI: `.github/workflows/checks.yml` describes the checks CI will run (format and rules, the CPU build on Linux, macOS
  and Windows, and the CUDA compile). It is linted and has run on GitHub (scratch branch `rebuild-ci`), with every
  job green on Linux, macOS, Windows and the CUDA compile, so the Windows CPU code has been compiled and its tests run.
  No CI job runs GPU code, and the Windows CUDA build has never been compiled. The local gate remains the commit rule.
- GPU targets: the CUDA presets compile for RTX 30 (`86-real`), RTX 40 (`89-real`) and RTX 50 (`120-real`, plus
  `120-virtual`). The owner's hardware run is on an RTX 5060 Ti; it is not recorded yet in `docs/gpu-runs.md`, so
  even that card is untested today. RTX 30 and 40 are compiled for, unverified. On a machine with a GPU, the tests
  that need it are `ctest --preset linux-release-cuda -L '^gpu$'` (a bare `-L gpu` would also match `no_gpu`).
- Run the tests of one module: every test carries its backend label and the name of its module, the folder of the
  `CMakeLists.txt` that registered it (`base`, `grid`, `contact`), so `ctest --preset macos-debug-cpu -L grid`
  runs the grid tests alone, and `ctest --preset macos-debug-cpu -L '^cpu$' -L grid` the grid tests labelled `cpu`.
  The gate runs ctest with one job per core (`--parallel <cores>`); run it the same way by hand, with
  `--parallel <cores>`. A preset cannot say "all cores" in a form every CMake version accepts.
- Run one functional TEST: the ctest entry is the whole program, so use the program with a gtest filter,
  `out/build/macos-debug-cpu/tests/<module>/<program> --gtest_filter='Suite.Name'` (`--gtest_brief=1` prints only
  failures), or `ctest --preset macos-debug-cpu -R '^<program>$' --output-on-failure` for the whole program. A failing
  entry prints the failing TEST by name (`[  FAILED  ] Suite.Name`). The number of ctest entries is smaller than the
  number of TESTs: the gate prints `NOTE functional tests run: <n> gtest tests in <m> programs (one ctest entry
  each)`, counted from the XML report of each program (`out/build/<preset>/gtest_reports/`). The count of
  `TEST(Suite, Name)` macros under `src/` and `tests/` does not change when programs are registered this way.
- Run the unit tests or the functional tests: every test also carries its tier, so `ctest --preset macos-debug-cpu -L unit`
  runs the unit tests of every module, `-L functional` the functional tests, and `-LE functional` everything but the
  functional tests. `-L grid` runs both tiers of one module. Two `-L` options are joined by "and":
  `ctest --preset macos-debug-cpu -L grid -L unit` runs the grid tests that are also unit tests.
- Test labels: `cpu` tests run everywhere, `no_gpu` tests run only in the CUDA container (a machine without a GPU,
  because they assert that no GPU is usable), and `gpu` tests run only on hardware.
- GPU run for the owner, on Windows with an NVIDIA GPU, in PowerShell 7, from the repository root:
  `pwsh -NoProfile -ExecutionPolicy Bypass -File tools/run_gpu_tests.ps1`. It builds the preset
  `windows-release-cuda`, runs the `cpu` tests (never the `no_gpu` tests, which fail on a machine with a GPU), runs
  the `gpu` tests twice and compares the result digests the two runs write (`docs/gpu-runs.md`), runs each `gpu`
  test program under `compute-sanitizer --tool memcheck`, and prints a block
  of eleven lines (a title and ten lines) ending in `result: PASS` or `result: FAIL`; the owner pastes that block
  back. Its full logs are in `out/gpu-run/<UTC time>/`. The script selects the GPU tests with `ctest -L '^gpu$'`: a
  label is a regular expression, so a bare `-L gpu` would also run the `no_gpu` tests. The script was parsed with
  PowerShell 7.4 in a Linux container and exercised there with stand-in tools; it has never run on Windows or
  against a GPU, so its first real run is the one on the owner's machine. A module's row in `docs/gpu-runs.md` is
  added in that module's hand-over task, as `compiled, not run`, and changes only when a run recorded by this
  script says otherwise.
- Session note, after each commit: `uv run python tools/add_session_note.py --session <url> --models <id>[,<id>...]
  [<commit>]` writes or replaces the note of `<commit>` (default `HEAD`) in `refs/notes/claude-session`: the lines
  `session: <url>` and `models: <ids>`, the ids de-duplicated and ordered opus, sonnet, haiku, fable, then others.
  It fails on an empty model list and on a link that is not https. Every clone needs this once, so notes follow amend
  and rebase and show in `git log`:

  ```sh
  git config notes.rewriteRef refs/notes/claude-session
  git config notes.rewriteMode overwrite
  git config --add notes.displayRef refs/notes/claude-session
  ```

  Notes stay local; they leave the machine only when pushed on purpose with
  `git push <remote> refs/notes/claude-session:refs/notes/claude-session`. The gate does not check for a note,
  because it runs before the commit exists.
