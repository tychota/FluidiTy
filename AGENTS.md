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

`native/fluidity/contact/solid_contact.h`, its types only. The file header and the docstrings follow the
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

**Folders by concept.** Namespaces mirror folders: `native/fluidity/grid/` is `fluidity::grid`.

```text
native/fluidity/
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
a header that is not itself `_gpu`.

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
- CPU and GPU agree within a tolerance. They are not bit-identical.

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
  Host code uses `-ffp-contract=off` (GCC, Clang) and `/fp:precise` (MSVC). No fast-math anywhere.
- **Format.** Google C++ style through the repository's clang-format 23.1 settings: 120 columns, two-space
  indent, `//` comments, braces on every control-flow body.

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

A commit passes every part of this gate that exists at that commit.

**GPU honesty.** Device code is "compiled, not run" until the owner's hardware run, which happens once, after
all modules are written. Until then every module's status reads "GPU: compiled, not run". `docs/gpu-runs.md`
records each run: the commit, the device and the result per module. Nothing is described as working on the GPU
before its record exists.

**Stop and ask.** When a name meets a word that the glossary does not define and `docs/design/programme.md`
section 9 lists as open for a module, work on that name stops and the owner is asked. Do not guess.

## Commands

- Gate for every commit: not written yet. It is added here together with the gate script.
- Windows GPU run for the owner: not written yet. It is added here together with its script.
- Session note, after each commit: `uv run python tools/add_session_note.py --session <url> --models <id>[,<id>...]
  [<commit>]` writes or replaces the note of `<commit>` (default `HEAD`) in `refs/notes/claude-session`: the lines
  `session: <url>` and `models: <ids>`, the ids de-duplicated and ordered opus, sonnet, haiku, fable, then others.
  It refuses an empty model list and a link that is not https. Every clone needs this once, so notes follow amend and
  rebase and show in `git log`:

  ```sh
  git config notes.rewriteRef refs/notes/claude-session
  git config notes.rewriteMode cat_sort_uniq
  git config --add notes.displayRef refs/notes/claude-session
  ```

  Notes stay local; they leave the machine only when pushed on purpose with
  `git push <remote> refs/notes/claude-session:refs/notes/claude-session`. The gate does not check for a note,
  because it runs before the commit exists.
