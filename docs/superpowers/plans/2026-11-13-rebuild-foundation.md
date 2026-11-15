# Rebuild Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the orphan `rebuild` branch with its rules, glossary, slim CPU+CUDA build, shared base math, the
local commit gate and the owner's GPU-run command, so that module 1 (contact) can start on a proven chain.

**Architecture:** A new git worktree holds an orphan branch that shares no commit with `main`. One static library
`fluidity` is built from `native/fluidity/<concept>/`; each operation has one public header and private `_cpu.cc`
/ `_gpu.cu` files; shared math headers compile for host and device. One script is the local gate for every
commit: format, rules check, strict CPU build and tests, CUDA compile in a Linux container.

**Tech Stack:** C++23 host, CUDA C++20 device, CMake presets + Ninja, Conan 2 (GoogleTest only for now), uv for
Python tools, clang-format and clang-tidy 19.1, Docker via colima, Typst for the method paper, PowerShell for the Windows run.

**Spec:** `docs/superpowers/specs/2026-11-13-production-rebuild-design.md` (in the `main` checkout, untracked;
Task 2 commits it on `rebuild` as `docs/design/programme.md`). Second input, read but never committed:
`/Users/tychota/Downloads/FluidiTy_Report_Source/inputs/owner_naming_comments_architecture.md` ("the naming
review").

## Global Constraints

- Work is local. Never `git push`, never open a pull request, never touch `main` or remote branches.
- Old code is read with `git show main:<path>` from the worktree. Nothing is checked out wholesale.
- Commit rule (spec section 6): one thing per commit; strict CPU build and all tests pass; CUDA target compiles
  in the container; clang-format 19.1 clean; clang-tidy 19.1 clean on host sources; every touched file passes the `AGENTS.md` checklist; new public
  names listed in the commit message with a one-line reason; tests arrive with the code. A commit must pass every
  part of the gate that exists at that commit.
- History on `rebuild` may be rewritten. A defect found later is fixed in the commit that introduced it.
- Commit messages carry no AI trailer (no `Co-Authored-By`, no `Assisted-by`, no session link); the owner is the
  author. After committing, write the commit's note with `uv run python tools/add_session_note.py --session
  <session link> --models <implementer's model id>,claude-fable-5-2`
  (a git note in `refs/notes/claude-session`; owner answer A55, 2026-11-15).
- Device code is "compiled, not run" until `docs/gpu-runs.md` records the owner's run. The owner runs on
  hardware once, after all modules are written, so Foundation ends without a hardware run. Never claim otherwise.
- Host C++23, exceptions and RTTI off in the library; device and shared headers C++20 subset; a shared header
  never includes `<expected>`, `<string>` or anything that allocates. No Abseil.
- Device numerical flags: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. Host: `-ffp-contract=off`,
  `/fp:precise`. No fast-math.
- Z-up, right-handed. Gravity points along `{0, 0, -1}`.
- Naming: no `3D` suffix, no `Cuda`/`Gpu`/`Device`/`host` in public names, no bare `bool` parameters, no
  abbreviations (MAC, PIC, APIC, P2G, G2P, EOS), no milestone codes. Function verbs: `Compute`, `Build`, `Check`,
  `Find`, `Apply`, `Solve`, `Read`, `Write`; anything else is listed for the owner's review.
- 120 columns, two-space indent, `//` comments, braces on every control-flow body.
- Tests are named `TEST(FunctionOrType, Situation_Outcome)`, one behaviour each, in `*_test.cc` beside the code.
- Deletion is the default: add a dependency, file or tool only when a task here names it.

## Task order

Run the tasks in this order: 1, 2, 3, **10**, 4, 5, **12**, 6, 7, **13, 14, 15, 16**, 8, 9, 11. The method paper
(Task 10) comes before any code so that no commit ever carries a `Read:` line pointing at a label that does
not exist yet. Tasks 12 to 16 were added on 2026-11-14 from the module designs: they are the `base/` pieces
every module needs before contact can start. Their exact declarations are in
`docs/superpowers/specs/modules/shared-interface.md` section (a) (in the `main` checkout); the declarations
there are the contract and each task below repeats only what it must decide beyond them.

## Review Focus

1. **CUDA build on a machine with no GPU** (the container, a CI runner): `CheckCanRunOn(RunOn::kGpu)` returns
   `RunOnError::kNoGpuFound`; it does not crash or hang. Pinned in Task 7.
2. **Gate run while Docker is not running:** the gate fails with a message naming Docker; it never reports the
   CUDA step as passed or skipped. Pinned in Task 7.
3. **Windows run without Compute Sanitizer installed:** the summary says the sanitizer step failed; the overall
   result is FAIL. Pinned in Task 8.
4. **A shared header pulled into device code with a host-only type:** compilation stops with a message naming
   the header. Pinned in Task 6.
5. **Rules check on files with CRLF line endings or a path containing a space:** same findings as with LF and
   no space. Pinned in Task 3.

---

### Task 1: Prove the CUDA compile container on this Mac

**Result, recorded 2026-11-14 (this task is done; later tasks use these facts):**

- `CUDA_IMAGE = nvidia/cuda@sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311`
  (tag `13.4.1-devel-ubuntu24.04`; nvcc 13.4.59, the toolkit line `main` was qualified on).
- `CUDA_PLATFORM = linux/arm64` (native on this Mac; no emulation).
- A one-function `.cu` file compiled with `nvcc -std=c++20 --fmad=false -arch=sm_120 -c` in under one second.
- Docker runs through colima (`colima start --cpu 4 --memory 8 --disk 60`). Colima shares only the home
  folder with containers: the worktree is under the home folder, so mounting it works; a path under `/tmp`
  or `/private/tmp` is not visible inside a container.
- What this proves and what it does not: device code and the CUDA host glue compile for `sm_120` with the
  Linux toolkit on an ARM host. It does not prove the code runs, and it is not the Windows x86-64 MSVC build;
  the Windows CPU lane in CI and the owner's hardware run cover those.

The steps below are kept as the record of how the result was obtained.

No commit. This decides commit-rule item 3, so it runs first.

**Files:**
- Create (temporary only): `<temp-dir>/cuda-probe/probe.cu`, `<temp-dir>/cuda-probe/Dockerfile`

**Interfaces:**
- Produces: three facts used by Task 7: `CUDA_IMAGE` (full image reference with digest), `CUDA_PLATFORM`
  (`linux/arm64` or `linux/amd64`), and the measured wall time of one compile.

- [ ] **Step 1: Start Docker.** Run `colima start --cpu 4 --memory 8 --disk 60`, then `docker version`.
  Expected: a server version is printed.
- [ ] **Step 2: Pick the image.** Toolkit must match `main`'s qualified line (13.4.x; see
  `git show main:docs/native/toolchains.md`). Run `docker manifest inspect nvidia/cuda:<tag>` for
  `13.4.1-devel-ubuntu24.04`, then the nearest 13.4.x devel tag if that one is absent. If the manifest lists
  `arm64`, `CUDA_PLATFORM=linux/arm64`; otherwise `linux/amd64`.
- [ ] **Step 3: Compile a probe.** `probe.cu`: one `__global__` function that adds two `double` arrays, built
  with `nvcc -std=c++20 --fmad=false -arch=sm_120 -c probe.cu` inside the image.
  Expected: exit 0 and an object file. Record the wall time.
- [ ] **Step 4: Apply the fallback rule from the spec (section 10, item 3).**
  - Native arm64 works: use it.
  - Only amd64 works: use it under emulation if the compile in Step 3 took under 60 s.
  - Neither, or slower: stop and tell the owner; commit-rule item 3 would become "CI CUDA compile is green",
    which needs their agreement because work is local.
- [ ] **Step 5: Report** the three facts to the owner in one line before continuing.

### Task 2: Orphan branch, rules and glossary

**Files:**
- Create worktree: `/Users/tychota/Documents/Code/openSource/FluidiTy-rebuild` (branch `rebuild`, orphan)
- Create: `docs/design/programme.md` (copy of the spec, unchanged apart from its own path references)
- Create: `AGENTS.md`, `docs/glossary.md`, `README.md`, `.gitignore`, `.gitattributes`

**Interfaces:**
- Produces: `docs/glossary.md` in a fixed table shape that Task 3 parses: every vocabulary table has exactly the
  header `| Say | Plain meaning | Never say |`. In the last column, entries are comma-separated; an entry that is
  banned in one sense only carries a parenthesised qualifier, for example `component (for this)` or `row (alone)`.
  `AGENTS.md` has a section titled `## Checklist for each file` and a section titled `## Allowed verbs`.

- [ ] **Step 1: Create the worktree.** From the `main` checkout:
  `git worktree add --orphan -b rebuild ../FluidiTy-rebuild`. Expected: an empty directory on branch `rebuild`;
  `git -C ../FluidiTy-rebuild log` reports no commits.
- [ ] **Step 2: Write `docs/glossary.md`** from the naming review section 4 (4.1 to 4.9, including the step
  stage list of 4.7 as a numbered list without its "Was" column).
  - Keep only rows marked Decided. Leave out every row or cell marked Verify or Open (they are listed in
    `docs/design/programme.md` section 9 and enter with their module).
  - No hit counts, no old type names except inside the "Never say" column.
  - Section 4.4 (fixed-point words) is left out entirely: that machinery does not exist on this branch. Add
    `quantum`, `quanta`, `quantized` to a final "Never say, no replacement" list.
- [ ] **Step 3: Write `AGENTS.md`.** Sections, in order:
  1. *The model in seven sentences* (naming review section 2).
  2. *Naming rules* (section 5, all 14; rule 6 without the `legacy2d` sentence).
  3. *Allowed verbs* (rule 8's list plus the stage verbs and particle verbs from the glossary), with two
     owner additions of 2026-11-13: `To<Type>` converts a value to that type (`ToMessage`); and inside
     `namespace math` a function that only returns a value is named with the plain math word for it (`Dot`,
     `Cross`, `Transpose`, `Determinant`).
  4. *Comment rules* (section 6.1), *File header* (6.2), *Function docstring* (6.3), *Function bodies* (6.4),
     *Citations* (6.5), each with its template.
  5. *One worked example*: only the "After" code of section 6.6.
  6. *Structure*: folders by concept as in `programme.md` section 8; one operation with two private
     implementations; `RunOn` as the caller's choice, never changed silently; one error enum per module returned
     through `std::expected` (used directly, no alias), text added by `ToMessage` at the outer boundary; the exactness requirement (section 7.4's two bullets); Z-up.
  7. *Language*: C++23 host, C++20 device and shared headers, header kinds, no exceptions or RTTI, never call
     `std::expected::value()` (it throws; test `has_value()` and use `*`); every CPU-only header includes
     `base/cpu_only.h` first.
  8. *Checklist for each file* (section 10, without the PR wording: "listed in the commit message").
  9. *Commit rule* (`programme.md` section 6, including the stop-and-ask rule and GPU honesty).
  10. *Commands*: placeholder lines for the gate and the Windows run, filled in by Tasks 7 and 8.
  It must contain no finding about old code, no hit count, no "was:" name, no roll-out table.
- [ ] **Step 4: Write `README.md`** (what FluidiTy is, that this branch is a rebuild in progress, where the rules
  and design live; under 40 lines), `.gitignore` (build outputs: `out/`, `.venv/`, `__pycache__/`, `.DS_Store`,
  `*.pdf` under `docs/method-paper/`), `.gitattributes` (`* text=auto eol=lf`, `*.ps1 text eol=crlf`).
- [ ] **Step 5: Verify.** Run
  `grep -nEi "was:|hits|[0-9]+ (hits|places)|m5_|AttemptWith|3D\b" AGENTS.md docs/glossary.md`.
  Expected: matches only inside "Never say" cells and inside the rule sentences that name the banned patterns.
  Read both files once from top to bottom as the stated reader.
- [ ] **Step 6: Commit.** `git add -A && git commit` with subject `Start the rebuild with its rules and glossary`.

### Task 3: Python tooling and the rules check

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.clang-format`, `.clang-tidy`, `tools/check_rules.py`, `tools/check_rules_test.py`,
  `tools/check_rules_allowed_names.txt`

**Interfaces:**
- Consumes: the glossary table shape and `AGENTS.md` section titles from Task 2.
- Produces: `uv run python tools/check_rules.py [PATH ...]` (default path `native`). Exit 0 when no failure,
  1 otherwise. Output lines `path:line: FAIL <rule-id> <text>` and `path:line: NOTE <rule-id> <text>`; notes do
  not change the exit code. Rule ids: `file-header`, `banned-word`, `one-sense-word`, `dimension-suffix`,
  `milestone-code`, `abbreviation`, `bare-bool`, `verb`, `role-suffix`, `backend-name`.
  `uv run clang-format` resolves to 19.1.5 and `uv run clang-tidy` to 19.1.x.
  `.clang-tidy` is used by Task 7's gate through `uv run clang-tidy -p out/build/<preset>` over host sources
  (`.cc` and the `.h` they include; `.cu` and `.cuh` are not tidied).

- [ ] **Step 1: `pyproject.toml`.** Project `fluidity-tools`, `requires-python = ">=3.13,<3.14"`, no runtime
  dependencies, dependency group `dev`: `clang-format==19.1.5`, `clang-tidy` (the newest 19.1.x wheel on
  PyPI; pin what resolves), `pytest==8.3.5`, `conan==2.34.0`, `cmake==4.4.4`,
  `ninja==1.13.2`, `actionlint-py` (latest; pin what resolves). `uv lock`. Copy `.clang-format` unchanged from
  `git show main:.clang-format`.
  Verify: `uv run clang-format --version` prints `19.1.5`; `uv run clang-tidy --version` prints `19.1`.
  `.clang-tidy`: `WarningsAsErrors: '*'`, `HeaderFilterRegex: 'native/fluidity/.*'`, checks
  `bugprone-*`, `performance-*`, `modernize-*`, `readability-*`, `cppcoreguidelines-init-variables`,
  `cppcoreguidelines-pro-type-member-init`, `misc-const-correctness`, `misc-include-cleaner`, minus
  `modernize-use-trailing-return-type`, `readability-identifier-length`, `readability-magic-numbers` (the rules
  check and review own that one) and `readability-math-missing-parentheses`. Add
  `readability-identifier-naming` options that encode Google style as `AGENTS.md` uses it: types and functions
  `CamelCase`, variables, parameters and fields `lower_case`, constants and enumerators `k` + `CamelCase`,
  macros `UPPER_CASE`. If a check cannot be satisfied in test files because of GoogleTest macros, disable it
  for `*_test.cc` only, with a comment saying why.
- [ ] **Step 2: Write failing tests** in `tools/check_rules_test.py`. Each builds a small tree in `tmp_path` and
  calls `check_rules.check_paths(paths, glossary_path) -> list[Finding]` where
  `Finding = (path, line, severity, rule_id, text)`.

  | Test | Input | Expected |
  |---|---|---|
  | `test_header_without_purpose_fails` | `a.h` starting with `#pragma once` | one FAIL `file-header` |
  | `test_header_with_all_labels_passes` | `a.h` with `// Purpose:`, `// Idea:`, `// Owns:`, `// Uses:`, `// Used by:`, `// Read:` before the first non-comment line | no finding |
  | `test_banned_word_in_identifier_fails` | `int stencil_count;` with glossary banning `stencil` | FAIL `banned-word` |
  | `test_banned_word_in_comment_fails` | `// the stencil` | FAIL `banned-word` |
  | `test_one_sense_word_is_a_note` | `int component;` with glossary entry `component (for this)` | NOTE `one-sense-word`, no FAIL |
  | `test_dimension_suffix_fails` | `struct Grid3D {};` | FAIL `dimension-suffix` |
  | `test_milestone_code_fails` | file named `m5_material.h`; comment `// certified F3 split` | two FAIL `milestone-code` |
  | `test_abbreviation_fails` | `double pic_velocity;` | FAIL `abbreviation` |
  | `test_bare_bool_parameter_fails` | `void ApplyForces(double x, bool bounded);` in a `.h` | FAIL `bare-bool` |
  | `test_bool_field_is_allowed` | `bool has_liquid = false;` inside a struct | no finding |
  | `test_unlisted_verb_fails` | `double EvaluateWeight(double x);` in a `.h` | FAIL `verb` |
  | `test_math_namespace_noun_is_allowed` | `namespace fluidity::math { double Dot(Vec3 a, Vec3 b); }` in a `.h` | no finding |
  | `test_to_conversion_is_allowed` | `std::string_view ToMessage(RunOnError error);` in a `.h` | no finding |
  | `test_allowed_name_file_exempts` | same, with `EvaluateWeight` in the allowed-names file | no finding |
  | `test_role_suffix_fails` | `struct StepOptions {};` | FAIL `role-suffix` |
  | `test_backend_word_in_header_fails` | `void WriteCudaBuffer();` in a `.h` | FAIL `backend-name` |
  | `test_crlf_file_gives_same_findings` | the `stencil` file written with `\r\n` | same single FAIL |
  | `test_path_with_space_is_checked` | the `stencil` file under `my dir/` | same single FAIL, path reported intact |
  | `test_glossary_with_wrong_header_is_an_error` | glossary table headed `| Word | Meaning |` | raises `GlossaryError` |

- [ ] **Step 3: Run** `uv run pytest tools/check_rules_test.py -q`. Expected: all fail (module missing).
- [ ] **Step 4: Implement `tools/check_rules.py`.** Decisions the tests do not fix:
  - Files scanned: `.h .cuh .cc .cu .py .lean`; `file-header`, `bare-bool`, `verb`, `backend-name` apply to
    `.h` and `.cuh` only.
  - Words are matched on tokens split at `_` and at lower-to-upper case changes, case-insensitively, so `cut`
    does not match `execute`.
  - `verb`: a free function declared at namespace scope in a header must start with a word from the
    `## Allowed verbs` section of `AGENTS.md`, or with `To` (converts to that type), or be an operator, or be
    declared inside `namespace math` (or `fluidity::math`), or be listed in
    `tools/check_rules_allowed_names.txt` (format: `Name  # one-line reason`, owner-approved names only).
  - `role-suffix` fails type names ending in `Input`, `Options`, `Parameters`, `Params`, `Config`, `Results`,
    `Outcome`.
  - `backend-name` fails `Cuda`, `Gpu`, `Device`, `Host` tokens in identifiers declared in `.h` files, except
    enumerators of `RunOn` and the macro `FLUIDITY_CPU_AND_GPU`.
  - `milestone-code`: tokens matching `m[0-9]+` or `f[0-9]+` standing alone, and the word `qualification`.
- [ ] **Step 5: Run** the tests. Expected: all pass. Run `uv run python tools/check_rules.py tools`.
  Expected: exit 0.
- [ ] **Step 6: Commit.** Subject: `Add the tool environment and the rules check`.

### Task 4: Build skeleton and `Vec3`

**Files:**
- Create: `conanfile.py`, `conan.lock`, `CMakeLists.txt`, `CMakePresets.json`, `cmake/fluidity_targets.cmake`
- Create: `native/fluidity/base/cpu_and_gpu.h`, `native/fluidity/base/vec3.h`, `native/fluidity/base/vec3_test.cc`

**Interfaces:**
- Produces:
  - Macro `FLUIDITY_CPU_AND_GPU` (expands to `__host__ __device__` under the CUDA compiler, to nothing otherwise).
  - `namespace fluidity { struct Vec3 { double x; double y; double z; }; }` with `constexpr` `FLUIDITY_CPU_AND_GPU`
    `noexcept` `operator+`, `operator-` (binary and unary), `operator*` (vector × scalar, scalar × vector),
    and, in `namespace fluidity::math`, `double Dot(Vec3, Vec3)` and `Vec3 Cross(Vec3, Vec3)`.
    Owner decision: math helpers are short nouns and live in `fluidity::math`, so a call reads
    `math::Dot(velocity, normal_out_of_solid)`. `AGENTS.md` (Task 2) carries the rule: "inside `math::`, a
    function that only returns a value is named with the plain math word for that value"; the rules check
    (Task 3) exempts namespace `math` from the verb rule.
  - CMake: static library target `fluidity`; `fluidity_add_test(<name> SOURCES <files> LABEL <cpu|gpu|no_gpu>)`
    in `cmake/fluidity_targets.cmake`; option `FLUIDITY_ENABLE_CUDA` (default `OFF`).
  - Presets `macos-debug-cpu`, `macos-release-cpu`, `linux-debug-cpu`, `linux-release-cpu`,
    `windows-debug-cpu`, `windows-release-cpu`; build dir `out/build/<preset>`, Conan output
    `out/conan/<preset>`; test presets of the same names.

- [ ] **Step 1: Build files.**
  - `conanfile.py`: `test_requires = "gtest/1.17.0"` and nothing else; `CMakeDeps` + `CMakeToolchain` (Ninja).
    `conan lock create` to produce `conan.lock`.
  - `CMakeLists.txt`: `cmake_minimum_required(VERSION 3.31)`; `CMAKE_CXX_STANDARD 23`, required, extensions off;
    a configure-time `check_cxx_source_compiles` that uses `std::expected<int, int>` and stops with
    "This compiler's standard library has no std::expected" when it fails; warning and floating-point flags
    as on `main` (`git show main:cmake/NativeOptions.cmake`, the `target_compile_options` block only), applied
    to our targets only; exceptions and RTTI off for `fluidity`, exceptions on for test targets.
  - None of `main`'s toolchain-receipt machinery, schema rules or fixture directories.
- [ ] **Step 2: Write the failing tests** in `vec3_test.cc`:

  ```cpp
  TEST(Vec3, AddTwoVectors_AddsEachAxis)            // {1,2,3}+{10,20,30} == {11,22,33}
  TEST(Vec3, SubtractTwoVectors_SubtractsEachAxis)  // {11,22,33}-{10,20,30} == {1,2,3}
  TEST(Vec3, Negate_FlipsEachAxis)                  // -{1,-2,3} == {-1,2,-3}
  TEST(Vec3, ScaleFromEitherSide_GivesSameVector)   // {1,2,3}*2 == 2*{1,2,3} == {2,4,6}
  TEST(Dot, PerpendicularVectors_IsZero)            // {1,0,0}.{0,1,0} == 0
  TEST(Dot, GeneralVectors_SumsAxisProducts)        // {1,2,3}.{4,5,6} == 32
  TEST(Cross, XThenY_IsZ)                           // {1,0,0}x{0,1,0} == {0,0,1}  (right-handed)
  TEST(Cross, SameVector_IsZero)                    // {1,2,3}x{1,2,3} == {0,0,0}
  ```
  Each also `static_assert`s its expression so `constexpr` is proven.
- [ ] **Step 3: Run.** `uv run conan install . --lockfile conan.lock -of out/conan/macos-debug-cpu -s build_type=Debug --build=missing`,
  then `uv run cmake --preset macos-debug-cpu && uv run cmake --build --preset macos-debug-cpu`.
  Expected: compile error, `vec3.h` not found.
- [ ] **Step 4: Implement** `cpu_and_gpu.h` and `vec3.h`, each with the file header from `AGENTS.md`. `vec3.h`'s
  header states: meters or any unit, right-handed, Z is up. `Read:` cites `docs/method-paper/fluidity-method.typ
  <sec-conventions>` (the label exists: Task 10 ran first).
- [ ] **Step 5: Run** build and `uv run ctest --preset macos-debug-cpu`; repeat for `macos-release-cpu`.
  Expected: 8 tests pass in each. Then `uv run clang-format --dry-run --Werror $(git ls-files 'native/*')` and
  `uv run python tools/check_rules.py`. Expected: both exit 0.
- [ ] **Step 6: Commit.** Subject: `Add the build and the three-axis vector`. Body lists new public names:
  `FLUIDITY_CPU_AND_GPU`, `Vec3`, `math::Dot`, `math::Cross`.

### Task 5: `Mat3` and the up direction

**Files:**
- Create: `native/fluidity/base/mat3.h`, `native/fluidity/base/mat3_test.cc`, `native/fluidity/base/axis.h`,
  `native/fluidity/base/axis_test.cc`

**Interfaces:**
- Consumes: `Vec3`, `FLUIDITY_CPU_AND_GPU`.
- Produces:
  - `struct Mat3 { double xx, xy, xz, yx, yy, yz, zx, zy, zz; }` (row-major; first letter is the row).
    `operator*(Mat3, Vec3)`, `operator*(Mat3, Mat3)`, `operator+`, `operator-`, `operator*(Mat3, double)`,
    `math::Transpose(Mat3) -> Mat3`, `math::Determinant(Mat3) -> double`, `inline constexpr Mat3 kIdentity`.
  - `enum class Axis { kX, kY, kZ };` and `inline constexpr Vec3 kUp{0, 0, 1};`,
    `constexpr Vec3 math::UnitVector(Axis)`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(Mat3, IdentityTimesVector_ReturnsVector)
  TEST(Mat3, TimesVector_UsesRows)            // rows {1,2,3},{4,5,6},{7,8,9} times {1,0,-1} == {-2,-2,-2}
  TEST(Mat3, TimesMatrix_IsNotCommutative)    // A*B != B*A for a rotation about X and one about Z
  TEST(Transpose, SwapsRowsAndColumns)        // Transpose(A).xy == A.yx for all six off-diagonal entries
  TEST(Determinant, Identity_IsOne)
  TEST(Determinant, SwappedYAndZ_IsMinusOne)  // a swap is a reflection, not a rotation
  TEST(Determinant, YUpToZUpRotation_IsOne)   // R = rows {1,0,0},{0,0,-1},{0,1,0}
  TEST(Mat3, YUpToZUpRotation_SendsOldGravityDown)  // R * {0,-9.81,0} == {0,0,-9.81}
  TEST(Axis, UnitVectorOfZ_IsUp)              // UnitVector(Axis::kZ) == kUp == {0,0,1}
  ```
- [ ] **Step 2: Run** the build. Expected: compile error, headers missing.
- [ ] **Step 3: Implement.** `axis.h`'s file header is where the convention is stated once for the code base:
  right-handed, Z up, gravity along minus Z, matches Blender; it gives the rotation above as the way to bring a
  value from the proof of concept (Y up) into this convention, and warns that swapping two axes is a reflection.
- [ ] **Step 4: Run** build, tests (both macOS presets), format, rules. Expected: 17 tests pass; checks exit 0.
- [ ] **Step 5: Commit.** Subject: `Add the 3x3 matrix and state the up direction`. New names: `Mat3`, `math::Transpose`, `math::Determinant`, `kIdentity`, `Axis`, `kUp`, `math::UnitVector`.

### Task 6: `RunOn` and the CPU side of `CheckCanRunOn`

**Files:**
- Create: `native/fluidity/base/cpu_only.h`, `native/fluidity/base/run_on.h` (shared: the enum only),
  `native/fluidity/base/check_can_run_on.h` (CPU only), `native/fluidity/base/check_can_run_on.cc`,
  `native/fluidity/base/run_on_gpu_check.h`, `native/fluidity/base/check_can_run_on_test.cc`

**Interfaces:**
- Produces:

  ```cpp
  // cpu_only.h: included first by every header that must never reach GPU code. No declarations.
  //   Under the CUDA compiler it stops the build with:
  //   "A CPU-only header was included in GPU code. Split the header that included it."

  // run_on.h  (shared CPU-and-GPU header, C++20 subset: only the enum, so views in other modules' shared
  //   headers can carry a RunOn member)
  //   enum class RunOn : std::uint8_t { kCpu, kGpu };
  // check_can_run_on.h  (CPU only; includes cpu_only.h then run_on.h; std::expected is used directly, no alias)
  enum class RunOnError {
    kGpuNotBuilt,     // This build has no GPU code.
    kNoGpuFound,      // The build has GPU code but the machine has no usable GPU.
    kGpuCallFailed,   // A GPU call returned an error.
    kGpuMathDiffers,  // The shared math gave different numbers on the GPU.
  };
  std::expected<void, RunOnError> CheckCanRunOn(RunOn run_on);
  std::string_view ToMessage(RunOnError error);
  // The enumerator without k, in lower snake case ("gpu_not_built"). The stable code the CLI and the
  // add-on use. Every module's error enum has ToName beside ToMessage.
  std::string_view ToName(RunOnError error);

  // run_on_gpu_check.h  (private to base/, C++20 subset, included by check_can_run_on.cc and by Task 7's .cu file)
  // Returns true when the GPU gave the same numbers as the CPU. Otherwise returns false and sets *error.
  bool CheckGpuMath(RunOnError* error);  // defined in Task 7; declared here so check_can_run_on.cc compiles without CUDA
  ```
  `check_can_run_on.cc` calls `CheckGpuMath` only when compiled with `FLUIDITY_HAS_GPU` defined.
  `RunOn` is a choice made by the caller (the add-on may ask for the CPU on a machine that has a GPU).
  `CheckCanRunOn` only answers whether the requested choice can run here. It never picks a processor and
  nothing ever falls back from GPU to CPU silently; the docstring says so.

- [ ] **Step 1: Failing tests** (`check_can_run_on_test.cc`, label `cpu`):

  ```cpp
  TEST(CheckCanRunOn, Cpu_Succeeds)                          // has_value()
  TEST(CheckCanRunOn, GpuInBuildWithoutGpuCode_IsNotBuilt)   // only when FLUIDITY_HAS_GPU is undefined:
                                                          // error() == RunOnError::kGpuNotBuilt
  TEST(ToMessage, EveryError_HasADistinctNonEmptyMessage) // four messages, all different, none empty
  TEST(ToName, EveryError_IsLowerSnakeCaseWithoutK)       // RunOnError::kGpuNotBuilt -> "gpu_not_built"
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** `cpu_only.h` holds the `#if defined(__CUDACC__)` / `#error` guard so Review
  Focus item 4 holds. `ToMessage` returns plain sentences that say what the user can do
  ("This build has no GPU code. Use a build made with the CUDA preset.").
- [ ] **Step 4: Prove the guard.** In a temporary directory, compile a one-line file that includes
  `check_can_run_on.h` with
  `-D__CUDACC__`. Expected: the `#error` text. (Not a committed test: it is a must-not-compile case.)
- [ ] **Step 5: Run** build, tests (20 pass), format, rules.
- [ ] **Step 6: Commit.** Subject: `Add RunOn and its check on the CPU`. New names: `RunOn`, `RunOnError` and its four values, `CheckCanRunOn`, `ToMessage`.

### Task 7: CUDA build, the GPU side of `CheckCanRunOn`, and the gate

**Files:**
- Create: `tools/cuda-container/Dockerfile`, `tools/check_commit.py`, `tools/check_commit_test.py`
- Create: `native/fluidity/base/check_can_run_on_gpu.cu`, `native/fluidity/base/check_can_run_on_gpu_test.cc`,
  `native/fluidity/base/check_can_run_on_no_gpu_test.cc`
- Modify: `CMakeLists.txt`, `cmake/fluidity_targets.cmake`, `CMakePresets.json`, `AGENTS.md` (Commands section)

**Interfaces:**
- Consumes: `CUDA_IMAGE`, `CUDA_PLATFORM` from Task 1; `CheckGpuMath` declaration from Task 6.
- Produces:
  - Presets `linux-release-cuda`, `windows-release-cuda`, `windows-debug-cuda` with
    `FLUIDITY_ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES="120-real;120-virtual"`; Windows adds
    `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`.
  - With CUDA on: `.cu` files join the `fluidity` target, `CUDA_STANDARD 20` required, extensions off, device
    flags from Global Constraints, `FLUIDITY_HAS_GPU` defined, `CUDA::cudart` linked.
  - `uv run python tools/check_commit.py [--only format|rules|cpu|tidy|paper|cuda]`: runs, in order, format check, rules
    check, `macos-debug-cpu` and `macos-release-cpu` configure/build/test (the host OS's presets when not on
    macOS), `tidy` (clang-tidy over host sources using the debug preset's compile commands), the `paper` step, then the container build of `linux-release-cuda` plus `ctest -L no_gpu` inside
    the container. The `paper` step compiles the method paper and, for every `Read:` line under `native/` that
    names a `<label>`, checks that the paper defines the label.
    Prints one line per step (`PASS format`, `FAIL cuda: <reason>`) and exits non-zero on any failure.

- [ ] **Step 1: Failing tests for the gate** (`check_commit_test.py`, with `subprocess.run` replaced by a fake):

  | Test | Condition | Expected |
  |---|---|---|
  | `test_all_steps_pass_exits_zero` | every command returns 0 | exit 0, seven `PASS` lines in order |
  | `test_read_line_with_unknown_label_fails_paper_step` | a header whose `Read:` line cites `<sec-missing>` | `FAIL paper: <path>:<line> cites <sec-missing>, which the paper does not define` |
  | `test_read_lines_with_known_labels_pass` | headers citing `<sec-conventions>` | `PASS paper` |
  | `test_first_failure_is_reported_and_exit_is_nonzero` | rules step returns 1 | `FAIL rules`, exit 1, later steps still listed as `NOT RUN` |
  | `test_docker_not_running_fails_cuda_step` | `docker info` returns non-zero | `FAIL cuda: Docker is not running (start it with: colima start)`, exit 1 |
  | `test_only_flag_runs_one_step` | `--only format` | one line, `PASS format` |
  | `test_cuda_is_never_reported_skipped` | Docker missing from `PATH` | output contains `FAIL cuda`, never `SKIP` |

- [ ] **Step 2: Failing GPU tests.**

  ```cpp
  // check_can_run_on_gpu_test.cc, label gpu (needs hardware; compiled here, run by the owner)
  TEST(CheckCanRunOn, GpuOnAMachineWithAGpu_Succeeds)
  TEST(CheckCanRunOn, GpuTwiceInARow_SucceedsBothTimes)

  // base_constants_gpu_compile_test.cu, label no_gpu: a GPU function (never launched) reads `kUp` and
  // `kIdentity` at run time (`gravity * kUp`, `matrix + kIdentity`); the test takes its address. It proves in
  // the container that device code may read these class-type constants. (A probe on 2026-11-14 with nvcc
  // 13.4.59 compiled this pattern; the test keeps it true.)
  // check_can_run_on_no_gpu_test.cc, label no_gpu (run in the container, which has no GPU)
  TEST(CheckCanRunOn, GpuOnAMachineWithoutAGpu_IsNoGpuFound)   // error() == RunOnError::kNoGpuFound
  ```
- [ ] **Step 3: Implement `check_can_run_on_gpu.cu`.** `CheckGpuMath`:
  1. Ask the runtime for the GPU count; zero or the "no device / no driver" errors give `kNoGpuFound`.
  2. For 32 fixed inputs (generated in code from a small integer formula, the same on both sides, and chosen
     so that a wrong numerical flag changes a result: several are subnormal, so flush-to-zero would show, and
     several products are picked so that a fused multiply-add would round differently; none produces a NaN or
     an infinity), compute on the GPU `math::Cross(a, b)`, `math::Dot(a, b)`, `rotation * a`,
     `math::Determinant` of a matrix built from them, and `math::Sqrt`, `math::Floor` and `math::Length` of
     values built from them. Only operations IEEE 754 defines as correctly rounded are used (add, subtract,
     multiply, divide, square root, floor), which is why the same bits are expected.
  3. Copy the 32 results back and compare with the same expressions evaluated on the host **bit for bit**.
     A difference gives `kGpuMathDiffers`; any failing runtime call gives `kGpuCallFailed`.
  Exact equality is intended here: each result is a short chain of IEEE add and multiply with contraction off
  on both sides. If the owner's run reports `kGpuMathDiffers`, that contradicts the numerical flags and is
  investigated, not relaxed. Free every allocation on every path.
- [ ] **Step 4: `Dockerfile`.** `FROM ${CUDA_IMAGE}`; install `uv`; copy nothing from the repository (it is
  mounted at run time, read-only, with `out/` on a named volume so Conan packages persist between runs).
  The gate invokes: `docker run --rm --platform ${CUDA_PLATFORM} -v <repo>:/src:ro -v fluidity-cuda-out:/out ...`
  then, inside, `conan install`, `cmake --preset linux-release-cuda`, build, `ctest -L no_gpu`.
- [ ] **Step 5: Implement `tools/check_commit.py`** until Step 1's tests pass.
- [ ] **Step 6: Run the real gate.** `uv run python tools/check_commit.py`.
  Expected: `PASS format`, `PASS rules`, `PASS cpu macos-debug-cpu`, `PASS cpu macos-release-cpu`, `PASS tidy`,
  `PASS paper`, `PASS cuda`
  (the last meaning: every `.cu` and every GPU test compiled and linked, and the one `no_gpu` test passed).
- [ ] **Step 7: Fill in `AGENTS.md` Commands** with the gate command and what each line of its output means.
- [ ] **Step 8: Commit.** Subject: `Compile for the GPU and gate every commit`. Body states
  `CUDA: compiled, not run` and lists `CheckGpuMath` (private).

### Task 8: The owner's GPU run

**Files:**
- Create: `tools/run_gpu_tests.ps1`, `docs/gpu-runs.md`
- Modify: `AGENTS.md` (Commands section)

**Interfaces:**
- Produces: `pwsh tools/run_gpu_tests.ps1` on Windows. It prints, last, a fenced block the owner pastes back:

  ```text
  FluidiTy GPU run
  commit:     <full hash> (<clean|modified>)
  gpu:        <name>, driver <version>
  toolkit:    <nvcc version>
  build:      PASS|FAIL
  gpu tests:  <passed>/<total> PASS|FAIL
  repeat run: PASS|FAIL
  sanitizer:  PASS|FAIL <error count or reason>
  result:     PASS|FAIL
  ```

- [ ] **Step 1: Write the script.** Steps: check `nvcc`, `nvidia-smi`, `cl`, `uv` are on `PATH` (each missing
  tool is a named FAIL line, not an exception trace); `uv sync`; `conan install` for `windows-release-cuda`;
  configure, build; `ctest -L gpu`; run the GPU test executable a second time and require the same result
  (`repeat run`); run it under `compute-sanitizer --tool memcheck --error-exitcode 1`. If `compute-sanitizer`
  is not found, `sanitizer: FAIL not installed` and `result: FAIL`. `result` is PASS only when every line
  above it is PASS and the commit is clean.
- [ ] **Step 2: Check what can be checked here.** `pwsh` is not expected on this Mac: if `pwsh` is absent, say
  so plainly in the hand-over ("the Windows script was written but not executed anywhere"). If present, run
  `pwsh -NoProfile -Command "[scriptblock]::Create((Get-Content -Raw tools/run_gpu_tests.ps1)) | Out-Null"`
  to prove it parses.
- [ ] **Step 3: `docs/gpu-runs.md`.** One table: module, commit, date, GPU and driver, result, notes. One row
  for Foundation with result `compiled, not run`. A sentence above the table says the hardware run happens
  once, after all modules are written.
- [ ] **Step 4: Fill in `AGENTS.md` Commands**; run the gate; commit. Subject:
  `Add the one-command GPU run for Windows`.

### Task 9: CI workflow file

**Files:**
- Create: `.github/workflows/checks.yml`

**Interfaces:**
- Consumes: the presets, `tools/check_rules.py`, `CUDA_IMAGE`.
- Produces: three jobs. `rules` (ubuntu: format check and rules check). `cpu` (matrix: `ubuntu-24.04` with
  GCC 13, `macos-15` with Xcode 16.4, `windows-2022` with the VS 2022 toolset; `debug` and `release`): conan
  install, configure, build, test for `<os>-<config>-cpu`. `cuda-compile` (ubuntu, `container: ${CUDA_IMAGE}`):
  build `linux-release-cuda`, `ctest -L no_gpu`.

- [ ] **Step 1: Write it.** Actions pinned by commit SHA; take the SHAs for `actions/checkout`,
  `astral-sh/setup-uv` and `actions/cache` from `git show main:.github/workflows/native-cpu.yml`. Triggers:
  `push`, `pull_request`, `workflow_dispatch`. `permissions: contents: read`. None of `main`'s qualification
  or evidence-upload steps.
- [ ] **Step 2: Verify** with `uv run actionlint .github/workflows/checks.yml`. Expected: no output, exit 0.
- [ ] **Step 3: Commit.** Subject: `Describe the checks CI will run once the branch is pushed`. Body says the
  workflow has been linted but has never run.

### Task 10: Method paper, started again

Load the `typesetting-with-typst`, `typesetting-documents` and `drawing-with-cetz` skills before writing.
The owner asked for time to be spent here on good drawn figures and pseudocode: this task sets the style every
later module section reuses.

**Files:**
- Create: `docs/method-paper/fluidity-method.typ`, `docs/method-paper/references.typ`,
  `docs/method-paper/figure_style.typ`, `docs/method-paper/pseudocode.typ`,
  `docs/method-paper/figures/axes.typ`, `docs/method-paper/figures/one_operation_two_processors.typ`

**Interfaces:**
- Produces:
  - `figure_style.typ`: the shared CeTZ look, used by every later figure: one stroke width for geometry and a
    lighter one for guides, a small named palette (liquid, solid, air, particle, highlight, muted) that still
    reads in greyscale, one label font and size matched to the body text, arrow style, and helpers
    `axes-3d(origin)` (draws the right-handed X, Y, Z triad with Z up) and `cell-grid(...)` left for the grid
    module to add.
  - `pseudocode.typ`: one function `procedure(title, label, ..steps)` that renders numbered steps whose numbers
    match the `// Step N:` comments in the code, with inputs, outputs and a "fails when" line, in the decided
    vocabulary and with no paper symbols.
  - Labels `<sec-conventions>`, `<sec-notation>`, `<sec-run-on>`, `<fig-axes>`, `<fig-run-on>`,
    `<proc-check-can-run-on>` in `docs/method-paper/fluidity-method.typ`, which
  compiles with `typst compile docs/method-paper/fluidity-method.typ out/paper/fluidity-method.pdf`.

- [ ] **Step 1: `references.typ`.** Extract the reference list verbatim:
  `git show main:docs/papers/fluidity-method.typ | sed -n '/^= References/,$p'`. Keep every entry's text, DOI
  and URL character for character and keep the numbers. Verify by diffing the extracted lines against the
  new file with whitespace ignored: no difference inside any entry.
- [ ] **Step 2: `fluidity-method.typ`.** Title, one-paragraph statement of what the paper is (the method the
  code implements, written section by section as modules arrive), then:
  - `= Conventions <sec-conventions>`: SI units; right-handed axes with Z up and gravity along minus Z; the
    rotation from a Y-up description; time labels used in names (`before_forces`, `before_pressure`,
    `after_pressure`); what "same GPU, same input, identical output" promises and what it does not.
  - `= Notation <sec-notation>`: a two-column table, paper symbol to code word, containing only rows for
    symbols this skeleton already uses.
  - `= References` including `references.typ`.
  - `= One operation, two processors <sec-run-on>`: why each operation has one public function and a CPU and
    a GPU implementation that share their formulas; what is compared between them and how.
  Decided vocabulary throughout; a literature term appears once, in parentheses, where its plain word is
  introduced. No section stubs for modules that do not exist yet.
- [ ] **Step 3: Figures.**
  - `<fig-axes>` (in Conventions): two triads side by side, the proof-of-concept Y-up frame and this Z-up
    frame, gravity drawn in each, and the quarter-turn about X that carries one to the other, with a small
    inset showing that swapping two axes instead would mirror a triangle's winding.
  - `<fig-run-on>` (in One operation, two processors): one public function at the top, the shared formula
    header in the middle, the CPU loop and the GPU function below it, and the check that compares their
    results.
  Each figure is drawn with `figure_style.typ`, has a caption that states the one thing to see, and is
  referenced from the text. Compile, look at the rendered pages, and fix overlaps, clipped labels and
  inconsistent sizes before moving on; a figure nobody looked at is not done.
- [ ] **Step 4: Pseudocode.** `<proc-check-can-run-on>`: the procedure of the GPU check as numbered steps
  (find a GPU; build the fixed inputs; compute on the GPU; compute on the CPU; compare bit for bit; report),
  written before Task 7 implements it, so Task 7's `// Step N:` comments follow the paper.
- [ ] **Step 5: Run** `typst compile`. Expected: exit 0, no warning about a missing label or font. Read every
  page of the PDF. Run `uv run python tools/check_rules.py`. Expected: exit 0.
- [ ] **Step 6: Commit.** Subject: `Start the method paper again from its conventions`.

### Task 12: Scalar math and the remaining vector and matrix helpers

**Files:**
- Create: `native/fluidity/base/scalar_math.h`, `native/fluidity/base/scalar_math_test.cc`
- Modify: `native/fluidity/base/vec3.h`, `vec3_test.cc`, `mat3.h`, `mat3_test.cc`

**Interfaces:**
- Produces, all in `namespace fluidity::math`, all `FLUIDITY_CPU_AND_GPU noexcept`: `Floor`, `Abs`, `Sqrt`,
  `Cbrt`, `Exp`, `Expm1`, `Log`, `Pow`, `Sin`, `Cos`, `Min`, `Max`, `IsFinite(double)`; `Length(Vec3)`,
  `Normalized(Vec3)`, `IsFinite(Vec3)`; `IsFinite(const Mat3&)`, `Rotation(Vec3 rotation_vector) -> Mat3`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(Floor, NegativeHalf_IsMinusOne)                 // Floor(-0.5) == -1.0
  TEST(Abs, NegativeZero_IsPositiveZero)               // !std::signbit(Abs(-0.0))
  TEST(Sqrt, PerfectSquare_IsExact)                    // Sqrt(9.0) == 3.0
  TEST(Min, OneNaN_ReturnsTheOtherValue)               // documented choice, same on both processors
  TEST(IsFinite, InfinityAndNaN_AreNotFinite)
  TEST(Length, ThreeFourZero_IsFive)                   // Length({3,4,0}) == 5.0
  TEST(Normalized, AnyNonZeroVector_HasLengthOne)      // within 1e-15
  TEST(IsFinite, VectorWithOneNaN_IsNotFinite)
  TEST(Rotation, ZeroVector_IsIdentity)
  TEST(Rotation, QuarterTurnAboutZ_SendsXToY)          // within 1e-15; right-handed
  TEST(Rotation, AnyVector_HasDeterminantOne)          // within 1e-14
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** `Floor`, `Abs`, `Sqrt`, `Min`, `Max`, `IsFinite` must give the same bits on CPU and
  GPU; say so in each docstring. `Cbrt`, `Exp`, `Expm1`, `Log`, `Pow`, `Sin`, `Cos` call the platform's function on each
  side and may differ in the last bits between CPU and GPU; the file header says this in plain words and tells
  the reader that any formula using them is compared with a tolerance. `Normalized` requires a positive
  length (documented precondition, not checked). `Rotation` uses Rodrigues' formula; cite it in `Read:`.
- [ ] **Step 4: Run** build, tests, format, rules. Extend `CheckGpuMath` inputs in Task 7 to cover `Sqrt`,
  `Floor` and `Length` in its bit-for-bit comparison (not the last-bits group).
- [ ] **Step 5: Commit.** Subject: `Add scalar math and vector length for both processors`.

### Task 13: `Buffer`, an array that lives where the simulation runs

**Files:**
- Create: `native/fluidity/base/buffer.h`, `buffer_cpu.cc`, `buffer_gpu.cu`, `buffer_gpu_calls.h`,
  `buffer_test.cc`, `buffer_gpu_test.cc`, `buffer_no_gpu_test.cc`

**Interfaces:**
- Consumes: `RunOn`, `CheckCanRunOn`, the CUDA wiring of Task 7.
- Produces: `BufferError`, `ToMessage(BufferError)`, `ToName(BufferError)`, `template <typename Value> class
  Buffer`, `BuildBuffer`, `WriteBuffer`, `ReadBuffer`, exactly as in `shared-interface.md`, plus the additions
  of `docs/superpowers/specs/modules/changes-to-apply.md` ("Additions to Foundation `base/`"):
  `CopyBuffer(from, to)` on the processor both live on; forms of `CopyBuffer`, `WriteBuffer` and `ReadBuffer`
  that take `(first, count)`; `enum class CpuMemory { kOrdinary, kLockedForCopies }` as an optional last
  argument of `BuildBuffer` (CPU buffers that are copied to the GPU every step; ignored for GPU buffers);
  `struct MemoryAvailable { std::uint64_t total_bytes; std::uint64_t free_bytes; };` (a total of 0 means
  unknown) and `std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on)`. `buffer_gpu_calls.h` is private: three
  untyped functions (allocate bytes, free bytes, copy bytes in either direction) that `buffer_gpu.cu` defines
  and the templates in `buffer.h` call, so `buffer.h` never includes a CUDA header.

- [ ] **Step 1: Failing tests.**

  ```cpp
  // buffer_test.cc, label cpu
  TEST(BuildBuffer, OnCpu_HasTheAskedSizeAndProcessor)
  TEST(BuildBuffer, SizeZero_Succeeds)                       // size() == 0, data() may be null
  TEST(WriteBuffer, ThenReadBuffer_ReturnsTheSameValues)     // 1000 doubles round trip, bit for bit
  TEST(WriteBuffer, SpanOfDifferentSize_IsSizeMismatch)
  TEST(ReadBuffer, SpanOfDifferentSize_IsSizeMismatch)
  TEST(Buffer, MovedFrom_HasSizeZero)                        // and destroying both frees once
  TEST(CopyBuffer, SameProcessor_CopiesEveryValue)
  TEST(CopyBuffer, DifferentSizes_IsSizeMismatch)
  TEST(CopyBuffer, SliceBeyondTheEnd_IsSizeMismatch)
  TEST(WriteBuffer, Slice_LeavesValuesOutsideTheSliceUntouched)
  TEST(BuildBuffer, LockedForCopiesOnCpu_BehavesLikeAnOrdinaryBuffer)
  TEST(ReadMemoryAvailable, OnCpu_FreeIsPositiveAndNotAboveTotal)
  TEST(BuildBuffer, ImpossiblySize_IsOutOfMemory)            // size_t max / sizeof(double): no crash, no throw
  // buffer_no_gpu_test.cc, label no_gpu
  TEST(BuildBuffer, OnGpuWithoutAGpu_IsRunOnCallFailed)
  TEST(ReadMemoryAvailable, OnGpuWithoutAGpu_IsRunOnCallFailed)
  // buffer_gpu_test.cc, label gpu
  TEST(WriteBuffer, OnGpuThenReadBuffer_ReturnsTheSameValues)
  TEST(BuildBuffer, OnGpuManyTimes_FreesEverything)          // 1000 build-and-destroy cycles, free memory unchanged
  ```
- [ ] **Step 2: Run.** Expected: compile error.
- [ ] **Step 3: Implement.** CPU storage is aligned to 64 bytes and zero-filled; GPU storage is zero-filled.
  A size whose byte count overflows `size_t` is `kOutOfMemory` before any allocation is tried. The class is
  move-only; copying is a compile error. No growth, no resize: a new size is a new buffer.
- [ ] **Step 4: Run** the gate. Expected: all steps pass; the `no_gpu` test passes in the container.
- [ ] **Step 5: Commit.** Subject: `Add Buffer, the array that lives on the chosen processor`.

### Task 14: Sums in one fixed order

**Files:**
- Create: `native/fluidity/base/tree_sum.h`, `tree_sum_test.cc`, `ordered_sums.h`, `ordered_sums_cpu.cc`,
  `ordered_sums_gpu.cu`, `ordered_sums_test.cc`, `ordered_sums_gpu_test.cc`
- Modify: `docs/method-paper/fluidity-method.typ` (new section and figure)

**Interfaces:**
- Consumes: `Buffer`.
- Produces: `math::TreeSum(const double* values, std::size_t count)`; `ComputeTreeSum`, `ComputeTreeDot`,
  `ComputeLargestMagnitude` over buffers; paper labels `<sec-fixed-order-sums>`, `<fig-tree-sum>`,
  `<proc-tree-sum>`.

- [ ] **Step 1: Paper first.** Section "Sums in a fixed order": why adding doubles in a different order gives
  different bits; the tree (pad with zeros to a power of two, each node is left plus right); a drawn figure of
  the tree over seven values showing the padding; the pseudocode; a worked example with numbers where the
  left-to-right sum and the tree sum differ in the last bit.
- [ ] **Step 2: Failing tests.**

  ```cpp
  TEST(TreeSum, Empty_IsZero)
  TEST(TreeSum, OneValue_IsThatValue)
  TEST(TreeSum, FourValues_AddsPairsThenPairs)          // {1e16, 1, -1e16, 1}: each pair rounds to +-1e16, so the
                                                        // tree gives exactly 0.0, while left to right gives 1.0.
                                                        // Assert 0.0, and assert the plain loop gives 1.0.
  TEST(TreeSum, SevenValues_EqualsEightWithATrailingZero)
  TEST(ComputeTreeSum, OnCpu_EqualsTreeSumBitForBit)    // 1000 values from a fixed integer formula
  TEST(ComputeTreeSum, CountSmallerThanBuffer_SumsOnlyTheFirstCount)
  TEST(ComputeTreeSum, CountLargerThanBuffer_IsSizeMismatch)
  TEST(ComputeTreeDot, DifferentSizes_IsSizeMismatch)
  TEST(ComputeTreeDot, OnCpu_EqualsTreeSumOfProducts)
  TEST(ComputeLargestMagnitude, MixedSigns_ReturnsLargestAbsoluteValue)
  TEST(ComputeLargestMagnitude, ContainsNaN_ReturnsNaN) // a NaN must never be hidden by a comparison
  // gpu label
  TEST(ComputeTreeSum, OnGpu_EqualsCpuBitForBit)        // same 1000 values and one of 1,000,003 values
  TEST(ComputeTreeSum, OnGpuTwice_GivesIdenticalBytes)
  ```
- [ ] **Step 3: Implement.** The CPU version may use threads only by splitting at tree nodes, so the result
  never depends on the thread count. The GPU version reduces level by level over the same tree; no
  floating-point atomics.
- [ ] **Step 4: Run** the gate; **commit.** Subject: `Add sums that give the same bits on every run`.

### Task 15: Stable ordering and grouping

**Files:**
- Create: `native/fluidity/base/stable_order.h`, `stable_order_cpu.cc`, `stable_order_gpu.cu`,
  `stable_order_test.cc`, `stable_order_gpu_test.cc`

**Interfaces:**
- Consumes: `Buffer`.
- Produces: `BuildStableOrder`, `BuildGroupsByDestination`, exactly as in `shared-interface.md`.
- **Amended 2026-11-14 (A17, grid built on the GPU):** also produces `OrderBuffers`, `BuildOrderBuffers`,
  `OrderBuffersSize`, `ComputeOrderBuffersSize`, `BuildStableOrder` taking its `OrderBuffers` (allocates nothing) and
  `ComputeExclusivePrefixSum`, with the exact declarations and comments of `plan-decisions.md`, section
  "Grid built on the GPU (A17)", item "Needed from Foundation".
- **Amended 2026-11-14 (A30, A31):** one form only, the caller owns the `OrderBuffers`; buffers on another processor
  are a `Requires:` precondition checked by a debug assertion (no `kRunOnMismatch`). Removed tests:
  `WithScratch_EqualsFormWithoutScratch` (replaced: one form, A31), `ScratchOnOtherProcessor_IsRunOnMismatch`
  (replaced: precondition, A30); `CountAboveScratchMaxCount_IsSizeMismatch` is now `CountAboveBuffersMaxCount_IsSizeMismatch`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(BuildOrderBuffers, MaxCountZero_Succeeds)
  TEST(ComputeOrderBuffersSize, EqualsBytesBuildOrderBuffersAllocates)   // cpu label and gpu label
  TEST(BuildStableOrder, CountAboveBuffersMaxCount_IsSizeMismatch)
  TEST(ComputeExclusivePrefixSum, FiveValues_GivesRunningSumsAndTotal)   // {3,0,2,5,1} -> {0,3,3,5,10,11}
  TEST(ComputeExclusivePrefixSum, InPlace_EqualsSeparateBuffers)
  TEST(ComputeExclusivePrefixSum, CountZero_WritesZeroTotal)
  TEST(ComputeExclusivePrefixSum, TotalAboveUint32Max_IsSizeMismatch)
  TEST(ComputeExclusivePrefixSum, SumsBufferTooSmall_IsSizeMismatch)
  // gpu label
  TEST(ComputeExclusivePrefixSum, OnGpu_EqualsCpuForAMillionValues)
  TEST(BuildStableOrder, DistinctKeys_OrdersByAscendingKey)
  TEST(BuildStableOrder, EqualKeys_KeepIndexOrder)
  TEST(BuildStableOrder, EqualKeysWithPreviousOrder_KeepPreviousOrder)
  TEST(BuildStableOrder, TwoPasses_SortByMajorThenMinorKey)
  TEST(BuildStableOrder, CountZero_Succeeds)
  TEST(BuildStableOrder, OrderBufferTooSmall_IsSizeMismatch)
  TEST(BuildGroupsByDestination, ThreeDestinations_ListsEachGroupInAscendingItemIndex)
  TEST(BuildGroupsByDestination, EmptyDestination_HasEqualNeighbouringStarts)
  TEST(BuildGroupsByDestination, DestinationOutOfRange_IsSizeMismatch)
  // gpu label
  TEST(BuildStableOrder, OnGpu_EqualsCpuForAMillionKeysWithManyTies)
  TEST(BuildGroupsByDestination, OnGpuTwice_GivesIdenticalBytes)
  ```
- [ ] **Step 2: Implement.** CPU: a counting or radix sort (stable by construction). GPU: a radix sort written
  here or the toolkit's sort **only** if its documentation for the pinned toolkit version states it is stable
  and deterministic; record which was chosen and the documentation line in the file header. Never an
  unstable sort followed by a fix-up.
- [ ] **Step 3: Run** the gate; **commit.** Subject: `Add ordering and grouping that never depend on timing`.

### Task 16: `PerParticleVelocity` and the memory count

**Files:**
- Create: `native/fluidity/base/per_particle_velocity.h`, `native/fluidity/base/processor_description.h`,
  `processor_description_cpu.cc`, `processor_description_gpu.cu`, `native/fluidity/base/memory_count.h`,
  `memory_count.cc`, `memory_count_test.cc`

**Interfaces:**
- Produces: `struct ProcessorDescription { std::string model; std::string driver; std::string runtime; }` and
  `ProcessorDescription ReadProcessorDescription(RunOn run_on)` in `base/processor_description.h` (CPU only;
  empty strings when the processor cannot be asked; test: on the CPU the model is not empty);
  `template <typename T> struct PerParticleVelocity { T for_next_step; T for_this_step; }` (owner
  answer A1 in `docs/superpowers/specs/modules/owner-answers.md`: the first is kept on the particle for the
  next step, the second moves the particle in this step);
  `CountedPer`, `KeptFor`, `MemoryWhere { kRunOn, kCpu }`, `MemoryEstimateEntry` (with a `where` field), `MemoryCounts`,
  `ComputeMemoryBytes(rows, counts)` and `ComputeMemoryBytes(rows, counts, where)`.

- [ ] **Step 1: Failing tests.**

  ```cpp
  TEST(ComputeMemoryBytes, NoRows_IsZero)
  TEST(ComputeMemoryBytes, RowPerParticleWithTwoCopies_MultipliesAll)  // 1000 particles x 24 bytes x 2 == 48000
  TEST(ComputeMemoryBytes, GivenCountRow_IgnoresSceneCounts)
  TEST(ComputeMemoryBytes, EveryCountedPerValue_UsesItsOwnCount)
  TEST(ComputeMemoryBytes, WithWhere_CountsOnlyRowsKeptThere)         // one kRunOn row and one kCpu row
  TEST(ComputeMemoryBytes, HugeCounts_SaturateInsteadOfWrapping)       // result == uint64 max, never a small number
  ```
- [ ] **Step 2: Implement; run the gate; commit.** Subject: `Add the memory count behind the sizing estimate`.

### Task 11: Comment pass and hand-over

**Files:**
- Modify: any file under `native/`, `tools/`, `AGENTS.md`, `docs/` that the pass changes

- [ ] **Step 1: Comment pass.** Reread every file on the branch from top to bottom as a C++ programmer who has
  never read a fluid paper. For each: is every comment true of the final code; does it say why and not what;
  is there a worked example with numbers where a contract is hard to picture; does each word appear in the
  glossary; are the `Words`, `Uses` and `Used by` lists correct now that all files exist.
- [ ] **Step 2: Fold each change into the commit that owns the file** (`git commit --fixup <hash>` per change,
  then `git rebase --autosquash --root`). Run the gate on the final tip and on each rewritten commit
  (`git rebase --root --exec "uv run python tools/check_commit.py --only format" --exec "uv run python tools/check_commit.py --only rules"`;
  the full gate on the tip).
  Expected: every commit passes format and rules; the tip passes everything.
- [ ] **Step 3: Check the spec's section 10 list** item by item against the branch and record, in the
  hand-over message, anything delivered differently and why.
- [ ] **Step 4: Hand over to the owner**, in one message: `git log --oneline` of `rebuild`; the gate's output;
  the list of all new public names with their one-line reasons; the Windows command that will be used at the end of the programme; and the statement that no device code
  has run yet.
Foundation is done at this point with its GPU status recorded as `compiled, not run`. The hardware run is
part of the end of the programme, not of this plan.

## Additions after Foundation closed

**Amended 2026-11-15 (A58):** three tasks added after Foundation's hand-over, from owner answer A58 (one reserved
memory block per run). Run them in the order 17, 18, 19, before module 1 (contact) starts: every module's building
function takes the run's `RunMemory`. The exact declarations and rules are in
`docs/superpowers/specs/modules/plan-decisions.md`, section "Owner answers A40–A64 applied (2026-11-15)", item 3;
`shared-interface.md` section (a) is re-copied from the headers after Task 19. Global Constraints above apply.

### Task 17: `RunMemory`, one reserved block that hands out buffers

**Files:**
- Create: `native/fluidity/base/run_memory.h`, `run_memory_cpu.cc`, `run_memory_test.cc`, `run_memory_gpu_test.cc`,
  `run_memory_no_gpu_test.cc`
- Modify: `native/fluidity/base/buffer.h` (a `BuildBuffer` form taking a `RunMemory&`; `Buffer` frees nothing when
  its bytes are a piece; `BufferError::kRunMemoryTooSmall` appended last; `ToMessage` and `ToName` for it),
  `buffer_bytes.h` (no new function: the block is built with `BuildBufferBytes` and freed with `FreeBufferBytes`),
  `memory_count.h` / `memory_count.cc` (`ComputeRunMemoryPieceBytes`, `ComputeRunMemoryBytes`),
  `docs/method-paper/fluidity-method.typ` (section `<sec-run-memory>`, figure `<fig-run-memory>`), `docs/glossary.md`
  (rows `RunMemory`, `piece`, `reserved bytes`, `handed out`), `native/fluidity/base/CMakeLists.txt`

**Interfaces:**
- Consumes: `Buffer`, `BuildBufferBytes`, `FreeBufferBytes`, `CpuMemory`, `MemoryEstimateEntry`, `MemoryCounts`.
- Produces: `class RunMemory` (move-only; `run_on()`, `reserved_bytes()`, `handed_out_bytes()`),
  `std::expected<RunMemory, BufferError> BuildRunMemory(RunOn run_on, std::size_t byte_count, CpuMemory cpu_memory =
  CpuMemory::kOrdinary)`, `template <BufferValue Value> std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn
  run_on, std::size_t size, RunMemory& run_memory)`, `inline constexpr std::size_t kRunMemoryPieceAlignmentBytes =
  256`, `std::size_t ComputeRunMemoryPieceBytes(std::size_t byte_count)`, `std::uint64_t
  ComputeRunMemoryBytes(std::span<const MemoryEstimateEntry> estimate, const MemoryCounts& counts,
  MemoryWhere where)` (the entries with that `where`; a GPU run reserves one block per processor),
  `BufferError::kRunMemoryTooSmall`.

**Kind:** mechanical

- [ ] **Step 1: Paper first.** Section "One block of memory per run": why one reservation (no fragmentation, one
  synchronizing GPU call, a footprint equal to the plan); the two rules (a piece is never freed alone; the block
  outlives its pieces); the figure: one block, pieces at multiples of 256 bytes, the next free offset; the worked
  example below.
- [ ] **Step 2: Failing tests.**

  ```cpp
  // run_memory_test.cc, label cpu
  TEST(ComputeRunMemoryPieceBytes, RoundsUpTo256)            // 0 -> 0, 1 -> 256, 256 -> 256, 257 -> 512
  TEST(ComputeRunMemoryBytes, CountsOnlyEntriesOfTheWhereRoundedPerCopy)  // 1000 particles x 24 B x 2 copies
                                                               // (kRunOn) -> 2 x 24064 = 48128; a kCpu entry adds 0
  TEST(ComputeRunMemoryBytes, HugeCounts_Saturate)
  TEST(BuildRunMemory, OnCpu_ReservesTheAskedBytes)          // reserved_bytes() == 4096, handed_out_bytes() == 0
  TEST(BuildBuffer, FromRunMemory_HandsOutAlignedZeroedPieces)  // three pieces of 10, 100, 3 doubles: data() a
                                                               // multiple of 256 apart from the block start, zeros,
                                                               // handed_out_bytes() == 256 + 1024 + 256
  TEST(BuildBuffer, FromRunMemoryPieceDoesNotFit_IsRunMemoryTooSmall)  // takes nothing: handed_out_bytes() unchanged
  TEST(BuildBuffer, FromRunMemorySizeZero_TakesNothing)
  TEST(Buffer, PieceDestroyed_FreesNothingAndBlockStaysUsable)
  TEST(RunMemory, Moved_PiecesStayValid)                     // move the RunMemory, write and read a piece
  TEST(ToName, RunMemoryTooSmall_IsRunMemoryTooSmall)        // "run_memory_too_small"
  // death test, debug builds only (label cpu): destroying a RunMemory while a piece lives asserts
  TEST(RunMemoryDeathTest, DestroyedBeforeItsPieces_Asserts)
  // run_memory_no_gpu_test.cc, label no_gpu
  TEST(BuildRunMemory, OnGpuWithoutAGpu_IsRunOnCallFailed)
  // run_memory_gpu_test.cc, label gpu
  TEST(BuildBuffer, FromGpuRunMemory_WriteThenReadReturnsTheSameValues)
  TEST(BuildRunMemory, OnGpu_FreeMemoryDropsByAtLeastTheReservedBytes)
  ```
- [ ] **Step 3: Implement.** `RunMemory` owns one `BufferBytes` block and a small heap cell holding the next free
  offset and, in every build, the count of live pieces (so a move keeps the pieces' pointer valid). A piece takes
  `ComputeRunMemoryPieceBytes(size * sizeof(Value))` bytes at the next free offset; the block's own start is a
  multiple of 256 on the GPU (CUDA) and is asked aligned to 256 on the CPU. `Buffer` gains one private pointer to
  that cell (null when the buffer owns its bytes); its destructor and move assignment free bytes only when it owns
  them and otherwise decrement the count. `~RunMemory` asserts the count is zero (`Requires:` in its comment, A30),
  then frees the block. Pieces are zero bytes: the block is zero-filled when built, and a piece is never reused.
  A piece that does not fit returns `kRunMemoryTooSmall`, writes nothing and also asserts in debug builds (it is
  a bug of the plan: the step reserves exactly `ComputeRunMemoryBytes` of the estimate).
- [ ] **Step 4: Run** the gate. **Commit.** Subject: `Add RunMemory, one reserved block that hands out buffers`.
  New public names in the message; GPU tests compiled, not run.

### Task 18: Working buffers built from a `RunMemory`

**Files:**
- Modify: `native/fluidity/base/ordered_sums.h`, `ordered_sums_cpu.cc`, `ordered_sums_test.cc`, `stable_order.h`,
  `stable_order_cpu.cc`, `stable_order_test.cc`

**Interfaces:**
- Consumes: Task 17.
- Produces: `std::expected<SumBuffers, BufferError> BuildSumBuffers(RunOn run_on, std::size_t max_count, RunMemory&
  run_memory)` and `std::expected<OrderBuffers, BufferError> BuildOrderBuffers(RunOn run_on, std::size_t max_count,
  RunMemory& run_memory)`; the plain forms stay (tests and tools). Rule every module copies: the last parameter is
  `RunMemory& run_memory`, `run_on == run_memory.run_on()` is a `Requires:` checked by assert.

**Kind:** mechanical

- [ ] **Step 1: Failing tests.** `BuildSumBuffers, IntoRunMemory_TakesOnePieceOfTheSize` (handed-out bytes ==
  `ComputeRunMemoryPieceBytes(ComputeSumBuffersSize(...))` on the CPU, label cpu; GPU, label gpu);
  `BuildOrderBuffers, IntoRunMemory_TakesOnePieceOfTheSize` (same); `BuildOrderBuffers, IntoRunMemory_SameResultsAsPlainForm`
  (a million keys ordered with either buffers give identical bytes).
- [ ] **Step 2: Implement** (each working-buffer object holds its memory in one `Buffer<std::byte>`, so one piece);
  run the gate; **commit.** Subject: `Build working buffers from a RunMemory`.

### Task 19: Free GPU memory read after the context exists

**Files:**
- Modify: `native/fluidity/base/processor_description_gpu.cu`, `processor_description.h` (comment),
  `processor_description_gpu_test.cc`

**Interfaces:**
- Produces: `ReadMemoryAvailable(RunOn::kGpu)` first makes sure the GPU context exists (`cudaFree(0)`, status through
  `ToRunOnCallStatus`), then calls `cudaMemGetInfo`, so the context and the loaded kernels are already outside
  "free" (literature-practice.md topic 2). Its comment says so and that the step computes the allowed bytes from it.

**Kind:** device

- [ ] **Step 1: Failing test** (label gpu): `ReadMemoryAvailable, OnGpuTwice_SecondFreeNotAboveFirst` (the context
  exists before the first read, so a second read does not show the context's bytes leaving "free").
- [ ] **Step 2: Implement; run the gate; commit.** Subject: `Read free GPU memory after the context exists`.
  Compiled, not run.
