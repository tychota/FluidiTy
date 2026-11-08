# FluidiTy coding rules

## Native C++ and CUDA

- Use C++20 and Google C++ style with the repository's clang-format settings. Indent blocks with two spaces and
  continuations with four. Keep lines at or below 120 columns and use braces for every control-flow body.
- Give each step its own line when a dense expression obscures validation, data flow, or a numerical invariant.
  Use named predicates or separate early returns for distinct validation concerns. Avoid double negatives.
  For a multiline aggregate initializer, put each member on its own indented line, add a trailing comma, and put
  the closing brace on its own line at the starting indentation. Prefer designated fields when they clarify the layout.
- Use consistent, searchable names that say what a value represents in the domain. Distinguish an index, coordinate,
  extent, and result. Keep standard coordinate and ABI names such as `Vec2::x` and `Grid2D::nx`.
- Name numerical intermediates so readers can distinguish a per-particle contribution from a face sum and a physical
  quantity from its scaled value or quantized integer. Public parameters should identify the quantity and, when relevant,
  its coordinate frame or time level without requiring the function body. Keep the quantity name consistent across stages, as in
  `stress_force_contribution`, `scaled_stress_force_contribution`, `quantized_stress_force_contribution`, and `stress_force_quanta_sum`.
- Prefer clear C++20 facilities where they express intent directly. Use `constexpr` for compile-time values, `std::array`
  for fixed-size host data, `std::span` for non-owning host views, and concepts when a template needs a type constraint.
  Ordinary loops are appropriate for numerical kernels whose iteration order matters.
- Use `#pragma once` and `.h` for ordinary headers. Headers under `native/include/fluidity/base/` can be shared by host
  and device when their contents are device-compatible. The `host/` directory is host-only. Use `.cuh` for headers
  that require nvcc, `.cu` for CUDA sources, and `.cc` for host sources.
- Return `absl::Status` or `absl::StatusOr` from checked host APIs. Keep Abseil and host containers out of device code.
  Keep production code free of exceptions, RTTI, modules, coroutines, and `using namespace` directives.
- Preserve documented numerical ordering, rounding, signed indices, and ABI layout. Validate extents and aliasing
  before access, and publish numerical outputs only on success.

## Function comments

- Document public functions and nonobvious data fields in the `.h` or `.cuh` file. State what the function computes or changes.
  Cover relevant units, ordering, bounds, preconditions, aliasing rules, failure statuses, and outputs on failure.
- In `.cc` or `.cu` implementations, give private helpers a short prologue when their assumptions need context. Use algorithm
  comments to explain invariants and precision choices. Leave straightforward operations to the code.

[Naming Things Principles](https://www.namingthings.co/naming-things-principles) · [Writing good comments](https://antirez.com/news/124)

## Formatting check

Use clang-format 19.1 (verified with 19.1.5). From the repository root in PowerShell, check the tracked
hand-written native sources.

```powershell
$nativeSources = git ls-files -- 'native/*.cc' 'native/*.h' 'native/*.cu' 'native/*.cuh'
clang-format --dry-run --Werror @nativeSources
```
