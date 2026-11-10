# M2 Native MotionInterval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal** — Give Mesh and Whitewater a checked host-side time sampler and bounded point transport over one accepted output interval.

**Architecture** — Own the immutable accepted timestamps and epochs in one checked value. Borrow a spatial sampler for the duration of each query. This leaves sparse grid ownership and versioned cache artifacts to the integration step.

**Tech Stack** — C++20, Abseil status and function references, GoogleTest, CMake.

**Spec** — [FluidiTy design](../specs/2026-11-07-fluidity-design.md) §§3, 9.1, 10 and 11. [M1c handoff](../../references/m1c-post-blender-audit.md) and [measured temporal controls](../../references/m2-fixed-cache-evidence.md#separate-motioninterval-study).

## Global Constraints

- Interpret positions and velocities in world metres and metres per second, with physical seconds for all timestamps.
- Use only accepted samples. Never extrapolate or treat missing carrier as zero. Require both bracketing spatial samples for interpolation and only one at an exact timestamp.
- Use RK2 midpoint integration, split at timestamp knots, and bound each accepted displacement by 0.5h. The caller owns the substep budget.
- Keep this slice host-only. Cache writer, occupancy, spatial sampling policy and solver configuration are separate parts of Task 4.

## Review Focus

- A duplicate, out-of-order or nonfinite stamp cannot become an interval.
- Exact endpoint access must succeed even when an unused adjacent sample has invalid spatial support.
- A query outside the covered time cannot invoke the spatial sampler.
- Rapidly increasing speed must shrink the midpoint step to preserve the displacement bound.
- A missing midpoint or exhausted budget must return no transported position.

## Task 1 — Checked temporal carrier and midpoint transport

**Files** — Create `native/include/fluidity/host/motion_interval_3d.h`, `native/src/motion_interval_3d.cc`
and `native/tests/test_motion_interval_3d.cc`. Register both translation units in `CMakeLists.txt`.

**Interfaces** — `MotionInterval3D::Create(LiquidClock3D first_frame, LiquidClock3D last_frame, std::span<const MotionStamp3D>)`
owns the exact samples for adjacent completed frames. Each `MotionStamp3D` contains `time_seconds` and
`accepted_epoch`. `MotionSpatialSampler3D` borrows a callable from sample index and world position to
`absl::StatusOr<Vec3>`. `SampleMotionCarrier3D(interval, position, time_seconds, sampler)` samples
and linearly interpolates the carrier. `AdvectMotionPoint3D(interval, position, from_time_seconds,
to_time_seconds, cell_width_m, maximum_substeps, sampler)` returns the transported world position,
accepted substep count and measured maximum displacement.

- [x] Add RED fixtures for invalid clocks and samples, exact endpoints and two-sample interpolation, missing and nonfinite carrier, a linear-in-time translation split at a sample knot, midpoint step reduction, and a zero-budget or exhausted call. A zero-duration call returns the starting position without sampling.
- [x] Build `fluidity_native_tests` and observe that it fails because the transport header is missing.
- [x] Implement validation, temporal sampling and bounded RK2. Own the supplied stamps. Use `std::lower_bound` on time and `std::lerp` for component interpolation. Before accepting a trial, require a finite midpoint, valid carrier, forward time progress and displacement ≤0.5h. Bound each retry loop to 32 halvings. Return `InvalidArgument`, `OutOfRange` or `ResourceExhausted` for the respective contract errors and propagate spatial status.
- [x] Run `ctest --test-dir out/build/m2-release-cpu -R '^MotionInterval3D\.' --no-tests=error --output-on-failure` and the full CPU CTest suite. Require the named fixtures to execute and pass.
- [x] Check all tracked hand-written native sources with clang-format 19.1.5. Commit the checked host contract and tests.
