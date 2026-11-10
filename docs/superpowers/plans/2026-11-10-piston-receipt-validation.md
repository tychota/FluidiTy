# Piston Receipt Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Reject malformed temporal and step geometry records before their values can qualify the frozen piston.

**Architecture:** Validate both temporal summary inputs before scoring. Assemble each geometry sample from verified trace metadata, the seed or preceding accepted geometry, and the four independent piston measurement fields.

**Tech Stack:** Python 3.13.18, uv 0.13.7, pytest.

**Spec:** docs/superpowers/specs/2026-11-07-fluidity-design.md §2.3 and docs/superpowers/plans/2026-11-09-real-3d-liquid-acceptance-and-throughput.md, Scene authority and Task 4.

## Global Constraints

- Exact base f7ad33a6b636e05a266430ebece6600e6dc5460c. The frozen tools depend on native trace fields absent from merged main. Work on the separate follow-on branch.
- Preserve physical limits of 1% and the temporal/spatial limit of 0.25%. The denominator is max(abs(independent wall flux), 1e-8 m³). Require the original 0.01/0.005 s temporal pair.
- Keep legitimate boolean False as a failed verdict and require boolean output. No GPU access or launches, changes to frozen physics, existing draft branch edits, remote operations, or acceptance claims from archived captures.
- Native executable, fixture and physical device binding is a separate follow-up before capture verdicts qualify.

## Review Focus

- Bad primary or refined types, huge numeric inputs and overflow during the comparison must fail cleanly.
- Negative signed wall flux still uses its magnitude. Near zero uses the frozen denominator floor.
- An otherwise valid measurement carrying a colliding field may not overwrite verified step geometry or clock.
- A subsequent sample must use the previous accepted end positions even if the current measured data carries forged positions.

### Task 1: Validate temporal summaries and protect step inputs

**Files**
- Modify tools/native/liquid_acceptance_metrics.py and tools/native/tests/run_liquid_acceptance.py.
- Extend tools/native/tests/test_liquid_acceptance_metrics.py with tests of the real comparator and prepare_tank_steps.

**Interfaces**
- Preserve compare_piston_time_refinement(primary, refined) -> dict and prepare_tank_steps(traces, frames, initial_state, particle_count, *, piston) -> list.
- Malformed present values raise ValueError. A valid negative physical verdict returns passed=False. Existing callers continue to fail closed when required fields are missing.

- [x] Write tests for non-booleans in each verdict (including "false", 1 and an object), denominators -0.008, 0, True, 1e6, nonfinite and huge integer, each in both receipts, and nonfinite, boolean, huge and overflowing volume arithmetic. Include valid negative wall flux, near-zero floor, distinct denominators, a true physical failure, and an unchanged valid case. Test two successive traces with a forged accepted_step, time_seconds and start_positions inside accepted_measurement, where the verified seed supplies the first start and verified accepted end supplies the second. Assert those records either reject the collision or derive all three protected values from their authoritative inputs. Include an ordinary two-trace positive control.
- [x] Run the focused test module and record the failing assertions before production edits.
- [x] Validate each temporal input boolean, all four numeric volume fields, the positive canonical denominator and representable relative changes. Use the canonical primary denominator. Assemble step geometry with an explicit allowlist of four measurement fields, without forwarding arbitrary metadata. Preserve the verified trace ordinal/time and seed/preceding geometry.
- [x] Rerun the focused module and locked Python suite. Run git diff --check. Report counts, exits, any skipped native integration tests, and separate RED/GREEN log paths. Do not claim native or GPU runs.
- [x] Commit the scoped code and tests, then hand the exact range to the independent reviewer.
