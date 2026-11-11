# M3 Iterative Pressure Parity Implementation Plan

**Goal** — Exercise the measured edge and corner graphs through the bounded CPU solver and the CUDA pressure solver.

**Baseline** — Native explicit-subface pressure PR #55 at `f7dca3b25a27315ce2fd2342447c05714cefa9d2`. Publish this follow-up only after #55 merges.

- [x] Share the four native graph fixtures between operator, CPU and CUDA tests. Assert the independent pressure, production divergence and residual limits, signed flux, and unchanged previously published CPU results on rejection.
- [x] Run tests against error-only entry points and observe rejection where a valid graph must solve.
- [x] Reuse the bounded CPU pressure hierarchy, PCG and result-bank publication with a read-only prepared graph. Reuse the CUDA operator upload and solver with the same prepared graph. Keep uniform assembly order and public entry points.
- [x] Run the CPU and CUDA pressure suites. Record the executed SM and numerical bounds. Run native formatting checks.
- [ ] Open the next PR and track platform checks. This slice does not exercise production adaptive topology discovery, transfers or remap.

## Results

The CPU and CUDA paths solved all four signed edge and corner cases in four or five iterations. Both peaked at 8.83e-7 s⁻¹ corrected divergence against the 2e-5 s⁻¹ production pressure limit. The largest integrated face-flux difference from the independent pressure solution was 3.0e-8 m³/s. The CPU cases also confirmed byte-identical pressure and velocity replay and preserved the earlier publication after a one-iteration rejection.

The CPU and CUDA parity tests first failed against error-only entry points. Review exposed that a uniform prepared system could enter the explicit-only CPU path and lose its wall-flux measurement. A failing regression established that behavior. The corrected API rejects uniform inputs and keeps their original entry point.

Clean MSVC Release builds passed 506 native CPU tests and 42 CUDA operator tests. The device receipt confirmed SM120 on the RTX 5060 Ti. A Windows incremental build had omitted Ninja header dependencies for French MSVC include output. The clean CUDA rebuild passed the boundary case that had terminated and then passed the full operator suite. The tracked native sources passed clang-format 19.1.5.

The local numerical receipts used the PR #55 head. This change was rebased onto the merge commit, which also includes the public CLI changes from PR #54. The native solver and test sources were identical across the two bases.
