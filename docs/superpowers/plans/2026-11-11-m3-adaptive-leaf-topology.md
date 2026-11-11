# M3 Balanced Leaf Topology Implementation Plan

**Goal** — Build the canonical interior MAC subfaces from a bounded set of signed 2:1 leaves and solve a manufactured pressure field on the resulting edge and corner graphs.

**Baseline** — Iterative pressure PR #56 head `9171d9ad3f7424b6a8f9aa7b46fcc4fca63a5e5e`.

- [x] Define signed finest-lattice lower corners and a width exponent. Verify coarse/fine face areas, distances, unique owners and deterministic ordering at translated edges and corners.
- [x] Reject dyadic misalignment, overlap, duplicate lower corners, signed overflow and level gaps larger than one at touching faces, edges or corners.
- [x] Run the pressure fixture and rejection tests against an error-only builder, then implement a checked neighbor search.
- [x] Build explicit pressure inputs from emitted subfaces and compare a bounded CPU solve against a gauge-fixed manufactured field.
- [x] Run the native formatting and focused CPU checks before opening the PR. Exterior apertures, sparse block grouping and adaptive substep integration remain separate.

## Evidence

Both translated junction orientations produced 12 fine interior faces and four uniquely owned 0.25 m² subfaces per incident coarse plane. Coarse/fine normal center spacing was 0.75 m. A separate pairwise-box oracle matched all face keys and endpoints on a sparse two-level layout with holes. The emitted corner graph solved a manufactured gauge-fixed pressure field within 2e-5 Pa and the production 2e-5 s⁻¹ local divergence limit.

The error-only stub failed all three original tests. The completed MSVC Debug native executable passed all 536 tests, including four topology tests. The native formatting check passed with clang-format 19.1.5. Independent review also enumerated 1,308 ungraded touching pairs to check that the fine-side probes detect contacts missed by coarse sampling.

The builder accepts an authored bounded set of leaves. Production sparse blocks, adaptive selection, exterior apertures and accepted-step publication are outside this result.

## Sibling changes

Refinement and coarsening now produce unpublished topology and pressure warm-start candidates. Refinement copies the parent's pressure. Coarsening uses a deterministic equal-volume mean. Retained leaves preserve their pressure and expose their old and new ordinals. Missing siblings, a malformed warm start and a 4:1 candidate fail before any committed state changes.

Both focused tests failed against the unimplemented entry points. All six topology tests and 538 full native CPU tests passed with the completed candidate builder. Native formatting passed with clang-format 19.1.5.
