# Exact pressure, transfer, integer and tile contracts

This pinned Lean project checks the M0 F5/F8 pilots and the first uniform-transfer slice of
[F1, F2, F8 and F10](../docs/references/formal-contracts.md). It does not discharge all ten contract
families or the full M1 milestone. All theorem arithmetic is over mathematical reals and integers.

## Build and independent verification

The toolchain is official Lean **4.34.1** (`leanprover/lean4:v4.34.1`, release commit
`5045d0056413266e57c625dcd7c365b10e377c52`). Mathlib is pinned to
`d13f23b723b8a846827a245b89c10fc7d3f11612`; `lake-manifest.json` locks inherited dependencies.
No third-party source is vendored. From `proofs/`, with the pinned toolchain and dependencies:

```sh
lake exe cache get
lake --wfail build
lake env lean Fluidity/Pressure.lean
lake env lean Fluidity/FixedPoint.lean
lake env lean Fluidity/UniformTransfer.lean
lake env lean Fluidity/UniformTile.lean
```

The default build includes typed consumers in `Fluidity.lean`: concrete tensor reconstruction,
nearest-rounding and count premises implying the inclusive integer budget, a signed negative tile
with both quadratic shifts and cubic normal, and distinct-row angular moments with a swapped-D
negative control. Build outputs and retained local evidence are ignored.

Independent verification uses `olean-export` **0.2.0** and experimental `nl-fast` from nano-lean
commit `54ce971a5ac2513a1bac0ac9bca73f672196c0f2`, built with Rust **1.97.1**. Build the tools from
their official sources using `cargo build --release --locked` (add `--bin nl-fast` for the checker).
Supply a JSON receipt with `schema_version: 1`, `nano_lean_commit`, `rust_version`, and
`checker_sha256` matching the executable, plus source URL, archive checksum and build command.
From the repository root:

```sh
uv run --project reference --locked python tools/verify_proofs.py --lean-bin /path/to/lean/bin \
  --exporter /path/to/olean-export --checker /path/to/nl-fast \
  --checker-provenance /path/to/receipt.json --output proofs/artifacts/verification
```

The verifier checks pins/provenance, builds with warnings treated as failures, audits every listed
closure, exports Init and the same target inventory, and runs the actual checker with an explicit
axiom allowlist and no cache. It writes `evidence.json` only on success. The accepted closure is a
subset of `{propext, Classical.choice, Quot.sound}`. No admissions or extra project axioms are allowed.
`MODULE_TARGETS` in `tools/verify_proofs.py` is the common module/audit/export inventory. The tests
check required target membership and reject missing/forbidden closures; process doubles test only
orchestration, never independent kernel acceptance.

`--lake-packages FILE` permits a verified local package override. Its contents and hash are recorded
as an additional trust input; use the exact manifest revisions. The checker is experimental and its
build receipt is a trusted external assertion, not a reproducible-build proof. A Lean build alone
is not independent acceptance. Keep local evidence private if its paths disclose local information.

## F1: actual uniform quadratic basis and reconstruction

All 1D basis coordinates here are normalized by cell width. For particle coordinate `X`, lattice
shift `s` equal to `0` or `1/2`, and `b = floor(X-s-1/2)`, let `t = X-s-1/2-b`. The three nodes are
`b+i`, `i : Fin 3`, with anchors `b+i+s` and offsets `r = anchor-X`.

| Definition | Exact values for i = 0, 1, 2 |
| --- | --- |
| `quadraticWeights t` | `((1-t)^2/2, 3/4-(t-1/2)^2, t^2/2)` |
| `quadraticOffsets t` | `(-1/2-t, 1/2-t, 3/2-t)` |
| `quadraticDerivatives t` | `(t-1, 1-2t, t)`; derivative with respect to particle coordinate |

The independently defined `centeredQuadratic` is zero outside `[-3/2,3/2]`, has central polynomial
`3/4-x^2` on `[-1/2,1/2]`, and quadratic tails `(x+3/2)^2/2` and `(x-3/2)^2/2`.

| Public audited target | Signature and conclusion |
| --- | --- |
| `quadratic_centered_local (t : ℝ) (0 ≤ t) (t ≤ 1) (i : Fin 3)` | `centeredQuadratic (-quadraticOffsets t i) = quadraticWeights t i`, including both endpoints. |
| `quadratic_derivative_polynomials (t : ℝ) (i : Fin 3)` | The local polynomial evaluates to the defined weight, and its formal polynomial derivative evaluates to the defined derivative. |
| `quadratic_boundary_matching` | Adjacent floor pieces agree in value and first derivative at `t=0,1`; the outer support weights and derivatives are zero. |
| `quadratic_shifted_basis (X s : ℝ) (s=0 ∨ s=1/2)` | Proves `0≤t<1`, the physical normalized anchor/offset relation, nonnegative weights, and all five 1D moment identities. Floor is signed, including negative coordinates. |
| `tensor_gradient_polynomials (t : Fin 3 → ℝ) (i : TensorNode) (axis : Fin 3)` | Holding two coordinates fixed, the polynomial in the remaining coordinate evaluates to the tensor weight; its derivative evaluates to the corresponding tensor gradient. |
| `quadratic_tensor_moments (t : Fin 3 → ℝ)` | The actual 27-node product basis satisfies `RowMoments` with `D_jk = (if j=k then 1/4 else 0)`. |
| `affine_reconstruction_of_moments` | Under the explicit moment and inverse hypotheses below, returns PIC velocity, the affine fit row C, and the derivative row L. |

`TensorNode = Fin 3 × Fin 3 × Fin 3`. `tensorWeight` multiplies the three weights;
`tensorOffset` collects their offsets; `tensorGradient` uses the product derivative on each axis.
The five identities are `Σw=1`, `Σw r_j=0`, `Σw r_j r_k=D_jk`, `Σg_j=0`, and
`Σr_j g_k=δ_jk`. The local/tensor polynomial identities are algebraically valid outside the local
interval too; the floor theorem and nonnegativity conclusion restrict the physical stencil to it.
Derivative certificates use formal polynomial differentiation and explicit endpoint matching;
no floating numerical differentiation is used in these proofs.

The reconstruction signature quantifies over any finite face type `ι`, dimension `d`,
`weight : ι → ℝ`, `offset gradient : ι → Fin d → ℝ`, and
`moment inverse : Fin d → Fin d → ℝ`. A `RowMoments weight offset gradient moment` proof supplies
the five identities. The additional hypothesis is the right inverse identity
`Σl moment j l * inverse l k = δ_jk`. All samples are explicitly the same affine continuation
`affineSamples v c r i = v + Σj c_j r_ij`. The conclusions are exactly:

```text
Σi w_i u_i = v
∀j, Σk (Σi w_i u_i r_ik) inverse_kj = c_j
∀j, Σi u_i gradient_ij = c_j
```

Thus C is a moment fit and L a derivative gather; they agree for these affine samples, not for
arbitrary fields. The root consumer discharges the moment and inverse premises for the actual
quadratic tensor basis with `D=I/4` and inverse `4I`. Constant fields are the `c=0` specialization.
For physical cell width `h`, runtime offsets have units m (`h*r`), gradients m^-1 (`g/h`), and
moments m² (`h²I/4`); the generic theorem accepts physical quantities when its stated moment and
inverse hypotheses hold. The bridge separately checks `h=1` and `h=2`.

This slice does not prove cubic or transition reconstruction, masked affine completeness,
conditioning/fallback, compatible-ghost generation, or that arbitrary ghost values are the same
physical affine continuation. Equal old/new grid samples imply zero grid increment; they say
nothing about a retained particle history residual. No history update is modeled here.

## F2: finite unmasked MAC deposition

`mac_deposit_conserves_row_moments` quantifies over `P` particles, `I` faces per component, and `d`
components/coordinates (`Fin P`, `Fin I`, `Fin d`). Its inputs are real masses `m_p` with a
nonnegativity premise, weights `w_pai`, positions `x_pj`, velocities `v_pa`, affine rows `c_pal`,
offsets `r_paij`, component face anchors `anchor_aij`, and moments `D_pajk`. It requires:

- `anchor_aij = x_pj + r_paij` for every particle, component, face and coordinate;
- `Σi w_pai = 1`, `Σi w_pai r_paij = 0`, and `Σi w_pai r_paij r_paik = D_pajk`.

`depositedMacMass a i = Σp m_p w_pai` and
`depositedMacPureMomentum a i = Σp m_p w_pai (v_pa + Σl c_pal r_pail)` are explicitly defined finite sums.
No force, contact, rounding, remapping or boundary modification appears in this operator. The
proof needs no inverse, no derivative hypotheses, and no common moment matrix. The equality is
algebraic (the nonnegative-mass premise restricts its intended physical domain).

The theorem returns all three conclusions together:

```text
∀a, Σi depositedMacMass a i = Σp m_p
∀a, Σi depositedMacPureMomentum a i = Σp m_p v_pa
∀j k, Σi anchor_kij depositedMacPureMomentum k i - Σi anchor_jik depositedMacPureMomentum j i
  = Σp m_p (x_pj v_pk - x_pk v_pj
             + Σl c_pkl D_pklj - Σl c_pjl D_pjlk)
```

The left angular expression uses different component face sets. Each component mass sum
represents the same inventory; summing the three inventories would triple count it.
The root consumer specializes the result to mass 2, zero translation/velocity, affine rows
`(2,4,-1), (-5,1,5), (7,-2,3)`, and distinct cubic-normal matrices with normal diagonal `1/3`
and transverse diagonals `1/4`. The z angular moment is `-9/2`; swapping `D_x` and `D_y` gives
`-6`. This is an algebraic negative control, not cubic runtime support or a cubic basis proof.

## F8: integer budget and nearest-rounding channel error

The M0 public targets are unchanged:

| Target | Exact conclusion and hypotheses |
| --- | --- |
| `fixed_point_order_independent (q r : List ℤ) (q.Perm r)` | `q.sum = r.sum`. |
| `fixed_point_partial_bound (q : List ℤ) (B : ℤ) (Σabs(q)≤B) (n : ℕ)` | `abs((q.take n).sum)≤B`, including empty/oversized prefixes. |
| `quantized_contribution_bound (scaled : ℝ) (Q B : ℤ)` | Nearest rounding and `abs(scaled)≤(B:ℝ)` imply `abs(Q)≤B`, even at the inclusive guard. |
| `fixed_point_cap_partial_bound (q : List ℤ) (n : ℕ) (n≤62)` | `q.length≤2^n` and every `abs(Q)≤2^(62-n)` imply `Σabs(q)≤2^62` and that bound for every prefix. |
| `quantized_sum_error_bound {N} (c : Fin N → ℝ) (Q : Fin N → ℤ) (s : ℝ)` | `s>0` and every Q nearest to c/s imply `abs(s * ((ΣQ : ℤ):ℝ) - Σc) ≤ N*s/2`. |

`IsNearestInteger scaled Q` means `∀candidate : ℤ, abs(scaled-Q)≤abs(scaled-candidate)`.
The helper `nearest_integer_error` derives the half-quantum bound by comparison with a nearest
integer; it does not assume the final error bound. Either nearest choice at an exact tie is
allowed. The count/range consumer combines nearest rounding with the scaled real guard before
applying the prefix theorem. Quantization happens per contribution; the decoded error expression
casts the actual integer sum. The inclusive `2^62` bound is inside signed i64, but conversion to a
machine word is not modeled. Associative integer reductions and the permutation theorem apply to
the same mathematical contributions; a GPU reduction tree is not modeled.

The implementation bridge checks ties to even, reversed particle order, nonfinite/range rejection,
capacity 3 rounded to a contributor bound of 4, and contribution-before-sum behavior. The exact
proof does not establish that CPU and CUDA form identical floating contributions, that a floating
guard is sound, or that scale validation and signed conversion are correctly implemented.

## F10: signed sort bins, motion and Cartesian tiles

`uniformSortBin sorted = floor((sorted-3/2)/4) : ℤ`.
`uniform_mac_tile_contains_stencil (sorted trial : ℝ)` assumes only
`abs(trial-sorted)≤1/2`. The proof first derives
`4B+1≤trial<4B+6` from the floor definition and motion, with no nonnegative-coordinate premise.
It concludes both:

```text
∀s, s=0 ∨ s=1/2 → ∀i : Fin 3,
  4B ≤ floor(trial-s-1/2)+i < 4B+8
∀i : Fin 4,
  4B ≤ floor(trial)-1+i < 4B+8
```

`uniform_mac_tile_product {d}` accepts axiswise sorted/trial/shift arrays, a Boolean cubic selector,
and natural ordinals. It assumes the same motion/shift bounds on each axis and ordinals `<4` on
cubic axes or `<3` otherwise. It proves every coordinate of the selected product node belongs to
its eight-index tile; this directly includes 2D and 3D mixed stencils. Tile inclusion says nothing
about allocation, signed machine-index representability, THB support, or whether an upstream
timestep estimator established the motion hypothesis. Runtime guards remain necessary.

## Retained F5 pressure pilots

`pressure_quadratic_form_eq` expands the quadratic form of the independently defined component
action `A x = R diag(k) Rᵀ x + diag(C) x` into
`Σf k_f (Σi R_if x_i)^2 + Σi C_i x_i^2`, without sign assumptions.
`pressure_psd` concludes nonnegativity when every `k_f` and `C_i` is nonnegative.
The finite real incidence model supports zero cells/faces and zero coefficients. No incidence
topology, connected-component kernel characterization, strict positive definiteness, solvability
or gauge theorem is proved. Those remain F5 obligations.

## Numerical hypothesis bridge and scope

`reference/tests/test_transfer_3d_fixtures.py` checks signed rational positions including knots,
both shifts, exact partition/first/second/derivative moments, and independent central differences
of the centered weight. At an inner knot the curvature jump is 3, giving exact central-difference
error at most `3*step/4`; away from knots the quadratic derivative is exact.

`SparseMacGrid3D.ExactBasisBoundaryAndFiniteDifferenceBridge` evaluates the centered weight
independently of native local-floor formulas at four signed/boundary positions, all three MAC
components, translated origins and `h=1,2`. It checks all tensor moments (absolute tolerance
`2e-12`) and finite differences with normalized step `1e-5` and gradient tolerance `1e-5/h`.
Existing corner affine and quadratic-field tests distinguish C from L, and existing missing-owner,
zero-weight support, tile boundary, capacity and quantization tests exercise the remaining runtime
premises. The exact oracle includes a deliberately wrong affine-row control.

These are separate gates: compiled exact proofs, independent kernel acceptance, Python exact
fixtures, and native runtime tests. None proves IEEE arithmetic, memory safety, CUDA race freedom,
arbitrary compatible ghosts, a full simulator, or overall simulation stability.
