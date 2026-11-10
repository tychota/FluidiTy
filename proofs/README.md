# Exact pressure, transfer, integer, tile and transaction contracts

This pinned Lean project checks the M0 F5/F8 pilots, integrated-pressure F5/F6, the uniform-transfer slice of
[F1, F2, F8 and F10](../docs/references/formal-contracts.md), and the F9 committed-state transition.
The proofs use exact reals and unbounded integer and natural-number counts.

## Build and independent verification

The toolchain is official Lean **4.34.1** (`leanprover/lean4:v4.34.1`, release commit
`5045d0056413266e57c625dcd7c365b10e377c52`). Mathlib is pinned to
`d13f23b723b8a846827a245b89c10fc7d3f11612`; `lake-manifest.json` locks inherited dependencies.
No third-party source is vendored. From `proofs/`, with the pinned toolchain and dependencies:

```sh
lake exe cache get
lake --wfail build
lake env lean Fluidity/Pressure.lean
lake env lean Fluidity/PressureTopology.lean
lake env lean Fluidity/PressureCorrection.lean
lake env lean Fluidity/FixedPoint.lean
lake env lean Fluidity/UniformTransfer.lean
lake env lean Fluidity/UniformTile.lean
lake env lean Fluidity/LiquidTransaction.lean
```

The default build includes typed consumers in `Fluidity.lean`: concrete tensor reconstruction,
nearest-rounding and count premises implying the inclusive integer budget, a signed negative tile
with both quadratic shifts and cubic normal, and distinct-row angular moments with a swapped-D
negative control. The transaction consumers require the full state equality, clock, replay and
adaptation signatures. Build outputs and retained local evidence are ignored.

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

The [Lean proofs workflow](../.github/workflows/lean-proofs.yml) checks the official Linux Lean
release archive against its pinned SHA-256 and fetches the Mathlib cache. It builds both
independent tools with the pinned Rust version and locked dependencies, then runs the same
verifier. The job log retains the axiom reports and checker output. The job summary records
the checked commit, elapsed verifier time, tool and export hashes, checker build receipt and
each theorem's axiom closure.

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
physical affine continuation.

`equal_samples_grid_increment_and_residual` uses one scalar component and the
same weights for both histories. Pointwise equal old and new face samples give a zero grid FLIP
increment (new weighted gather minus old weighted gather). The retained residual (old particle
velocity minus old weighted gather) is zero exactly when those two velocities agree. The root
consumer has equal histories with gather 5 and old particle velocity 7, so the residual is nonzero.

## Finite signed MAC trace reindexing

`finite_face_sum_eq_weighted_forward_difference` takes a signed origin `lo : ℤ`, cell count
`n : ℕ`, and exact real face velocities `u` and cell weights `w` indexed by `ℤ`. With exterior
weights `w(lo-1) = w(lo+n) = 0`, it proves

```text
Σ(j=0…n)   u(lo+j) [w(lo+j-1) - w(lo+j)]
  = Σ(i=0…n-1) w(lo+i) [u(lo+i+1) - u(lo+i)]
```

The cell range can be empty. In that case the face range contains one term, which vanishes by
the same two endpoint hypotheses. Typed consumers in `Fluidity/UniformTransfer.lean` check the
full signature, the empty case and asymmetric signed data. Changing either exterior weight alone
gives a counterexample to equality.

The theorem covers the one-dimensional finite reindexing step in the mixed MAC trace. The
cubic derivative identity, transverse tensor factors and physical `1/h` scaling are separate
steps. The [M1 numerical bridge](../docs/references/m1-transport-evidence.md) tests the assembled
trace on complete uniform support. Cut stencils, topology and floating-point behavior are
outside the finite identity.

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

## F5: integrated pressure topology, kernel, gauge and solvability

The retained `pressure_quadratic_form_eq` and `pressure_psd` targets describe
`pressureAction R k C x i = Σf R_if k_f (Σj R_jf x_j) + C_i x_i` and its quadratic form
`Σi x_i (A x)_i`. They support arbitrary real incidence and zero cells/faces.

`PressureGraph n m` supplies `tail head : Fin m → Fin n`,
`conductance : Fin m → ℝ`, `anchor : Fin n → ℝ`, and proofs that both coefficient arrays
are nonnegative. `anchor` is the sum of Dirichlet and compliance diagonals. Its incidence is
`R_if = (if i=tail_f then 1 else 0) - (if i=head_f then 1 else 0)`.
`Reachable i j` is the reflexive, symmetric, transitive closure (`Relation.EqvGen`) of
actual oriented edges with strictly positive conductance. Zero edges do not connect cells;
self-loops cancel in the incidence. No supplied partition or kernel premise is accepted.

The three new topology signatures quantify over any `g : PressureGraph n m`:

| Audited target | Exact signature and conclusion |
| --- | --- |
| `pressure_kernel_eq_component_constants (g) (x : Fin n → ℝ)` | `(∀i, pressureAction g.incidence g.conductance g.anchor x i = 0) ↔ g.ComponentConstants x`. |
| `pressure_gauge_positive_definite (g) (pin : Fin n → Prop) (covers : ∀i, g.Unanchored i → ∃j, g.Reachable i j ∧ pin j) (x : Fin n → ℝ) (zeroPins : ∀i, pin i → x i = 0) (nonzero : x ≠ 0)` | `0 < pressureQuadraticForm g.incidence g.conductance g.anchor x`. |
| `pressure_solvable_iff_component_compatible (g) (b : Fin n → ℝ)` | `(∃p, ∀i, pressureAction g.incidence g.conductance g.anchor p i = b i) ↔ g.Compatible b`. |

`ComponentConstants x` is precisely `(∀i j, Reachable i j → x_i=x_j)` together with
`∀i, 0<anchor_i → x_i=0`. Hence an anchor forces zero throughout its whole component.
`Unanchored i` means every anchor reachable from i is zero. `Compatible b` means
`∀i, Unanchored i → Σj (if Reachable i j then b_j else 0)=0`. Repeated representatives state
the same component condition and do not weaken it.

Zero energy forces every nonnegative edge/anchor square to vanish; actual reachability then
propagates constants. Conversely, those constants annihilate the original action. Gauge pins
meet every unanchored component; choosing no pins proves strict positivity when all components
are anchored. If a component is unanchored, its nonzero characteristic vector lies in the kernel,
so the unrestricted matrix is not SPD. The runtime chooses one lowest-ordinal pin, while the
proof allows any covering pin set. Restriction means zero pressure at pins; retained neighboring
pin conductance is still part of the original quadratic form.

For solvability, a linear map on real Euclidean space is built from the same `pressureAction`.
Finite-sum transposition proves it symmetric. The proof uses
`LinearMap.IsSymmetric.orthogonal_range` and finite-dimensional double orthogonal complement
to identify range with kernel orthogonal complement. Characteristic vectors prove necessity;
partitioning the dot product by the quotient of actual reachability classes proves sufficiency.
No symmetry, component partition, kernel characterization or solvability conclusion is assumed.

## F6: shared integrated links and arbitrary-pressure physical defect

`PressureLinks n m` stores one frozen incidence `R`, physical open/full areas `a, A`, velocity
scale `q`, predicted/wall velocities `u*, u_wall`, signed lift `J`, compliance `C`, integrated
source `S=V s`, and previous pressure `p_old`. The same object defines all of:

```text
k_f = a_f q_f
F_corrected,f(p) = a_f [u*_f + q_f (Σi R_if p_i + J_f)]
W_i = Σf R_if (A_f-a_f) u_wall,f
b_i = S_i - Σf R_if (a_f u*_f + k_f J_f) - W_i + C_i p_old,i
r_i(p) = b_i - pressureAction R k C p i
defect_i(p) = Σf R_if F_corrected,f(p) + W_i - S_i + C_i(p_i-p_old,i)
```

The exact audited signature is
`pressure_corrected_flux_defect_eq_neg_residual {n m : ℕ} (l : PressureLinks n m)
(p : Fin n → ℝ) (i : Fin n) : l.defect p i = -l.residual p i`.
No solved-pressure, residual bound, positivity, or convergence premise occurs. This algebraic
identity is valid for arbitrary real frozen data. Physical use additionally requires the runtime
validation below. Dividing both sides by a strictly positive cell volume gives the physical
constraint defect in s^-1 and the signed local residual criterion.

For internal links `J` is the oriented pressure jump; for a one-cell Dirichlet link it is
`-p_ext` at a tail or `+p_ext` at a head, with no extra internal jump. Sealed links have `a=0`;
blocked-area wall flux remains. Internal columns have +1/-1, Dirichlet columns one signed entry.
In F5 the latter contributions become the nonnegative Dirichlet anchor diagonal; F6 retains
those columns and their lifts explicitly. Units are pressure/lift Pa, areas m², velocity m/s,
`q=dt β/d` in m/(Pa s), `k,C` in m³/(Pa s), and `S,W,b,r,defect` in m³/s.

The default-build consumers require all four complete signatures. Concrete consumers use a
closed two-cell component plus an independently anchored singleton, permitting a nonzero RHS
sum on the anchored component. An unsolved Dirichlet/compliance candidate has residual -3 and
physical integrated defect +3. Changing only correction's scale makes the defect 2 while the
original minus-residual remains 3; the negative control proves these differ.

The pressure slice added three targets in `PressureTopology` and one in `PressureCorrection`.
Missing any required closure fails the audit. Every target is independently exported and checked.
Only `propext`,
`Classical.choice`, and `Quot.sound` are allowed. No IEEE arithmetic, numerical convergence rate,
geometry accuracy, multigrid effectiveness, or GPU correctness is established by these theorems.

## Pressure runtime hypothesis bridge

The existing Task 1–4 cases supply the separate numerical bridge:

- `reference/tests/test_pressure_3d_fixtures.py` assembles independent dense incidence/matrices
  and tests all three signed block seams, both partial-area moving-wall Dirichlet incidences,
  zero-area separation, isolated compliance, and mixed disconnected components. Its
  `test_incompatible_components_cannot_cancel_across_closed_face` checks compatibility per
  actual component. `test_mismatched_correction_scale_breaks_the_signed_f6_identity` uses the
  unsolved two-cell candidate `(0,0.75)`, checks residual `(-0.25,0.25)` and defects
  `(0.25,-0.25)`, then changes only correction mobility and obtains a nonzero identity error.
- `MacPressureSystem3D.EvaluatesAllIndependentFixtureCases` and
  `ChecksPhysicalCompatibilityPerComponentWithoutAbsoluteFloor` cover frozen assembly,
  anchors and component gauges. `EvaluatesUnconvergedCandidateOnEveryOriginalRow` checks the
  original pin row too, and confirms frozen data survive changes to caller inputs.
- `MacPressure3D.SolvesAllIndependentFixtureCases`,
  `RetainsAnchoredPreviousPressureMean`, and
  `EliminatedArticulationPinRetainsConductanceInSeparateSingletons` exercise original-row
  solutions, compliant means, and pin restriction without losing neighboring diagonals.
  `TinyCellMustPassLocalGateEvenWhenInitialGlobalGatePasses` checks the volume-scaled criterion;
  the independent Python tiny-cell case has integrated error 1e-7 m³/s but defect 0.1 s^-1.
- `CudaMacPressure3D.MixedLinksMatchOracle`, `MultilevelProjectionMatchesCpu`, and
  `RejectsIncompatibleAndUnconverged` qualify the same inventory and actual device execution,
  multilevel iteration, original residual/physical defect, and transactional failures.

Runtime validates finite positive volumes/dt, coefficient representability, typed endpoints,
owned faces, and shared frozen link coefficients; it enforces both global/local residual gates
and the independent physical defect/identity roundoff checks. Numerical tests and device evidence
remain separate from exact-real theorem acceptance; the formal definitions do not prove that
floating implementations or geometry preparation meet these hypotheses automatically.
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

## F9 committed state and attempt control

`LiquidCommittedState` represents the coherent accepted value. Its clock corresponds to
`LiquidClock3D`: accepted time in seconds, completed-frame ordinal, within-frame substep ordinal,
accepted-step ordinal and epoch. The fields below mirror the valid ordered extents of
`LiquidState3D` in `native/include/fluidity/host/liquid_state_3d.h`.

| Lean fields | Runtime correspondence |
|---|---|
| `geometry` | World origin and cell width in `UniformMacGeometry3D`. |
| `particles` | Every `LiquidParticleState3D` field, including stable ID/tag, position/velocity, APIC slopes and enable flag, mass, reference volume, volume ratio, side and PIC initialization. |
| `blocks`, `cells`, `faces` | Owned signed coordinates, generation and hysteresis, occupancy and previous pressure, and face normal axis, slot, carrier velocity, validity, side and eligibility. |
| `sources`, `randomSeed`, `nextParticleId` | Ordered source ordinals, child counters and fractional remainders, Philox seed and next particle ID. |
| `particleMaterialHistory`, `bodyState`, `cellPhysicalInventories`, `componentLineage`, `gasScalarsAndEnthalpy`, `whitewaterEvents`, `whitewaterAccumulators` | Separately typed later-domain fields from the design's persistent inventory. Each can be a unit value for the M1 liquid-only model. |

The model treats numeric fields as exact real, signed integer or natural-number values as
appropriate. Fixed-width limits, validity checks, unused capacity, checkpoint envelope and resume
reservations belong to the runtime. The trial carries physical fields and an accepted duration,
with no proposed clock. Retry duration and diagnostic values are separate from committed state.
The runtime supplies a validated positive duration and a correct frame-boundary decision.

| Audited target | Result |
|---|---|
| `liquid_reject_preserves_committed_state` | One reject returns the entire committed value, including its clock. |
| `liquid_finite_rejects_preserve_committed_state` | Folding any finite list of rejected attempts returns that value, regardless of retry durations or diagnostics. |
| `liquid_accept_advances_clock_once` | One acceptance installs the trial fields, adds the accepted duration once, advances accepted-step ordinal and epoch once, and updates frame/substep at the same boundary. |
| `liquid_same_state_key_replays_trial` | Evaluations of one pure trial function with equal state and accepted-step key have equal outcomes and committed results, even when diagnostics differ. The shared function fixes settings, execution artifact and requested duration. |
| `liquid_adaptation_preserves_input_snapshot` | A separate pure adaptation proposes new fields and retains its original committed snapshot. The proposal has no clock or publication operation. |

All five targets are included in the 28-target independent export and axiom audit. Runtime
fault injection and serialized-state comparison are separate evidence for the C++ ownership
discipline, allocation/retry rules, CUDA ordering, floating-point behavior and actual byte replay.
