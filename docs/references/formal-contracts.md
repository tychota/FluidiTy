# Independent numerical reference and ten formal contracts

This document specifies acceptance obligations; it does not claim a Python reference solver, Lean project, proof, or production implementation exists. Development arithmetic fixtures are not the required NumPy/SciPy reference solver or Lean proofs.

## Delivery and commitment model

The Python reference is a prerequisite for optimizing each corresponding numerical feature. Build a small, independent **2D NumPy/SciPy MAC reference**, with direct assembly and a direct sparse solve for tiny cases, then compare the production C++ CPU and CUDA implementations against exported fixtures. It includes transfers, closed/free-surface pressure, moving-wall flux, source compatibility, weak compliance, explicit/implicit viscosity, and small resampling cases. Preserve separate 3D algebraic fixtures for angular momentum and cubic MAC split; a 2D picture cannot validate these identities. Do not implement Blender, sparse production allocation, meshing, or an alternative full simulator in Python. Python and production must not share their assembly implementation.

Use the ten contracts below as the **entire initial Lean 4 scope**. They concern finite sums, real matrices, integer reductions, and a pure transition model. They do not prove Navier–Stokes, overall simulation stability, rendered quality, IEEE arithmetic, memory safety, CUDA race freedom, or interface reconstruction topology. Each named contract is a theorem family: a PSD lemma alone does not discharge its compatibility, kernel, or correction conclusions.

Start M0 with a pinned, minimal Lean/mathlib project and **two pilots**: F5's pressure sum-of-squares identity and F8's integer partial-sum bound/order invariance. Record their actual proof effort before expanding the project. These pilots establish feasibility; they do not replace the ten-contract program. Transfer and pressure identities, the range guard, and each energy claim must be discharged before the corresponding production feature claims those guarantees. A noncritical proof obligation may be deferred only by an explicit ADR naming the missing theorem, the reason, a numerical test substitute, and the milestone at which the obligation will be revisited. There is no blanket “Lean optional” exemption or permission to label numerical examples formal proofs. Adopt the following feature acceptance gates:

| Gate | Required formal targets | Independent numerical bridge |
|---|---|---|
| M0 | Pilots F5 and F8; reproducible pinned proof build; effort review | Directly assembled tiny pressure matrices and known nullspaces |
| M1 | F1 uniform basis, F2, F5, F6, F8, F9, F10 | Uniform transfer, projection, quantized reductions, rollback and allocation fixtures |
| M3 | F1 cubic/transition hypothesis checks, F3, F4 | 3D MAC split/merge and transition patch tests |
| M4 | F7; activate F6 finite-compliance cases | Viscosity with small dual masses, moving walls, compliant pressure |

F5/F6 already quantify over nonnegative compliance at M1; M4 changes which production cases exercise the parameter. F1 does not promise a proof of arbitrary THB construction: numerical transition patch tests must establish that the chosen implementation satisfies its moment assumptions. A changed discretization must update its named contract and fixtures before its feature passes. Every theorem must compile without `sorry`, `admit`, or project-defined unapproved axioms. Inspect and retain `#print axioms` output for each public target; allow only the explicitly recorded foundational assumptions of the pinned Lean/mathlib environment. A proof that imports `sorryAx` fails the gate. Report which obligations are checked, not merely that Lean builds. Pin versions when the project is created; this proposal makes no claim about a particular mathlib API.

## Common finite model

Use finite particle, face and cell index types (`Fin n` or finite types), vectors `Fin d → ℝ`, matrices over ℝ, and finite sums. Most algebraic fixtures also admit exact rational instances. The affine row for velocity component a is c_a; its own second-moment matrix is D_a. On a MAC grid D_a may differ by row. Do not silently replace these matrices by one isotropic D. Work in exact arithmetic in theorem statements and state numerical tolerances separately in the tests.

### F1 — Partition, first moments, and affine reconstruction

**Hypotheses.** For each complete row stencil, weights w_i satisfy Σw_i = 1, offsets r_i = x_i−x_p satisfy Σw_i r_i = 0, and D_a = Σw_i r_i r_iᵀ is invertible. For a derivative reconstruction additionally require Σ∇w_i = 0 and Σr_i(∇w_i)ᵀ = I. For an affine field u_a(x_i) = v_a + c_a·r_i, all gathered samples, including ghosts, equal that same affine continuation.

**Conclusions.** Weighted gather returns v_a; D_a⁻¹Σw_i u_a(x_i)r_i returns the affine row c_a; the derivative gather returns the prescribed affine gradient. Constant fields have zero reconstructed derivative. Identical old/new sample arrays produce zero grid FLIP increment; they do not imply a zero retained particle residual unless the old particle value equals the old gather. These are distinct consequences of the moment identities.

**Proof.** Substitute the affine expression, distribute the finite sums and use the first/second moments. F1 delivery also defines the actual uniform quadratic/cubic weights and proves their identities by piecewise polynomial reasoning; a generic reconstruction theorem that only assumes those identities is not the complete uniform-basis target. The general transition reconstruction theorem remains conditional on its explicit moment hypotheses.

**Bridge.** Uniform rational-position fixtures, affine patches on all MAC components, independent finite differences of weight derivatives, complete compatible-ghost gather, THB junction patches and conditioning/fallback tests. Renormalizing a masked stencil establishes partition but generally destroys its original first moment. F1 cannot certify masked affine completeness without new hypotheses. A ghost interpolation fit is not proof that its values equal the physical affine continuation.

### F2 — Unmasked MAC P2G mass and linear/angular momentum

**Hypotheses.** Nonnegative particle masses; complete unmasked stencils satisfy F1's partition/first moment; q_ia = Σ_p m_p w_ipa(v_pa+c_pa·r_ipa). No force, contact, level remap, quantization, or boundary modification is included.

**Conclusions.** For each velocity component a, Σ_i m_ia = Σ_p m_p and Σ_i q_ia = Σ_p m_p v_pa. The three staggered sums represent one particle inventory, not three masses to add. When Gas uses numerical marker weights, this is conservation of those quadrature weights, not a proof of physical Eulerian gas mass conservation. The antisymmetric spatial moment is also conserved:

`Σ_faces (x_j q_k − x_k q_j) = Σ_p m_p[x_pj v_pk − x_pk v_pj + c_pk·D_pk e_j − c_pj·D_pj e_k]`.

The left side means the sum on component k's faces minus the sum on component j's faces. This is the discrete angular momentum; its affine part uses the correct row-dependent D.

**Proof.** Expand each deposited component, write x_i = x_p+r_i, and collapse the zero first moments and second moments. No inverse is required.

**Bridge.** Random and rational exact fixtures in 2D/3D, all component placements, nonzero affine rows, both uniform kernels; separately bound fixed-point deposition error using F8. Ghost gather and CPIC contact have their own reaction/work budgets and are outside this theorem.

### F3 — Symmetric split conservation and the cubic MAC energy deficit

**Hypotheses.** Uniform unmasked parent/children, no affine-off flag, masses and reference volumes divided equally; symmetric offsets have mean zero and covariance S; child position x+δ, velocity v+Cδ, unchanged C and J. For each row let R_a = D_parent,a − D_child,a − S.

**Conclusions.** Mass, reference/current volume and linear momentum are conserved. Child kinetic-plus-affine energy equals parent energy minus `(m/2) Σ_a c_a R_a c_aᵀ`. If each R_a is supported only in coordinate a, the angular momentum from F2 is conserved because the missing off-diagonal spatial moments vanish. Equal isotropic quadratic supports have R_a=0. For 2:1 cubic-normal/quadratic-transverse MAC, S=3h²I/16 and R_a=h² e_a e_aᵀ/16, giving the **nonzero** exact energy loss `m h² Σ_a C_aa²/32`. Do not call this split energy preserving.

**Proof.** Expand squared velocities and affine energies, cancel odd terms, and compare the row moments. The physical admissibility of children is a separate acceptance predicate.

**Bridge.** 3D exact/rational covariance cases, pure diagonal and off-diagonal C, angular momentum checks, energy-deficit accounting, masked/transition split refusal. The theorem does not prove a child lies outside an obstacle or remains in a permitted phase.

### F4 — Merge moment conservation and covariance energy gate

**Hypotheses.** Positive child masses, M=Σm; positive-definite D_child,a and D_target,a; unmasked uniform admissible supports. Let r_c=x_c−x̄, h_a=Σm_c[(v_ca−v̄_a)r_cᵀ+c_ca D_child,ca], K_a=Σm_c[r_c r_cᵀ+D_child,ca], c_target,a=h_a D_target,a⁻¹/M. Require `M D_target,a − K_a` PSD **for every a**. Set V0_target=ΣV0_c, J_target=Σ(J_c V0_c)/V0_target and each constituent fraction f_target=Σ(f_c J_c V0_c)/Σ(J_c V0_c), with positive denominators. These definitions are part of the hypotheses for the volume conclusions.

**Conclusions.** Preserve mass, reference volume, current volume, each constituent volume, linear momentum and F2 angular momentum. Target kinetic-plus-affine energy is no greater than the summed child energy. The gate is sufficient, not necessary: rejecting a group is allowed even if a more elaborate merge could dissipate energy safely.

**Proof.** First moments follow from the definitions. For each row the block Gram matrix with covariance K_a, cross moment h_a and scalar variance `T_a=Σm_c[(v_ca−v̄_a)²+c_ca D_child,ca c_caᵀ]` is PSD. Cauchy–Schwarz/Schur complement gives `h_a K_a⁻¹ h_aᵀ ≤ T_a`; the covariance gate and inverse order give `h_a(M D_target,a)⁻¹h_aᵀ ≤ T_a`. Add mean translational energy. Strict positive definiteness avoids a pseudoinverse proof in the initial project.

**Bridge.** Counterstreaming children, different affine rows, rejected negative-eigenvalue gates, mixture-volume accounting and exact pre/post energy. Floating eigenvalue tolerances must conservatively refuse uncertain cases; they are not established by a theorem about exact PSD matrices.

### F5 — Integrated FV pressure quadratic form, kernel and solvability

**Hypotheses.** Finite oriented cell/face incidence B, positive internal-face conductances k; nonnegative Dirichlet anchor and compliance diagonals d,c. A=`B diag(k) Bᵀ + diag(d+c)`. Cell rows use integrated flux; no asymmetric row division by cell volume.

**Conclusions.** `pᵀAp=Σ_f k_f(Bᵀp)_f²+Σ_i(d_i+c_i)p_i²≥0`. The kernel consists precisely of vectors constant on each positive-conductance connected component and zero on components having any positive anchor/compliance. Thus A is SPD exactly when all components are anchored, or after gauge restriction on unanchored components. `Ap=b` is solvable exactly when b sums to zero on every unanchored component. An incompressible closed component therefore needs a compatibility correction; a component anchored by finite compliance does not have the same zero-sum requirement.

**Proof.** Matrix transpose/finite sum expansion; every nonnegative term must vanish. For solvability use the finite-dimensional symmetric range/kernel orthogonality result. A practical first pilot may represent components by a supplied partition and assume their kernel characterization; discharging the full F5 family still requires the characterization theorem.

**Bridge.** Independently assembled two-cell/multicomponent matrices, free-surface anchors, zero-aperture edge removal, singular closed boxes, compliant constant-pressure modes and coarse/fine aggregate flux checks. This does not prove geometric apertures are correct or that a multigrid preconditioner is effective.

### F6 — Pressure correction residual is the physical constraint defect

**Hypotheses.** Fix geometry and reuse exactly the same integrated incidence, apertures, conductances, jumps, wall flux and boundary lifting in assembly and correction. Let F* be predicted integrated fluid flux, W prescribed wall flux, S=V s source, C the compliance diagonal. With pressure/boundary jumps absorbed consistently into F*, take b=`S−B F*−W+C p_old`, A=`L+C`, corrected integrated divergence `B F_new=B F*+L p`. Define algebraic residual r=`b−A p`.

**Conclusion.** `B F_new+W−S+C(p−p_old)=−r`. At C=0, residual is minus the integrated divergence/source defect. At C>0, the compliant volume-change term is part of the physical constraint; a nonzero divergence alone is not a failed solve. Dividing this identity by strictly positive cell volumes supplies a physical per-cell stopping criterion; small integrated residual can hide large divergence in tiny cells.

**Proof.** Substitute A,b and cancel. No exact solve is assumed.

**Bridge.** Direct versus iterative solve, deliberate loose residuals, two-phase signed jumps, moving walls, tiny cut cells and compliance. A mismatched correction coefficient must fail this identity. A two-cell SPD fixture covers only a small part of F5 and cannot substitute for F6.

### F7 — Diffusion energy: explicit timestep condition and implicit wall work

**Hypotheses.** Positive diagonal M; K=EᵀWE, W PSD; fixed coefficients. Let energy E_M(u)=uᵀMu/2. For explicit homogeneous walls require Δt≥0 and a certified Λ with `K M⁻¹ K ≼ ΛK`, ΔtΛ≤2. One construction is a row-sum bound on the symmetric mass-scaled matrix, with an independently verified upper rounding margin in production.

**Conclusions.** Explicit Euler `u+=u−Δt M⁻¹Ku` has `E_M(u+)−E_M(u)=−Δt uᵀKu+Δt² uᵀKM⁻¹Ku/2≤0`. Positive K alone is insufficient. For backward Euler `(M+ΔtK)u+=Mu+Δt b_wall`, with no timestep upper bound:

`E_M(u+)−E_M(u)=Δt u+ᵀb_wall−Δt u+ᵀKu+−||u+−u||_M²/2`.

With moving walls, energy may increase by boundary work. The corresponding explicit forced identity is `ΔE=Δt uᵀ(b−Ku)+Δt²||M⁻¹(b−Ku)||_M²/2`; no absolute contraction is claimed.

**Proof.** Expand the quadratic energy for Euler; multiply the backward equation by u+ and use polarization. Prove the row-sum-to-spectral bound as a separate F7 lemma or supply the matrix inequality as a certified hypothesis; naming Gershgorin without establishing this bridge is not the full chosen contract.

**Bridge.** Small dual masses, discontinuous coefficients, matrix PSD, explicit cases just below/above the bound, forced-wall balances and implicit residual effects. The old small-mass and wall fixtures motivate this contract but do not verify production quadratures, geometric assembly or rounding of Λ.

### F8 — Integer reduction order, range, and quantization error

**Hypotheses.** N≤2ⁿ with integer 0≤n≤62; per-contribution integer Q_j is nearest rounding of c_j/s, s>0, and `|c_j/s|≤2^(62−n)`. Each contribution is quantized before summation. State ties-to-even behavior explicitly in the implementation mapping.

**Conclusions.** Every partial sum has absolute value at most `Σ|Q_j|≤2^62`, safely inside signed i64. Any permutation/reduction tree gives the same integer total; decoded total error is `|sΣQ_j−Σc_j|≤Ns/2`. Equality at 2^62 is safe: the old wording “strictly less than 2^62” does not follow from non-strict guards and should be corrected to “at most 2^62.” This proof does not imply CPU/CUDA compute identical pre-quantization floating contributions.

**Proof.** Integer triangle inequality, finite-sum commutativity and the nearest-rounding error bound. Keeping the range proof in ℤ makes the separate signed-word representation obligation explicit.

**Bridge.** Adversarial cancellation, full-capacity same-sign inputs, all channels including force, randomized orders, half-integer ties and fatal range rejection. Include a production check that conversion receives a finite value; NaN comparisons cannot certify the hypothesis.

### F9 — Reject preserves committed state and accepted time advances once

**Hypotheses.** A pure transition model has committed persistent state A, isolated trial B, and explicit scratch/diagnostics. Trial construction cannot mutate A. Rejection returns A; acceptance replaces A by B exactly once, assigning accepted_time(B)=accepted_time(A)+accepted_dt. A proposed trial may not advance the committed clock before acceptance. Persistent state includes particles, topology, pressure history, gas scalars, whitewater events/accumulators, fractional source counters, next IDs, accepted-step RNG keys and accepted time. Attempt diagnostics and the smaller retry dt are separate control outputs, so they may change on reject.

**Conclusions.** Any finite sequence of rejected trials preserves the entire committed state by equality, including accepted time. A single acceptance advances accepted time by its accepted Δt once; replaying a deterministic trial from the same A and accepted-step key yields the same result. Adaptation is a separate transaction and cannot retroactively mutate its input snapshot.

**Proof.** Case analysis for one transition and induction over a finite transition list. Deterministic replay additionally assumes the trial function is deterministic; this theorem cannot establish CUDA determinism.

**Bridge.** Serialized persistent-state hashes before/after injected failures at every stage; remap/warm-start retries; source/RNG/ID counters; snapshot and slow-copy fence tests on real GPU. The formal state record must enumerate all persistent fields. A pure model cannot prove C++ aliasing discipline or asynchronous buffer lifetime.

### F10 — Complete uniform MAC stencil fits the declared shared tile

**Hypotheses.** In normalized one-dimensional coordinates, sorted position X₀ has B=`floor((X₀−1.5)/4)` and trial position satisfies `|X−X₀|≤0.5`. Quadratic basis starts at `floor(X−s−0.5)` with three samples, s∈{0,0.5}; cubic normal basis starts at `floor(X)−1` with four samples. The tile contains integer indices `[4B,4B+8)`.

**Conclusion.** Every sample of these stencils lies in the tile, for every real position including bin boundaries and negative coordinates. Tensor products establish the 2D/3D 8^d inclusion. A conservative velocity/acceleration interval can establish the displacement hypothesis upstream; the tile theorem itself assumes that bound and says nothing about whether the timestep estimator supplied it correctly.

**Proof.** The bin gives `X₀∈[4B+1.5,4B+5.5)` and therefore `X∈[4B+1,4B+6)`. Bound each floor expression and add the largest stencil ordinal. This is floor arithmetic, independent of CUDA and floating interpolation.

**Bridge.** Rational boundary cases, next-representable floats, negative positions, all components/kernels, x=2.1 and 5.9, old/new positions, and the measured displacement extrema. THB stencils and changed support radii require new bounds. The runtime inclusion guard remains mandatory before any write; allocation and out-of-tile fallback correctness are separate GPU obligations.

## Evidence reporting

Report four distinct results: compiled exact-arithmetic theorems; independent Python fixture results; C++/CUDA parity and runtime-invariant results; real-device end-to-end quality/performance results. A theorem cannot stand in for the other three. The implementation bridge should name which hypothesis each check exercises and retain at least one counterexample that fails if the hypothesis is removed. No milestone should claim the ten contracts are proved merely because their numerical examples pass.
