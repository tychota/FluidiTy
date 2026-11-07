# M0 Numerical Reference and Proof Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent numerical oracle and two exact theorem pilots cross-checked with pinned nano-lean.

**Architecture:** Small NumPy/SciPy operators describe uniform MAC geometry, particle transfers, pressure, viscosity and conservative particle operations. Versioned fixtures support later C++/CUDA comparisons. Lean proves algebraic pilots; an independent kernel checks their exported proof terms with a matching trusted prelude.

**Tech Stack:** Python 3.13, NumPy/SciPy/pytest; Lean 4.34.1; mathlib d13f23b723b8a846827a245b89c10fc7d3f11612; olean-export 0.2.0; nano-lean 54ce971a5ac2513a1bac0ac9bca73f672196c0f2; Rust 1.97.1.

**Spec:** `docs/superpowers/specs/2026-11-07-fluidity-design.md`; `docs/references/formal-contracts.md` defines theorem boundaries.

## Global Constraints

- All public technical prose and identifiers are English.
- One shared MAC velocity representation.
- A missing or skipped test is not a success.
- A formal proof of the whole simulator is deferred.
- No implementation, proof, neural inference or CUDA performance claim without corresponding evidence.
- No third-party source or weights are included by this design.
- No `sorry`, `admit`, `sorryAx`, new axioms or native_decide in accepted proofs; exact-real proofs do not certify IEEE/CUDA arithmetic.
- Scope: independent reference and F5/F8 pilots, not the full M0 CUDA gate or all ten theorem families.

## Review Focus

- Invalid/nonfinite geometry and coefficients fail without mutating inputs (Tasks 1–6).
- Sealed/disconnected pressure components check physical compatibility before deterministic gauges, including boundary flux (Task 3).
- PSD viscosity alone is insufficient: reject unsafe explicit steps and account for moving-wall work (Task 4).
- Resampling rejects infeasible moment/EOS states rather than claiming conservation (Task 5).
- Verification fails closed on tool/prelude mismatch, forbidden axioms, missing tools and checker failure (Tasks 7–8).

## File map and scope

`reference/src/fluidity_reference/` contains focused `grid.py`, `basis.py`, `transfers.py`, `pressure.py`, `viscosity.py`, `resampling.py`, `state.py`, `fixtures.py`, `cli.py`; tests in `reference/tests/`. Package metadata/lock and README live in `reference/`. `proofs/` contains the pinned Lake project, `Fluidity.lean`, `Fluidity/Pressure.lean`, `Fluidity/FixedPoint.lean` and README. `tools/proof-toolchain.json`, `tools/verify_proofs.py`, `tools/tests/test_verify_proofs.py` own checker orchestration.

Later plans cover production sparse MAC/CUDA, complete scenes, AMR, meshing, whitewater, Blender and performance gates. These algebraic fixtures are not a production simulator. Implement tasks serially with fresh implementer/reviewer gates.

### Task 1: Installable reference, uniform MAC geometry and basis

**Files:** Create `reference/pyproject.toml`, `reference/requirements.lock`, `reference/src/fluidity_reference/{__init__,grid,basis}.py`, `reference/tests/test_grid_basis.py`, `reference/README.md`.

**Interfaces:** Immutable `Grid2D(nx: int, ny: int, h: float, origin: tuple[float,float]=(0.,0.))` exposes `cell_count`, `face_count`, `cell_centers`, `face_centers`, `incidence` CSR, `face_areas`, `face_distances`, `cell_volumes`. `quadratic_stencil(position: ndarray[d], h: float, offset: ndarray[d]) -> (nodes: ndarray[n,d], weights: ndarray[n], gradients: ndarray[n,d])`; nodes integer, physical samples h*(nodes+offset), position relative to origin.

- [x] **Step 1:** Tests Grid2D(2,1,1): centers (.5,.5),(1.5,.5); x faces first `(ny,nx+1)`, then y `(ny+1,nx)`, row-major. Internal incidence +left/below, −right/above; boundary low −1, high +1. Areas h, volumes h² (unit depth), distances h internal/h/2 boundary. Reject zero/bool dims, bad origin, nonpositive/nonfinite h. Random subcell quadratic tests: unity, zero first moment, derivative sum zero, affine reproduction, moment h²/4 I in 2D/3D.
- [x] **Step 2:** Run `python -m pytest reference/tests/test_grid_basis.py -q`; record import failure before code.
- [x] **Step 3:** Implement interfaces; editable setuptools package Python >=3.13,<3.14. Pin runtime/test/build dependencies with artifact hashes; no machine paths. Document isolated install/testing.
- [x] **Step 4:** Install lock/editable package in ignored venv; run tests, require all pass.
- [x] **Step 5:** Commit `feat(reference): add uniform MAC geometry and quadratic basis`.

### Task 2: Independent staggered APIC transfer oracle

**Files:** Create `reference/src/fluidity_reference/transfers.py`, `reference/tests/test_transfers.py`.

**Interfaces:** Consumes Grid2D/basis. Produces `local_apic_fit(position, sample_positions, weights, values) -> (value, affine_row)` using centered full moment; `local_velocity_gradient(values, gradients) -> ndarray[d]`; `p2g(grid, positions[n,2], masses[n], velocities[n,2], affine[n,2,2]) -> (face_mass, face_momentum)`; `g2p(grid, positions, old_faces, new_faces) -> (pic[n,2], increments[n,2], affine[n,2,2], gradient[n,2,2])`; `energy_velocity(pic, residual, energy, phase, separation=1.) -> (particle_velocity, advection_velocity)`. Float64, complete uniform supports only; incomplete support raises ValueError.

- [x] **Step 1:** Test affine reproduction, linear/angular moments, each component mass sum separately (do not add component masses). Nonlinear field gives C different from true derivative L. Identical old/new gives zero increment while supplied residual may persist. Liquid alpha=.99e,beta=e; gas alpha=beta=0; e in [0,1]. Invalid/nonfinite data and incomplete supports reject without mutation.
- [x] **Step 2:** Run `python -m pytest reference/tests/test_transfers.py -q`; record failure.
- [x] **Step 3:** Implement independent old/new gathers and full moment solve. Particle result pic+alpha*residual; advection result pic+beta*alpha*residual, beta<=e and elastic beta=0. Grid increment is diagnostic only, not added separately; residual=old_particle−gather(old_grid). No CPIC/boundary or masked-stencil conservation claim.
- [x] **Step 4:** Run transfer and prior tests; all pass.
- [x] **Step 5:** Commit `feat(reference): add staggered APIC transfer oracle`.

### Task 3: Finite-volume pressure, boundaries and residual identity

**Files:** Create `reference/src/fluidity_reference/pressure.py`, `reference/tests/test_pressure.py`.

**Interfaces:** `PressureInputs(grid, dt, velocity, aperture=ones, mobility=ones, distance=grid_distances, wall_velocity=zeros, dirichlet_mask=False, dirichlet_pressure=zeros, jump=zeros, source=zeros, fluid_volume=grid_volumes, bulk_modulus=inf, previous_pressure=zeros)`; face vs cell shapes follow names. `solve_pressure(inputs) -> PressureResult(pressure, velocity, matrix, rhs, residual, constraint_defect, gauges)`; `bounded_band_source(error, volumes, band, maximum) -> ndarray`.

- [x] **Step 1:** Hand-check two-cell pressure/jump signs; one-cell capillary Dirichlet gives p=pGamma,J=0. Compatible Neumann components pin lowest index; incompatible source/boundary flux rejects before gauge. Compliance allows mean source/no pin. Verify residual identity r/V at tiny cut volumes and blocked moving-wall flux. Reject internal Dirichlet, jump on Dirichlet, bad shapes/coefficient domains/nonfinite data. Band correction is clipped, zero in core and volume-weighted zero mean.
- [x] **Step 2:** Run `python -m pytest reference/tests/test_pressure.py -q`; record failure.
- [x] **Step 3:** D=R diag(a), a=area*aperture; active faces=open internal or explicit boundary Dirichlet, other boundaries prescribed flux. k=dt*a*beta/d, C=V/(K*dt); A=R_active diag(k) R_active.T+diag(C). b=V*s−D*u−R((area−a)*wall_velocity)−R_active(k*J)+Dirichlet lift+C*pstar. Correct active u by dt*beta/d*(R.T*p+J−external pressure lift). Keep other u prescribed. Components use positive internal k; compliance/Dirichlet anchors prevent gauges. Pin row AND column, retain original A,b for r=b−Ap. defect=(D*u_new+blocked_flux)/V−s+(p−pstar)/(K*dt)=−r/V. Bisection solves clipped zero-mean band correction; physical source compatibility separate.
- [x] **Step 4:** Run pressure/prior tests; all pass, no singular warnings.
- [x] **Step 5:** Commit `feat(reference): add conservative pressure projection and gauges`.

### Task 4: Tensorial viscosity and certified time step

**Files:** Create `reference/src/fluidity_reference/viscosity.py`, `reference/tests/test_viscosity.py`.

**Interfaces:** `mac_strain(grid, viscosity: float) -> (E: csr_matrix, W: ndarray)`; `viscous_step(velocity, mass, E, W, boundary_strain, dt, mode='auto', safety=.01) -> ViscosityResult(velocity, method, spectral_bound, energy_before, energy_after, wall_work, numerical_dissipation)`.

- [x] **Step 1:** Translation/rotation null modes, Couette shear, K=E.T W E symmetry/PSD and tensor shear factor. M=diag(.01,1), K=[[1,−1],[−1,1]] has row-sum bound110: dt=.1 selects implicit; forced explicit raises; dt=.99*2/110 permits explicit and dissipates. Scalar BE moving-wall energy/work identity; zero viscosity identity; invalid masses/weights/data reject.
- [x] **Step 2:** Run `python -m pytest reference/tests/test_viscosity.py -q`; record failure.
- [x] **Step 3:** E normal rows du/dx,dv/dy at cells, shear Dxy=.5(du/dy+dv/dx) at interior vertices. W=2*mu*h² normal,4*mu*h² shear. load=E.T W d_wall. Bound symmetric S=M^−1/2 K M^−1/2 by max absolute row sum; dt*bound<=2*(1−safety) required explicit. Auto otherwise solves (M+dt*K)u_new=M*u_old+dt*load. BE identity: Δenergy+.5||du||M²+dt||E*u_new||W²=dt*u_new.T*load.
- [x] **Step 4:** Run viscosity/prior tests; all pass.
- [x] **Step 5:** Commit `feat(reference): add tensorial viscosity with stability certificate`.

### Task 5: Conservative resampling and EOS adapters

**Files:** Create `reference/src/fluidity_reference/resampling.py`, `reference/tests/test_resampling.py`.

**Interfaces:** `ParticleBatch` arrays positions[n,d], mass[n], current_volume[n], reference_volume[n], velocity[n,d], affine[n,d,d], composition[n,k], ids[n]. `split_particle(batch, index, offsets, child_ids) -> ParticleBatch`; `merge_particles(batch, child_moments[n,d,d,d], target_moments[d,d,d], merged_id) -> ParticleBatch`; `reseed_cell(volume,density,J,count,composition) -> dict`; `tait_J(pressure,bulk_modulus,exponent) -> float`.

- [x] **Step 1:** Symmetric equal-child split preserves mass/COM/momentum; test angular identity under explicit uniform moment hypotheses. Cubic-normal MAC offsets ±sqrt(3)*h/4 per axis (S=3h²/16 I) and child h/2 loses m*h²*sum(Caa²)/32 (no energy preservation claim). Merge rejects singular target or non-PSD M*D_target,a−K_a; feasible constant affine case preserves first moments. Composition current-volume weighted. EOS rejects p<=−K/gamma; reseed m=rho*V/n,V0=V/(nJ). Duplicate IDs, invalid counts/fractions/shapes/nonfinite data reject without mutation.
- [x] **Step 2:** Run `python -m pytest reference/tests/test_resampling.py -q`; record failure.
- [x] **Step 3:** Uniform algebra only: child v=v_parent+C*offset; merge COM/velocity mass weighted; K_a=sum m[(x−xbar)(x−xbar).T+D_child,a]; affine row from exact first-moment accumulation/target. No inferred phase geometry. Tait p=(K/gamma)*(J^−gamma−1), J=(1+gamma*p/K)^−1/gamma. Validate EOS separately from moment feasibility.
- [x] **Step 4:** Run resampling/prior tests; all pass.
- [x] **Step 5:** Commit `feat(reference): add conservative resampling and EOS checks`.

### Task 6: Transaction state and deterministic fixture exports

**Files:** Create `reference/src/fluidity_reference/{state,fixtures,cli}.py`, `reference/tests/test_state_fixtures.py`; update package scripts/README.

**Interfaces:** `ReferenceState(arrays: Mapping[str,ndarray], counters: Mapping[str,int])` copies inputs; `trial_state(state, update) -> ReferenceState`; `accept_trial(committed,trial,validator) -> ReferenceState`, false validator raises `TrialRejected`. `write_fixture(path,name,arrays,metadata) -> Path`. CLI `fluidity-reference export DIRECTORY [--overwrite]`.

- [x] **Step 1:** Rejection leaves arrays/counters byte-identical; acceptance aliases no writable caller array. JSON schema_version=1, dtype/shape/data/units, no NaN/Infinity, canonical UTF-8 hash. CLI exports hand-checked two-cell pressure, compliant pressure, affine APIC and stiff-viscosity inputs/expected outputs. Repeated exports identical; malformed payload rejects; overwrite only explicit.
- [x] **Step 2:** Run `python -m pytest reference/tests/test_state_fixtures.py -q`; record failure.
- [x] **Step 3:** Implement JSON (no pickle), validated atomic writes, explicit units. Validator remains caller-owned; this is not GPU rollback implementation. Add CLI entry point.
- [x] **Step 4:** Run all reference tests and compare manifests from two temporary CLI exports; all pass.
- [x] **Step 5:** Commit `feat(reference): export numerical fixtures and trial state`.

### Task 7: Exact Lean pressure and integer pilots

**Files:** Create `proofs/{lean-toolchain,lakefile.toml,lake-manifest.json,Fluidity.lean}`, `proofs/Fluidity/{Pressure,FixedPoint}.lean`, `proofs/README.md`.

**Interfaces:** `Fluidity.pressure_psd`: finite real R,k,C, k,C>=0 imply x.T*(R diag(k) R.T+diag(C))*x=sum_f k_f*(sum_i R_if*x_i)^2+sum_i C_i*x_i²>=0 (include separate equality lemma). `Fluidity.fixed_point_order_independent`: integer list sum invariant under permutation. `Fluidity.fixed_point_partial_bound`: sum |q_i|<=B implies absolute sum of any prefix<=B. Exact hypotheses exposed; no quantizer/hardware claim.

- [x] **Step 1:** Add statements with deliberate failing proof holes (never sorry) and run `lake build` to record failure. Document finite dimensions/empty sets.
- [x] **Step 2:** Prove via finite sum rearrangement, nonnegative squares, permutation and integer triangle inequality. Pin official toolchain and mathlib above. Add #print axioms for public targets; no new axioms/native_decide/admissions.
- [x] **Step 3:** Run `lake build`, save command/output/versions in ignored artifacts; public axiom closure subset {propext,Classical.choice,Quot.sound}. Compile consumers including empty face sets and zero coefficients.
- [x] **Step 4:** Commit `feat(proofs): prove pressure PSD and integer accumulation pilots`.

### Task 8: Independent pinned checker and evidence boundaries

**Files:** Create `tools/proof-toolchain.json`, `tools/verify_proofs.py`, `tools/tests/test_verify_proofs.py`; update root/proof/reference READMEs and method paper when claims change.

**Interfaces:** `python tools/verify_proofs.py --lean-bin DIR --exporter PATH --checker PATH --checker-provenance FILE --output DIR [--lake-packages FILE]` builds/audits Lean, exports Init prelude and theorem closures via exporter 0.2.0, invokes nl-fast with `--prelude` and `--axioms propext,Classical.choice,Quot.sound`, no trusted cache. The pinned checker has no version flag: require a build receipt binding its SHA256 to the nano commit and Rust version, explicitly treated as a trusted external build assertion. Optional Lake package overrides support verified archive bootstrapping and are recorded in evidence. Toolchain manifest pins nano commit, Lean tag/commit 5045d0056413266e57c625dcd7c365b10e377c52, mathlib commit, exporter/Rust versions.

- [x] **Step 1:** Tests wrong tool versions, missing executable, forbidden axiom/admission, failed export/checker, stale/missing prelude fail. Clearly labeled process doubles check orchestration only, never kernel evidence; assert prelude/allowlist/no-cache arguments.
- [x] **Step 2:** Run `python -m pytest tools/tests -q`; record failure.
- [x] **Step 3:** Implement fail-closed workflow; record version/provenance, binary/prelude/export SHA256 and commands in evidence JSON only after actual success. Build external pinned sources with documented commands, no vendoring. Same official Lean environment exports Init and our closures. Do not bypass primitive-prelude validation.
- [x] **Step 4:** Run full Python tests, Lean build/axiom audit, actual exported-proof verification. If upstream cannot execute, preserve blocker and leave gate incomplete; no smoke-test substitution. Synchronize public claims with actual evidence.
- [x] **Step 5:** Commit `feat(proofs): cross-check pilots with pinned nano-lean`.

## Completion and self-review

Run all Python tests, fixture export, Lean build/axiom audit, actual nano-lean checks, git diff --check. Fresh whole-branch review covers signs, units, ownership, validation and trust. Commit fixes separately; keep evidence/ledger ignored. Preserve feature branch, do not merge/push. Restricted workspace delivery may mirror public tracked files without Git internals and export feature bundle.

Coverage: Tasks1–6 cover independent operator reference, not full physical scenes/CUDA/AMR. Tasks7–8 cover F5/F8 only; other eight families remain future work. Interfaces consistently use Grid2D ordering, original-system residual, explicit row-specific merge moments. Each Review Focus line has owning tests. Owner authorized automatic execution after passing review and explicitly selected SDD; no redundant handoff confirmation is needed.

Execution status: all eight bounded tasks passed their task reviews. The whole-branch review and final delivery checks are recorded separately; the full production M0/CUDA gate remains open.
