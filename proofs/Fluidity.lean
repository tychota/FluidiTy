import Fluidity.Pressure
import Fluidity.PressureTopology
import Fluidity.PressureCorrection
import Fluidity.Viscosity
import Fluidity.FixedPoint
import Fluidity.UniformTransfer
import Fluidity.UniformTile
import Fluidity.LiquidTransaction
import Fluidity.VolumeTransport
import Fluidity.Resampling
import Fluidity.Merge

open scoped BigOperators
attribute [local instance] Classical.propDecidable

namespace Fluidity

-- Consumer checks are part of the default build.
example {n : ℕ} (R : Fin n → Fin 0 → ℝ) (C x : Fin n → ℝ) :
    pressureQuadraticForm R (fun _ => 0) C x = ∑ i, C i * x i ^ 2 := by
  simp [pressure_quadratic_form_eq]

example {n : ℕ} (R : Fin n → Fin 0 → ℝ) (C x : Fin n → ℝ)
    (hC : ∀ i, 0 ≤ C i) : 0 ≤ pressureQuadraticForm R (fun _ => 0) C x := by
  exact pressure_psd R _ C (fun f => Fin.elim0 f) hC x

example {m : ℕ} (R : Fin 0 → Fin m → ℝ) (k : Fin m → ℝ)
    (C x : Fin 0 → ℝ) : pressureQuadraticForm R k C x = 0 := by
  simp [pressureQuadraticForm]

example {n m : ℕ} (R : Fin n → Fin m → ℝ) (x : Fin n → ℝ) :
    pressureQuadraticForm R (fun _ => 0) (fun _ => 0) x = 0 := by
  simp [pressure_quadratic_form_eq]

example (q : List ℤ) : q.sum = q.reverse.sum := by
  exact fixed_point_order_independent q q.reverse (List.reverse_perm q).symm

example (q r : List ℤ) (h : q.Perm r) (B : ℤ)
    (hB : (q.map (fun a => |a|)).sum ≤ B) (n : ℕ) :
    |(r.take n).sum| ≤ B := by
  apply fixed_point_partial_bound r B ?_ n
  rw [← (h.map (fun a => |a|)).sum_eq]
  exact hB

example (n : ℕ) : |(([-7, 7, 3] : List ℤ).take n).sum| ≤ 17 := by
  exact fixed_point_partial_bound _ 17 (by decide) n

example : |(([] : List ℤ).take 4).sum| ≤ 0 := by
  exact fixed_point_partial_bound [] 0 (by decide) 4

example (n : ℕ) : |(([2 ^ 62] : List ℤ).take n).sum| ≤ 2 ^ 62 := by
  exact fixed_point_partial_bound _ (2 ^ 62) (by decide) n

end Fluidity

namespace Fluidity

-- Both shifts select the centered piecewise spline, including support endpoints.
example (position shift : ℝ) (shifts : shift = 0 ∨ shift = 1 / 2) (i : Fin 3) :
    let t := position - shift - 1 / 2 - (⌊position - shift - 1 / 2⌋ : ℝ)
    centeredQuadratic (-quadraticOffsets t i) = quadraticWeights t i := by
  have basis := quadratic_shifted_basis position shift shifts
  exact quadratic_centered_local _ basis.1 basis.2.1.le i

-- The concrete 27-node basis discharges every generic reconstruction hypothesis.
example (t : Fin 3 → ℝ) (velocity : ℝ) (row : Fin 3 → ℝ) :
    (∑ i, tensorWeight t i * affineSamples velocity row (tensorOffset t) i) = velocity ∧
    (∀ j, ∑ k, (∑ i, tensorWeight t i * affineSamples velocity row (tensorOffset t) i *
      tensorOffset t i k) * (if k = j then 4 else 0) = row j) ∧
    (∀ j, ∑ i, affineSamples velocity row (tensorOffset t) i * tensorGradient t i j = row j) := by
  apply affine_reconstruction_of_moments _ _ _
    (fun j k => if j = k then 1 / 4 else 0) (fun j k => if j = k then 4 else 0)
    (quadratic_tensor_moments t)
  intro j k
  fin_cases j <;> fin_cases k <;> norm_num [Fin.sum_univ_succ]

-- Equal face histories give zero grid increment while the old particle can differ from old PIC.
example :
    let weight : Fin 2 → ℝ := ![1 / 4, 3 / 4]
    let sample : Fin 2 → ℝ := ![2, 6]
    ((∑ i, weight i * sample i) - (∑ i, weight i * sample i) = 0) ∧
      ((7 : ℝ) - (∑ i, weight i * sample i) ≠ 0) := by
  intro weight sample
  have history :=
    equal_samples_grid_increment_and_residual weight sample sample 7 (fun _ => rfl)
  refine ⟨history.1, ?_⟩
  intro zeroResidual
  have particleMatchesGather := history.2.mp zeroResidual
  norm_num [weight, sample, Fin.sum_univ_succ] at particleMatchesGather

-- Count cap and guarded nearest rounding imply every prefix's inclusive i64 budget.
example (q : List ℤ) (scaled : ℤ → ℝ) (n : ℕ) (exponent : n ≤ 62)
    (count : q.length ≤ 2 ^ n)
    (nearest : ∀ value ∈ q, IsNearestInteger (scaled value) value)
    (guard : ∀ value ∈ q, |scaled value| ≤ ((2 : ℤ) ^ (62 - n) : ℝ)) (prefixLength : ℕ) :
    |(q.take prefixLength).sum| ≤ (2 : ℤ) ^ 62 := by
  apply (fixed_point_cap_partial_bound q n exponent count ?_).2
  intro value member
  exact quantized_contribution_bound _ _ _ (nearest value member) (by exact_mod_cast guard value member)

example {N : ℕ} (c : Fin N → ℝ) (q : Fin N → ℤ) (s : ℝ) (hs : 0 < s)
    (nearest : ∀ i, IsNearestInteger (c i / s) (q i)) :
    |s * ((∑ i, q i : ℤ) : ℝ) - ∑ i, c i| ≤ N * s / 2 :=
  quantized_sum_error_bound c q s hs nearest

-- Negative signed bin -1; both quadratic shifts and cubic normal hit [-4,4).
example :
    (∀ shift : ℝ, shift = 0 ∨ shift = 1 / 2 → ∀ ordinal : Fin 3,
      -4 ≤ ⌊(-3 : ℝ) - shift - 1 / 2⌋ + ordinal.val ∧
      ⌊(-3 : ℝ) - shift - 1 / 2⌋ + ordinal.val < 4) ∧
    (∀ ordinal : Fin 4, -4 ≤ (-3 : ℤ) - 1 + ordinal.val ∧
      (-3 : ℤ) - 1 + ordinal.val < 4) := by
  have contained := uniform_mac_tile_contains_stencil (-5 / 2) (-3) (by norm_num)
  norm_num [uniformSortBin] at contained ⊢
  exact contained

-- Cubic-normal moment matrices only: no cubic runtime reconstruction is claimed.
private noncomputable def cubicMoment (a j k : Fin 3) : ℝ :=
  if j = k then (if j = a then 1 / 3 else 1 / 4) else 0

private def distinctRows : Fin 3 → Fin 3 → ℝ :=
  ![![2, 4, -1], ![-5, 1, 5], ![7, -2, 3]]

example {I : ℕ} (weight : Fin 1 → Fin 3 → Fin I → ℝ)
    (offset : Fin 1 → Fin 3 → Fin I → Fin 3 → ℝ)
    (anchor : Fin 3 → Fin I → Fin 3 → ℝ)
    (anchors : ∀ p a i j, anchor a i j = 0 + offset p a i j)
    (partition : ∀ p a, ∑ i, weight p a i = 1)
    (first : ∀ p a j, ∑ i, weight p a i * offset p a i j = 0)
    (second : ∀ p a j k, ∑ i, weight p a i * offset p a i j * offset p a i k = cubicMoment a j k) :
    (∑ i, anchor 1 i 0 * depositedMacPureMomentum (fun _ => 2) weight (fun _ _ => 0)
      (fun _ => distinctRows) offset 1 i) -
    (∑ i, anchor 0 i 1 * depositedMacPureMomentum (fun _ => 2) weight (fun _ _ => 0)
      (fun _ => distinctRows) offset 0 i) = -9 / 2 := by
  have conserved := mac_deposit_conserves_row_moments (fun _ => 2) (by intro p; norm_num)
    weight (fun _ _ => 0) (fun _ _ => 0) (fun _ => distinctRows) (fun _ => cubicMoment)
    offset anchor anchors partition first second
  convert conserved.2.2 0 1 using 1
  norm_num [Fin.sum_univ_succ, distinctRows, cubicMoment]

-- Swapping D_x and D_y changes the z angular moment from -9/2 to -6.
example :
    (2 * ((∑ l, distinctRows 1 l * cubicMoment 1 l 0) -
      ∑ l, distinctRows 0 l * cubicMoment 0 l 1) : ℝ) = -9 / 2 ∧
    (2 * ((∑ l, distinctRows 1 l * cubicMoment 0 l 0) -
      ∑ l, distinctRows 0 l * cubicMoment 1 l 1) : ℝ) = -6 ∧
    (-9 / 2 : ℝ) ≠ -6 := by
  norm_num [Fin.sum_univ_succ, distinctRows, cubicMoment]

end Fluidity

namespace Fluidity

-- These typed consumers require component conclusions, not merely exported names.
example {n m : ℕ} (g : PressureGraph n m) (p : Fin n → ℝ)
    (kernel : ∀ i, pressureAction g.incidence g.conductance g.anchor p i = 0)
    (i j a : Fin n) (same : g.Reachable i j) (anchored : 0 < g.anchor a) :
    p i = p j ∧ p a = 0 := by
  have result := (pressure_kernel_eq_component_constants g p).mp kernel
  exact ⟨result.1 i j same, result.2 a anchored⟩

-- Disconnected components may carry different constants. Pressure vanishes on anchored components.
example {n m : ℕ} (g : PressureGraph n m) (p : Fin n → ℝ)
    (constant : ∀ i j, g.Reachable i j → p i = p j)
    (anchored : ∀ i, 0 < g.anchor i → p i = 0) :
    ∀ i, pressureAction g.incidence g.conductance g.anchor p i = 0 := by
  exact (pressure_kernel_eq_component_constants g p).mpr ⟨constant, anchored⟩

example {n m : ℕ} (g : PressureGraph n m) (pin : Fin n → Prop)
    (covers : ∀ i, g.Unanchored i → ∃ j, g.Reachable i j ∧ pin j)
    (p : Fin n → ℝ) (zeroPins : ∀ i, pin i → p i = 0) (nonzero : p ≠ 0) :
    0 < pressureQuadraticForm g.incidence g.conductance g.anchor p := by
  exact pressure_gauge_positive_definite g pin covers p zeroPins nonzero

example {n m : ℕ} (g : PressureGraph n m) (b : Fin n → ℝ)
    (sums : ∀ i, g.Unanchored i → ∑ j, (if g.Reachable i j then b j else 0) = 0) :
    ∃ p, ∀ i, pressureAction g.incidence g.conductance g.anchor p i = b i := by
  exact (pressure_solvable_iff_component_compatible g b).mpr sums

example {n m : ℕ} (g : PressureGraph n m) (b p : Fin n → ℝ)
    (solves : ∀ i, pressureAction g.incidence g.conductance g.anchor p i = b i)
    (i : Fin n) (unanchored : g.Unanchored i) :
    ∑ j, (if g.Reachable i j then b j else 0) = 0 := by
  exact (pressure_solvable_iff_component_compatible g b).mp ⟨p, solves⟩ i unanchored

-- No solve or residual-smallness premise is permitted here.
example {n m : ℕ} (links : PressureLinks n m) (candidate : Fin n → ℝ) (i : Fin n) :
    links.defect candidate i = -links.residual candidate i := by
  exact pressure_corrected_flux_defect_eq_neg_residual links candidate i

end Fluidity

namespace Fluidity

-- A closed two-cell component and an independent positively anchored singleton.
private def disconnectedPressureGraph : PressureGraph 3 1 where
  tail := fun _ => 0
  head := fun _ => 1
  conductance := fun _ => 2
  anchor := ![0, 0, 3]
  conductance_nonneg := by intro f; norm_num
  anchor_nonneg := by intro i; fin_cases i <;> norm_num

example (componentPressure : ℝ) :
    disconnectedPressureGraph.ComponentConstants ![componentPressure, componentPressure, 0] := by
  apply (pressure_kernel_eq_component_constants disconnectedPressureGraph _).mp
  intro i
  fin_cases i <;>
    simp [pressureAction, PressureGraph.incidence, disconnectedPressureGraph, Fin.sum_univ_succ]

-- The anchored singleton RHS may have nonzero sum; only the closed pair must balance.
example : disconnectedPressureGraph.Compatible ![-2, 2, 7] := by
  apply (pressure_solvable_iff_component_compatible disconnectedPressureGraph _).mp
  refine ⟨![0, 1, 7 / 3], ?_⟩
  intro i
  fin_cases i <;>
    norm_num [pressureAction, PressureGraph.incidence, disconnectedPressureGraph, Fin.sum_univ_succ]

private noncomputable def defectConsumerLinks : PressureLinks 1 1 where
  incidence := fun _ _ => 1
  openArea := fun _ => 1 / 2
  fullArea := fun _ => 1
  velocityScale := fun _ => 2
  predictedVelocity := fun _ => 1
  wallVelocity := fun _ => 2
  lift := fun _ => -3
  compliance := fun _ => 1 / 2
  source := fun _ => 1
  previousPressure := fun _ => 4

-- Unsolved p=5 has original residual -3 and independently accumulated defect +3.
example : defectConsumerLinks.residual (fun _ => 5) 0 = -3 ∧
    defectConsumerLinks.defect (fun _ => 5) 0 = 3 := by
  constructor
  · norm_num [PressureLinks.residual, PressureLinks.rhs, PressureLinks.wallFlux,
      PressureLinks.conductance, pressureAction, defectConsumerLinks, Fin.sum_univ_succ]
  · rw [pressure_corrected_flux_defect_eq_neg_residual]
    norm_num [PressureLinks.residual, PressureLinks.rhs, PressureLinks.wallFlux,
      PressureLinks.conductance, pressureAction, defectConsumerLinks, Fin.sum_univ_succ]

-- Change only correction's velocity scale: defect becomes 2, original -residual stays 3.
example : ({ defectConsumerLinks with velocityScale := fun _ => 1 } : PressureLinks 1 1).defect
      (fun _ => 5) 0 ≠ -defectConsumerLinks.residual (fun _ => 5) 0 := by
  norm_num [PressureLinks.defect, PressureLinks.correctedFlux, PressureLinks.residual,
    PressureLinks.rhs, PressureLinks.wallFlux, PressureLinks.conductance,
    pressureAction, defectConsumerLinks, Fin.sum_univ_succ]

end Fluidity

namespace Fluidity

-- These consumers check the full published state and explicit trial contracts.
example {domains : LiquidDomainTypes} {Diagnostics : Type}
    (committed : LiquidCommittedState domains) (diagnostics : Diagnostics) (retryDuration : Option ℝ) :
    finishLiquidAttempt committed ⟨.rejected retryDuration, diagnostics⟩ = committed :=
  liquid_reject_preserves_committed_state committed diagnostics retryDuration

example {domains : LiquidDomainTypes} {Diagnostics : Type}
    (committed : LiquidCommittedState domains) (attempts : List (LiquidAttempt domains Diagnostics))
    (allRejected : ∀ attempt ∈ attempts, attempt.outcome.IsRejected) :
    attempts.foldl finishLiquidAttempt committed = committed :=
  liquid_finite_rejects_preserve_committed_state committed attempts allRejected

example {domains : LiquidDomainTypes} {Diagnostics : Type}
    (committed : LiquidCommittedState domains) (trialFields : LiquidPersistentFields domains)
    (diagnostics : Diagnostics) (acceptedDuration : ℝ) (completesFrame : Bool) :
    let accepted := finishLiquidAttempt committed
      ⟨.accepted trialFields acceptedDuration completesFrame, diagnostics⟩
    accepted.fields = trialFields ∧
      accepted.clock.time = committed.clock.time + acceptedDuration ∧
      accepted.clock.acceptedStepOrdinal = committed.clock.acceptedStepOrdinal + 1 ∧
      accepted.clock.epoch = committed.clock.epoch + 1 ∧
      accepted.clock.frameOrdinal = (if completesFrame then committed.clock.frameOrdinal + 1
        else committed.clock.frameOrdinal) ∧
      accepted.clock.substepOrdinal = (if completesFrame then 0 else committed.clock.substepOrdinal + 1) :=
  liquid_accept_advances_clock_once committed trialFields diagnostics acceptedDuration completesFrame

example {domains : LiquidDomainTypes} {Diagnostics : Type}
    (runTrial : LiquidCommittedState domains → LiquidAcceptedStepKey → LiquidAttemptOutcome domains)
    (firstInput replayInput : LiquidCommittedState domains) (firstKey replayKey : LiquidAcceptedStepKey)
    (firstAttempt replayAttempt : LiquidAttempt domains Diagnostics)
    (sameInput : firstInput = replayInput) (sameKey : firstKey = replayKey)
    (firstResult : firstAttempt.outcome = runTrial firstInput firstKey)
    (replayedResult : replayAttempt.outcome = runTrial replayInput replayKey) :
    firstAttempt.outcome = replayAttempt.outcome ∧
      finishLiquidAttempt firstInput firstAttempt = finishLiquidAttempt replayInput replayAttempt :=
  liquid_same_state_key_replays_trial
    runTrial firstInput replayInput firstKey replayKey firstAttempt replayAttempt
    sameInput sameKey firstResult replayedResult

example {domains : LiquidDomainTypes} (inputSnapshot : LiquidCommittedState domains)
    (adapt : LiquidCommittedState domains → LiquidPersistentFields domains) :
    (proposeLiquidAdaptation inputSnapshot adapt).1 = inputSnapshot ∧
      (proposeLiquidAdaptation inputSnapshot adapt).2 = adapt inputSnapshot :=
  liquid_adaptation_preserves_input_snapshot inputSnapshot adapt

end Fluidity
