import Fluidity.Resampling
import Mathlib.LinearAlgebra.Matrix.PosDef

open scoped BigOperators
open Matrix

noncomputable section

namespace Fluidity

private def rowBilinear {d : ℕ} (left right : Fin d → ℝ) (moment : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  resamplingDot (resamplingRowMoment left moment) right

private theorem row_bilinear_symm {d : ℕ} (left right : Fin d → ℝ)
    (moment : Matrix (Fin d) (Fin d) ℝ) (symmetric : moment.IsHermitian) :
    rowBilinear left right moment = rowBilinear right left moment := by
  unfold rowBilinear resamplingDot resamplingRowMoment
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro k _
  have transpose : moment j k = moment k j := by simpa using symmetric.apply k j
  rw [transpose]
  ring

private theorem row_quadratic_nonneg {d : ℕ} (row : Fin d → ℝ)
    (moment : Matrix (Fin d) (Fin d) ℝ) (psd : moment.PosSemidef) :
    0 ≤ resamplingQuadratic row moment := by
  have same : resamplingQuadratic row moment = star row ⬝ᵥ (moment *ᵥ row) := by
    change (∑ j, (∑ k, row k * moment k j) * row j) =
      ∑ j, star (row j) * (∑ k, moment j k * row k)
    simp_rw [Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j _
    apply Finset.sum_congr rfl
    intro k _
    rw [star_trivial]
    ring
  rw [same]
  exact psd.dotProduct_mulVec_nonneg row

private theorem row_square_difference {d : ℕ} (left right : Fin d → ℝ)
    (moment : Matrix (Fin d) (Fin d) ℝ) (symmetric : moment.IsHermitian) :
    resamplingQuadratic (fun j => left j - right j) moment =
      resamplingQuadratic left moment + resamplingQuadratic right moment -
        2 * rowBilinear left right moment := by
  have expand : resamplingQuadratic (fun j => left j - right j) moment =
      resamplingQuadratic left moment - rowBilinear left right moment -
        rowBilinear right left moment + resamplingQuadratic right moment := by
    simp only [resamplingQuadratic, rowBilinear, resamplingDot, resamplingRowMoment,
      sub_mul, mul_sub, Finset.sum_sub_distrib]
    ring
  rw [expand, row_bilinear_symm right left moment symmetric]
  ring

/-- Positive mass weighted center, for either spatial position or velocity. -/
def mergeCenter {n d : ℕ} (mass : Fin n → ℝ) (sample : Fin n → Fin d → ℝ) (j : Fin d) : ℝ :=
  (∑ i, mass i * sample i j) / ∑ i, mass i

/-- F4's centered cross moment of one velocity row. -/
def mergeCrossMoment {n d : ℕ} (mass : Fin n → ℝ) (offset : Fin n → Fin d → ℝ)
    (relativeVelocity : Fin n → ℝ) (affine : Fin n → Fin d → ℝ)
    (child : Fin n → Matrix (Fin d) (Fin d) ℝ) (j : Fin d) : ℝ :=
  ∑ i, mass i * (relativeVelocity i * offset i j + resamplingRowMoment (affine i) (child i) j)

/-- F4's mass-weighted spatial covariance of one velocity row. -/
def mergeCovariance {n d : ℕ} (mass : Fin n → ℝ) (offset : Fin n → Fin d → ℝ)
    (child : Fin n → Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  fun j k => ∑ i, mass i * (offset i j * offset i k + child i j k)

/-- The target affine row is h D_target⁻¹ / M. -/
def mergeAffineRow {d : ℕ} (totalMass : ℝ) (crossMoment : Fin d → ℝ)
    (target : Matrix (Fin d) (Fin d) ℝ) : Fin d → ℝ :=
  totalMass⁻¹ • (crossMoment ᵥ* target⁻¹)

/-- The target's mass-weighted affine spatial moment is exactly the child
cross moment. Positive definiteness ensures the inverse exists. -/
theorem merge_affine_row_matches_cross_moment {d : ℕ} (totalMass : ℝ)
    (positiveMass : 0 < totalMass) (crossMoment : Fin d → ℝ)
    (target : Matrix (Fin d) (Fin d) ℝ) (positiveTarget : target.PosDef) (j : Fin d) :
    totalMass * resamplingRowMoment (mergeAffineRow totalMass crossMoment target) target j =
      crossMoment j := by
  have invDet := (Matrix.isUnit_iff_isUnit_det target).mp positiveTarget.isUnit
  change totalMass * ((totalMass⁻¹ • (crossMoment ᵥ* target⁻¹)) ᵥ* target) j = _
  rw [Matrix.smul_vecMul, Matrix.vecMul_vecMul, Matrix.nonsing_inv_mul _ invDet, Matrix.vecMul_one]
  simp [Pi.smul_apply, smul_eq_mul, positiveMass.ne']

#print axioms merge_affine_row_matches_cross_moment

private theorem dot_cross_moment {n d : ℕ} (mass : Fin n → ℝ) (offset : Fin n → Fin d → ℝ)
    (velocity : Fin n → ℝ) (affine : Fin n → Fin d → ℝ)
    (child : Fin n → Matrix (Fin d) (Fin d) ℝ) (row : Fin d → ℝ) :
    resamplingDot (mergeCrossMoment mass offset velocity affine child) row =
      ∑ i, mass i * (velocity i * resamplingDot row (offset i) + rowBilinear (affine i) row (child i)) := by
  unfold resamplingDot mergeCrossMoment rowBilinear resamplingDot
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  calc
    _ = ∑ j, mass i * (velocity i * (row j * offset i j) +
        resamplingRowMoment (affine i) (child i) j * row j) := by
      apply Finset.sum_congr rfl
      intro j _
      ring
    _ = _ := by
      rw [← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.mul_sum]

private theorem quadratic_covariance {n d : ℕ} (mass : Fin n → ℝ) (offset : Fin n → Fin d → ℝ)
    (child : Fin n → Matrix (Fin d) (Fin d) ℝ) (row : Fin d → ℝ) :
    resamplingQuadratic row (mergeCovariance mass offset child) =
      ∑ i, mass i * (resamplingDot row (offset i) ^ 2 + resamplingQuadratic row (child i)) := by
  have entry (j : Fin d) :
      resamplingRowMoment row (mergeCovariance mass offset child) j =
        ∑ i, mass i * (resamplingDot row (offset i) * offset i j +
          resamplingRowMoment row (child i) j) := by
    change (∑ k, row k * ∑ i, mass i * (offset i k * offset i j + child i k j)) =
      ∑ i, mass i * ((∑ k, row k * offset i k) * offset i j + ∑ k, row k * child i k j)
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    simp_rw [Finset.sum_mul, ← Finset.sum_add_distrib, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k _
    ring
  unfold resamplingQuadratic resamplingDot
  simp_rw [entry, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  rw [pow_two, Finset.mul_sum, ← Finset.sum_add_distrib, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  unfold resamplingDot
  ring

private theorem quadratic_gap {d : ℕ} (totalMass : ℝ)
    (target covariance : Matrix (Fin d) (Fin d) ℝ) (row : Fin d → ℝ) :
    resamplingQuadratic row (totalMass • target - covariance) =
      totalMass * resamplingQuadratic row target - resamplingQuadratic row covariance := by
  have entry (j : Fin d) : resamplingRowMoment row (totalMass • target - covariance) j =
      totalMass * resamplingRowMoment row target j - resamplingRowMoment row covariance j := by
    simp only [resamplingRowMoment, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul,
      mul_sub, Finset.sum_sub_distrib, Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro k _
    ring
  simp only [resamplingQuadratic, resamplingDot, entry, sub_mul, Finset.sum_sub_distrib, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Rowwise F4 energy bound. Completing squares at the target row uses the
same block Gram positivity as a Schur complement. The PSD covariance gap
then bounds the target affine energy by the child's centered kinetic plus
affine energy, with no pseudoinverse. -/
theorem merge_covariance_gate_bounds_row_energy {n d : ℕ} (mass : Fin n → ℝ)
    (massNonnegative : ∀ i, 0 ≤ mass i) (totalMass : ℝ) (positiveMass : 0 < totalMass)
    (offset : Fin n → Fin d → ℝ) (relativeVelocity : Fin n → ℝ)
    (affine : Fin n → Fin d → ℝ) (child : Fin n → Matrix (Fin d) (Fin d) ℝ)
    (positiveChild : ∀ i, (child i).PosDef)
    (target : Matrix (Fin d) (Fin d) ℝ) (positiveTarget : target.PosDef)
    (gate : (totalMass • target - mergeCovariance mass offset child).PosSemidef) :
    let row := mergeAffineRow totalMass (mergeCrossMoment mass offset relativeVelocity affine child) target
    totalMass * resamplingQuadratic row target ≤
      ∑ i, mass i * (relativeVelocity i ^ 2 + resamplingQuadratic (affine i) (child i)) := by
  intro row
  have cross : totalMass * resamplingQuadratic row target =
      resamplingDot (mergeCrossMoment mass offset relativeVelocity affine child) row := by
    unfold resamplingQuadratic resamplingDot
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [← mul_assoc, merge_affine_row_matches_cross_moment totalMass positiveMass _ target positiveTarget]
  have covarianceBound := row_quadratic_nonneg row _ gate
  rw [quadratic_gap] at covarianceBound
  have term (i : Fin n) :
      mass i * (2 * (relativeVelocity i * resamplingDot row (offset i) +
        rowBilinear (affine i) row (child i))) ≤
      mass i * ((relativeVelocity i ^ 2 + resamplingQuadratic (affine i) (child i)) +
        (resamplingDot row (offset i) ^ 2 + resamplingQuadratic row (child i))) := by
    apply mul_le_mul_of_nonneg_left _ (massNonnegative i)
    have affineSquare := row_quadratic_nonneg (fun j => affine i j - row j) _
      (positiveChild i).posSemidef
    rw [row_square_difference _ _ _ (positiveChild i).isHermitian] at affineSquare
    nlinarith [sq_nonneg (relativeVelocity i - resamplingDot row (offset i))]
  have gram := Finset.sum_le_sum (s := Finset.univ) fun i _ => term i
  simp_rw [mul_comm (mass _) (2 * _), mul_assoc, mul_comm _ (mass _), ← Finset.mul_sum,
    mul_add, Finset.sum_add_distrib] at gram
  rw [dot_cross_moment] at cross
  rw [quadratic_covariance] at covarianceBound
  simp only [mul_add, Finset.sum_add_distrib] at cross covarianceBound ⊢
  linarith

#print axioms merge_covariance_gate_bounds_row_energy

private theorem merge_center_moment {n d : ℕ} (mass : Fin n → ℝ) (sample : Fin n → Fin d → ℝ)
    (positiveMass : 0 < ∑ i, mass i) (j : Fin d) :
    (∑ i, mass i) * mergeCenter mass sample j = ∑ i, mass i * sample i j := by
  unfold mergeCenter
  field_simp [positiveMass.ne']

private theorem merge_centered_sum {n d : ℕ} (mass : Fin n → ℝ) (sample : Fin n → Fin d → ℝ)
    (positiveMass : 0 < ∑ i, mass i) (j : Fin d) :
    ∑ i, mass i * (sample i j - mergeCenter mass sample j) = 0 := by
  simp_rw [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul,
    merge_center_moment mass sample positiveMass]
  simp

private theorem merge_velocity_square {n d : ℕ} (mass : Fin n → ℝ) (velocity : Fin n → Fin d → ℝ)
    (positiveMass : 0 < ∑ i, mass i) (a : Fin d) :
    (∑ i, mass i * velocity i a ^ 2) =
      (∑ i, mass i) * mergeCenter mass velocity a ^ 2 +
        ∑ i, mass i * (velocity i a - mergeCenter mass velocity a) ^ 2 := by
  let center := mergeCenter mass velocity a
  have expand (i : Fin n) : velocity i a ^ 2 =
      center ^ 2 + (2 * center * (velocity i a - center) + (velocity i a - center) ^ 2) := by ring
  conv_lhs => simp only [expand]
  simp_rw [mul_add, Finset.sum_add_distrib, ← Finset.sum_mul]
  have cancel : (∑ i, mass i * (2 * center * (velocity i a - center))) = 0 := by
    calc
      _ = (2 * center) * ∑ i, mass i * (velocity i a - center) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = 0 := by rw [merge_centered_sum mass velocity positiveMass, mul_zero]
  rw [cancel, zero_add]

/-- F4 for a nonempty collection of positive-mass children with positive
definite row supports. Every target row must pass its own PSD covariance
gate. The target preserves both first moments and the unsymmetrized F2
spatial moments, hence their angular differences. -/
theorem merge_preserves_moments_and_bounds_energy {n d : ℕ} (nonempty : 0 < n)
    (mass : Fin n → ℝ) (massPositive : ∀ i, 0 < mass i)
    (position velocity : Fin n → Fin d → ℝ) (affine : Fin n → Fin d → Fin d → ℝ)
    (child : Fin n → Fin d → Matrix (Fin d) (Fin d) ℝ)
    (target : Fin d → Matrix (Fin d) (Fin d) ℝ)
    (childPositive : ∀ i a, (child i a).PosDef) (targetPositive : ∀ a, (target a).PosDef)
    (gate : ∀ a, ((∑ i, mass i) • target a - mergeCovariance mass
      (fun i j => position i j - mergeCenter mass position j) (fun i => child i a)).PosSemidef) :
    let totalMass := ∑ i, mass i
    let center := mergeCenter mass position
    let meanVelocity := mergeCenter mass velocity
    let offset (i : Fin n) (j : Fin d) := position i j - center j
    let relativeVelocity (i : Fin n) (a : Fin d) := velocity i a - meanVelocity a
    let targetAffine (a : Fin d) := mergeAffineRow totalMass
      (mergeCrossMoment mass offset (fun i => relativeVelocity i a)
        (fun i => affine i a) (fun i => child i a)) (target a)
    (∀ j, totalMass * center j = ∑ i, mass i * position i j) ∧
    (∀ a, totalMass * meanVelocity a = ∑ i, mass i * velocity i a) ∧
    (∀ a j, resamplingSpatialMomentum totalMass center meanVelocity targetAffine target a j =
      ∑ i, resamplingSpatialMomentum (mass i) (position i) (velocity i) (affine i) (child i) a j) ∧
    resamplingEnergy totalMass meanVelocity targetAffine target ≤
      ∑ i, resamplingEnergy (mass i) (velocity i) (affine i) (child i) := by
  intro totalMass center meanVelocity offset relativeVelocity targetAffine
  have : NeZero n := ⟨nonempty.ne'⟩
  have positive : 0 < totalMass :=
    Finset.sum_pos (fun i _ => massPositive i) Finset.univ_nonempty
  refine ⟨merge_center_moment mass position positive, merge_center_moment mass velocity positive, ?_, ?_⟩
  · intro a j
    have expand (i : Fin n) : position i j * velocity i a =
        center j * velocity i a + (meanVelocity a * offset i j + relativeVelocity i a * offset i j) := by
      dsimp [offset, relativeVelocity]
      ring
    have cancel : (∑ i, mass i * (meanVelocity a * offset i j)) = 0 := by
      calc
        _ = meanVelocity a * ∑ i, mass i * offset i j := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro i _
          ring
        _ = 0 := by rw [merge_centered_sum mass position positive, mul_zero]
    have centerProduct :
        (∑ i, mass i * (center j * velocity i a)) = totalMass * (center j * meanVelocity a) := by
      calc
        _ = center j * ∑ i, mass i * velocity i a := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro i _
          ring
        _ = _ := by
          rw [← merge_center_moment mass velocity positive]
          change center j * (totalMass * meanVelocity a) = _
          ring
    unfold resamplingSpatialMomentum
    rw [mul_add]
    rw [merge_affine_row_matches_cross_moment totalMass positive _ _ (targetPositive a)]
    unfold mergeCrossMoment
    simp_rw [expand, mul_add, Finset.sum_add_distrib, cancel, centerProduct, zero_add]
    ring
  · have row (a : Fin d) :
        totalMass * (meanVelocity a ^ 2 + resamplingQuadratic (targetAffine a) (target a)) ≤
          ∑ i, mass i * (velocity i a ^ 2 + resamplingQuadratic (affine i a) (child i a)) := by
      have bound := merge_covariance_gate_bounds_row_energy mass (fun i => (massPositive i).le)
        totalMass positive offset (fun i => relativeVelocity i a) (fun i => affine i a)
        (fun i => child i a) (fun i => childPositive i a) (target a) (targetPositive a) (gate a)
      change totalMass * resamplingQuadratic (targetAffine a) (target a) ≤
        ∑ i, mass i * (relativeVelocity i a ^ 2 + resamplingQuadratic (affine i a) (child i a)) at bound
      simp only [mul_add, Finset.sum_add_distrib] at bound ⊢
      rw [merge_velocity_square mass velocity positive]
      change totalMass * meanVelocity a ^ 2 + _ ≤
        (totalMass * meanVelocity a ^ 2 + ∑ i, mass i * relativeVelocity i a ^ 2) + _
      linarith
    have summed := Finset.sum_le_sum (s := Finset.univ) fun a _ => row a
    rw [← Finset.mul_sum, Finset.sum_comm] at summed
    simp_rw [← Finset.mul_sum] at summed
    unfold resamplingEnergy
    have scaled := mul_le_mul_of_nonneg_right summed (by norm_num : (0 : ℝ) ≤ 2⁻¹)
    simp only [div_eq_mul_inv]
    convert scaled using 1
    · ring
    · rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i _
      ring

#print axioms merge_preserves_moments_and_bounds_energy

/-- F4's target reference volume, volume ratio and constituent fractions
preserve each corresponding inventory. The fractions use current volumes
`J * V0`, even when mass and reference-volume fractions differ. -/
theorem merge_preserves_volume_inventories {n constituents : ℕ}
    (referenceVolume volumeRatio : Fin n → ℝ) (fraction : Fin n → Fin constituents → ℝ)
    (positiveReference : 0 < ∑ i, referenceVolume i)
    (positiveCurrent : 0 < ∑ i, volumeRatio i * referenceVolume i) :
    let targetReference := ∑ i, referenceVolume i
    let current := ∑ i, volumeRatio i * referenceVolume i
    let targetRatio := current / targetReference
    let targetFraction (k : Fin constituents) :=
      (∑ i, fraction i k * (volumeRatio i * referenceVolume i)) / current
    targetRatio * targetReference = current ∧
    (∀ k, targetFraction k * (targetRatio * targetReference) =
      ∑ i, fraction i k * (volumeRatio i * referenceVolume i)) := by
  intro targetReference current targetRatio targetFraction
  have currentBalance : targetRatio * targetReference = current := by
    dsimp [targetRatio]
    exact div_mul_cancel₀ current positiveReference.ne'
  refine ⟨currentBalance, ?_⟩
  intro k
  rw [currentBalance]
  dsimp [targetFraction]
  exact div_mul_cancel₀ _ positiveCurrent.ne'

#print axioms merge_preserves_volume_inventories


end Fluidity
