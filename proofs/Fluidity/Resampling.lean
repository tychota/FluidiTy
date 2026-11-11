import Fluidity.UniformTransfer

open scoped BigOperators

noncomputable section

namespace Fluidity

/-- Component of an affine velocity increment at a spatial offset. -/
def resamplingDot {d : ℕ} (row offset : Fin d → ℝ) : ℝ :=
  ∑ j, row j * offset j

/-- One affine row's spatial moment. Both indices refer to spatial coordinates. -/
def resamplingRowMoment {d : ℕ} (row : Fin d → ℝ) (moment : Fin d → Fin d → ℝ)
    (j : Fin d) : ℝ :=
  ∑ k, row k * moment k j

/-- The row's affine squared speed in a support with second moment `moment`. -/
def resamplingQuadratic {d : ℕ} (row : Fin d → ℝ) (moment : Fin d → Fin d → ℝ) : ℝ :=
  resamplingDot (resamplingRowMoment row moment) row

/-- Particle kinetic-plus-affine energy, using each velocity row's own support moment. -/
def resamplingEnergy {d : ℕ} (mass : ℝ) (velocity : Fin d → ℝ)
    (affine : Fin d → Fin d → ℝ) (moment : Fin d → Fin d → Fin d → ℝ) : ℝ :=
  mass / 2 * ∑ a, (velocity a ^ 2 + resamplingQuadratic (affine a) (moment a))

/-- F2's unsymmetrized spatial momentum for velocity component `a` and spatial coordinate `j`. -/
def resamplingSpatialMomentum {d : ℕ} (mass : ℝ) (position velocity : Fin d → ℝ)
    (affine : Fin d → Fin d → ℝ) (moment : Fin d → Fin d → Fin d → ℝ) (a j : Fin d) : ℝ :=
  mass * (position j * velocity a + resamplingRowMoment (affine a) (moment a) j)

private theorem split_increment_sum {n d : ℕ} (offset : Fin n → Fin d → ℝ)
    (zeroMean : ∀ j, ∑ i, offset i j = 0) (row : Fin d → ℝ) :
    ∑ i, resamplingDot row (offset i) = 0 := by
  unfold resamplingDot
  rw [Finset.sum_comm]
  simp [← Finset.mul_sum, zeroMean]

private theorem split_increment_moment {n d : ℕ} (offset : Fin n → Fin d → ℝ)
    (covariance : Fin d → Fin d → ℝ)
    (second : ∀ k j, ∑ i, offset i k * offset i j = n * covariance k j)
    (row : Fin d → ℝ) (j : Fin d) :
    (∑ i, resamplingDot row (offset i) * offset i j) =
      n * resamplingRowMoment row covariance j := by
  unfold resamplingDot resamplingRowMoment
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  simp_rw [mul_assoc, ← Finset.mul_sum, second]
  ring

private theorem split_increment_square {n d : ℕ} (offset : Fin n → Fin d → ℝ)
    (covariance : Fin d → Fin d → ℝ)
    (second : ∀ k j, ∑ i, offset i k * offset i j = n * covariance k j)
    (row : Fin d → ℝ) :
    (∑ i, resamplingDot row (offset i) ^ 2) = n * resamplingQuadratic row covariance := by
  simp_rw [pow_two]
  calc
    _ = ∑ i, ∑ j, (resamplingDot row (offset i) * offset i j) * row j := by
      apply Finset.sum_congr rfl
      intro i _
      conv_lhs => rhs; unfold resamplingDot
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      ring
    _ = _ := by
      rw [Finset.sum_comm]
      simp_rw [← Finset.sum_mul, split_increment_moment offset covariance second]
      unfold resamplingQuadratic resamplingDot
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      ring

private theorem row_moment_deficit {d : ℕ} (row : Fin d → ℝ)
    (parent child covariance : Fin d → Fin d → ℝ) (j : Fin d) :
    resamplingRowMoment row (fun k l => parent k l - child k l - covariance k l) j =
      resamplingRowMoment row parent j - resamplingRowMoment row child j -
        resamplingRowMoment row covariance j := by
  simp [resamplingRowMoment, mul_sub, Finset.sum_sub_distrib]

private theorem quadratic_deficit {d : ℕ} (row : Fin d → ℝ)
    (parent child covariance : Fin d → Fin d → ℝ) :
    resamplingQuadratic row (fun k l => parent k l - child k l - covariance k l) =
      resamplingQuadratic row parent - resamplingQuadratic row child -
        resamplingQuadratic row covariance := by
  simp [resamplingQuadratic, resamplingDot, row_moment_deficit, sub_mul, Finset.sum_sub_distrib]

/-- F3's inventory, momentum, energy and rowwise spatial-momentum identities.
The `n` equal children have positions `x + offset`, velocities `v + C offset`,
the same affine matrix and volume ratio as the parent, and common rowwise support
moments. The offsets have zero mean and the stated covariance. Child placement
and unmasked support admissibility are runtime predicates. -/
theorem symmetric_split_moments {n d : ℕ} (count : 0 < n)
    (mass referenceVolume volumeRatio : ℝ) (x v : Fin d → ℝ)
    (affine : Fin d → Fin d → ℝ) (parent child : Fin d → Fin d → Fin d → ℝ)
    (offset : Fin n → Fin d → ℝ) (covariance : Fin d → Fin d → ℝ)
    (zeroMean : ∀ j, ∑ i, offset i j = 0)
    (second : ∀ k j, ∑ i, offset i k * offset i j = n * covariance k j) :
    let velocity (i : Fin n) (a : Fin d) := v a + resamplingDot (affine a) (offset i)
    let deficit (a k j : Fin d) := parent a k j - child a k j - covariance k j
    (∑ _ : Fin n, mass / n) = mass ∧
      (∑ _ : Fin n, referenceVolume / n) = referenceVolume ∧
      (∑ _ : Fin n, volumeRatio * (referenceVolume / n)) = volumeRatio * referenceVolume ∧
      (∀ a, (∑ i, (mass / n) * velocity i a) = mass * v a) ∧
      (∑ i, resamplingEnergy (mass / n) (velocity i) affine child) =
        resamplingEnergy mass v affine parent -
          mass / 2 * ∑ a, resamplingQuadratic (affine a) (deficit a) ∧
      (∀ a j, (∑ i, resamplingSpatialMomentum (mass / n) (fun k => x k + offset i k)
          (velocity i) affine child a j) =
        resamplingSpatialMomentum mass x v affine parent a j -
          mass * resamplingRowMoment (affine a) (deficit a) j) := by
  intro velocity deficit
  have countNonzero : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr count.ne'
  have scaled (value : ℝ) : (n : ℝ) * (value / n) = value := by
    field_simp
  refine ⟨by simpa using scaled mass, by simpa using scaled referenceVolume, ?_, ?_, ?_, ?_⟩
  · rw [← Finset.mul_sum]
    congr 1
    simpa using scaled referenceVolume
  · intro a
    rw [← Finset.mul_sum]
    simp only [velocity, Finset.sum_add_distrib, split_increment_sum offset zeroMean,
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, add_zero]
    field_simp
  · have rowEnergy (a : Fin d) :
        (∑ i, (velocity i a ^ 2 + resamplingQuadratic (affine a) (child a))) =
          n * (v a ^ 2 + resamplingQuadratic (affine a) (covariance) +
            resamplingQuadratic (affine a) (child a)) := by
      have expand (i : Fin n) :
          velocity i a ^ 2 = v a ^ 2 +
            (2 * v a * resamplingDot (affine a) (offset i) +
              resamplingDot (affine a) (offset i) ^ 2) := by
        dsimp [velocity]
        ring
      simp only [expand, Finset.sum_add_distrib, ← Finset.mul_sum,
        split_increment_sum offset zeroMean, split_increment_square offset covariance second,
        mul_zero, zero_add, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      ring
    unfold resamplingEnergy
    rw [← Finset.mul_sum, Finset.sum_comm]
    simp only [deficit, quadratic_deficit]
    simp_rw [rowEnergy, ← Finset.mul_sum]
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib]
    field_simp
    ring
  · intro a j
    have expand (i : Fin n) :
        (x j + offset i j) * velocity i a =
          x j * v a + (x j * resamplingDot (affine a) (offset i) +
            v a * offset i j) + resamplingDot (affine a) (offset i) * offset i j := by
      dsimp [velocity]
      ring
    unfold resamplingSpatialMomentum
    rw [← Finset.mul_sum]
    simp only [expand, Finset.sum_add_distrib, ← Finset.mul_sum,
      split_increment_sum offset zeroMean, split_increment_moment offset covariance second,
      zeroMean, mul_zero, add_zero, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    simp only [deficit, row_moment_deficit]
    field_simp
    ring

#print axioms symmetric_split_moments

/-- The F2 angular component for a pair of axes is the antisymmetric part of
the spatial momentum. A split deficit confined to the row's normal coordinate
leaves every off-diagonal entry and therefore every angular component unchanged. -/
theorem split_normal_deficit_preserves_angular_momentum {n d : ℕ}
    (before : Fin d → Fin d → ℝ) (after : Fin n → Fin d → Fin d → ℝ)
    (mass : ℝ) (affine : Fin d → Fin d → ℝ) (deficit : Fin d → Fin d → Fin d → ℝ)
    (spatialBalance : ∀ a j, (∑ i, after i a j) =
      before a j - mass * resamplingRowMoment (affine a) (deficit a) j)
    (normalOnly : ∀ a k j, j ≠ a → deficit a k j = 0) (j k : Fin d) :
    (∑ i, (after i k j - after i j k)) = before k j - before j k := by
  by_cases same : j = k
  · subst k
    simp
  · simp [Finset.sum_sub_distrib, spatialBalance, resamplingRowMoment, normalOnly, same, Ne.symm same]

#print axioms split_normal_deficit_preserves_angular_momentum

/-- Uniform quadratic support has the same variance in each spatial direction. -/
def quadraticSplitMoment (h : ℝ) (k j : Fin 3) : ℝ :=
  if k = j then h ^ 2 / 4 else 0

/-- Uniform cubic-normal/quadratic-transverse MAC support has variance h²/3
on the row's normal and h²/4 on the other two spatial axes. -/
def cubicMacSplitMoment (h : ℝ) (a k j : Fin 3) : ℝ :=
  if k = j then (if k = a then h ^ 2 / 3 else h ^ 2 / 4) else 0

/-- The 2:1 eight-child split's isotropic spatial covariance in parent units. -/
def splitCornerCovariance (h : ℝ) (k j : Fin 3) : ℝ :=
  if k = j then 3 * h ^ 2 / 16 else 0

def splitCornerOffset (s : ℝ) (corner : Fin 2 × Fin 2 × Fin 2) (j : Fin 3) : ℝ :=
  ![![ -s, s] corner.1, ![ -s, s] corner.2.1, ![ -s, s] corner.2.2] j

/-- The eight symmetric corners with squared coordinate displacement 3h²/16
have zero mean and the stated exact covariance, including zero cross moments. -/
theorem split_corner_covariance (h s : ℝ) (displacement : s ^ 2 = 3 * h ^ 2 / 16) :
    (∀ j, ∑ corner, splitCornerOffset s corner j = 0) ∧
    (∀ k j, ∑ corner, splitCornerOffset s corner k * splitCornerOffset s corner j =
      8 * splitCornerCovariance h k j) := by
  constructor
  · intro j
    fin_cases j <;>
      simp [splitCornerOffset, Fintype.sum_prod_type, Fin.sum_univ_succ]
  · intro k j
    fin_cases k <;> fin_cases j <;>
      simp [splitCornerOffset, splitCornerCovariance, Fintype.sum_prod_type, Fin.sum_univ_succ] <;>
      nlinarith

#print axioms split_corner_covariance

/-- Halving an isotropic quadratic support leaves no covariance deficit.
For the mixed MAC support the only deficit is its normal entry h²/16. -/
theorem split_uniform_mac_deficits (h : ℝ) (a k j : Fin 3) :
    quadraticSplitMoment h k j - quadraticSplitMoment (h / 2) k j -
      splitCornerCovariance h k j = 0 ∧
    cubicMacSplitMoment h a k j - cubicMacSplitMoment (h / 2) a k j -
      splitCornerCovariance h k j = (if k = a ∧ j = a then h ^ 2 / 16 else 0) := by
  by_cases same : k = j
  · subst j
    by_cases normal : k = a <;>
      simp [quadraticSplitMoment, cubicMacSplitMoment, splitCornerCovariance, normal] <;>
      (try constructor) <;> ring
  · have notBoth : ¬(k = a ∧ j = a) := fun both => same (both.1.trans both.2.symm)
    simp [quadraticSplitMoment, cubicMacSplitMoment, splitCornerCovariance, same, notBoth]

#print axioms split_uniform_mac_deficits

/-- The F3 energy-deficit term vanishes for the quadratic support, and for
mixed MAC support it equals m h² Σ C_aa²/32. This expression vanishes on
off-diagonal rows alone and is positive for positive m,h and any nonzero C_aa. -/
theorem split_uniform_mac_energy_deficits (mass h : ℝ) (affine : Fin 3 → Fin 3 → ℝ) :
    mass / 2 * (∑ a, resamplingQuadratic (affine a) (fun k j =>
      quadraticSplitMoment h k j - quadraticSplitMoment (h / 2) k j -
        splitCornerCovariance h k j)) = 0 ∧
    mass / 2 * (∑ a, resamplingQuadratic (affine a) (fun k j =>
      cubicMacSplitMoment h a k j - cubicMacSplitMoment (h / 2) a k j -
        splitCornerCovariance h k j)) = mass * h ^ 2 * (∑ a, affine a a ^ 2) / 32 := by
  constructor
  · have zero (k j : Fin 3) := (split_uniform_mac_deficits h 0 k j).1
    simp [resamplingQuadratic, resamplingDot, resamplingRowMoment, zero]
  · simp only [resamplingQuadratic, resamplingDot, resamplingRowMoment]
    have normal (a k j : Fin 3) := (split_uniform_mac_deficits h a k j).2
    simp_rw [normal, mul_ite, mul_zero]
    have row (a : Fin 3) :
        (∑ j, (∑ k, if k = a ∧ j = a then affine a k * (h ^ 2 / 16) else 0) * affine a j) =
          h ^ 2 / 16 * (affine a a ^ 2) := by
      simp_rw [ite_and]
      simp [Finset.sum_ite_eq', ite_mul, zero_mul]
      ring
    simp_rw [row, ← Finset.mul_sum]
    ring

#print axioms split_uniform_mac_energy_deficits


end Fluidity
