import Mathlib.Algebra.Polynomial.Derivative
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

open scoped BigOperators
open Polynomial

noncomputable section

namespace Fluidity

/-- Quadratic B-spline weights on the three nodes starting at floor(X-s-1/2).
The local coordinate t=X-s-1/2-floor(X-s-1/2) lies in [0,1). -/
def quadraticWeights (t : ℝ) : Fin 3 → ℝ :=
  ![(1 - t) ^ 2 / 2, 3 / 4 - (t - 1 / 2) ^ 2, t ^ 2 / 2]

/-- Derivatives with respect to normalized particle position, not face position. -/
def quadraticDerivatives (t : ℝ) : Fin 3 → ℝ := ![t - 1, 1 - 2 * t, t]

/-- Face-minus-particle offsets in cells. -/
def quadraticOffsets (t : ℝ) : Fin 3 → ℝ := ![-1 / 2 - t, 1 / 2 - t, 3 / 2 - t]

/-- Independently written centered piecewise quadratic. -/
def centeredQuadratic (x : ℝ) : ℝ :=
  if x ≤ -3 / 2 then 0 else
  if x ≤ -1 / 2 then (x + 3 / 2) ^ 2 / 2 else
  if x ≤ 1 / 2 then 3 / 4 - x ^ 2 else
  if x ≤ 3 / 2 then (x - 3 / 2) ^ 2 / 2 else 0

theorem quadratic_centered_local (t : ℝ) (lower : 0 ≤ t) (upper : t ≤ 1) (i : Fin 3) :
    centeredQuadratic (-quadraticOffsets t i) = quadraticWeights t i := by
  fin_cases i <;> dsimp [centeredQuadratic, quadraticOffsets, quadraticWeights]
  all_goals split_ifs <;> nlinarith

def quadraticPolynomial : Fin 3 → Polynomial ℝ :=
  ![C (1 / 2) * (1 - X) ^ 2, C (3 / 4) - (X - C (1 / 2)) ^ 2, C (1 / 2) * X ^ 2]

theorem quadratic_derivative_polynomials (t : ℝ) (i : Fin 3) :
    (quadraticPolynomial i).eval t = quadraticWeights t i ∧
    (quadraticPolynomial i).derivative.eval t = quadraticDerivatives t i := by
  fin_cases i <;>
    simp [quadraticPolynomial, quadraticWeights, quadraticDerivatives, derivative_mul,
      derivative_pow] <;> ring

/-- Adjacent floor pieces agree in value and first derivative, including the two
zero support endpoints. The centered quadratic is C1 at stencil changes. -/
theorem quadratic_boundary_matching :
    quadraticWeights 0 0 = quadraticWeights 1 1 ∧
    quadraticWeights 0 1 = quadraticWeights 1 2 ∧
    quadraticWeights 0 2 = 0 ∧ quadraticWeights 1 0 = 0 ∧
    quadraticDerivatives 0 0 = quadraticDerivatives 1 1 ∧
    quadraticDerivatives 0 1 = quadraticDerivatives 1 2 ∧
    quadraticDerivatives 0 2 = 0 ∧ quadraticDerivatives 1 0 = 0 := by
  norm_num [quadraticWeights, quadraticDerivatives]
/-- Exact polynomial moment identities on the local interval (in fact for all t). -/
theorem quadratic_local_moments (t : ℝ) :
    (∑ i, quadraticWeights t i) = 1 ∧
    (∑ i, quadraticWeights t i * quadraticOffsets t i) = 0 ∧
    (∑ i, quadraticWeights t i * quadraticOffsets t i ^ 2) = 1 / 4 ∧
    (∑ i, quadraticDerivatives t i) = 0 ∧
    (∑ i, quadraticOffsets t i * quadraticDerivatives t i) = 1 := by
  simp [quadraticWeights, quadraticOffsets, quadraticDerivatives, Fin.sum_univ_succ]
  constructor
  · ring
  constructor
  · ring
  constructor
  · ring
  constructor <;> ring

/-- The floor-selected local coordinate, physical anchors and nonnegative weights
for either MAC lattice. Identities are inherited from the actual local polynomials. -/
theorem quadratic_shifted_basis (position shift : ℝ) (_shifts : shift = 0 ∨ shift = 1 / 2) :
    let start := ⌊position - shift - 1 / 2⌋
    let t := position - shift - 1 / 2 - start
    0 ≤ t ∧ t < 1 ∧
    (∀ i : Fin 3, quadraticOffsets t i = (start : ℝ) + i.val + shift - position) ∧
    (∀ i, 0 ≤ quadraticWeights t i) ∧
    (∑ i, quadraticWeights t i) = 1 ∧
    (∑ i, quadraticWeights t i * quadraticOffsets t i) = 0 ∧
    (∑ i, quadraticWeights t i * quadraticOffsets t i ^ 2) = 1 / 4 ∧
    (∑ i, quadraticDerivatives t i) = 0 ∧
    (∑ i, quadraticOffsets t i * quadraticDerivatives t i) = 1 := by
  dsimp
  have lower := Int.floor_le (position - shift - 1 / 2)
  have upper := Int.lt_floor_add_one (position - shift - 1 / 2)
  have nonnegative : 0 ≤ position - shift - 1 / 2 - ⌊position - shift - 1 / 2⌋ := by linarith
  have belowOne : position - shift - 1 / 2 - ⌊position - shift - 1 / 2⌋ < 1 := by linarith
  refine ⟨nonnegative, belowOne, ?_, ?_, quadratic_local_moments _⟩
  · intro i
    fin_cases i <;> dsimp [quadraticOffsets] <;> norm_num only [Nat.cast_ofNat, Nat.cast_zero, Nat.cast_one] <;> unfold Int.fract <;> ring
  · intro i
    fin_cases i <;> dsimp [quadraticWeights] <;> unfold Int.fract
    · positivity
    · nlinarith [mul_nonneg nonnegative (sub_nonneg.mpr belowOne.le)]
    · positivity

abbrev TensorNode := Fin 3 × Fin 3 × Fin 3

/-- Three-dimensional separable stencil, with a separate local coordinate on each axis. -/
def tensorWeight (t : Fin 3 → ℝ) (i : TensorNode) : ℝ :=
  quadraticWeights (t 0) i.1 * quadraticWeights (t 1) i.2.1 * quadraticWeights (t 2) i.2.2

def tensorOffset (t : Fin 3 → ℝ) (i : TensorNode) : Fin 3 → ℝ :=
  ![quadraticOffsets (t 0) i.1, quadraticOffsets (t 1) i.2.1, quadraticOffsets (t 2) i.2.2]

/-- Product-rule gradient in normalized coordinates. Physical gradients divide by h. -/
def tensorGradient (t : Fin 3 → ℝ) (i : TensorNode) : Fin 3 → ℝ :=
  ![quadraticDerivatives (t 0) i.1 * quadraticWeights (t 1) i.2.1 * quadraticWeights (t 2) i.2.2,
    quadraticWeights (t 0) i.1 * quadraticDerivatives (t 1) i.2.1 * quadraticWeights (t 2) i.2.2,
    quadraticWeights (t 0) i.1 * quadraticWeights (t 1) i.2.1 * quadraticDerivatives (t 2) i.2.2]

/-- Hold two local coordinates fixed and vary one. -/
def tensorAxisPolynomial (t : Fin 3 → ℝ) (i : TensorNode) : Fin 3 → Polynomial ℝ :=
  ![quadraticPolynomial i.1 * C (quadraticWeights (t 1) i.2.1) * C (quadraticWeights (t 2) i.2.2),
    C (quadraticWeights (t 0) i.1) * quadraticPolynomial i.2.1 * C (quadraticWeights (t 2) i.2.2),
    C (quadraticWeights (t 0) i.1) * C (quadraticWeights (t 1) i.2.1) * quadraticPolynomial i.2.2]

/-- The tensor gradient is the actual polynomial partial derivative, independently
of the moment identities used to recover affine fields. -/
theorem tensor_gradient_polynomials (t : Fin 3 → ℝ) (i : TensorNode) (axis : Fin 3) :
    (tensorAxisPolynomial t i axis).eval (t axis) = tensorWeight t i ∧
    (tensorAxisPolynomial t i axis).derivative.eval (t axis) = tensorGradient t i axis := by
  fin_cases axis <;>
    simp [tensorAxisPolynomial, tensorWeight, tensorGradient, derivative_mul,
      (quadratic_derivative_polynomials _ _).1, (quadratic_derivative_polynomials _ _).2]
/-- Complete exact row moments. D remains row-dependent; derivatives are separate
from the weighted affine fit and must satisfy their own moment identities. -/
structure RowMoments {ι : Type} [Fintype ι] {d : ℕ}
    (weight : ι → ℝ) (offset gradient : ι → Fin d → ℝ) (moment : Fin d → Fin d → ℝ) : Prop where
  partition : ∑ i, weight i = 1
  first : ∀ j, ∑ i, weight i * offset i j = 0
  second : ∀ j k, ∑ i, weight i * offset i j * offset i k = moment j k
  derivativeZero : ∀ j, ∑ i, gradient i j = 0
  derivativeFirst : ∀ j k, ∑ i, offset i j * gradient i k = if j = k then 1 else 0

/-- Tensor products preserve partition, zero first moments, isotropic second moment
I/4, zero gradient sum and the identity derivative moment. -/
theorem quadratic_tensor_moments (t : Fin 3 → ℝ) :
    RowMoments (tensorWeight t) (tensorOffset t) (tensorGradient t)
      (fun j k => if j = k then 1 / 4 else 0) := by
  constructor
  · simp [tensorWeight, Fintype.sum_prod_type, Fin.sum_univ_succ, quadraticWeights]
    ring
  · intro j
    fin_cases j <;>
      simp [tensorWeight, tensorOffset, Fintype.sum_prod_type, Fin.sum_univ_succ,
        quadraticWeights, quadraticOffsets] <;> ring
  · intro j k
    fin_cases j <;> fin_cases k <;>
      simp [tensorWeight, tensorOffset, Fintype.sum_prod_type, Fin.sum_univ_succ,
        quadraticWeights, quadraticOffsets] <;> ring
  · intro j
    fin_cases j <;>
      simp [tensorGradient, Fintype.sum_prod_type, Fin.sum_univ_succ,
        quadraticWeights, quadraticDerivatives] <;> ring
  · intro j k
    fin_cases j <;> fin_cases k <;>
      simp [tensorGradient, tensorOffset, Fintype.sum_prod_type, Fin.sum_univ_succ,
        quadraticWeights, quadraticOffsets, quadraticDerivatives] <;> ring

/-- Exact affine continuation at every sample of a row. -/
def affineSamples {ι : Type} {d : ℕ} (velocity : ℝ) (row : Fin d → ℝ)
    (offset : ι → Fin d → ℝ) (i : ι) : ℝ := velocity + ∑ j, row j * offset i j

private theorem affine_sum {ι : Type} [Fintype ι] {d : ℕ} (a : ι → ℝ)
    (velocity : ℝ) (row : Fin d → ℝ) (offset : ι → Fin d → ℝ) :
    (∑ i, a i * affineSamples velocity row offset i) =
      velocity * (∑ i, a i) + ∑ j, row j * (∑ i, a i * offset i j) := by
  simp only [affineSamples, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
  rw [Finset.sum_comm]
  congr 1
  · apply Finset.sum_congr rfl
    intro i _
    ring
  · apply Finset.sum_congr rfl
    intro j _
    apply Finset.sum_congr rfl
    intro i _
    ring

/-- PIC, the weighted first-velocity moment followed by a right inverse of D,
and the independent derivative gather reconstruct the same affine row.
All samples must equal the specified affine continuation. -/
theorem affine_reconstruction_of_moments {ι : Type} [Fintype ι] {d : ℕ}
    (weight : ι → ℝ) (offset gradient : ι → Fin d → ℝ)
    (moment inverse : Fin d → Fin d → ℝ) (moments : RowMoments weight offset gradient moment)
    (rightInverse : ∀ j k, ∑ l, moment j l * inverse l k = if j = k then 1 else 0)
    (velocity : ℝ) (row : Fin d → ℝ) :
    (∑ i, weight i * affineSamples velocity row offset i) = velocity ∧
    (∀ j, ∑ k, (∑ i, weight i * affineSamples velocity row offset i * offset i k) * inverse k j = row j) ∧
    (∀ j, ∑ i, affineSamples velocity row offset i * gradient i j = row j) := by
  have weightedVelocity := affine_sum weight velocity row offset
  rw [moments.partition] at weightedVelocity
  simp only [moments.first, mul_zero, Finset.sum_const_zero, mul_one, add_zero] at weightedVelocity
  refine ⟨weightedVelocity, ?_, ?_⟩
  · intro j
    have weightedOffsetVelocity : ∀ k, (∑ i, weight i * affineSamples velocity row offset i * offset i k) =
        ∑ l, row l * moment l k := by
      intro k
      calc
        _ = ∑ i, (weight i * offset i k) * affineSamples velocity row offset i := by
          apply Finset.sum_congr rfl
          intro i _
          ring
        _ = _ := by
          rw [affine_sum, moments.first]
          simp only [mul_zero, zero_add]
          congr 1
          ext l
          congr 1
          rw [← moments.second l k]
          apply Finset.sum_congr rfl
          intro i _
          ring
    simp_rw [weightedOffsetVelocity, Finset.sum_mul]
    rw [Finset.sum_comm]
    simp_rw [mul_assoc, ← Finset.mul_sum, rightInverse]
    simp
  · intro j
    calc
      _ = ∑ i, gradient i j * affineSamples velocity row offset i := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = _ := by
        rw [affine_sum, moments.derivativeZero]
        simp only [mul_zero, zero_add]
        have mixed : ∀ k, ∑ i, gradient i j * offset i k = if k = j then 1 else 0 := by
          intro k
          simpa only [mul_comm] using moments.derivativeFirst k j
        simp_rw [mixed]
        simp

/-- With the same weights and equal old and new face samples, new PIC minus old PIC is zero.
The retained residual is the old particle velocity minus old PIC and vanishes exactly when
the particle velocity equals old PIC. -/
theorem equal_samples_grid_increment_and_residual {ι : Type} [Fintype ι]
    (weight oldSample newSample : ι → ℝ) (oldParticleVelocity : ℝ)
    (samplesEqual : ∀ i, newSample i = oldSample i) :
    ((∑ i, weight i * newSample i) - (∑ i, weight i * oldSample i) = 0) ∧
    (oldParticleVelocity - (∑ i, weight i * oldSample i) = 0 ↔
      oldParticleVelocity = ∑ i, weight i * oldSample i) := by
  constructor
  · simp [samplesEqual]
  · exact sub_eq_zero

#print axioms quadratic_boundary_matching
#print axioms tensor_gradient_polynomials
#print axioms quadratic_centered_local
#print axioms quadratic_derivative_polynomials
#print axioms quadratic_shifted_basis
#print axioms quadratic_tensor_moments
#print axioms affine_reconstruction_of_moments
#print axioms equal_samples_grid_increment_and_residual


/-- Unrounded, force-free per-component mass and pure momentum deposited by a
finite particle population. Faces and moments may differ between components. -/
def depositedMacMass {particleCount faceCount d : ℕ} (mass : Fin particleCount → ℝ) (weight : Fin particleCount → Fin d → Fin faceCount → ℝ)
    (axis : Fin d) (face : Fin faceCount) : ℝ := ∑ particle, mass particle * weight particle axis face

def depositedMacPureMomentum {particleCount faceCount d : ℕ} (mass : Fin particleCount → ℝ) (weight : Fin particleCount → Fin d → Fin faceCount → ℝ)
    (velocity : Fin particleCount → Fin d → ℝ) (row : Fin particleCount → Fin d → Fin d → ℝ)
    (offset : Fin particleCount → Fin d → Fin faceCount → Fin d → ℝ) (axis : Fin d) (face : Fin faceCount) : ℝ :=
  ∑ particle, mass particle * weight particle axis face * affineSamples (velocity particle axis) (row particle axis) (offset particle axis) face

/-- F2: each component represents the same particle inventory; angular momentum
is the difference of two distinct component-face spatial sums. No common D or
inverse is assumed. Anchors are bound to particle positions and row offsets. -/
theorem mac_deposit_conserves_row_moments {particleCount faceCount d : ℕ}
    (mass : Fin particleCount → ℝ) (_nonnegativeMass : ∀ particle, 0 ≤ mass particle)
    (weight : Fin particleCount → Fin d → Fin faceCount → ℝ)
    (position velocity : Fin particleCount → Fin d → ℝ)
    (row : Fin particleCount → Fin d → Fin d → ℝ)
    (moment : Fin particleCount → Fin d → Fin d → Fin d → ℝ)
    (offset : Fin particleCount → Fin d → Fin faceCount → Fin d → ℝ)
    (anchor : Fin d → Fin faceCount → Fin d → ℝ)
    (anchors : ∀ particle faceNormal faceSample spatialAxis, anchor faceNormal faceSample spatialAxis = position particle spatialAxis + offset particle faceNormal faceSample spatialAxis)
    (partition : ∀ particle faceNormal, ∑ faceSample, weight particle faceNormal faceSample = 1)
    (first : ∀ particle faceNormal spatialAxis, ∑ faceSample, weight particle faceNormal faceSample * offset particle faceNormal faceSample spatialAxis = 0)
    (second : ∀ particle faceNormal spatialAxis otherAxis, ∑ faceSample, weight particle faceNormal faceSample * offset particle faceNormal faceSample spatialAxis * offset particle faceNormal faceSample otherAxis = moment particle faceNormal spatialAxis otherAxis) :
    (∀ faceNormal, ∑ faceSample, depositedMacMass mass weight faceNormal faceSample = ∑ particle, mass particle) ∧
    (∀ faceNormal, ∑ faceSample, depositedMacPureMomentum mass weight velocity row offset faceNormal faceSample =
      ∑ particle, mass particle * velocity particle faceNormal) ∧
    (∀ spatialAxis otherAxis,
      (∑ faceSample, anchor otherAxis faceSample spatialAxis * depositedMacPureMomentum mass weight velocity row offset otherAxis faceSample) -
      (∑ faceSample, anchor spatialAxis faceSample otherAxis * depositedMacPureMomentum mass weight velocity row offset spatialAxis faceSample) =
      ∑ particle, mass particle * (position particle spatialAxis * velocity particle otherAxis - position particle otherAxis * velocity particle spatialAxis +
        (∑ slopeAxis, row particle otherAxis slopeAxis * moment particle otherAxis slopeAxis spatialAxis) - ∑ slopeAxis, row particle spatialAxis slopeAxis * moment particle spatialAxis slopeAxis otherAxis)) := by
  have weightedVelocity : ∀ particle faceNormal, (∑ faceSample, weight particle faceNormal faceSample *
      affineSamples (velocity particle faceNormal) (row particle faceNormal) (offset particle faceNormal) faceSample) = velocity particle faceNormal := by
    intro particle faceNormal
    rw [affine_sum, partition]
    simp [first]
  have weightedOffsetVelocity : ∀ particle faceNormal spatialAxis, (∑ faceSample, weight particle faceNormal faceSample *
      affineSamples (velocity particle faceNormal) (row particle faceNormal) (offset particle faceNormal) faceSample * offset particle faceNormal faceSample spatialAxis) =
      ∑ slopeAxis, row particle faceNormal slopeAxis * moment particle faceNormal slopeAxis spatialAxis := by
    intro particle faceNormal spatialAxis
    calc
      _ = ∑ faceSample, (weight particle faceNormal faceSample * offset particle faceNormal faceSample spatialAxis) *
          affineSamples (velocity particle faceNormal) (row particle faceNormal) (offset particle faceNormal) faceSample := by
        apply Finset.sum_congr rfl
        intro faceSample _
        ring
      _ = _ := by
        rw [affine_sum, first]
        simp only [mul_zero, zero_add]
        congr 1
        ext slopeAxis
        congr 1
        rw [← second particle faceNormal slopeAxis spatialAxis]
        apply Finset.sum_congr rfl
        intro faceSample _
        ring
  have faceMomentumMoment : ∀ faceNormal spatialAxis,
      (∑ faceSample, anchor faceNormal faceSample spatialAxis * depositedMacPureMomentum mass weight velocity row offset faceNormal faceSample) =
      ∑ particle, mass particle * (position particle spatialAxis * velocity particle faceNormal + ∑ slopeAxis, row particle faceNormal slopeAxis * moment particle faceNormal slopeAxis spatialAxis) := by
    intro faceNormal spatialAxis
    simp only [depositedMacPureMomentum, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro particle _
    calc
      _ = mass particle * (position particle spatialAxis * (∑ faceSample, weight particle faceNormal faceSample *
            affineSamples (velocity particle faceNormal) (row particle faceNormal) (offset particle faceNormal) faceSample) +
          ∑ faceSample, weight particle faceNormal faceSample * affineSamples (velocity particle faceNormal) (row particle faceNormal) (offset particle faceNormal) faceSample * offset particle faceNormal faceSample spatialAxis) := by
        simp only [mul_add, Finset.mul_sum, ← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro faceSample _
        rw [anchors particle faceNormal faceSample spatialAxis]
        ring
      _ = _ := by rw [weightedVelocity, weightedOffsetVelocity]
  refine ⟨?_, ?_, ?_⟩
  · intro faceNormal
    simp only [depositedMacMass]
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, partition, mul_one]
  · intro faceNormal
    simp only [depositedMacPureMomentum]
    rw [Finset.sum_comm]
    simp_rw [mul_assoc, ← Finset.mul_sum, weightedVelocity]
  · intro spatialAxis otherAxis
    rw [faceMomentumMoment, faceMomentumMoment, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro particle _
    ring

#print axioms mac_deposit_conserves_row_moments

/-- On `n` consecutive signed cells starting at `lo`, the face sum against
backward weight differences equals the cell sum against forward velocity
differences. The two exterior cell weights at `lo - 1` and `lo + n` must vanish,
including when `n = 0`. No physical cell-width factor is included. -/
theorem finite_face_sum_eq_weighted_forward_difference (lo : ℤ) (n : ℕ)
    (velocity weight : ℤ → ℝ) (leftZero : weight (lo - 1) = 0)
    (rightZero : weight (lo + (n : ℤ)) = 0) :
    (∑ j ∈ Finset.range (n + 1), velocity (lo + (j : ℤ)) *
      (weight (lo + (j : ℤ) - 1) - weight (lo + (j : ℤ)))) =
      ∑ i ∈ Finset.range n, weight (lo + (i : ℤ)) *
        (velocity (lo + (i : ℤ) + 1) - velocity (lo + (i : ℤ))) := by
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib,
    Finset.sum_range_succ' (fun j => velocity (lo + (j : ℤ)) * weight (lo + (j : ℤ) - 1)) n,
    Finset.sum_range_succ (fun j => velocity (lo + (j : ℤ)) * weight (lo + (j : ℤ))) n]
  simp only [Nat.cast_zero, add_zero, leftZero, rightZero, mul_zero]
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  have nextFace : lo + ((i + 1 : ℕ) : ℤ) = lo + (i : ℤ) + 1 := by omega
  rw [nextFace, add_sub_cancel_right]
  ring

#print axioms finite_face_sum_eq_weighted_forward_difference

-- Pin the public signature, including both exterior weights and signed indices.
example (lo : ℤ) (n : ℕ) (u w : ℤ → ℝ)
    (leftZero : w (lo - 1) = 0) (rightZero : w (lo + (n : ℤ)) = 0) :
    (∑ j ∈ Finset.range (n + 1), u (lo + (j : ℤ)) * (w (lo + (j : ℤ) - 1) - w (lo + (j : ℤ)))) =
      ∑ i ∈ Finset.range n, w (lo + (i : ℤ)) * (u (lo + (i : ℤ) + 1) - u (lo + (i : ℤ))) :=
  finite_face_sum_eq_weighted_forward_difference lo n u w leftZero rightZero

private def traceTestVelocity (face : ℤ) : ℝ :=
  if face = -2 then 2 else if face = -1 then -3 else 5

private def traceTestWeight (leftExterior rightExterior : ℝ) (cell : ℤ) : ℝ :=
  if cell = -3 then leftExterior else if cell = -2 then 3 else if cell = -1 then -2
  else if cell = 0 then rightExterior else 0

example :
    (∑ j ∈ Finset.range 3, traceTestVelocity (-2 + (j : ℤ)) *
      (traceTestWeight 0 0 (-2 + (j : ℤ) - 1) - traceTestWeight 0 0 (-2 + (j : ℤ)))) = -31 := by
  rw [finite_face_sum_eq_weighted_forward_difference (-2) 2 traceTestVelocity (traceTestWeight 0 0)
    (by norm_num [traceTestWeight]) (by norm_num [traceTestWeight])]
  norm_num [Finset.sum_range_succ, traceTestVelocity, traceTestWeight]

example (lo : ℤ) (u w : ℤ → ℝ) (leftZero : w (lo - 1) = 0) (rightZero : w lo = 0) :
    u lo * (w (lo - 1) - w lo) = 0 := by
  simpa using finite_face_sum_eq_weighted_forward_difference lo 0 u w leftZero (by simpa using rightZero)

-- A nonzero exterior weight on either side changes the face sum, while the cell sum is still -31.
example :
    (∑ j ∈ Finset.range 3, traceTestVelocity (-2 + (j : ℤ)) *
      (traceTestWeight 7 0 (-2 + (j : ℤ) - 1) - traceTestWeight 7 0 (-2 + (j : ℤ)))) ≠
      ∑ i ∈ Finset.range 2, traceTestWeight 7 0 (-2 + (i : ℤ)) *
        (traceTestVelocity (-2 + (i : ℤ) + 1) - traceTestVelocity (-2 + (i : ℤ))) := by
  norm_num [Finset.sum_range_succ, traceTestVelocity, traceTestWeight]

example :
    (∑ j ∈ Finset.range 3, traceTestVelocity (-2 + (j : ℤ)) *
      (traceTestWeight 0 7 (-2 + (j : ℤ) - 1) - traceTestWeight 0 7 (-2 + (j : ℤ)))) ≠
      ∑ i ∈ Finset.range 2, traceTestWeight 0 7 (-2 + (i : ℤ)) *
        (traceTestVelocity (-2 + (i : ℤ) + 1) - traceTestVelocity (-2 + (i : ℤ))) := by
  norm_num [Finset.sum_range_succ, traceTestVelocity, traceTestWeight]

end Fluidity
