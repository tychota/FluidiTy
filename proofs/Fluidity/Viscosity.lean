import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

open scoped BigOperators

namespace Fluidity

/-- Kinetic energy for the diagonal face-inertia metric. -/
noncomputable def viscousEnergy {n : ℕ} (mass velocity : Fin n → ℝ) : ℝ :=
  ∑ i, mass i * velocity i ^ 2 / 2

/-- Explicit energy change for a frozen force, including prescribed wall work. -/
theorem viscous_forward_energy {n : ℕ} (mass u v force : Fin n → ℝ) (dt : ℝ)
    (hm : ∀ i, 0 < mass i)
    (step : ∀ i, mass i * (v i - u i) = dt * force i) :
    viscousEnergy mass v - viscousEnergy mass u =
      dt * (∑ i, u i * force i) + dt ^ 2 / 2 * (∑ i, force i ^ 2 / mass i) := by
  unfold viscousEnergy
  rw [← Finset.sum_sub_distrib, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  have velocity : v i = u i + dt * force i / mass i := by
    have h : v i - u i = dt * force i / mass i := (eq_div_iff (ne_of_gt (hm i))).2 (by simpa [mul_comm] using step i)
    linarith
  rw [velocity]
  field_simp [ne_of_gt (hm i)]
  ring

/-- Explicit Euler contracts kinetic energy with homogeneous walls when the
operator quadratic forms satisfy the certified bound and the step fits it. -/
theorem viscous_forward_contraction {n : ℕ} (mass u v stiffness : Fin n → ℝ)
    (dt bound : ℝ) (hm : ∀ i, 0 < mass i) (ht : 0 ≤ dt)
    (nonnegative : 0 ≤ ∑ i, u i * stiffness i)
    (certificate : (∑ i, stiffness i ^ 2 / mass i) ≤ bound * ∑ i, u i * stiffness i)
    (stable : dt * bound ≤ 2)
    (step : ∀ i, mass i * (v i - u i) = dt * -stiffness i) :
    viscousEnergy mass v ≤ viscousEnergy mass u := by
  have energy := viscous_forward_energy mass u v (fun i => -stiffness i) dt hm step
  simp only [mul_neg, Finset.sum_neg_distrib, neg_sq] at energy
  have term := mul_le_mul_of_nonneg_left certificate (show 0 ≤ dt ^ 2 / 2 by positivity)
  have limit := mul_le_mul_of_nonneg_right stable
    (mul_nonneg ht nonnegative)
  nlinarith only [energy, term, limit]

/-- Backward Euler energy balance. The frozen stiffness action is evaluated
at the new velocity, which is also the velocity paired with the wall load. -/
theorem viscous_backward_energy {n : ℕ} (mass u v stiffness wall : Fin n → ℝ) (dt : ℝ)
    (step : ∀ i, mass i * (v i - u i) = dt * (wall i - stiffness i)) :
    viscousEnergy mass v - viscousEnergy mass u =
      dt * (∑ i, v i * wall i) - dt * (∑ i, v i * stiffness i) -
        viscousEnergy mass (fun i => v i - u i) := by
  unfold viscousEnergy
  rw [← Finset.sum_sub_distrib, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib,
    ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  linear_combination v i * step i

#print axioms viscous_forward_energy
#print axioms viscous_forward_contraction
#print axioms viscous_backward_energy

/-- Trace certificate for a factored PSD mass-scaled stiffness. Set A to
sqrt(W) E invsqrt(M) and strain to A times the mass-scaled velocity. Then
Aᵀ times strain is the mass-scaled stiffness action. -/
theorem viscous_trace_certificate {n m : ℕ} (A : Fin m → Fin n → ℝ) (strain : Fin m → ℝ) :
    (∑ i, (∑ j, A j i * strain j) ^ 2) ≤
      (∑ i, ∑ j, A j i ^ 2) * (∑ j, strain j ^ 2) := by
  rw [Finset.sum_mul]
  apply Finset.sum_le_sum
  intro i _
  exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun j => A j i) strain

#print axioms viscous_trace_certificate

end Fluidity
