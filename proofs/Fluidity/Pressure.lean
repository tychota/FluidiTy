import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring

open scoped BigOperators

namespace Fluidity

/-- Component action of `R diag(k) Rᵀ + diag(C)` in exact real arithmetic. -/
def pressureAction {n m : ℕ} (R : Fin n → Fin m → ℝ)
    (k : Fin m → ℝ) (C x : Fin n → ℝ) (i : Fin n) : ℝ :=
  (∑ f, R i f * k f * (∑ j, R j f * x j)) + C i * x i

/-- The actual operator quadratic form `xᵀ A x`. -/
def pressureQuadraticForm {n m : ℕ} (R : Fin n → Fin m → ℝ)
    (k : Fin m → ℝ) (C x : Fin n → ℝ) : ℝ :=
  ∑ i, x i * pressureAction R k C x i

/-- Finite operator expansion, including zero cells/faces. -/
theorem pressure_quadratic_form_eq {n m : ℕ} (R : Fin n → Fin m → ℝ)
    (k : Fin m → ℝ) (C x : Fin n → ℝ) :
    pressureQuadraticForm R k C x =
      (∑ f, k f * (∑ i, R i f * x i) ^ 2) + ∑ i, C i * x i ^ 2 := by
  unfold pressureQuadraticForm pressureAction
  calc
    (∑ i, x i * ((∑ f, R i f * k f * (∑ j, R j f * x j)) + C i * x i)) =
        (∑ i, ∑ f, (k f * (∑ j, R j f * x j)) * (R i f * x i)) +
          ∑ i, C i * x i ^ 2 := by
      simp only [mul_add, Finset.sum_add_distrib]
      congr 1
      · apply Finset.sum_congr rfl
        intro i _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro f _
        ring
      · apply Finset.sum_congr rfl
        intro i _
        ring
    _ = (∑ f, (k f * (∑ j, R j f * x j)) * (∑ i, R i f * x i)) +
          ∑ i, C i * x i ^ 2 := by
      rw [Finset.sum_comm]
      simp only [Finset.mul_sum]
    _ = _ := by
      congr 1
      apply Finset.sum_congr rfl
      intro f _
      ring

theorem pressure_psd {n m : ℕ} (R : Fin n → Fin m → ℝ)
    (k : Fin m → ℝ) (C : Fin n → ℝ)
    (hk : ∀ f, 0 ≤ k f) (hC : ∀ i, 0 ≤ C i) (x : Fin n → ℝ) :
    0 ≤ pressureQuadraticForm R k C x := by
  rw [pressure_quadratic_form_eq]
  exact add_nonneg
    (Finset.sum_nonneg fun f _ => mul_nonneg (hk f) (sq_nonneg _))
    (Finset.sum_nonneg fun i _ => mul_nonneg (hC i) (sq_nonneg _))

#print axioms pressure_quadratic_form_eq
#print axioms pressure_psd

end Fluidity
