import Mathlib.Algebra.Order.Ring.Int
import Mathlib.Algebra.Order.Group.Abs
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Data.List.Perm.Basic

import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Round
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

open scoped BigOperators

namespace Fluidity

/-- Permuting mathematical integers preserves the total. -/
theorem fixed_point_order_independent (q r : List ℤ) (h : q.Perm r) :
    q.sum = r.sum := by
  exact h.sum_eq

private theorem abs_sum_le_sum_abs (q : List ℤ) :
    |q.sum| ≤ (q.map (fun a => |a|)).sum := by
  induction q with
  | nil => simp
  | cons a q ih =>
    simpa only [List.sum_cons, List.map_cons] using
      (abs_add_le a q.sum).trans (add_le_add (le_refl |a|) ih)

/-- Any prefix is bounded by the absolute-contribution budget, in `ℤ`. -/
theorem fixed_point_partial_bound (q : List ℤ) (B : ℤ)
    (hB : (q.map (fun a => |a|)).sum ≤ B) (n : ℕ) :
    |(q.take n).sum| ≤ B := by
  apply (abs_sum_le_sum_abs (q.take n)).trans
  apply le_trans ?_ hB
  apply (List.take_sublist n q).map (fun a => |a|) |>.sum_le_sum
  intro a ha
  obtain ⟨b, _, rfl⟩ := List.mem_map.mp ha
  exact abs_nonneg b

#print axioms fixed_point_order_independent
#print axioms fixed_point_partial_bound


/-- Any nearest integer is allowed: the error/range proofs do not select a tie.
The runtime bridge separately checks ties to even. -/
def IsNearestInteger (scaled : ℝ) (quanta : ℤ) : Prop :=
  ∀ candidate : ℤ, |scaled - quanta| ≤ |scaled - candidate|

/-- Nearest rounding is within half a quantum, including either choice at ties. -/
theorem nearest_integer_error (scaled : ℝ) (quanta : ℤ)
    (nearest : IsNearestInteger scaled quanta) : |scaled - quanta| ≤ 1 / 2 :=
  (nearest (round scaled)).trans (abs_sub_round scaled)

/-- An inclusive integer magnitude guard is preserved by nearest rounding. -/
theorem quantized_contribution_bound (scaled : ℝ) (quanta bound : ℤ)
    (nearest : IsNearestInteger scaled quanta) (guard : |scaled| ≤ (bound : ℝ)) :
    |quanta| ≤ bound := by
  obtain ⟨errLower, errUpper⟩ := abs_le.mp (nearest_integer_error scaled quanta nearest)
  obtain ⟨guardLower, guardUpper⟩ := abs_le.mp guard
  have upper : (quanta : ℝ) < (bound : ℝ) + 1 := by linarith
  have lower : -(bound : ℝ) - 1 < (quanta : ℝ) := by linarith
  have upperInt : quanta < bound + 1 := by exact_mod_cast upper
  have lowerInt : -bound - 1 < quanta := by exact_mod_cast lower
  exact abs_le.mpr (by omega)

private theorem absolute_budget_of_count (q : List ℤ) (bound : ℤ)
    (perContribution : ∀ value ∈ q, |value| ≤ bound) :
    (q.map (fun value => |value|)).sum ≤ q.length * bound := by
  induction q with
  | nil => simp
  | cons value rest ih =>
    have head := perContribution value (by simp)
    have tail := ih (fun x hx => perContribution x (by simp [hx]))
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    nlinarith

/-- A count cap N ≤ 2^n and per-integer guard 2^(62-n) imply the inclusive
2^62 absolute budget and every prefix bound. `n` is a natural with n ≤ 62. -/
theorem fixed_point_cap_partial_bound (q : List ℤ) (n : ℕ) (exponent : n ≤ 62)
    (count : q.length ≤ 2 ^ n)
    (perContribution : ∀ value ∈ q, |value| ≤ (2 : ℤ) ^ (62 - n)) :
    (q.map (fun value => |value|)).sum ≤ (2 : ℤ) ^ 62 ∧
      ∀ prefixLength, |(q.take prefixLength).sum| ≤ (2 : ℤ) ^ 62 := by
  have countInt : (q.length : ℤ) ≤ (2 : ℤ) ^ n := by exact_mod_cast count
  have powers : (2 : ℤ) ^ n * 2 ^ (62 - n) = 2 ^ 62 := by
    rw [← pow_add, Nat.add_sub_of_le exponent]
  have budget := (absolute_budget_of_count q _ perContribution).trans
    (mul_le_mul_of_nonneg_right countInt (by positivity : (0 : ℤ) ≤ 2 ^ (62 - n)))
  rw [powers] at budget
  exact ⟨budget, fixed_point_partial_bound q _ budget⟩

/-- Quantize contributions individually, then decode the integer sum. This exact
channel error bound assumes only s > 0 and nearest rounding, not a parity rule. -/
theorem quantized_sum_error_bound {N : ℕ} (contribution : Fin N → ℝ)
    (quanta : Fin N → ℤ) (scale : ℝ) (positiveScale : 0 < scale)
    (nearest : ∀ i, IsNearestInteger (contribution i / scale) (quanta i)) :
    |scale * ((∑ i, quanta i : ℤ) : ℝ) - ∑ i, contribution i| ≤ N * scale / 2 := by
  have each : ∀ i, |scale * (quanta i : ℝ) - contribution i| ≤ scale / 2 := by
    intro i
    have error := nearest_integer_error _ _ (nearest i)
    have identity : scale * (quanta i : ℝ) - contribution i =
        -(scale * (contribution i / scale - (quanta i : ℝ))) := by
      field_simp
      ring
    rw [identity, abs_neg, abs_mul, abs_of_pos positiveScale]
    nlinarith
  calc
    _ = |∑ i, (scale * (quanta i : ℝ) - contribution i)| := by
      rw [Int.cast_sum, Finset.sum_sub_distrib, ← Finset.mul_sum]
    _ ≤ ∑ i, |scale * (quanta i : ℝ) - contribution i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin N, scale / 2 := Finset.sum_le_sum (fun i _ => each i)
    _ = _ := by simp; ring

#print axioms fixed_point_cap_partial_bound
#print axioms quantized_contribution_bound
#print axioms quantized_sum_error_bound

end Fluidity
