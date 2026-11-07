import Mathlib.Algebra.Order.Ring.Int
import Mathlib.Algebra.Order.Group.Abs
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Data.List.Perm.Basic

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

end Fluidity
