import Fluidity.Pressure
import Fluidity.FixedPoint

open scoped BigOperators

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
