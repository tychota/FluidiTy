import Fluidity.Pressure
import Fluidity.FixedPoint
import Fluidity.UniformTransfer
import Fluidity.UniformTile

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
