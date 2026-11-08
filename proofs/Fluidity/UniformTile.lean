import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace Fluidity

/-- Signed sort bin in normalized coordinates, with no truncation toward zero. -/
noncomputable def uniformSortBin (sortedPosition : ℝ) : ℤ := ⌊(sortedPosition - 3 / 2) / 4⌋

/-- Sorting and the half-cell motion bound imply this half-open trial interval. -/
theorem uniform_trial_interval (sortedPosition trialPosition : ℝ) (halfCellMotion : |trialPosition - sortedPosition| ≤ 1 / 2) :
    4 * (uniformSortBin sortedPosition : ℝ) + 1 ≤ trialPosition ∧
      trialPosition < 4 * (uniformSortBin sortedPosition : ℝ) + 6 := by
  have lower := Int.floor_le ((sortedPosition - 3 / 2) / 4)
  have upper := Int.lt_floor_add_one ((sortedPosition - 3 / 2) / 4)
  obtain ⟨moveLower, moveUpper⟩ := abs_le.mp halfCellMotion
  unfold uniformSortBin
  constructor <;> linarith

/-- All three quadratic nodes at either MAC shift and all four cubic-normal nodes
fit the same eight-index tile. This is an exact-real inclusion, not allocation. -/
theorem uniform_mac_tile_contains_stencil (sortedPosition trialPosition : ℝ)
    (halfCellMotion : |trialPosition - sortedPosition| ≤ 1 / 2) :
    (∀ shift : ℝ, shift = 0 ∨ shift = 1 / 2 → ∀ stencilOrdinal : Fin 3,
      4 * uniformSortBin sortedPosition ≤ ⌊trialPosition - shift - 1 / 2⌋ + stencilOrdinal.val ∧
      ⌊trialPosition - shift - 1 / 2⌋ + stencilOrdinal.val < 4 * uniformSortBin sortedPosition + 8) ∧
    (∀ stencilOrdinal : Fin 4,
      4 * uniformSortBin sortedPosition ≤ ⌊trialPosition⌋ - 1 + stencilOrdinal.val ∧
      ⌊trialPosition⌋ - 1 + stencilOrdinal.val < 4 * uniformSortBin sortedPosition + 8) := by
  obtain ⟨lower, upper⟩ := uniform_trial_interval sortedPosition trialPosition halfCellMotion
  constructor
  · intro shift shifts stencilOrdinal
    have shiftBounds : 0 ≤ shift ∧ shift ≤ 1 / 2 := by
      rcases shifts with rfl | rfl <;> norm_num
    have firstLower : 4 * uniformSortBin sortedPosition ≤ ⌊trialPosition - shift - 1 / 2⌋ := by
      apply Int.le_floor.mpr
      push_cast
      linarith [shiftBounds.2]
    have firstUpper : ⌊trialPosition - shift - 1 / 2⌋ < 4 * uniformSortBin sortedPosition + 6 := by
      apply Int.floor_lt.mpr
      push_cast
      linarith [shiftBounds.1]
    have := stencilOrdinal.isLt
    omega
  · intro stencilOrdinal
    have firstLower : 4 * uniformSortBin sortedPosition + 1 ≤ ⌊trialPosition⌋ := by
      apply Int.le_floor.mpr
      push_cast
      exact lower
    have firstUpper : ⌊trialPosition⌋ < 4 * uniformSortBin sortedPosition + 6 := by
      apply Int.floor_lt.mpr
      push_cast
      exact upper
    have := stencilOrdinal.isLt
    omega

/-- Axiswise inclusion gives the full Cartesian product (including 2D and 3D).
`usesCubicNormal` selects the normal cubic stencil; other axes use quadratic MAC shifts. -/
theorem uniform_mac_tile_product {d : ℕ} (sortedPosition trialPosition shift : Fin d → ℝ)
    (usesCubicNormal : Fin d → Bool) (stencilOrdinal : Fin d → ℕ)
    (halfCellMotion : ∀ axis, |trialPosition axis - sortedPosition axis| ≤ 1 / 2)
    (shifts : ∀ axis, shift axis = 0 ∨ shift axis = 1 / 2)
    (ordinals : ∀ axis, stencilOrdinal axis < if usesCubicNormal axis then 4 else 3) :
    ∀ axis, let index := if usesCubicNormal axis then ⌊trialPosition axis⌋ - 1 + stencilOrdinal axis
                        else ⌊trialPosition axis - shift axis - 1 / 2⌋ + stencilOrdinal axis
      4 * uniformSortBin (sortedPosition axis) ≤ index ∧
      index < 4 * uniformSortBin (sortedPosition axis) + 8 := by
  intro axis
  have h := uniform_mac_tile_contains_stencil (sortedPosition axis) (trialPosition axis) (halfCellMotion axis)
  cases hc : usesCubicNormal axis
  · simpa [hc] using h.1 (shift axis) (shifts axis) ⟨stencilOrdinal axis, by simpa [hc] using ordinals axis⟩
  · simpa [hc] using h.2 ⟨stencilOrdinal axis, by simpa [hc] using ordinals axis⟩

#print axioms uniform_mac_tile_contains_stencil
#print axioms uniform_mac_tile_product

end Fluidity
