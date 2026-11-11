import Fluidity.UniformTransfer

open scoped BigOperators

noncomputable section

namespace Fluidity

/-- The two voxel-centre weights on one axis. Runtime checks require the
fraction in [0,1] and complete positive-weight support. -/
def volumeAxisWeights (fraction : ℝ) : Fin 2 → ℝ := ![1 - fraction, fraction]

/-- Eight unmasked trilinear weights. Coordinates are fractions, not metres;
the product order matches the native x/y/z deposition convention. -/
def trilinearVolumeWeight (x y z : ℝ) (corner : Fin 2 × Fin 2 × Fin 2) : ℝ :=
  volumeAxisWeights x corner.1 * volumeAxisWeights y corner.2.1 *
    volumeAxisWeights z corner.2.2

/-- Complete eight-corner support partitions unity in exact arithmetic. -/
theorem trilinear_volume_weight_partition (x y z : ℝ) :
    (∑ corner : Fin 2 × Fin 2 × Fin 2, trilinearVolumeWeight x y z corner) = 1 := by
  simp [trilinearVolumeWeight, volumeAxisWeights, Fintype.sum_prod_type, Fin.sum_univ_succ] <;> ring

#print axioms trilinear_volume_weight_partition

private theorem volume_axis_weight_nonnegative (fraction : ℝ)
    (lower : 0 ≤ fraction) (upper : fraction ≤ 1) (corner : Fin 2) :
    0 ≤ volumeAxisWeights fraction corner := by
  fin_cases corner <;> simp [volumeAxisWeights] <;> linarith

/-- Valid fractions give nonnegative weights. This does not authorize clipping,
dropping unknown donors, or renormalizing a partial stencil. -/
theorem trilinear_volume_weight_nonnegative (x y z : ℝ)
    (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1)
    (corner : Fin 2 × Fin 2 × Fin 2) :
    0 ≤ trilinearVolumeWeight x y z corner := by
  unfold trilinearVolumeWeight
  exact mul_nonneg
    (mul_nonneg (volume_axis_weight_nonnegative x hx.1 hx.2 corner.1)
      (volume_axis_weight_nonnegative y hy.1 hy.2 corner.2.1))
    (volume_axis_weight_nonnegative z hz.1 hz.2 corner.2.2)

#print axioms trilinear_volume_weight_nonnegative

/-- Depositing one represented count or gas/material volume over the complete
stencil preserves that inventory. Density integrates after multiplication by
voxel volume; floating conversion and accumulation require separate tests. -/
theorem trilinear_deposition_conserves_inventory (inventory x y z : ℝ) :
    (∑ corner : Fin 2 × Fin 2 × Fin 2, inventory * trilinearVolumeWeight x y z corner) = inventory := by
  rw [← Finset.mul_sum, trilinear_volume_weight_partition, mul_one]

#print axioms trilinear_deposition_conserves_inventory

/-- Summing any finite batch, including the empty batch, preserves its total
inventory. This is not a proof of liquid-to-gas mass transfer or physical gas
thermodynamics; whitewater is a one-way visual representation. -/
theorem trilinear_batch_conserves_inventory {particleCount : ℕ}
    (inventory x y z : Fin particleCount → ℝ) :
    (∑ particle, ∑ corner : Fin 2 × Fin 2 × Fin 2,
      inventory particle * trilinearVolumeWeight (x particle) (y particle) (z particle) corner) =
      ∑ particle, inventory particle := by
  simp_rw [trilinear_deposition_conserves_inventory]

#print axioms trilinear_batch_conserves_inventory

end Fluidity
