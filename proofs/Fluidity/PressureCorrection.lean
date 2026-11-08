import Fluidity.Pressure

open scoped BigOperators

namespace Fluidity

/-- Frozen integrated link data. The lift is J on an internal face, -p_ext at a
Dirichlet tail, +p_ext at a Dirichlet head. Sealed faces have zero open area.
Area is m², scale is m/(Pa s), velocities are m/s, pressure/lift are Pa. -/
structure PressureLinks (n m : ℕ) where
  incidence : Fin n → Fin m → ℝ
  openArea : Fin m → ℝ
  fullArea : Fin m → ℝ
  velocityScale : Fin m → ℝ
  predictedVelocity : Fin m → ℝ
  wallVelocity : Fin m → ℝ
  lift : Fin m → ℝ
  compliance : Fin n → ℝ
  source : Fin n → ℝ
  previousPressure : Fin n → ℝ

/-- Conductance is frozen from the same open area and velocity scale as correction. -/
def PressureLinks.conductance {n m : ℕ} (l : PressureLinks n m) (f : Fin m) : ℝ :=
  l.openArea f * l.velocityScale f

/-- Corrected open fluid flux in m³/s, for any candidate pressure. -/
def PressureLinks.correctedFlux {n m : ℕ} (l : PressureLinks n m) (p : Fin n → ℝ) (f : Fin m) : ℝ :=
  l.openArea f * (l.predictedVelocity f + l.velocityScale f *
    ((∑ i, l.incidence i f * p i) + l.lift f))

/-- Prescribed blocked-area wall flux, integrated outward in m³/s. -/
def PressureLinks.wallFlux {n m : ℕ} (l : PressureLinks n m) (i : Fin n) : ℝ :=
  ∑ f, l.incidence i f * ((l.fullArea f - l.openArea f) * l.wallVelocity f)

/-- Original integrated RHS, including the signed link lift and old compliance pressure. -/
def PressureLinks.rhs {n m : ℕ} (l : PressureLinks n m) (i : Fin n) : ℝ :=
  l.source i - (∑ f, l.incidence i f *
    (l.openArea f * l.predictedVelocity f + l.conductance f * l.lift f)) -
    l.wallFlux i + l.compliance i * l.previousPressure i

/-- Physical integrated defect, independently evaluated from corrected flux. -/
def PressureLinks.defect {n m : ℕ} (l : PressureLinks n m) (p : Fin n → ℝ) (i : Fin n) : ℝ :=
  (∑ f, l.incidence i f * l.correctedFlux p f) + l.wallFlux i - l.source i +
    l.compliance i * (p i - l.previousPressure i)

/-- Algebraic residual of the original integrated system in m³/s. -/
def PressureLinks.residual {n m : ℕ} (l : PressureLinks n m) (p : Fin n → ℝ) (i : Fin n) : ℝ :=
  l.rhs i - pressureAction l.incidence l.conductance l.compliance p i

/-- For any candidate pressure, the independently defined physical integrated defect equals
minus the original algebraic residual. Assembly and correction share all frozen link data;
no exact solve, convergence, positivity, or small-residual premise is used. -/
theorem pressure_corrected_flux_defect_eq_neg_residual {n m : ℕ} (l : PressureLinks n m)
    (p : Fin n → ℝ) (i : Fin n) : l.defect p i = -l.residual p i := by
  have flux : (∑ f, l.incidence i f * l.correctedFlux p f) =
      (∑ f, l.incidence i f * (l.openArea f * l.predictedVelocity f + l.conductance f * l.lift f)) +
      ∑ f, l.incidence i f * l.conductance f * (∑ j, l.incidence j f * p j) := by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro f _
    simp only [PressureLinks.correctedFlux, PressureLinks.conductance]
    ring
  simp only [PressureLinks.defect, PressureLinks.residual, PressureLinks.rhs, pressureAction, flux]
  ring

#print axioms pressure_corrected_flux_defect_eq_neg_residual

end Fluidity
