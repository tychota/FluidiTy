import Mathlib.Basic.Real.Basic

namespace Fluidity

/-- Value types for persistent fields owned by later simulation stages. The
transaction never inspects these values, but they participate in state equality. -/
structure LiquidDomainTypes where
  particleMaterialHistory : Type
  bodyState : Type
  cellPhysicalInventories : Type
  componentLineage : Type
  gasScalarsAndEnthalpy : Type
  whitewaterEvents : Type
  whitewaterAccumulators : Type

/-- Exact geometry of the uniform MAC lattice, in meters. -/
structure LiquidGridGeometry where
  worldOrigin : Fin 3 → ℝ
  cellWidth : ℝ

/-- Ordered particle records include the APIC velocity slopes and PIC history. -/
structure LiquidParticleFields where
  id : ℕ
  tag : ℕ
  worldPosition : Fin 3 → ℝ
  worldVelocity : Fin 3 → ℝ
  velocitySlopes : Fin 3 → Fin 3 → ℝ
  affineEnabled : Bool
  mass : ℝ
  referenceVolume : ℝ
  volumeRatio : ℝ
  sideTag : ℤ
  picInitialized : Bool

/-- Owned block keys and hysteresis ages, in accepted steps. -/
structure LiquidBlockFields where
  blockIndex : Fin 3 → ℤ
  generation : ℕ
  activationAge : ℕ
  inactiveAge : ℕ

inductive LiquidCellOccupancy where
  | air | liquid | solid

/-- The warm-start pressure belongs to the committed cell, in pascals. -/
structure LiquidCellFields where
  cellIndex : Fin 3 → ℤ
  occupancy : LiquidCellOccupancy
  previousPressure : ℝ

inductive LiquidCarrierValidity where
  | missing | valid | predicted | lowMass

/-- A signed lattice coordinate and normal axis identify an owned face. Its
carrier velocity in meters per second and eligibility survive to the next step. -/
structure LiquidFaceFields where
  faceLatticeIndex : Fin 3 → ℤ
  normalAxis : Fin 3
  faceSlot : ℕ
  carrierVelocity : ℝ
  validity : LiquidCarrierValidity
  sideTag : ℤ
  structurallyEligible : Bool

/-- The source's child ordinal and fractional remainder advance on acceptance. -/
structure LiquidSourceFields where
  sourceOrdinal : ℕ
  nextChildOrdinal : ℕ
  fractionalRemainder : ℝ

/-- Accepted time in seconds and the persistent version and schedule identities. -/
structure LiquidAcceptedClock where
  time : ℝ
  frameOrdinal : ℕ
  substepOrdinal : ℕ
  acceptedStepOrdinal : ℕ
  epoch : ℕ

/-- Particle, topology, pressure, carrier and source records retain their order.
The lists describe valid extents. Reservation sizes and unused slots are outside this value. -/
structure LiquidPersistentFields (domains : LiquidDomainTypes) where
  geometry : LiquidGridGeometry
  randomSeed : ℕ
  nextParticleId : ℕ
  particles : List LiquidParticleFields
  blocks : List LiquidBlockFields
  cells : List LiquidCellFields
  faces : List LiquidFaceFields
  sources : List LiquidSourceFields
  particleMaterialHistory : domains.particleMaterialHistory
  bodyState : domains.bodyState
  cellPhysicalInventories : domains.cellPhysicalInventories
  componentLineage : domains.componentLineage
  gasScalarsAndEnthalpy : domains.gasScalarsAndEnthalpy
  whitewaterEvents : domains.whitewaterEvents
  whitewaterAccumulators : domains.whitewaterAccumulators

/-- An accepted snapshot pairs physical fields with a clock. A trial supplies the
physical fields and duration without proposing an already advanced clock. -/
structure LiquidCommittedState (domains : LiquidDomainTypes) where
  fields : LiquidPersistentFields domains
  clock : LiquidAcceptedClock

/-- Shared inputs for the accepted-step RNG key. Source and child ordinals for
individual draws are supplied by the persistent source records. -/
structure LiquidAcceptedStepKey where
  randomSeed : ℕ
  acceptedStepOrdinal : ℕ

/-- Rejection may carry a smaller retry duration. Capacity and fatal rejection
also preserve the committed state and need no retry duration in this model. -/
inductive LiquidAttemptOutcome (domains : LiquidDomainTypes) where
  | rejected (retryDuration : Option ℝ)
  | accepted (trialFields : LiquidPersistentFields domains) (acceptedDuration : ℝ) (completesFrame : Bool)

def LiquidAttemptOutcome.IsRejected {domains : LiquidDomainTypes} :
    LiquidAttemptOutcome domains → Prop
  | .rejected _ => True
  | .accepted .. => False

/-- Diagnostics and control belong to the attempt outside the committed snapshot. -/
structure LiquidAttempt (domains : LiquidDomainTypes) (Diagnostics : Type) where
  outcome : LiquidAttemptOutcome domains
  diagnostics : Diagnostics

/-- Publish a successful isolated trial once. Trial validation supplies the
accepted duration and frame boundary. Rejected control never enters the state. -/
noncomputable def finishLiquidAttempt {domains : LiquidDomainTypes} {Diagnostics : Type}
    (committed : LiquidCommittedState domains) (attempt : LiquidAttempt domains Diagnostics) :
    LiquidCommittedState domains :=
  match attempt.outcome with
  | .rejected _ => committed
  | .accepted trialFields acceptedDuration completesFrame =>
      { fields := trialFields
        clock :=
          { time := committed.clock.time + acceptedDuration
            frameOrdinal := if completesFrame then committed.clock.frameOrdinal + 1 else committed.clock.frameOrdinal
            substepOrdinal := if completesFrame then 0 else committed.clock.substepOrdinal + 1
            acceptedStepOrdinal := committed.clock.acceptedStepOrdinal + 1
            epoch := committed.clock.epoch + 1 } }

/-- A single reject preserves the entire committed value, including the clock. -/
theorem liquid_reject_preserves_committed_state {domains : LiquidDomainTypes} {Diagnostics : Type}
    (committed : LiquidCommittedState domains) (diagnostics : Diagnostics) (retryDuration : Option ℝ) :
    finishLiquidAttempt committed ⟨.rejected retryDuration, diagnostics⟩ = committed := rfl

/-- Every finite sequence of rejects preserves the same value even when the
attempt diagnostics or retry durations vary. -/
theorem liquid_finite_rejects_preserve_committed_state {domains : LiquidDomainTypes} {Diagnostics : Type}
    (committed : LiquidCommittedState domains) (attempts : List (LiquidAttempt domains Diagnostics))
    (allRejected : ∀ attempt ∈ attempts, attempt.outcome.IsRejected) :
    attempts.foldl finishLiquidAttempt committed = committed := by
  induction attempts with
  | nil => rfl
  | cons first remaining remainingUnchanged =>
      have firstRejected := allRejected first (by simp)
      have firstUnchanged : finishLiquidAttempt committed first = committed := by
        cases outcome : first.outcome with
        | rejected retryDuration => simp [finishLiquidAttempt, outcome]
        | accepted trialFields acceptedDuration completesFrame =>
            simp [outcome, LiquidAttemptOutcome.IsRejected] at firstRejected
      simp only [List.foldl_cons, firstUnchanged]
      apply remainingUnchanged
      intro attempt inRemaining
      exact allRejected attempt (List.mem_cons_of_mem first inRemaining)

/-- One acceptance installs the trial fields and increments the clock from the
old committed clock. Frame closure resets the substep in that same transition. -/
theorem liquid_accept_advances_clock_once {domains : LiquidDomainTypes} {Diagnostics : Type}
    (committed : LiquidCommittedState domains) (trialFields : LiquidPersistentFields domains)
    (diagnostics : Diagnostics) (acceptedDuration : ℝ) (completesFrame : Bool) :
    let accepted := finishLiquidAttempt committed
      ⟨.accepted trialFields acceptedDuration completesFrame, diagnostics⟩
    accepted.fields = trialFields ∧
      accepted.clock.time = committed.clock.time + acceptedDuration ∧
      accepted.clock.acceptedStepOrdinal = committed.clock.acceptedStepOrdinal + 1 ∧
      accepted.clock.epoch = committed.clock.epoch + 1 ∧
      accepted.clock.frameOrdinal = (if completesFrame then committed.clock.frameOrdinal + 1
        else committed.clock.frameOrdinal) ∧
      accepted.clock.substepOrdinal = (if completesFrame then 0 else committed.clock.substepOrdinal + 1) :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Both outcomes must be evaluations of the same deterministic trial function
under the same input and key. Its fixed environment includes the execution
artifact, settings and requested duration. Diagnostics may differ. -/
theorem liquid_same_state_key_replays_trial {domains : LiquidDomainTypes} {Diagnostics : Type}
    (runTrial : LiquidCommittedState domains → LiquidAcceptedStepKey → LiquidAttemptOutcome domains)
    (firstInput replayInput : LiquidCommittedState domains) (firstKey replayKey : LiquidAcceptedStepKey)
    (firstAttempt replayAttempt : LiquidAttempt domains Diagnostics)
    (sameInput : firstInput = replayInput) (sameKey : firstKey = replayKey)
    (firstResult : firstAttempt.outcome = runTrial firstInput firstKey)
    (replayedResult : replayAttempt.outcome = runTrial replayInput replayKey) :
    firstAttempt.outcome = replayAttempt.outcome ∧
      finishLiquidAttempt firstInput firstAttempt = finishLiquidAttempt replayInput replayAttempt := by
  rw [sameInput, sameKey] at firstResult
  have sameOutcome := firstResult.trans replayedResult.symm
  exact ⟨sameOutcome, by simp only [finishLiquidAttempt, sameInput, sameOutcome]⟩

/-- Prepare a separate adaptation candidate while retaining the input snapshot.
The proposal has no clock, and this operation cannot publish it. -/
def proposeLiquidAdaptation {domains : LiquidDomainTypes} (inputSnapshot : LiquidCommittedState domains)
    (adapt : LiquidCommittedState domains → LiquidPersistentFields domains) :
    LiquidCommittedState domains × LiquidPersistentFields domains :=
  (inputSnapshot, adapt inputSnapshot)

/-- Adaptation produces only a candidate and leaves its original snapshot intact. -/
theorem liquid_adaptation_preserves_input_snapshot {domains : LiquidDomainTypes}
    (inputSnapshot : LiquidCommittedState domains)
    (adapt : LiquidCommittedState domains → LiquidPersistentFields domains) :
    (proposeLiquidAdaptation inputSnapshot adapt).1 = inputSnapshot ∧
      (proposeLiquidAdaptation inputSnapshot adapt).2 = adapt inputSnapshot :=
  ⟨rfl, rfl⟩

#print axioms liquid_reject_preserves_committed_state
#print axioms liquid_finite_rejects_preserve_committed_state
#print axioms liquid_accept_advances_clock_once
#print axioms liquid_same_state_key_replays_trial
#print axioms liquid_adaptation_preserves_input_snapshot

end Fluidity
