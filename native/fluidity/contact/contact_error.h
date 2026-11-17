// Why a contact call failed, and the context of the failure.
//
// Purpose: give the contact module its error enum, the reason behind a close call, and one small record that
//   carries both with the particle and the solid they are about.
//
// Idea: contact either moves a particle safely or says why it cannot. A failure is a ContactError. The most common
//   one, kTooCloseToCall, means the answer for this step length is not safe, so the step is shortened and tried
//   again. A close call is the reason behind it. ContactFailure holds both, plus the ids and one length. It is a
//   plain struct that never allocates, so GPU code can fill it. The text of each value is in
//   contact/contact_error_message.h, which only the CPU may include.
//
// Words:
//   close call  a step that contact cannot answer safely; the step is shortened and tried again
// Owns:    ContactError, ContactCloseCall, ContactFailure
// Uses:    nothing
// Used by: every contact function that can fail, and the step, which maps a failure to a rejected try
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>

#pragma once

#include <cstddef>
#include <cstdint>

namespace fluidity::contact {

// Why a contact call failed. The first value is no error. Values keep their numbers: a new value is added last.
enum class ContactError : std::uint8_t {
  kNone,  // No error. No function returns it as an error.
  kBadInput,  // A value is not finite or is out of range.
  kStartsInsideSolid,  // The start is deeper inside a solid than start_overlap_tolerance allows.
  kTooCloseToCall,  // No safe answer for this step length. The caller should shorten the step.
  kOnWrongSideOfPlate,  // The start is on the other side of the plate the particle is committed to.
  kSolidTooThin,  // A solid is thinner, or a cell narrower, than 1024 contact tolerances.
  kBuffersTooSmall,  // More particles or solids than the contact buffers were built for.
  kRunOnCallFailed,  // A call to the chosen processor failed.
  // The run's RunMemory or the processor could not hold a buffer. Only BuildContactBuffers returns it; the context
  // magnitude is the number of bytes asked for.
  kOutOfMemory,
  // A start has no point at the contact tolerance from every surface it is near within the move cap: two solids facing
  // each other less than 2 contact tolerances apart, a start within rounding of the contact point of a ball resting on
  // a floor, a third solid within the rounding window of such a contact's rest point, or a pocket too small; or the
  // start is near surfaces facing more than four different ways, or near more than 16 surfaces, which one move cannot
  // read at once. A scene error that a shorter step does not cure; only MoveParticle returns it, at the start. The
  // context names the near solid latest in array order, magnitude 0. Last value.
  kStartsInNarrowGap,
};

// Why a call returned kTooCloseToCall. The first value means the failure is not a close call.
enum class ContactCloseCall : std::uint8_t {
  kNone,  // The failure is not a close call.
  // A fifth hit in one step (kMaxHitsPerStep), or more surfaces or directions at one hit than one response reads.
  kTooManyHits,
  // No velocity leaves every surface the particle is on (solids closing on it), the velocity that leaves them is
  // faster than kMaxSqueezeOutSpeedFactor times the speeds involved, or, with several surfaces, the rounding of the
  // response over the time left could bring the particle more than a sixty-fourth of the tolerance nearer a solid.
  kSqueezedBetweenSolids,
  // The stepping search toward a turning solid ran out of advances: the particle hovers near the solid or grazes it,
  // with almost no motion along its normal.
  kSlowApproachToTurningSolid,
  // After a hit on a turning solid, another solid is within the closing bound of the carry in the turning solid's axes,
  // which sweeps nothing, so the carry is not safe.
  kTurningSolidNearOtherSolid,
  kSecondReactingSolid,  // A second solid that reports a reaction is pushed in one step; one record cannot hold both.
  kSolidTurnsTooFar,  // A solid turns more than one step allows.
  kEndsTooCloseToSolid,  // The end position is closer to a solid than half the contact tolerance (the end check).
  // The particle, or the solid it meets, moves too far in one step for the rounding of a response to stay below a
  // sixty-fourth of the contact tolerance over the time left (MoveParticle, at a hit on one surface). Last value.
  kMovesTooFastForStep,
};

// A failure with its context. Fields that do not apply keep their default.
struct ContactFailure {
  ContactError error = ContactError::kNone;
  ContactCloseCall close_call = ContactCloseCall::kNone;  // Set only when error is kTooCloseToCall.
  // The failure is about one particle, named by particle_id. false for a failure of a whole call or of one move of
  // MoveParticle, which knows no id; particle_id is then 0, which is also a valid id, so this flag tells them apart.
  bool has_particle_id = false;
  std::uint64_t particle_id = 0;  // The particle's stable id when has_particle_id; 0 otherwise.
  std::uint32_t solid_id = 0;  // 0 when the failure is not about one solid.
  // m. The overlap depth or the offending length; the bytes asked for kOutOfMemory. 0 when not relevant.
  double magnitude = 0;
};
// The GPU writes one failure per particle and the CPU reads the reported one back byte for byte, so both compilers must
// lay it out alike: error at 0, close_call at 1, has_particle_id at 2, particle_id at 8, solid_id at 16, magnitude at
// 24.
static_assert(sizeof(ContactFailure) == 32, "the contact buffers charge 32 bytes per failure");
static_assert(offsetof(ContactFailure, has_particle_id) == 2, "the CPU and the GPU read one layout of ContactFailure");
static_assert(offsetof(ContactFailure, particle_id) == 8, "the CPU and the GPU read one layout of ContactFailure");
static_assert(offsetof(ContactFailure, solid_id) == 16, "the CPU and the GPU read one layout of ContactFailure");
static_assert(offsetof(ContactFailure, magnitude) == 24, "the CPU and the GPU read one layout of ContactFailure");

}  // namespace fluidity::contact
