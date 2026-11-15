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

#include <cstdint>

namespace fluidity::contact {

// Why a contact call failed. The first value is no error. Values keep their numbers: a new value is added last.
enum class ContactError : std::uint8_t {
  kNone,  // No error. No function returns it as an error.
  kBadInput,  // A value is not finite or is out of range.
  kStartsInsideSolid,  // The start is deeper inside a solid than start_overlap_tolerance allows.
  kTooCloseToCall,  // No safe answer for this step length. The caller should shorten the step.
  kOnWrongSideOfPlate,  // The start is on the other side of the plate the particle is committed to.
  kSolidTooThin,  // A solid or cell is thinner than 1024 contact tolerances: not enough resolution.
  kBuffersTooSmall,  // More particles or solids than the contact buffers were built for.
  kRunOnCallFailed,  // A call to the chosen processor failed.
  // The run's RunMemory or the processor could not hold a buffer. Only BuildContactBuffers returns it; the context
  // magnitude is the number of bytes asked for. Last value.
  kOutOfMemory,
};

// Why a call returned kTooCloseToCall. The first value means the failure is not a close call.
enum class ContactCloseCall : std::uint8_t {
  kNone,  // The failure is not a close call.
  kTooManyHits,  // The path hit more solids than one step allows.
  kSqueezedBetweenSolids,  // Two solids close on the particle and no position is safe between them.
  kSlowApproachToTurningSolid,  // The particle approaches a turning solid too slowly to find the hit safely.
  kTurningSolidNearOtherSolid,  // A turning solid is near another solid, so the slide along it is not safe.
  kSecondReactingSolid,  // A second solid that reports a reaction was hit; one record cannot hold both.
  kSolidTurnsTooFar,  // A solid turns more than one step allows.
  kEndsTooCloseToSolid,  // The end position is closer to a solid than the contact tolerance allows.
};

// A failure with its context. Fields that do not apply keep their default.
struct ContactFailure {
  ContactError error = ContactError::kNone;
  ContactCloseCall close_call = ContactCloseCall::kNone;  // Set only when error is kTooCloseToCall.
  std::uint64_t particle_id = 0;  // 0 when the failure is not about one particle.
  std::uint32_t solid_id = 0;  // 0 when the failure is not about one solid.
  // m. The overlap depth or the offending length; the bytes asked for kOutOfMemory. 0 when not relevant.
  double magnitude = 0;
};

}  // namespace fluidity::contact
