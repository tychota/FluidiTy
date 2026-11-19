#include "fluidity/contact/contact_error_message.h"

#include <string_view>

#include "fluidity/contact/contact_error.h"

namespace fluidity::contact {

std::string_view ToMessage(ContactError error) {
  switch (error) {
    case ContactError::kNone:
      return "There is no contact error.";
    case ContactError::kBadInput:
      return "A contact value is not a finite number or is out of range. Check the solids, the friction and the "
             "step length in the scene.";
    case ContactError::kStartsInsideSolid:
      return "A particle starts deeper inside a solid than the overlap allowed. Move the liquid or the solid so they "
             "do not overlap.";
    case ContactError::kTooCloseToCall:
      return "Contact cannot answer this step length safely. Use a shorter step.";
    case ContactError::kOnWrongSideOfPlate:
      return "A particle is on the other side of the plate it belongs to. Keep the liquid on one side of each thin "
             "plate, or move the plate.";
    case ContactError::kSolidTooThin:
      return "A solid is thinner, or a grid cell narrower, than 1024 contact tolerances. Make the solid thicker or "
             "the cells larger, or move the scene closer to the origin.";
    case ContactError::kBuffersTooSmall:
      return "The scene has more particles or solids than the contact buffers were built for. Use a smaller scene or "
             "raise the particle limit.";
    case ContactError::kRunOnCallFailed:
      return "The chosen processor cannot run here, or a call to it failed during contact. Check the GPU and its "
             "driver, or choose the CPU.";
    case ContactError::kOutOfMemory:
      return "There is not enough memory for the contact buffers. Use a smaller scene, free memory, or choose the "
             "other processor.";
    case ContactError::kStartsInNarrowGap:
      return "A particle starts where it cannot be kept one contact tolerance from every solid around it: the gap is "
             "too narrow, or the particle is near solid surfaces facing more than four different ways, or near more "
             "than 16 surfaces. Move the solids apart, or the liquid out of the gap.";
  }
  return "An unknown contact error happened.";
}

std::string_view ToMessage(ContactCloseCall close_call) {
  switch (close_call) {
    case ContactCloseCall::kNone:
      return "There is no close call.";
    case ContactCloseCall::kTooManyHits:
      return "A particle met solids too many times in one step, or too many solid surfaces at once, for contact to "
             "handle. Use a shorter step or move the solids apart.";
    case ContactCloseCall::kSqueezedBetweenSolids:
      return "A particle is squeezed between solids that close on it, or would be pushed out of them too fast. Use a "
             "shorter step or keep the solids apart.";
    case ContactCloseCall::kSlowApproachToTurningSolid:
      return "A particle hovers near a turning solid or grazes it, too slowly for the hit to be found safely. Use a "
             "shorter step or turn the solid more slowly.";
    case ContactCloseCall::kTurningSolidNearOtherSolid:
      return "A turning solid is too near another solid to carry a particle safely along it. Use a shorter step or "
             "keep the two solids apart.";
    case ContactCloseCall::kSecondReactingSolid:
      return "A particle touched a second solid that takes a reaction in the same step. Use a shorter step or keep "
             "those solids apart.";
    case ContactCloseCall::kSolidTurnsTooFar:
      return "A solid turns too far in one step. Use a shorter step or turn the solid more slowly.";
    case ContactCloseCall::kEndsTooCloseToSolid:
      return "A particle would end the step too close to a solid. Use a shorter step or slow the solid down.";
    case ContactCloseCall::kMovesTooFastForStep:
      return "A particle, or a solid near it, moves too far in one step for contact to keep the particle outside the "
             "solids safely. Use a shorter step or slow the particle or the solid down.";
  }
  return "An unknown close call happened.";
}

std::string_view ToName(ContactError error) {
  switch (error) {
    case ContactError::kNone:
      return "none";
    case ContactError::kBadInput:
      return "bad_input";
    case ContactError::kStartsInsideSolid:
      return "starts_inside_solid";
    case ContactError::kTooCloseToCall:
      return "too_close_to_call";
    case ContactError::kOnWrongSideOfPlate:
      return "on_wrong_side_of_plate";
    case ContactError::kSolidTooThin:
      return "solid_too_thin";
    case ContactError::kBuffersTooSmall:
      return "buffers_too_small";
    case ContactError::kRunOnCallFailed:
      return "run_on_call_failed";
    case ContactError::kOutOfMemory:
      return "out_of_memory";
    case ContactError::kStartsInNarrowGap:
      return "starts_in_narrow_gap";
  }
  return "unknown";
}

std::string_view ToName(ContactCloseCall close_call) {
  switch (close_call) {
    case ContactCloseCall::kNone:
      return "none";
    case ContactCloseCall::kTooManyHits:
      return "too_many_hits";
    case ContactCloseCall::kSqueezedBetweenSolids:
      return "squeezed_between_solids";
    case ContactCloseCall::kSlowApproachToTurningSolid:
      return "slow_approach_to_turning_solid";
    case ContactCloseCall::kTurningSolidNearOtherSolid:
      return "turning_solid_near_other_solid";
    case ContactCloseCall::kSecondReactingSolid:
      return "second_reacting_solid";
    case ContactCloseCall::kSolidTurnsTooFar:
      return "solid_turns_too_far";
    case ContactCloseCall::kEndsTooCloseToSolid:
      return "ends_too_close_to_solid";
    case ContactCloseCall::kMovesTooFastForStep:
      return "moves_too_fast_for_step";
  }
  return "unknown";
}

}  // namespace fluidity::contact
