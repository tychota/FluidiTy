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
      return "A solid is thinner than 1024 contact tolerances. Make it thicker or move the scene closer to the "
             "origin.";
    case ContactError::kBuffersTooSmall:
      return "The scene has more particles or solids than the contact buffers were built for. Use a smaller scene or "
             "raise the particle limit.";
    case ContactError::kRunOnCallFailed:
      return "The chosen processor cannot run here, or a call to it failed during contact. Check the GPU and its "
             "driver, or choose the CPU.";
    case ContactError::kOutOfMemory:
      return "There is not enough memory for the contact buffers. Use a smaller scene, free memory, or choose the "
             "other processor.";
  }
  return "An unknown contact error happened.";
}

std::string_view ToMessage(ContactCloseCall close_call) {
  switch (close_call) {
    case ContactCloseCall::kNone:
      return "There is no close call.";
    case ContactCloseCall::kTooManyHits:
      return "A particle hit more solids in one step than contact handles. Use a shorter step or move the solids "
             "apart.";
    case ContactCloseCall::kSqueezedBetweenSolids:
      return "A particle is squeezed between two solids that close on it. Use a shorter step or keep the solids "
             "apart.";
    case ContactCloseCall::kSlowApproachToTurningSolid:
      return "A particle approaches a turning solid too slowly to find the hit safely. Use a shorter step or turn "
             "the solid more slowly.";
    case ContactCloseCall::kTurningSolidNearOtherSolid:
      return "A turning solid is too near another solid for a safe slide. Use a shorter step or keep the two "
             "solids apart.";
    case ContactCloseCall::kSecondReactingSolid:
      return "A particle touched a second solid that takes a reaction in the same step. Use a shorter step or keep "
             "those solids apart.";
    case ContactCloseCall::kSolidTurnsTooFar:
      return "A solid turns too far in one step. Use a shorter step or turn the solid more slowly.";
    case ContactCloseCall::kEndsTooCloseToSolid:
      return "A particle would end the step too close to a solid. Use a shorter step or slow the solid down.";
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
  }
  return "unknown";
}

}  // namespace fluidity::contact
