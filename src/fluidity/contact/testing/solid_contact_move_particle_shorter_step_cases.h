// Test-only: the helpers that the unit tests and the functional tests of the shorter-step rule of MoveParticle share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    MoveParticleCall, ComputeMoveWithHalvings, ComputeCountHalvingsToSuccess, CatalogueFailureCounts
// Uses:    the headers it includes
// Used by: src/fluidity/contact/solid_contact.move_particle_shorter_step.test.cc and
//          tests/contact/move_particle_shorter_step_on_catalogue.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <set>
#include <string>

#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"

namespace fluidity::contact {

// What one MoveParticle call returned.
struct MoveParticleCall {
  ContactError error = ContactError::kNone;
  SolidContactResult result{};
  ContactFailure failure{};
};

// Runs MoveParticle on a scene with its step divided by 2^halving_count, every other input kept, the contact tolerance
// included.
inline MoveParticleCall ComputeMoveWithHalvings(const MoveParticleCase& scene, int halving_count) {
  SolidContactInputs inputs = ToSolidContactInputs(scene);
  inputs.step_duration = std::ldexp(inputs.step_duration, -halving_count);
  MoveParticleCall call;
  call.error = MoveParticle(inputs, &call.result, &call.failure);
  return call;
}

// Halves the scene's step until MoveParticle succeeds, at most max_halving_count times, and returns how many halvings
// that took, or max_halving_count + 1 when none succeeded. Fails the calling test unless every failure on the way is
// kTooCloseToCall with that close call and can_shorter_step_cure true.
inline int
ComputeCountHalvingsToSuccess(const MoveParticleCase& scene, ContactCloseCall close_call, int max_halving_count) {
  for (int halving_count = 0; halving_count <= max_halving_count; ++halving_count) {
    const MoveParticleCall call = ComputeMoveWithHalvings(scene, halving_count);
    if (call.error == ContactError::kNone) {
      return halving_count;
    }
    EXPECT_EQ(call.error, ContactError::kTooCloseToCall) << "after " << halving_count << " halvings";
    EXPECT_EQ(call.failure.close_call, close_call) << "after " << halving_count << " halvings";
    EXPECT_TRUE(call.failure.can_shorter_step_cure) << "after " << halving_count << " halvings";
  }
  return max_halving_count + 1;
}

// What the failures of the catalogue gave.
struct CatalogueFailureCounts {
  int lasting = 0;  // can_shorter_step_cure false.
  int cured = 0;  // can_shorter_step_cure true.
  int most_halvings = 0;  // The most halvings a cured failure needed.
  std::set<ContactError> errors;  // Every error met.
};

}  // namespace fluidity::contact
