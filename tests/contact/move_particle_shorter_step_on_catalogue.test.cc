#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <set>
#include <string>

#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/solid_contact_move_particle_shorter_step_cases.h"

namespace fluidity::contact {
namespace {

// Fails the calling test unless a failure with can_shorter_step_cure false comes back at T / 2, T / 4 and T / 16:
// the same error, close call and solid, still false.
void CheckLastingFailure(const MoveParticleCase& scene, const ContactFailure& failure, ContactError error) {
  for (const int halving_count : {1, 2, 4}) {
    SCOPED_TRACE("step divided by 2^" + std::to_string(halving_count));
    const MoveParticleCall shorter = ComputeMoveWithHalvings(scene, halving_count);
    EXPECT_EQ(shorter.error, error);
    EXPECT_EQ(shorter.failure.close_call, failure.close_call);
    EXPECT_EQ(shorter.failure.solid_id, failure.solid_id);
    EXPECT_FALSE(shorter.failure.can_shorter_step_cure);
  }
}

// The most halvings a failure of the catalogue with can_shorter_step_cure true needs: GrazingWedgeClosingFast's hit
// comes 2^-26 of 0.1 s into the step, and every longer step meets it.
constexpr int kMostCatalogueHalvings = 26;

// Checks one scene of the catalogue and counts it: a failure with can_shorter_step_cure false comes back at shorter
// steps (CheckLastingFailure); one with it true is a close call cured within kMostCatalogueHalvings halvings, every
// failure on the way true.
void CheckCatalogueScene(const MoveParticleCase& scene, CatalogueFailureCounts& counts) {
  const MoveParticleCall call = ComputeMoveWithHalvings(scene, 0);
  if (call.error == ContactError::kNone) {
    return;
  }
  counts.errors.insert(call.error);
  if (!call.failure.can_shorter_step_cure) {
    ++counts.lasting;
    CheckLastingFailure(scene, call.failure, call.error);
    return;
  }
  ++counts.cured;
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  const int halving_count = ComputeCountHalvingsToSuccess(scene, call.failure.close_call, kMostCatalogueHalvings);
  EXPECT_LE(halving_count, kMostCatalogueHalvings);
  counts.most_halvings = std::max(counts.most_halvings, halving_count);
}

TEST(MoveParticle, EveryCatalogueFailure_ShorterStepCuresExactlyWhenFlagSays) {
  // Every scene of BuildMoveParticleCases that fails (CheckCatalogueScene). Every error other than kTooCloseToCall is
  // false, and the four errors of a scene are each met.
  CatalogueFailureCounts counts;
  for (const MoveParticleCase& scene : BuildMoveParticleCases()) {
    SCOPED_TRACE(std::string(scene.name));
    CheckCatalogueScene(scene, counts);
  }
  std::cout << "catalogue failures: lasting " << counts.lasting << ", cured " << counts.cured << " (most halvings "
            << counts.most_halvings << ")\n";
  EXPECT_GT(counts.lasting, 0);
  EXPECT_GT(counts.cured, 0);
  EXPECT_EQ(counts.most_halvings, kMostCatalogueHalvings);
  const std::set<ContactError> scene_errors = {
      ContactError::kBadInput,
      ContactError::kStartsInsideSolid,
      ContactError::kOnWrongSideOfPlate,
      ContactError::kStartsInNarrowGap,
  };
  EXPECT_TRUE(std::ranges::includes(counts.errors, scene_errors));
}

}  // namespace
}  // namespace fluidity::contact
