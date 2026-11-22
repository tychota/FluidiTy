#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <numbers>
#include <string>

#include "fluidity/base/vec3.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/solid_contact_move_particle_cases.h"

namespace fluidity::contact {
namespace {

// Builds one pair of the residual test: n_a = {0, 0, 1} and n_b = {s cos p, s sin p, -(1 - one_plus_dot)} with
// s = Sqrt(one_plus_dot (2 - one_plus_dot)), so 1 + Dot(n_a, n_b) is one_plus_dot exactly. Draws: p = 2 pi Next, then
// the two targets, each Next.
CorrectionTargets BuildResidualTestPair(CaseGenerator& generator, double one_plus_dot) {
  const double around = 2 * std::numbers::pi * generator.Next();  // rad
  const double across = std::sqrt(one_plus_dot * (2 - one_plus_dot));
  const Vec3 second_normal = {
      .x = across * std::cos(around),
      .y = across * std::sin(around),
      .z = -(1 - one_plus_dot),
  };
  const double first_target = generator.Next();
  const double second_target = generator.Next();
  return BuildCorrectionTargets({{.x = 0, .y = 0, .z = 1}, second_normal}, {first_target, second_target});
}

// Computes the larger of a pair's two residuals |Dot(n_i, change) - target_i|.
double ComputeLargestPairResidual(const CorrectionTargets& pair, Vec3 change) {
  const double first_residual = std::abs(math::Dot(pair.normals[0], change) - pair.targets[0]);
  const double second_residual = std::abs(math::Dot(pair.normals[1], change) - pair.targets[1]);
  return std::max(first_residual, second_residual);
}

// Computes the larger residual of one pair solved for a position, in rounding units over 1 + c, and fails the calling
// test unless the pair was solved through both surfaces.
double ComputePairResidualShare(const CorrectionTargets& pair, double one_plus_dot) {
  const SmallestCorrection first_solve = SolveSmallestCorrection(pair, 0x1p-40, CorrectedQuantity::kPosition);
  EXPECT_TRUE(first_solve.is_found && first_solve.used_count == 2);
  return ComputeLargestPairResidual(pair, first_solve.change) * one_plus_dot / 0x1p-52;
}

// Fails the calling test unless the pair, solved for a velocity with a slack of 2^-21, needed at most one refinement to
// come within a quarter of that slack, 2^-23.
void CheckOneRefinementWithinQuarterSlack(const CorrectionTargets& pair) {
  const SmallestCorrection velocity_change = SolveSmallestCorrection(pair, 0x1p-21, CorrectedQuantity::kVelocity);
  EXPECT_TRUE(velocity_change.is_found);
  EXPECT_LE(velocity_change.refinement_count, 1);
  EXPECT_LE(ComputeLargestPairResidual(pair, velocity_change.change), 0x1p-23);
}

// Draws a velocity of the sharp-surface set: 2^(60 Next - 30) m/s times three NextSigned.
Vec3 BuildWideRangeVelocity(CaseGenerator& generator) {
  const double scale = std::exp2(60 * generator.Next() - 30);  // m/s
  return scale * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
}

// Builds the normals of one case of the sharp-surface set: by Next < 0.5 a pair with 1 + c = 2^(-40 + 36 Next) (n_a a
// BuildUnitVector, a second one made square to it, and n_b = -(1 - s) n_a + Sqrt(s (2 - s)) t), else a triple of
// BuildUnitVector drawn again until |Dot(n_1, Cross(n_2, n_3))| >= 2^-20.
CorrectionTargets BuildSharpSurfaceNormals(CaseGenerator& generator) {
  CorrectionTargets built;
  if (generator.Next() < 0.5) {
    const double one_plus_dot = std::exp2(-40 + 36 * generator.Next());
    const Vec3 first_normal = BuildUnitVector(generator);
    Vec3 square = BuildUnitVector(generator);
    square = math::Normalized(square - math::Dot(first_normal, square) * first_normal);
    built.normals[0] = first_normal;
    built.normals[1] = -(1 - one_plus_dot) * first_normal + std::sqrt(one_plus_dot * (2 - one_plus_dot)) * square;
    built.count = 2;
    return built;
  }
  double triple_product = 0;
  while (std::abs(triple_product) < kVelocityTripleLimit) {
    for (int surface_index = 0; surface_index < 3; ++surface_index) {
      built.normals[surface_index] = BuildUnitVector(generator);
    }
    triple_product = math::Dot(built.normals[0], math::Cross(built.normals[1], built.normals[2]));
  }
  built.count = 3;
  return built;
}

// Draws one case of the sharp-surface set (BuildSharpSurfaceNormals, then v and each u_i by BuildWideRangeVelocity),
// solves it for a velocity with slack 0, and fails the calling test unless a change is found that leaves at every
// surface an approach -Dot(n_i, v + w - u_i) within 2^-46 (|v|_1 + |w|_1 + sum |u_i|_1).
SharpSurfaceResult CheckSharpSurfaceCase(CaseGenerator& generator) {
  CorrectionTargets targets = BuildSharpSurfaceNormals(generator);
  const Vec3 velocity = BuildWideRangeVelocity(generator);
  std::array<Vec3, 3> solid_velocities{};
  double speeds = std::abs(velocity.x) + std::abs(velocity.y) + std::abs(velocity.z);  // m/s
  for (int surface_index = 0; surface_index < targets.count; ++surface_index) {
    const Vec3 solid_velocity = BuildWideRangeVelocity(generator);
    solid_velocities.at(static_cast<std::size_t>(surface_index)) = solid_velocity;
    speeds += std::abs(solid_velocity.x) + std::abs(solid_velocity.y) + std::abs(solid_velocity.z);
    targets.targets[surface_index] = -math::Dot(targets.normals[surface_index], velocity - solid_velocity);
  }
  targets.speeds_before_change = speeds;
  const SmallestCorrection correction = SolveSmallestCorrection(targets, 0, CorrectedQuantity::kVelocity);
  SharpSurfaceResult checked;
  if (!correction.is_found) {
    ADD_FAILURE() << "no change found";
    return checked;
  }
  checked.refinement_count = correction.refinement_count;
  const Vec3 changed = velocity + correction.change;
  const Vec3 change = correction.change;
  const double summed_speeds = speeds + std::abs(change.x) + std::abs(change.y) + std::abs(change.z);  // m/s
  for (int surface_index = 0; surface_index < targets.count; ++surface_index) {
    const Vec3 solid_velocity = solid_velocities.at(static_cast<std::size_t>(surface_index));
    const double approach_left = -math::Dot(targets.normals[surface_index], changed - solid_velocity);  // m/s
    EXPECT_LE(approach_left, 0x1p-46 * summed_speeds);
    // A velocity already within the slack keeps its approach, up to 64 rounding units: count only the changes.
    const double share_left = correction.used_count > 0 ? approach_left / (0x1p-52 * summed_speeds) : 0;
    checked.largest_share_left = std::max(checked.largest_share_left, share_left);
  }
  return checked;
}

TEST(SolveSmallestCorrection, PairResidualWithoutRefinement_IsAboutEpsOverOnePlusC) {
  // 3000 pairs at each 1 + c = 2^-40, 2^-28, 2^-16, 2^-4 (BuildResidualTestPair). Each target is met to within
  // 4 * 2^-52 / (1 + c) by the pair alone, a position's solve; with a velocity's slack of 2^-21, at most one refinement
  // brings it within a quarter of that slack, 2^-23, at 2^-40 (ComputePairResidualShare).
  CaseGenerator generator;
  for (const double one_plus_dot : {0x1p-40, 0x1p-28, 0x1p-16, 0x1p-4}) {
    double largest_residual_share = 0;  // In 2^-52 / (1 + c).
    for (int pair_index = 0; pair_index < 3000; ++pair_index) {
      const CorrectionTargets pair = BuildResidualTestPair(generator, one_plus_dot);
      largest_residual_share = std::max(largest_residual_share, ComputePairResidualShare(pair, one_plus_dot));
      if (one_plus_dot == 0x1p-40) {
        CheckOneRefinementWithinQuarterSlack(pair);
      }
    }
    std::cout << "pair residual at 1 + c = " << one_plus_dot << ": at most " << largest_residual_share
              << " rounding units over 1 + c\n";
    EXPECT_LE(largest_residual_share, 4);
  }
}

TEST(SolveSmallestCorrection, SharpPairsAndCornersNearLimit_ApproachLeftWithinShare) {
  // 20,000 cases (BuildSharpSurfaceNormals, then v and each u_i by BuildWideRangeVelocity, in that order), the targets
  // -Dot(n_i, v - u_i), solved for a velocity with slack 0, so the slack is the change's own allowance. The approach
  // left at every surface, -Dot(n_i, v + w - u_i), is within 2^-46 (|v|_1 + |w|_1 + sum |u_i|_1), and at most 3
  // refinements are used (the model: at most 16 * 2^-52 of that sum with the refinements; 177 for pairs near 0.45
  // degrees and 2697 for triples without them).
  CaseGenerator generator;
  int largest_refinement_count = 0;
  double largest_share_left = 0;  // In 2^-52 of the summed speeds.
  for (int case_index = 0; case_index < 20000; ++case_index) {
    const SharpSurfaceResult checked = CheckSharpSurfaceCase(generator);
    largest_refinement_count = std::max(largest_refinement_count, checked.refinement_count);
    largest_share_left = std::max(largest_share_left, checked.largest_share_left);
  }
  std::cout << "sharp pairs and corners: at most " << largest_share_left << " rounding units of the speeds left by a "
            << "change, " << largest_refinement_count << " refinements\n";
  EXPECT_LE(largest_refinement_count, 3);
}

}  // namespace
}  // namespace fluidity::contact
