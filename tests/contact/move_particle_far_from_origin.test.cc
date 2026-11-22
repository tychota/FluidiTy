#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>

#include "fluidity/contact/testing/contact_cases.h"

// The tests that replace the rounding enclosures: a particle that meets a solid stops at the rest gap, one contact
// tolerance outside it, and the tolerance is set from the scene's size, so rounding far from the origin never brings it
// inside. They run with the contact tolerance of a real scene, not the test tolerance of the analytic tests. They show
// what the rest gap does; the settle at the end of a step, for solids that close on a particle, has tests of its own
// (ApplySettleAtEnd in solid_contact.move_particle.test.cc), and a settle that does nothing passes these.

namespace fluidity::contact {
namespace {

// --- Far from the origin: every end outside ---

TEST(MoveParticle, ObliqueMovingSolidFarFromOrigin_NeverEndsInside) {
  // BuildObliqueCases(10000): one moving flat solid up to 1 km from the origin, hit at 0.1 to 0.9 of the step, with
  // the production tolerance there (about 1e-9 m, so a coordinate's rounding unit is about tol / 9000). Every move
  // succeeds and ends at least tol / 2 outside the solid at its end pose. The control: the same paths hit at the
  // surface itself and moved on without a settle end a few rounding units on either side of it, so about half of them
  // inside (measured 4,886); the bounds 4,000 and 6,000 leave about 20 standard deviations of a fair coin on each side.
  // The rest gap is what keeps the particles outside.
  const EndDistanceCounts counts = ComputeEndDistanceCounts(BuildObliqueCases(10000));
  std::cout << "oblique far from origin: cases " << counts.cases << ", failures " << counts.failures << ", hits "
            << counts.hits << ", ends closer than tol / 2 " << counts.ends_closer_than_half_tolerance
            << ", least end distance " << counts.least_end_distance_share << " tol; naive control ends inside "
            << counts.naive_ends_inside << "\n";
  EXPECT_EQ(counts.cases, 10000);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_EQ(counts.ends_closer_than_half_tolerance, 0);
  EXPECT_EQ(counts.hits, 10000);  // Every path is aimed at the solid.
  EXPECT_GE(counts.least_end_distance_share, 0.5);
  EXPECT_GE(counts.naive_ends_inside, 4000);
  EXPECT_LE(counts.naive_ends_inside, 6000);
}

TEST(MoveParticle, RoundSolidsFarFromOrigin_NeverEndInside) {
  // BuildRoundCases(10000): one moving sphere or cylinder up to 1 km from the origin, with the production tolerance
  // there. Every move succeeds and ends at least tol / 2 outside the solid at its end pose. The rest gap's proof does
  // not read the shape, so the control of the oblique set is not repeated as a check; it is printed.
  const EndDistanceCounts counts = ComputeEndDistanceCounts(BuildRoundCases(10000));
  std::cout << "round far from origin: cases " << counts.cases << ", failures " << counts.failures << ", hits "
            << counts.hits << ", ends closer than tol / 2 " << counts.ends_closer_than_half_tolerance
            << ", least end distance " << counts.least_end_distance_share << " tol; naive control ends inside "
            << counts.naive_ends_inside << "\n";
  EXPECT_EQ(counts.cases, 10000);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_EQ(counts.ends_closer_than_half_tolerance, 0);
  EXPECT_GE(counts.hits, 4500);
  EXPECT_GE(counts.least_end_distance_share, 0.5);
}

}  // namespace
}  // namespace fluidity::contact
