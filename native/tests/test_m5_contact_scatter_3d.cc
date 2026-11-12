#include <array>
#include <cmath>
#include <cstdio>

#include "fluidity/host/contact_face_bodies_3d.h"

namespace {

bool TestContactFaceBatch() {
  constexpr std::array particles{
      fluidity::ContactParticleIdentity3D{1, 9, 2},
      fluidity::ContactParticleIdentity3D{2, 11, 6},
  };
  constexpr std::array contributors{
      fluidity::ContactFaceParticle3D{1, 9, 2, 0.5, {-2, 3, 0}},
      fluidity::ContactFaceParticle3D{2, 11, 6, 0.5, {2, 0, 0}},
  };
  const std::array faces{
      fluidity::DetectedContactFace3D{3, contributors, 9, 11, {1, 0, 0}, 0.25},
      fluidity::DetectedContactFace3D{6, contributors, 9, 11, {1, 0, 0}, 0.25},
  };
  const auto batch = fluidity::StageContactFaceBatch3D(particles, faces);
  if (!batch.ok() || batch->face_contacts.size() != 2 || batch->velocity_deltas.size() != 2
      || std::abs(batch->velocity_deltas[0].x - 3) > 1e-12 || std::abs(batch->velocity_deltas[1].x + 1) > 1e-12
      || std::abs(2 * batch->velocity_deltas[0].y + 6 * batch->velocity_deltas[1].y) > 1e-12) {
    return false;
  }
  auto invalid = faces;
  invalid[1].second_body_id = 99;
  if (fluidity::StageContactFaceBatch3D(particles, invalid).ok()) {
    return false;
  }
  invalid = faces;
  invalid[1].face_slot = invalid[0].face_slot;
  if (fluidity::StageContactFaceBatch3D(particles, invalid).ok()) {
    return false;
  }
  auto excessive = contributors;
  excessive[0].weight = 0.75;
  invalid = faces;
  invalid[1].contributors = excessive;
  return !fluidity::StageContactFaceBatch3D(particles, invalid).ok();
}

bool TestContactFaceParticleScatter() {
  constexpr std::array particles{
      fluidity::ContactFaceParticle3D{1, 9, 2, 0.5, {-2, 3, 0}},
      fluidity::ContactFaceParticle3D{2, 9, 4, 0.25, {-2, 3, 0}},
      fluidity::ContactFaceParticle3D{3, 11, 6, 1, {2, 0, 0}},
      fluidity::ContactFaceParticle3D{4, 13, 1, 1, {9, 0, 0}},
  };
  const auto scattered = fluidity::ResolveContactFaceParticles3D(particles, 9, 11, {1, 0, 0}, 0.25);
  if (!scattered.ok() || scattered->velocity_deltas.size() != particles.size()) {
    return false;
  }
  const auto& deltas = scattered->velocity_deltas;
  const double impulse_x = 2 * deltas[0].x + 4 * deltas[1].x;
  const double impulse_y = 2 * deltas[0].y + 4 * deltas[1].y;
  if (std::abs(impulse_x - 6) > 1e-12 || std::abs(impulse_y + 1.5) > 1e-12
      || std::abs(impulse_x + 6 * deltas[2].x) > 1e-12 || std::abs(impulse_y + 6 * deltas[2].y) > 1e-12
      || deltas[3].x != 0 || deltas[3].y != 0) {
    return false;
  }
  auto invalid = particles;
  invalid.back().particle_id = 3;
  return !fluidity::ResolveContactFaceParticles3D(invalid, 9, 11, {1, 0, 0}, 0.25).ok();
}

}  // namespace

int main() {
  if (!TestContactFaceParticleScatter()) {
    std::fprintf(stderr, "M5 contact scatter fixture failed\n");
    return 1;
  }
  if (!TestContactFaceBatch()) {
    std::fprintf(stderr, "M5 contact batch fixture failed\n");
    return 1;
  }
  return 0;
}
