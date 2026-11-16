// The GPU half of MoveParticles, private to contact/: MoveParticlesOneThreadPerParticleOnGpu of
// contact/move_particles_gpu.h.
//
// The GPU loop is not written yet: this half launches nothing and reports a failed call, so a GPU run stops with
// kRunOnCallFailed instead of returning particles that did not move. The CPU half, its checks and the per-particle
// functions it will launch (contact/move_particles_steps.h) are in place.
// Read: docs/method-paper/fluidity-method.typ <sec-contact>

#include "fluidity/base/run_on_call.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles_gpu.h"
#include "fluidity/contact/move_particles_steps.h"

namespace fluidity::contact {

RunOnCallStatus MoveParticlesOneThreadPerParticleOnGpu(
    const MoveParticlesCall& call,
    MoveParticleCounts* counts,
    ContactFailure* first_failure
) {
  static_cast<void>(call);
  static_cast<void>(counts);
  static_cast<void>(first_failure);
  return RunOnCallStatus::kCallFailed;
}

}  // namespace fluidity::contact
