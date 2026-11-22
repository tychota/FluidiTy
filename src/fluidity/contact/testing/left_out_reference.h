// Test-only: the answers of the contact code on the left-out sets when no solid is ever left out.
//
// Purpose: give the identity test of the rule that leaves out a solid a particle cannot come near the answers of
//   the same scenes with the rule switched off, as bits.
//
// Idea: the contact code is header-only, so testing/left_out_reference.cc compiles its own copy with the rule
//   switched off (FLUIDITY_CONTACT_READ_EVERY_SOLID), under a renamed namespace so that it never shares a name with
//   the copy that has the rule. The tests call this one function across the two copies.
//
// Owns:    ComputeReferenceResultBits
// Uses:    CPU-only guard (base/cpu_only.h)
// Used by: src/fluidity/contact/solid_contact.move_particle_left_out.test.cc and
//          tests/contact/move_particle_left_out_every_set.test.cc
// Read:    docs/method-paper/fluidity-method.typ <proc-move-particle>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <vector>

namespace left_out_reference {

// The bits of the answers on one set with no solid ever left out. `set_index` is a LeftOutSet of
// contact/testing/left_out_solids_sets.h.
std::vector<std::uint64_t> ComputeReferenceResultBits(int set_index);

}  // namespace left_out_reference
