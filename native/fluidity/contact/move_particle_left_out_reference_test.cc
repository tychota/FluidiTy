// The reference of the identity test of the rule that leaves out a solid a particle cannot come near: the same scenes
// run by the contact code built with FLUIDITY_CONTACT_READ_EVERY_SOLID, so that no solid is ever left out.
//
// The contact code is header-only, so this file compiles its own copy of it. To keep that copy apart from the one the
// tests of move_particle_left_out_test.cc compile with the rule, every name of namespace fluidity here is renamed to
// fluidity_read_every_solid before the headers are read: the two copies never share a name, and each program part
// calls its own.

#include <cstdint>
#include <vector>

namespace left_out_reference {
std::vector<std::uint64_t> ComputeReferenceResultBits(int set_index);
}  // namespace left_out_reference

// NOLINTBEGIN(cppcoreguidelines-macro-usage): the renaming that keeps the two copies of the headers apart.
#define FLUIDITY_CONTACT_READ_EVERY_SOLID
// NOLINTNEXTLINE(readability-identifier-naming): the namespace's own name, which this macro renames.
#define fluidity fluidity_read_every_solid
#include "fluidity/contact/left_out_solids_test_sets.h"
#undef fluidity
// NOLINTEND(cppcoreguidelines-macro-usage)

namespace left_out_reference {

std::vector<std::uint64_t> ComputeReferenceResultBits(int set_index) {
  return fluidity_read_every_solid::contact::ComputeResultBits(
      static_cast<fluidity_read_every_solid::contact::LeftOutSet>(set_index)
  );
}

}  // namespace left_out_reference
