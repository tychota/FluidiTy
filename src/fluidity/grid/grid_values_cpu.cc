#include <expected>

#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_values.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {

std::expected<void, GridFailure> CheckGridVersion(const StaggeredGrid& grid, GridVersion version) {
  if (version.value != grid.version().value) {
    return std::unexpected(GridFailure{.error = GridError::kWrongGridVersion});
  }
  return {};
}

}  // namespace fluidity::grid
