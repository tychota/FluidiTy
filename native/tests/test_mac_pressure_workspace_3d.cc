#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include <vector>

#include "../src/mac_pressure_3d_workspace_internal.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "gtest/gtest.h"
#include "mac_pressure_3d_fixture.h"

namespace fluidity {
namespace {
constexpr UniformMacGeometry3D kUnitGeometry{{0.0, 0.0, 0.0}, 1.0};
constexpr double kInfinity = std::numeric_limits<double>::infinity();
std::int32_t FixtureOwnerAxis(std::int32_t index) {
  return index / 4 - (index % 4 < 0 ? 1 : 0);
}
// Build an independent affine solution: each predicted face velocity is the
// analytic pressure slope, so correction leaves zero flux on every open face.
struct AffineProblem {
  std::vector<MacPressureCell3D> cells;
  std::vector<MacPressureFace3D> faces;
  std::vector<BlockIndex3D> blocks;
};

double AffinePressure(CellIndex3D index) {
  return index.x + 2.0 * index.y + 4.0 * index.z;
}

AffineProblem MakeAffineProblem(std::int32_t minimum, std::int32_t extent, bool anchored) {
  AffineProblem problem;
  std::set<std::tuple<int, int, int, int>> face_keys;
  for (int z = minimum; z < minimum + extent; ++z) {
    for (int y = minimum; y < minimum + extent; ++y) {
      for (int x = minimum; x < minimum + extent; ++x) {
        problem.cells.push_back({
            .cell_index = {x, y, z},
            .fluid_volume = 1.0,
            .bulk_modulus = kInfinity,
        });
        for (int axis = 0; axis < 3; ++axis) {
          for (int side = 0; side < 2; ++side) {
            face_keys.emplace(axis, x + (axis == 0 ? side : 0), y + (axis == 1 ? side : 0), z + (axis == 2 ? side : 0));
          }
        }
      }
    }
  }
  std::set<std::tuple<int, int, int>> block_keys;
  for (const auto& [axis, x, y, z] : face_keys) {
    const std::array coordinates{x, y, z};
    const bool low = coordinates[axis] == minimum;
    const bool high = coordinates[axis] == minimum + extent;
    const bool boundary = low || high;
    MacPressureFace3D face{
        .face_index = {static_cast<FaceNormalAxis>(axis), {x, y, z}},
        .open_area = boundary && !anchored ? 0.0 : 1.0,
        .inverse_density = 1.0,
        .pressure_gradient_distance = 1.0,
        .predicted_normal_velocity = boundary && !anchored ? 0.0 : static_cast<double>(1 << axis),
        .pressure_link = InternalMacPressureLink3D{0.0},
    };
    if (boundary && anchored) {
      const CellIndex3D exterior{
          x - (axis == 0 && low ? 1 : 0),
          y - (axis == 1 && low ? 1 : 0),
          z - (axis == 2 && low ? 1 : 0),
      };
      face.pressure_link = DirichletMacPressureLink3D{AffinePressure(exterior)};
    } else if (boundary) {
      face.pressure_link = SealedMacPressureLink3D{};
    }
    problem.faces.push_back(face);
    block_keys.emplace(FixtureOwnerAxis(x), FixtureOwnerAxis(y), FixtureOwnerAxis(z));
  }
  for (const auto& [x, y, z] : block_keys) {
    problem.blocks.push_back({x, y, z});
  }
  return problem;
}

void SortFaces(AffineProblem& problem, const SparseMacGrid3D& grid) {
  std::ranges::sort(problem.faces, [&](const auto& left, const auto& right) {
    return *grid.FaceSlot(left.face_index) < *grid.FaceSlot(right.face_index);
  });
}

TEST(MacPressureWorkspace3D, MatchesIndependentFrozenPressureFixtures) {
  std::vector<pressure_test::MacPressureFixtureCase3D> cases;
  ASSERT_NO_FATAL_FAILURE(pressure_test::LoadMacPressureFixtureCases3D(cases));
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(workspace.ok());
  for (const auto& fixture : cases) {
    SCOPED_TRACE(fixture.name);
    const auto result = ProjectMacPressure3D(
        fixture.grid.view(),
        fixture.step_duration,
        {fixture.cells, fixture.faces},
        {},
        *workspace
    );
    ASSERT_TRUE(result.ok()) << result.status();
    const MacPressureProjection3D owned{
        .cell_pressures = {result->cell_pressures.begin(), result->cell_pressures.end()},
        .corrected_face_velocities =
            {result->corrected_face_velocities.begin(), result->corrected_face_velocities.end()},
        .integrated_rhs = {result->integrated_rhs.begin(), result->integrated_rhs.end()},
        .true_residuals = {result->true_residuals.begin(), result->true_residuals.end()},
        .cell_constraint_defects = {result->cell_constraint_defects.begin(), result->cell_constraint_defects.end()},
        .gauge_cell_ordinals = {result->gauge_cell_ordinals.begin(), result->gauge_cell_ordinals.end()},
        .diagnostics = result->diagnostics,
    };
    pressure_test::ExpectMacPressureFixtureProjection3D(
        owned,
        fixture.expected_projection,
        fixture.absolute_tolerance,
        fixture.relative_tolerance
    );
  }
}
TEST(MacPressureWorkspace3D, RejectsInsufficientCapacityAndOverflow) {
  std::vector<pressure_test::MacPressureFixtureCase3D> cases;
  ASSERT_NO_FATAL_FAILURE(pressure_test::LoadMacPressureFixtureCases3D(cases));
  auto workspace = MacPressureWorkspace3D::Create(0, 0);
  ASSERT_TRUE(workspace.ok());
  const auto& fixture = cases.front();
  const auto result =
      ProjectMacPressure3D(fixture.grid.view(), fixture.step_duration, {fixture.cells, fixture.faces}, {}, *workspace);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(
      QueryMacPressureWorkspaceBytes3D(std::numeric_limits<std::size_t>::max(), 6).status().code(),
      absl::StatusCode::kResourceExhausted
  );
}
TEST(MacPressureWorkspace3D, MultilevelWarmStartsAndRejectionsPreservePublishedProjection) {
  for (const bool anchored : {false, true}) {
    auto problem = MakeAffineProblem(-3, 6, anchored);
    const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
    ASSERT_TRUE(grid.ok());
    SortFaces(problem, *grid);
    auto workspace = MacPressureWorkspace3D::Create(problem.cells.size(), problem.faces.size());
    ASSERT_TRUE(workspace.ok());
    const auto first = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
    ASSERT_TRUE(first.ok()) << first.status();
    EXPECT_GE(first->diagnostics.multigrid_level_count, 3);
    EXPECT_GT(first->diagnostics.iteration_count, 1);
    EXPECT_LE(first->diagnostics.max_cell_constraint_defect, 2e-5);
    const auto* published_address = first->cell_pressures.data();
    const std::vector<double> published(first->cell_pressures.begin(), first->cell_pressures.end());
    const std::vector<double> velocities(
        first->corrected_face_velocities.begin(),
        first->corrected_face_velocities.end()
    );
    auto rejected =
        ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {.max_iterations = 1}, *workspace);
    EXPECT_EQ(rejected.status().code(), absl::StatusCode::kAborted);
    rejected = ProjectMacPressure3D(grid->view(), -1, {problem.cells, problem.faces}, {}, *workspace);
    EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
    const auto good_face = problem.faces.back();
    problem.faces.back().predicted_normal_velocity = std::numeric_limits<double>::infinity();
    rejected = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
    EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
    problem.faces.back() = good_face;
    if (!anchored) {
      problem.cells.front().target_divergence = 1;
      rejected = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
      EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
      problem.cells.front().target_divergence = 0;
    }
    auto excessive_cells = problem.cells;
    excessive_cells.push_back(problem.cells.back());
    rejected = ProjectMacPressure3D(grid->view(), 1, {excessive_cells, problem.faces}, {}, *workspace);
    EXPECT_EQ(rejected.status().code(), absl::StatusCode::kResourceExhausted);
    EXPECT_EQ(first->cell_pressures.data(), published_address);
    EXPECT_TRUE(std::ranges::equal(first->cell_pressures, published));
    EXPECT_TRUE(std::ranges::equal(first->corrected_face_velocities, velocities));
    for (auto& cell : problem.cells) {
      EXPECT_DOUBLE_EQ(cell.previous_pressure, 0);
      cell.previous_pressure = AffinePressure(cell.cell_index) + (anchored ? 0 : 100);
    }
    const auto warm =
        ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {.max_iterations = 1}, *workspace);
    ASSERT_TRUE(warm.ok()) << warm.status();
    EXPECT_EQ(warm->diagnostics.iteration_count, 0);
    for (std::size_t ordinal = 0; ordinal < problem.cells.size(); ++ordinal) {
      EXPECT_DOUBLE_EQ(
          warm->cell_pressures[ordinal],
          AffinePressure(problem.cells[ordinal].cell_index) + (anchored ? 0 : 21)
      );
      EXPECT_NEAR(warm->true_residuals[ordinal], -warm->cell_constraint_defects[ordinal], 1e-12);
    }
    EXPECT_LE(workspace->used_bytes(), workspace->high_water_bytes());
    EXPECT_LE(workspace->high_water_bytes(), workspace->bytes().total_bytes);
  }
}
TEST(MacPressureWorkspace3D, BoundedHierarchyMatchesAcceptedOwningHierarchyExactlyAtMaximumCells) {
  auto problem = MakeAffineProblem(-9, 16, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  const auto prepared = pressure_internal::PrepareMacPressureSystem3D({*grid, 1, problem.cells, problem.faces});
  ASSERT_TRUE(prepared.ok());
  const auto expected = pressure_internal::BuildMacPressureHierarchy3D(*prepared);
  ASSERT_TRUE(expected.ok());
  const auto bytes = QueryMacPressureWorkspaceBytes3D(problem.cells.size(), problem.faces.size());
  ASSERT_TRUE(bytes.ok());
  auto storage = LiquidRecords3D<std::max_align_t>::Create(bytes->scratch_bytes / sizeof(std::max_align_t));
  ASSERT_TRUE(storage.ok());
  ASSERT_TRUE(storage->Resize(storage->capacity()).ok());
  pressure_internal::MacPressureScratch3D scratch(storage->values());
  // The builder reads only cells/links/gauges; duplicate scalar spans are not needed.
  std::vector<double> rhs(prepared->integrated_rhs().begin(), prepared->integrated_rhs().end());
  std::vector<double> terms(prepared->rhs_absolute_term_sums().begin(), prepared->rhs_absolute_term_sums().end());
  std::vector<std::size_t> components(prepared->component_ordinals().begin(), prepared->component_ordinals().end());
  std::vector<std::size_t> gauges(prepared->gauge_cell_ordinals().begin(), prepared->gauge_cell_ordinals().end());
  const pressure_internal::PreparedMacPressureView3D system{
      .cell_records = prepared->cells(),
      .link_records = prepared->links(),
      .rhs = rhs,
      .rhs_absolute_terms = terms,
      .components = components,
      .gauges = gauges,
  };
  const auto actual = pressure_internal::BuildBoundedMacPressureHierarchy3D(system, scratch);
  ASSERT_TRUE(actual.ok()) << actual.status();
  ASSERT_EQ(actual->levels.size(), expected->levels.size());
  EXPECT_TRUE(std::ranges::equal(actual->unknown_to_cell, expected->unknown_to_cell));
  EXPECT_TRUE(std::ranges::equal(actual->cell_to_unknown, expected->cell_to_unknown));
  for (std::size_t ordinal = 0; ordinal < actual->levels.size(); ++ordinal) {
    const auto& level = actual->levels[ordinal];
    const auto& want = expected->levels[ordinal];
    ASSERT_EQ(level.coordinates.size(), want.coordinates.size());
    for (std::size_t row = 0; row < level.coordinates.size(); ++row) {
      EXPECT_EQ(level.coordinates[row].x, want.coordinates[row].x);
      EXPECT_EQ(level.coordinates[row].y, want.coordinates[row].y);
      EXPECT_EQ(level.coordinates[row].z, want.coordinates[row].z);
    }
    EXPECT_TRUE(std::ranges::equal(level.anchor_diagonal, want.anchor_diagonal));
    EXPECT_TRUE(std::ranges::equal(level.diagonal, want.diagonal));
    EXPECT_TRUE(std::ranges::equal(level.component_ordinals, want.component_ordinals));
    EXPECT_TRUE(std::ranges::equal(level.row_offsets, want.row_offsets));
    EXPECT_TRUE(std::ranges::equal(level.aggregate_ordinals, want.aggregate_ordinals));
    ASSERT_EQ(level.edges.size(), want.edges.size());
    for (std::size_t edge = 0; edge < level.edges.size(); ++edge) {
      EXPECT_EQ(level.edges[edge].lower_unknown_ordinal, want.edges[edge].lower_unknown_ordinal);
      EXPECT_EQ(level.edges[edge].higher_unknown_ordinal, want.edges[edge].higher_unknown_ordinal);
      EXPECT_DOUBLE_EQ(level.edges[edge].conductance, want.edges[edge].conductance);
    }
    ASSERT_EQ(level.neighbors.size(), want.neighbors.size());
    for (std::size_t neighbor = 0; neighbor < level.neighbors.size(); ++neighbor) {
      EXPECT_EQ(level.neighbors[neighbor].unknown_ordinal, want.neighbors[neighbor].unknown_ordinal);
      EXPECT_EQ(level.neighbors[neighbor].edge_ordinal, want.neighbors[neighbor].edge_ordinal);
    }
    std::size_t factor_offset = 0;
    for (std::size_t block = 0; block < want.terminal_blocks.size(); ++block) {
      const auto& terminal = want.terminal_blocks[block];
      const auto begin = level.terminal_offsets[block];
      EXPECT_TRUE(std::ranges::equal(
          level.terminal_unknowns.subspan(begin, terminal.unknown_ordinals.size()),
          terminal.unknown_ordinals
      ));
      EXPECT_TRUE(std::ranges::equal(
          level.terminal_factors.subspan(factor_offset, terminal.cholesky_lower.size()),
          terminal.cholesky_lower
      ));
      factor_offset += terminal.cholesky_lower.size();
    }
    EXPECT_EQ(factor_offset, level.terminal_factors.size());
  }
  auto workspace = MacPressureWorkspace3D::Create(problem.cells.size(), problem.faces.size());
  ASSERT_TRUE(workspace.ok());
  const auto projected = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(projected.ok()) << projected.status();
  EXPECT_LE(workspace->high_water_bytes(), bytes->total_bytes);
  EXPECT_EQ(bytes->maximum_terminal_factor_entries, 32 * 4096);
}
TEST(MacPressureWorkspace3D, CheckedScratchExhaustionDoesNotOverwriteLiveSpans) {
  std::array<std::max_align_t, 2> storage{};
  pressure_internal::MacPressureScratch3D scratch(storage);
  const auto first = scratch.Take<double>(1);
  ASSERT_TRUE(first.ok());
  (*first)[0] = 17;
  const auto used = scratch.used();
  const auto rejected = scratch.Take<double>(std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scratch.used(), used);
  EXPECT_DOUBLE_EQ((*first)[0], 17);
  {
    pressure_internal::MacPressureScratchMark3D mark(scratch);
    const auto second = scratch.Take<double>(1);
    ASSERT_TRUE(second.ok());
    (*second)[0] = 23;
  }
  EXPECT_EQ(scratch.used(), used);
  EXPECT_DOUBLE_EQ((*first)[0], 17);
}
#if defined(_WIN32) && defined(_DEBUG)
std::size_t allocations = 0;
std::size_t allocated_bytes = 0;
std::size_t frees = 0;
std::size_t fail_request = 0;
_CRT_ALLOC_HOOK previous_hook = nullptr;
int AllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int block_type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++allocations;
    allocated_bytes += size;
    if (allocations == fail_request) {
      return 0;
    }
  }
  if (operation == _HOOK_FREE) {
    ++frees;
  }
  return previous_hook == nullptr ? 1 : previous_hook(operation, data, size, block_type, request, filename, line);
}
TEST(MacPressureWorkspace3D, SuccessRejectionsAndStatusDestructionHaveNoHeapActivity) {
  auto problem = MakeAffineProblem(-3, 6, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  auto workspace = MacPressureWorkspace3D::Create(problem.cells.size(), problem.faces.size());
  auto insufficient = MacPressureWorkspace3D::Create(problem.cells.size() - 1, problem.faces.size());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(insufficient.ok());
  std::array<absl::StatusCode, 5> codes{};
  allocations = 0;
  frees = 0;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    auto result = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
    codes[0] = result.status().code();
    result = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {.max_iterations = 1}, *workspace);
    codes[1] = result.status().code();
    problem.cells.front().target_divergence = 1;
    result = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
    codes[2] = result.status().code();
    result = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *insufficient);
    codes[3] = result.status().code();
    problem.cells.front().target_divergence = 0;
    problem.faces.back().predicted_normal_velocity = std::numeric_limits<double>::infinity();
    result = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
    codes[4] = result.status().code();
  }
  _CrtSetAllocHook(previous_hook);
  EXPECT_EQ(allocations, 0);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(codes[0], absl::StatusCode::kOk);
  EXPECT_EQ(codes[1], absl::StatusCode::kAborted);
  EXPECT_EQ(codes[2], absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(codes[3], absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(codes[4], absl::StatusCode::kInvalidArgument);
}
TEST(MacPressureWorkspace3D, QueryExactlyAccountsForActualDynamicReservations) {
  for (const auto capacities : {std::pair<std::size_t, std::size_t>{0, 0}, {1, 6}, {4096, 24576}}) {
    const auto bytes = QueryMacPressureWorkspaceBytes3D(capacities.first, capacities.second);
    ASSERT_TRUE(bytes.ok());
    bool successful = false;
    allocations = 0;
    frees = 0;
    allocated_bytes = 0;
    previous_hook = _CrtSetAllocHook(AllocationHook);
    {
      const auto workspace = MacPressureWorkspace3D::Create(capacities.first, capacities.second);
      successful = workspace.ok();
    }
    _CrtSetAllocHook(previous_hook);
    EXPECT_TRUE(successful);
    EXPECT_EQ(allocated_bytes, bytes->total_bytes);
    EXPECT_EQ(frees, allocations);
    EXPECT_EQ(bytes->scratch_bytes + bytes->projection_bytes, bytes->total_bytes);
  }
}
TEST(MacPressureWorkspace3D, ActualPartialReservationFailureReturnsInlineStatusAndCleansUp) {
  allocations = 0;
  frees = 0;
  fail_request = 4;
  absl::StatusCode code{};
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    const auto workspace = MacPressureWorkspace3D::Create(64, 400);
    code = workspace.status().code();
  }
  _CrtSetAllocHook(previous_hook);
  fail_request = 0;
  EXPECT_EQ(code, absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(allocations, 4);
  EXPECT_EQ(frees, 3);
}
#endif
}  // namespace
}  // namespace fluidity
