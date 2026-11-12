#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "../src/liquid_adaptive_state_internal_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include "gtest/gtest.h"
namespace fluidity {
namespace {
struct AdaptiveValidationFixture {
  std::vector<AdaptiveMacCellKey3D> keys;
  mutable std::size_t cell_reads = 0;

  absl::Status Validate(std::size_t face_count) const {
    return liquid_state_internal::ValidateAdaptiveRecordView3D({
        .context = this,
        .finest_cell_width = 0.5,
        .cell_count = keys.size(),
        .face_count = face_count,
        .read_cell =
            [](const void* context, std::size_t ordinal) {
              const auto& fixture = *static_cast<const AdaptiveValidationFixture*>(context);
              ++fixture.cell_reads;
              return LiquidAdaptiveCellState3D{.key = fixture.keys[ordinal]};
            },
        .read_face =
            [](const void*, std::size_t) {
              return LiquidAdaptiveFaceState3D{};
            },
    });
  }
};

// Directly intersect closed lattice boxes. On a valid 2:1 mesh each covered
// side is either whole or a union of disjoint quarters.
std::size_t ReferenceAdaptiveFaceCount(std::span<const AdaptiveMacCellKey3D> keys) {
  std::vector<std::array<int, 6>> covered_area(keys.size());
  std::size_t face_count = 0;
  for (std::size_t first = 0; first < keys.size(); ++first) {
    const auto a = keys[first].lower_corner;
    const std::array lower{a.x, a.y, a.z};
    const int width = 1 << keys[first].width_exponent;
    for (std::size_t second = first + 1; second < keys.size(); ++second) {
      const auto b = keys[second].lower_corner;
      const std::array other_lower{b.x, b.y, b.z};
      const int other_width = 1 << keys[second].width_exponent;
      for (std::size_t axis = 0; axis < lower.size(); ++axis) {
        const bool touches_high = lower[axis] + width == other_lower[axis];
        const bool touches_low = other_lower[axis] + other_width == lower[axis];
        if (!touches_high && !touches_low) {
          continue;
        }
        int face_area = 1;
        for (std::size_t tangent = 0; tangent < lower.size(); ++tangent) {
          if (tangent != axis) {
            const int high = std::min(lower[tangent] + width, other_lower[tangent] + other_width);
            const int low = std::max(lower[tangent], other_lower[tangent]);
            face_area *= std::max(0, high - low);
          }
        }
        if (face_area > 0) {
          ++face_count;
          covered_area[first][2 * axis + touches_high] += face_area;
          covered_area[second][2 * axis + touches_low] += face_area;
        }
      }
    }
  }
  for (std::size_t cell = 0; cell < keys.size(); ++cell) {
    const int width = 1 << keys[cell].width_exponent;
    for (const int covered : covered_area[cell]) {
      face_count += covered == 0 ? 1 : 4 - 4 * covered / (width * width);
    }
  }
  return face_count;
}

TEST(LiquidAdaptiveRecordView3D, MatchesGeometricOwnershipOnSeededSignedLeaves) {
  using absl::StatusCode;
  std::mt19937 random(8917);
  for (int trial = 0; trial < 16; ++trial) {
    SCOPED_TRACE(trial);
    AdaptiveValidationFixture fixture;
    for (int z = -4; z < 4; z += 2) {
      for (int y = -4; y < 4; y += 2) {
        for (int x = -4; x < 4; x += 2) {
          if (random() % 3 == 0) {
            fixture.keys.push_back({.lower_corner = {x, y, z}, .width_exponent = 1});
            continue;
          }
          for (int child = 0; child < 8; ++child) {
            if (random() % 3 != 0) {
              fixture.keys.push_back({.lower_corner = {x + (child & 1), y + ((child >> 1) & 1), z + (child >> 2)}});
            }
          }
        }
      }
    }
    std::ranges::sort(fixture.keys, {}, [](const AdaptiveMacCellKey3D& key) {
      const auto lower = key.lower_corner;
      return std::tuple{lower.z, lower.y, lower.x};
    });
    const auto face_count = ReferenceAdaptiveFaceCount(fixture.keys);
    EXPECT_TRUE(fixture.Validate(face_count).ok());
    EXPECT_EQ(fixture.cell_reads, fixture.keys.size());
    EXPECT_EQ(fixture.Validate(face_count - 1).code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(fixture.Validate(face_count + 1).code(), StatusCode::kInvalidArgument);
  }
}

TEST(LiquidAdaptiveRecordView3D, RejectsSignedOverlapAndFaceEdgeCornerGrading) {
  using absl::StatusCode;
  for (const CellIndex3D fine_lower :
       {CellIndex3D{-3, -3, -3}, CellIndex3D{0, -3, -3}, CellIndex3D{0, 0, -3}, CellIndex3D{0, 0, 0}}) {
    AdaptiveValidationFixture fixture{
        .keys = {{.lower_corner = {-4, -4, -4}, .width_exponent = 2}, {.lower_corner = fine_lower}},
    };
    EXPECT_EQ(fixture.Validate(12).code(), StatusCode::kInvalidArgument);
  }

  AdaptiveValidationFixture separated{
      .keys = {{.lower_corner = {-4, -4, -4}, .width_exponent = 2}, {.lower_corner = {1, 0, 0}}},
  };
  EXPECT_TRUE(separated.Validate(12).ok());
}

TEST(LiquidAdaptiveRecordView3D, AcceptsStrictlySeparatedSignedSlabs) {
  const auto signed_min = std::numeric_limits<std::int32_t>::min();
  const auto signed_max = std::numeric_limits<std::int32_t>::max();
  AdaptiveValidationFixture fixture{
      .keys =
          {
              {.lower_corner = {signed_min, 0, signed_min}, .width_exponent = 30},
              {.lower_corner = {0, 0, 0}, .width_exponent = 29},
              {.lower_corner = {0, 0, signed_max - 1}},
          },
  };
  EXPECT_TRUE(fixture.Validate(18).ok());
  EXPECT_EQ(fixture.cell_reads, fixture.keys.size());
  EXPECT_EQ(fixture.Validate(17).code(), absl::StatusCode::kInvalidArgument);
}
constexpr LiquidStateCapacities3D kCapacity{4, 2, 2, 4, 2};
LiquidState3D InitialState() {
  LiquidState3D state;
  state.particles = std::move(*LiquidRecords3D<LiquidParticleState3D>::Create(2));
  state.blocks = std::move(*LiquidRecords3D<LiquidBlockState3D>::Create(1));
  state.cells = std::move(*LiquidRecords3D<LiquidCellState3D>::Create(1));
  state.faces = std::move(*LiquidRecords3D<LiquidFaceState3D>::Create(1));
  state.sources = std::move(*LiquidRecords3D<LiquidSourceState3D>::Create(1));
  EXPECT_TRUE(state.particles.Resize(2).ok());
  EXPECT_TRUE(state.blocks.Resize(1).ok());
  EXPECT_TRUE(state.cells.Resize(1).ok());
  EXPECT_TRUE(state.faces.Resize(1).ok());
  EXPECT_TRUE(state.sources.Resize(1).ok());
  state.particles[0].id = 2;
  state.particles[1].id = 1;
  state.next_particle_id = 3;
  state.random_seed = 73;
  state.blocks[0].block_index = {-1, 0, 0};
  state.cells[0].cell_index = {-4, 0, 0};
  state.faces[0].face_index = {FaceNormalAxis::kX, {-4, 0, 0}};
  state.faces[0].validity = LiquidFaceValidity3D::kValid;
  return state;
}
void Poison(LiquidState3D& trial) {
  trial.geometry = {{1, 2, 3}, 4};
  trial.clock = {2, 3, 4, 5, 6};
  trial.random_seed = 91;
  trial.next_particle_id = 90;
  trial.particles[0] = {
      .id = 11,
      .tag = 12,
      .world_position = {13, 14, 15},
      .world_velocity = {16, 17, 18},
      .velocity_slopes = {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}},
      .affine_enabled = false,
      .mass = 28,
      .reference_volume = 29,
      .volume_ratio = 30,
      .side_tag = 31,
      .pic_initialized = true,
  };
  trial.blocks[0] = {{2, 3, 4}, 5, 6, 7};
  trial.cells[0] = {{8, 9, 10}, LiquidCellOccupancy3D::kSolid, 11};
  trial.faces[0] = {{FaceNormalAxis::kZ, {12, 13, 14}}, 15, 16, LiquidFaceValidity3D::kPredicted, 17, false};
  trial.sources[0] = {18, 19, 0.25};
}
TEST(LiquidFrameTransaction3D, CreatesIndependentCommittedVersion) {
  auto initial = InitialState();
  auto result = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->committed(), initial);
  EXPECT_NE(result->committed().particles.values().data(), initial.particles.values().data());
  initial.random_seed = 91;
  EXPECT_EQ(result->committed().random_seed, 73);
}
TEST(LiquidFrameTransaction3D, RejectionPreservesEveryPhysicalField) {
  for (auto decision :
       {LiquidAttemptDisposition3D::kRetryDt, LiquidAttemptDisposition3D::kCapacity, LiquidAttemptDisposition3D::kFatal
       }) {
    auto initial = InitialState();
    auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
    ASSERT_TRUE(transaction.ok());
    ASSERT_TRUE(transaction->BeginFrame(1).ok());
    auto trial = transaction->ResetTrialFromCommitted();
    ASSERT_TRUE(trial.ok());
    Poison(**trial);
    EXPECT_TRUE(transaction->FinishAttempt(decision, 0.5).ok());
    EXPECT_EQ(transaction->committed(), initial);
    EXPECT_EQ(transaction->frame_start(), initial);
  }
}
TEST(LiquidFrameTransaction3D, RetryHalvesTwiceThenStopsWithoutConsumingIdentity) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  for (double duration : {1.0, 0.5}) {
    auto trial = transaction->ResetTrialFromCommitted();
    ASSERT_TRUE(trial.ok());
    EXPECT_EQ(**trial, initial);
    (**trial).next_particle_id = 100;
    auto outcome = transaction->FinishAttempt(LiquidAttemptDisposition3D::kRetryDt, duration);
    ASSERT_TRUE(outcome.ok());
    EXPECT_EQ(outcome->retry_duration, duration / 2);
    EXPECT_EQ(transaction->committed(), initial);
  }
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  auto exhausted = transaction->FinishAttempt(LiquidAttemptDisposition3D::kRetryDt, 0.25);
  ASSERT_TRUE(exhausted.ok());
  EXPECT_EQ(exhausted->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_FALSE(exhausted->retry_duration.has_value());
  EXPECT_FALSE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_EQ(transaction->committed(), initial);
}
TEST(LiquidFrameTransaction3D, AcceptCarriesHistoryAfterRetryAndAdvancesExactlyOnce) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  auto trial = transaction->ResetTrialFromCommitted();
  ASSERT_TRUE(trial.ok());
  (**trial).cells[0].previous_pressure = 8;
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kRetryDt, 1).ok());
  trial = transaction->ResetTrialFromCommitted();
  ASSERT_TRUE(trial.ok());
  EXPECT_EQ((**trial).cells[0].previous_pressure, 0);
  (**trial).cells[0].previous_pressure = 17;
  (**trial).sources[0].fractional_remainder = 0.25;
  (**trial).sources[0].next_child_ordinal = 2;
  (**trial).next_particle_id = 5;
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.5).ok());
  EXPECT_EQ(transaction->committed().clock, (LiquidClock3D{0.5, 0, 1, 1, 1}));
  EXPECT_EQ(transaction->frame_start(), initial);
  EXPECT_FALSE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.5).ok());
  trial = transaction->ResetTrialFromCommitted();
  ASSERT_TRUE(trial.ok());
  EXPECT_EQ((**trial).cells[0].previous_pressure, 17);
  EXPECT_EQ((**trial).sources[0].fractional_remainder, 0.25);
  EXPECT_EQ((**trial).next_particle_id, 5);
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.5).ok());
  ASSERT_TRUE(transaction->FinishFrame().ok());
  EXPECT_EQ(transaction->committed().clock, (LiquidClock3D{1, 1, 0, 2, 2}));
  EXPECT_FALSE(transaction->FinishFrame().ok());
  EXPECT_EQ(transaction->frame_start(), initial);
  ASSERT_TRUE(transaction->BeginFrame(2).ok());
  EXPECT_EQ(transaction->frame_start(), transaction->committed());
}
TEST(LiquidFrameTransaction3D, CapacityRestartsWholeFrameOnceAfterAcceptedSubstep) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  auto trial = transaction->ResetTrialFromCommitted();
  ASSERT_TRUE(trial.ok());
  (**trial).next_particle_id = 10;
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.25).ok());
  auto accepted = CloneLiquidState3D(transaction->committed(), kCapacity);
  ASSERT_TRUE(accepted.ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kCapacity, 0.25).ok());
  EXPECT_FALSE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->RestartFrameWithCapacity(kCapacity).ok());
  EXPECT_FALSE(transaction->RestartFrameWithCapacity({8, 1, 2, 4, 2}).ok());
  EXPECT_FALSE(transaction->RestartFrameWithCapacity({8, 2, 2, 4, std::numeric_limits<std::size_t>::max()}).ok());
  EXPECT_EQ(transaction->committed(), *accepted);
  EXPECT_EQ(transaction->frame_start(), initial);
  ASSERT_TRUE(transaction->RestartFrameWithCapacity({8, 2, 2, 4, 2}).ok());
  EXPECT_EQ(transaction->committed(), initial);
  EXPECT_EQ(transaction->frame_start(), initial);
  trial = transaction->ResetTrialFromCommitted();
  ASSERT_TRUE(trial.ok());
  EXPECT_EQ((**trial).particles.capacity(), 8);
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kCapacity, 0.5).ok());
  EXPECT_FALSE(transaction->RestartFrameWithCapacity({16, 2, 2, 4, 2}).ok());
  EXPECT_FALSE(transaction->ResetTrialFromCommitted().ok());
}
TEST(LiquidFrameTransaction3D, RejectsMalformedPersistentStates) {
  using Mutator = void (*)(LiquidState3D&);
  const std::array<Mutator, 23> corruptions = {
      [](auto& s) {
        s.particles[0].id = 0;
      },
      [](auto& s) {
        s.particles[0].id = 1;
      },
      [](auto& s) {
        s.next_particle_id = 2;
      },
      [](auto& s) {
        s.particles[0].volume_ratio = 0;
      },
      [](auto& s) {
        s.particles[0].volume_ratio = std::numeric_limits<double>::infinity();
      },
      [](auto& s) {
        s.particles[0].world_velocity.x = std::nan("");
      },
      [](auto& s) {
        s.particles[0].mass = -1;
      },
      [](auto& s) {
        s.particles[0].reference_volume = 0;
      },
      [](auto& s) {
        s.particles[0].velocity_slopes.z_row.y = std::nan("");
      },
      [](auto& s) {
        s.geometry.cell_width = 0;
      },
      [](auto& s) {
        s.blocks[0].block_index.x = std::numeric_limits<std::int32_t>::max();
      },
      [](auto& s) {
        s.cells[0].cell_index.x = 0;
      },
      [](auto& s) {
        s.cells[0].occupancy = static_cast<LiquidCellOccupancy3D>(42);
      },
      [](auto& s) {
        s.cells[0].previous_pressure = std::nan("");
      },
      [](auto& s) {
        s.faces[0].face_slot = 1;
      },
      [](auto& s) {
        s.faces[0].face_index.axis = static_cast<FaceNormalAxis>(42);
      },
      [](auto& s) {
        s.faces[0].validity = static_cast<LiquidFaceValidity3D>(42);
      },
      [](auto& s) {
        s.faces[0].carrier_velocity = std::nan("");
      },
      [](auto& s) {
        s.sources[0].fractional_remainder = 1;
      },
      [](auto& s) {
        s.clock.time = -1;
      },
      [](auto& s) {
        s.clock.substep_ordinal = 1;
      },
      [](auto& s) {
        s.clock.accepted_step_ordinal = 1;
      },
      [](auto& s) {
        s.clock.frame_ordinal = 1;
      },
  };
  for (std::size_t index = 0; index < std::size(corruptions); ++index) {
    SCOPED_TRACE(index);
    auto initial = InitialState();
    corruptions[index](initial);
    EXPECT_FALSE(LiquidFrameTransaction3D::Create(initial, kCapacity).ok());
  }
}
TEST(LiquidFrameTransaction3D, InvalidAcceptedTrialCannotPublishOrRetry) {
  using Mutator = void (*)(LiquidState3D&);
  for (Mutator corrupt :
       {+[](LiquidState3D& s) {
          s.clock.time = 0.1;
        },
        +[](LiquidState3D& s) {
          s.particles[0].volume_ratio = -1;
        },
        +[](LiquidState3D& s) {
          s.geometry.cell_width = 2;
        },
        +[](LiquidState3D& s) {
          s.random_seed = 1;
        }}) {
    auto initial = InitialState();
    auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
    ASSERT_TRUE(transaction.ok());
    ASSERT_TRUE(transaction->BeginFrame(1).ok());
    auto trial = transaction->ResetTrialFromCommitted();
    ASSERT_TRUE(trial.ok());
    corrupt(**trial);
    EXPECT_FALSE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.5).ok());
    EXPECT_EQ(transaction->committed(), initial);
    EXPECT_FALSE(transaction->ResetTrialFromCommitted().ok());
  }
}
TEST(LiquidFrameTransaction3D, EnforcesFrameAndAttemptLifecycle) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  EXPECT_FALSE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->BeginFrame(0).ok());
  EXPECT_FALSE(transaction->BeginFrame(std::nan("")).ok());
  EXPECT_FALSE(transaction->RestartFrameWithCapacity({8, 2, 2, 4, 2}).ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  EXPECT_FALSE(transaction->BeginFrame(2).ok());
  EXPECT_FALSE(transaction->FinishFrame().ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->FinishFrame().ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kFatal, 0.5).ok());
  EXPECT_FALSE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->FinishFrame().ok());
}
TEST(LiquidFrameTransaction3D, RejectsInvalidDurationAndUnrepresentableProgress) {
  for (double duration : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::nan(""), 1.1}) {
    auto initial = InitialState();
    auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
    ASSERT_TRUE(transaction.ok());
    ASSERT_TRUE(transaction->BeginFrame(1).ok());
    ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
    EXPECT_FALSE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, duration).ok());
    EXPECT_EQ(transaction->committed(), initial);
  }
  auto initial = InitialState();
  initial.clock.time = 1e100;
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(2e100).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 1).ok());
  EXPECT_EQ(transaction->committed(), initial);
}
TEST(LiquidFrameTransaction3D, EnforcesRetryDurationAndCounterRange) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kRetryDt, 1).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.25).ok());
  EXPECT_EQ(transaction->committed(), initial);
  initial.clock.epoch = std::numeric_limits<std::uint64_t>::max();
  transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.5).ok());
  EXPECT_EQ(transaction->committed(), initial);
}
TEST(LiquidFrameTransaction3D, BoundedStoragePreservesExtentAndAddressOnFailure) {
  auto records = LiquidRecords3D<LiquidParticleState3D>::Create(2);
  ASSERT_TRUE(records.ok());
  ASSERT_TRUE(records->Resize(1).ok());
  (*records)[0].id = 9;
  const auto* address = records->values().data();
  EXPECT_FALSE(records->Resize(3).ok());
  EXPECT_EQ(records->size(), 1);
  EXPECT_EQ((*records)[0].id, 9);
  EXPECT_EQ(records->values().data(), address);
  ASSERT_TRUE(records->Resize(2).ok());
  EXPECT_EQ(records->values().data(), address);
}
TEST(LiquidFrameTransaction3D, EqualityIncludesEachPersistentField) {
  using Mutator = void (*)(LiquidState3D&);
  const std::array<Mutator, 62> changes = {
      [](auto& s) {
        ++s.random_seed;
      },
      [](auto& s) {
        ++s.next_particle_id;
      },
      [](auto& s) {
        ++s.geometry.world_origin.x;
      },
      [](auto& s) {
        ++s.geometry.world_origin.y;
      },
      [](auto& s) {
        ++s.geometry.world_origin.z;
      },
      [](auto& s) {
        ++s.geometry.cell_width;
      },
      [](auto& s) {
        ++s.clock.time;
      },
      [](auto& s) {
        ++s.clock.frame_ordinal;
      },
      [](auto& s) {
        ++s.clock.substep_ordinal;
      },
      [](auto& s) {
        ++s.clock.accepted_step_ordinal;
      },
      [](auto& s) {
        ++s.clock.epoch;
      },
      [](auto& s) {
        ++s.particles[0].id;
      },
      [](auto& s) {
        ++s.particles[0].tag;
      },
      [](auto& s) {
        ++s.particles[0].world_position.x;
      },
      [](auto& s) {
        ++s.particles[0].world_position.y;
      },
      [](auto& s) {
        ++s.particles[0].world_position.z;
      },
      [](auto& s) {
        ++s.particles[0].world_velocity.x;
      },
      [](auto& s) {
        ++s.particles[0].world_velocity.y;
      },
      [](auto& s) {
        ++s.particles[0].world_velocity.z;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.x_row.x;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.x_row.y;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.x_row.z;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.y_row.x;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.y_row.y;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.y_row.z;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.z_row.x;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.z_row.y;
      },
      [](auto& s) {
        ++s.particles[0].velocity_slopes.z_row.z;
      },
      [](auto& s) {
        s.particles[0].affine_enabled = false;
      },
      [](auto& s) {
        ++s.particles[0].mass;
      },
      [](auto& s) {
        ++s.particles[0].reference_volume;
      },
      [](auto& s) {
        ++s.particles[0].volume_ratio;
      },
      [](auto& s) {
        ++s.particles[0].side_tag;
      },
      [](auto& s) {
        s.particles[0].pic_initialized = true;
      },
      [](auto& s) {
        ++s.blocks[0].block_index.x;
      },
      [](auto& s) {
        ++s.blocks[0].block_index.y;
      },
      [](auto& s) {
        ++s.blocks[0].block_index.z;
      },
      [](auto& s) {
        ++s.blocks[0].generation;
      },
      [](auto& s) {
        ++s.blocks[0].activation_age;
      },
      [](auto& s) {
        ++s.blocks[0].inactive_age;
      },
      [](auto& s) {
        ++s.cells[0].cell_index.x;
      },
      [](auto& s) {
        ++s.cells[0].cell_index.y;
      },
      [](auto& s) {
        ++s.cells[0].cell_index.z;
      },
      [](auto& s) {
        s.cells[0].occupancy = LiquidCellOccupancy3D::kAir;
      },
      [](auto& s) {
        ++s.cells[0].previous_pressure;
      },
      [](auto& s) {
        s.faces[0].face_index.axis = FaceNormalAxis::kY;
      },
      [](auto& s) {
        ++s.faces[0].face_index.lattice_index.x;
      },
      [](auto& s) {
        ++s.faces[0].face_index.lattice_index.y;
      },
      [](auto& s) {
        ++s.faces[0].face_index.lattice_index.z;
      },
      [](auto& s) {
        ++s.faces[0].face_slot;
      },
      [](auto& s) {
        ++s.faces[0].carrier_velocity;
      },
      [](auto& s) {
        s.faces[0].validity = LiquidFaceValidity3D::kLowMass;
      },
      [](auto& s) {
        ++s.faces[0].side_tag;
      },
      [](auto& s) {
        s.faces[0].structurally_eligible = false;
      },
      [](auto& s) {
        ++s.sources[0].source_ordinal;
      },
      [](auto& s) {
        ++s.sources[0].next_child_ordinal;
      },
      [](auto& s) {
        s.sources[0].fractional_remainder = 0.5;
      },
      [](auto& s) {
        EXPECT_TRUE(s.particles.Resize(1).ok());
      },
      [](auto& s) {
        EXPECT_TRUE(s.blocks.Resize(0).ok());
      },
      [](auto& s) {
        EXPECT_TRUE(s.cells.Resize(0).ok());
      },
      [](auto& s) {
        EXPECT_TRUE(s.faces.Resize(0).ok());
      },
      [](auto& s) {
        EXPECT_TRUE(s.sources.Resize(0).ok());
      },
  };
  const auto initial = InitialState();
  for (std::size_t index = 0; index < std::size(changes); ++index) {
    SCOPED_TRACE(index);
    auto different = CloneLiquidState3D(initial, kCapacity);
    ASSERT_TRUE(different.ok());
    EXPECT_EQ(initial, *different);  // Different capacity alone does not change physical equality.
    changes[index](*different);
    EXPECT_NE(initial, *different);
    auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
    ASSERT_TRUE(transaction.ok());
    ASSERT_TRUE(transaction->BeginFrame(1).ok());
    auto trial = transaction->ResetTrialFromCommitted();
    ASSERT_TRUE(trial.ok());
    changes[index](**trial);
    ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kRetryDt, 1).ok());
    EXPECT_EQ(transaction->committed(), initial);
    EXPECT_EQ(transaction->frame_start(), initial);
  }
}
TEST(LiquidFrameTransaction3D, RequiresCanonicalUniqueTopologyAndSourceKeys) {
  using Mutator = void (*)(LiquidState3D&);
  const std::array<Mutator, 4> duplicates = {
      [](auto& s) {
        ASSERT_TRUE(s.blocks.Resize(2).ok());
        s.blocks[1] = s.blocks[0];
      },
      [](auto& s) {
        ASSERT_TRUE(s.cells.Resize(2).ok());
        s.cells[1] = s.cells[0];
      },
      [](auto& s) {
        ASSERT_TRUE(s.faces.Resize(2).ok());
        s.faces[1] = s.faces[0];
      },
      [](auto& s) {
        ASSERT_TRUE(s.sources.Resize(2).ok());
        s.sources[1] = s.sources[0];
      },
  };
  for (auto duplicate : duplicates) {
    auto state = CloneLiquidState3D(InitialState(), kCapacity);
    ASSERT_TRUE(state.ok());
    duplicate(*state);
    EXPECT_FALSE(ValidateLiquidState3D(*state).ok());
  }
}
TEST(LiquidFrameTransaction3D, ActiveClockMustDescribePossibleCompletedFramesAndSteps) {
  auto state = InitialState();
  state.clock = {1, 1, 1, 1, 1};
  EXPECT_FALSE(ValidateLiquidState3D(state).ok());
  state.clock = {1, 0, 1, 1, 1};
  auto transaction = LiquidFrameTransaction3D::Create(state, kCapacity);
  ASSERT_TRUE(transaction.ok());
  EXPECT_FALSE(transaction->BeginFrame(2).ok());
}
TEST(LiquidFrameTransaction3D, FinalStepReachesBoundaryAndOvershootIsNeverSnapped) {
  auto initial = InitialState();
  initial.clock.time = 0.1;
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(0.3).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.3 - 0.1).ok());
  EXPECT_EQ(transaction->committed().clock.time, 0.3);
  ASSERT_TRUE(transaction->FinishFrame().ok());
  transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(0.3).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_FALSE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, std::nextafter(0.3 - 0.1, 1.0)).ok());
  EXPECT_EQ(transaction->committed(), initial);
}
TEST(LiquidFrameTransaction3D, AcceptedSourceIdentityCannotMoveBackwards) {
  auto state = InitialState();
  state.sources[0].next_child_ordinal = 3;
  auto transaction = LiquidFrameTransaction3D::Create(state, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  auto trial = transaction->ResetTrialFromCommitted();
  ASSERT_TRUE(trial.ok());
  (**trial).sources[0].next_child_ordinal = 2;
  EXPECT_FALSE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 1).ok());
  EXPECT_EQ(transaction->committed(), state);
}
#if defined(_MSC_VER) && defined(_DEBUG)
// The CRT hook observes all Debug heap allocation/free calls, including standard
// library and status allocations. No assertion/logging runs while it is installed.
std::size_t allocation_count = 0;
std::size_t deallocation_count = 0;
std::size_t fail_allocation = 0;
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
    ++allocation_count;
    if (fail_allocation == allocation_count) {
      return 0;
    }
  }
  if (operation == _HOOK_FREE) {
    ++deallocation_count;
  }
  if (previous_hook != nullptr) {
    return previous_hook(operation, data, size, block_type, request, filename, line);
  }
  return 1;
}
TEST(LiquidFrameTransaction3D, SuccessfulAttemptDoesNotAllocateOrFreeHeapStorage) {
  auto state = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(state, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  allocation_count = 0;
  deallocation_count = 0;
  fail_allocation = 0;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  auto trial = transaction->ResetTrialFromCommitted();
  auto outcome = transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 1);
  _CrtSetAllocHook(previous_hook);
  EXPECT_TRUE(trial.ok());
  EXPECT_TRUE(outcome.ok());
  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(deallocation_count, 0);
}
TEST(LiquidFrameTransaction3D, CapacityFailureAndStatusDestructionDoNotAllocateOrFree) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  absl::StatusCode resize_code = absl::StatusCode::kOk;
  bool capacity_outcome = false;
  allocation_count = 0;
  deallocation_count = 0;
  fail_allocation = 0;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    auto trial = transaction->ResetTrialFromCommitted();
    auto resize_status = (**trial).particles.Resize(kCapacity.particles + 1);
    resize_code = resize_status.code();
    auto outcome = transaction->FinishAttempt(LiquidAttemptDisposition3D::kCapacity, 1);
    capacity_outcome = outcome.ok() && outcome->disposition == LiquidAttemptDisposition3D::kCapacity;
  }
  _CrtSetAllocHook(previous_hook);
  EXPECT_EQ(resize_code, absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(capacity_outcome);
  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(deallocation_count, 0);
  EXPECT_EQ(transaction->committed(), initial);
}
TEST(LiquidFrameTransaction3D, InvalidCandidateAndStatusDestructionDoNotAllocateOrFree) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  absl::StatusCode validation_code = absl::StatusCode::kOk;
  absl::StatusCode acceptance_code = absl::StatusCode::kOk;
  allocation_count = 0;
  deallocation_count = 0;
  fail_allocation = 0;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    auto trial = transaction->ResetTrialFromCommitted();
    (**trial).particles[0].volume_ratio = 0;
    auto validation_status = ValidateLiquidState3D(**trial);
    validation_code = validation_status.code();
    auto outcome = transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 1);
    acceptance_code = outcome.status().code();
  }
  _CrtSetAllocHook(previous_hook);
  EXPECT_EQ(validation_code, absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(acceptance_code, absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(deallocation_count, 0);
  EXPECT_EQ(transaction->committed(), initial);
}
TEST(LiquidFrameTransaction3D, AllocationFailureStatusDoesNotRequestAdditionalHeapStorage) {
  absl::StatusCode allocation_code = absl::StatusCode::kOk;
  allocation_count = 0;
  deallocation_count = 0;
  fail_allocation = 1;
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    auto records = LiquidRecords3D<LiquidParticleState3D>::Create(2);
    allocation_code = records.status().code();
  }
  _CrtSetAllocHook(previous_hook);
  fail_allocation = 0;
  EXPECT_EQ(allocation_code, absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(allocation_count, 1);  // Only the failed storage request; no allocation for its error.
  EXPECT_EQ(deallocation_count, 0);
}
TEST(LiquidFrameTransaction3D, PartialAllocationFailureRetainsVersionsAndGrowthAuthority) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.25).ok());
  auto accepted = CloneLiquidState3D(transaction->committed(), kCapacity);
  ASSERT_TRUE(accepted.ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kCapacity, 0.25).ok());
  const auto* committed_particles = transaction->committed().particles.values().data();
  const auto* snapshot_particles = transaction->frame_start().particles.values().data();
  allocation_count = 0;
  deallocation_count = 0;
  absl::StatusCode growth_code = absl::StatusCode::kOk;
  fail_allocation = 7;  // A complete temporary version and part of the next are already owned.
  previous_hook = _CrtSetAllocHook(AllocationHook);
  {
    auto status = transaction->RestartFrameWithCapacity({8, 2, 2, 4, 2});
    growth_code = status.code();
  }
  _CrtSetAllocHook(previous_hook);
  fail_allocation = 0;
  EXPECT_EQ(growth_code, absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(allocation_count, 7);  // No error allocation after the injected failed request.
  EXPECT_EQ(deallocation_count, 6);  // Only the six successfully allocated temporary record owners.
  EXPECT_EQ(transaction->committed(), *accepted);
  EXPECT_EQ(transaction->frame_start(), initial);
  EXPECT_EQ(transaction->committed().particles.values().data(), committed_particles);
  EXPECT_EQ(transaction->frame_start().particles.values().data(), snapshot_particles);
  ASSERT_TRUE(transaction->RestartFrameWithCapacity({8, 2, 2, 4, 2}).ok());
  EXPECT_EQ(transaction->committed(), initial);
}
#endif

TEST(LiquidFrameTransaction3D, FrameCompletionPublishesOneImmutableEpoch) {
  auto transaction = LiquidFrameTransaction3D::Create(InitialState(), kCapacity);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.25).ok());
  EXPECT_EQ(transaction->committed().clock, (LiquidClock3D{0.25, 0, 1, 1, 1}));
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.75).ok());
  EXPECT_EQ(transaction->committed().clock, (LiquidClock3D{1, 1, 0, 2, 2}));
  auto published = CloneLiquidState3D(transaction->committed(), kCapacity);
  ASSERT_TRUE(published.ok());
  ASSERT_TRUE(transaction->FinishFrame().ok());
  EXPECT_EQ(transaction->committed(), *published);
  EXPECT_FALSE(transaction->FinishFrame().ok());
  EXPECT_EQ(transaction->committed(), *published);
}

}  // namespace
TEST(LiquidFrameTransaction3D, EmptyStateCompletesFrameAndMoveTransfersIndependentOwnership) {
  auto created = LiquidFrameTransaction3D::Create(LiquidState3D{}, {});
  ASSERT_TRUE(created.ok());
  auto transaction = std::move(*created);
  ASSERT_TRUE(transaction.BeginFrame(1).ok());
  auto trial = transaction.ResetTrialFromCommitted();
  ASSERT_TRUE(trial.ok());
  EXPECT_EQ((**trial).particles.size(), 0);
  ASSERT_TRUE(transaction.FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 1).ok());
  EXPECT_EQ(transaction.committed().clock, (LiquidClock3D{1, 1, 0, 1, 1}));
  ASSERT_TRUE(transaction.FinishFrame().ok());
  EXPECT_EQ(transaction.frame_start().clock, LiquidClock3D{});
}

TEST(LiquidFrameTransaction3D, PreviewAcceptedClockIsExactReadOnlyAndChecksRetry) {
  auto initial = InitialState();
  auto transaction = LiquidFrameTransaction3D::Create(initial, kCapacity);
  ASSERT_TRUE(transaction.ok());
  EXPECT_EQ(transaction->PreviewAcceptedClock(1).status().code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(transaction->BeginFrame(1).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  const auto first = transaction->PreviewAcceptedClock(1);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(*first, (LiquidClock3D{1, 1, 0, 1, 1}));
  EXPECT_EQ(transaction->committed(), initial);
  EXPECT_EQ(transaction->PreviewAcceptedClock(0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(transaction->PreviewAcceptedClock(2).status().code(), absl::StatusCode::kOutOfRange);
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kRetryDt, 1).ok());
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  EXPECT_EQ(transaction->PreviewAcceptedClock(1).status().code(), absl::StatusCode::kInvalidArgument);
  const auto retry = transaction->PreviewAcceptedClock(0.5);
  ASSERT_TRUE(retry.ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.5).ok());
  EXPECT_EQ(*retry, transaction->committed().clock);
  ASSERT_TRUE(transaction->ResetTrialFromCommitted().ok());
  const auto final = transaction->PreviewAcceptedClock(0.5);
  ASSERT_TRUE(final.ok());
  ASSERT_TRUE(transaction->FinishAttempt(LiquidAttemptDisposition3D::kAccepted, 0.5).ok());
  EXPECT_EQ(*final, transaction->committed().clock);
}
}  // namespace fluidity
