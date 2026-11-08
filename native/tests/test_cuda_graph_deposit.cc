#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "cuda_graph_replay_support.h"
#include "gtest/gtest.h"
#include "transfer_bridge.h"

namespace fluidity::graph_spike {
namespace {

using cuda_internal::DepositError;
using cuda_internal::DepositFaceAccumulator;
using cuda_internal::DepositKernelState;
using cuda_internal::DepositParticle;
using cuda_internal::DepositScales;
using cuda_internal::LaunchDeposit;
using cuda_internal::PrepareTransferStencil;

constexpr Grid2D kMacGrid{4, 4, 1.0};
constexpr std::size_t kFaceCount = 40;
constexpr std::size_t kParticleCapacity = 4;
constexpr DepositScales kDepositScales{0x1p-20, 0x1p-20, 0x1p-20};
constexpr Vec2 kFirstParticleWorldPosition{1.5, 1.5};
constexpr std::size_t kFirstParticleXNormalFaceIndex = 6;
constexpr int kExpectedCompiledArchitecture = 1200;

struct DepositTotals {
  std::array<std::int64_t, kFaceCount> mass_quanta_sums{};
  std::array<std::int64_t, kFaceCount> pure_momentum_quanta_sums{};
  std::array<std::int64_t, kFaceCount> stress_force_quanta_sums{};
};

enum class DepositSubmission {
  kDirect,
  kReplay,
};

class DepositReplayFixture {
 public:
  absl::Status Initialize() {
    const auto face_count = FaceCount(kMacGrid);
    if (!face_count.ok()) {
      return face_count.status();
    }
    if (*face_count != kFaceCount) {
      return absl::InternalError("unexpected deposit fixture face count");
    }
    // All physical fields are zero, but inactive slots still visit a valid stencil.
    if (auto status = PrepareTransferStencil(kMacGrid, kFirstParticleWorldPosition, neutral_particle_.stencil);
        !status.ok()) {
      return status;
    }
    if (auto status = particles_.Allocate(); !status.ok()) {
      return status;
    }
    if (auto status = trial_accumulators_.Allocate(); !status.ok()) {
      return status;
    }
    if (auto status = trial_state_.Allocate(); !status.ok()) {
      return status;
    }
    return stream_.Initialize();
  }

  absl::Status Capture() {
    return replay_.Capture(stream_.handle(), [this] {
      return EnqueueDeposit();
    });
  }

  // Host capacity rejection leaves staging and accepted totals untouched. Trial
  // outputs are published only after the sequence and device checks succeed.
  absl::Status Deposit(std::span<const DepositParticle> live_particles, DepositSubmission submission) {
    if (live_particles.size() > kParticleCapacity) {
      return absl::ResourceExhaustedError("live deposit particles exceed the fixed replay capacity");
    }

    // The previous sequence has fenced, so pinned storage is safe to rewrite.
    auto staged_particles = particles_.host_staging();
    std::ranges::copy(live_particles, staged_particles.begin());
    std::ranges::fill(staged_particles.subspan(live_particles.size()), neutral_particle_);
    std::ranges::fill(trial_accumulators_.host_staging(), DepositFaceAccumulator{});
    trial_state_.host_staging().front() = {};

    absl::Status status =
        submission == DepositSubmission::kDirect ? EnqueueDeposit() : replay_.Launch(stream_.handle());
    status.Update(stream_.Synchronize());
    if (!status.ok()) {
      return status;
    }

    const auto& deposit_state = trial_state_.host_staging().front();
    if (deposit_state.error == static_cast<int>(DepositError::kOutOfRange)) {
      return absl::OutOfRangeError("device rejected an unquantizable deposit contribution");
    }
    if (deposit_state.error != static_cast<int>(DepositError::kNone)) {
      return absl::InternalError("device rejected the deposit stencil");
    }

    const auto face_accumulators = trial_accumulators_.host_staging();
    for (std::size_t face_index = 0; face_index < kFaceCount; ++face_index) {
      const auto& accumulator = face_accumulators[face_index];
      accepted_totals_.mass_quanta_sums[face_index] = std::bit_cast<std::int64_t>(accumulator.mass_quanta_sum);
      accepted_totals_.pure_momentum_quanta_sums[face_index] =
          std::bit_cast<std::int64_t>(accumulator.pure_momentum_quanta_sum);
      accepted_totals_.stress_force_quanta_sums[face_index] =
          std::bit_cast<std::int64_t>(accumulator.stress_force_quanta_sum);
    }
    accepted_compiled_architecture_ = deposit_state.compiled_architecture;
    return absl::OkStatus();
  }

  const DepositTotals& accepted_totals() const {
    return accepted_totals_;
  }

  int accepted_compiled_architecture() const {
    return accepted_compiled_architecture_;
  }

 private:
  // Direct submission and graph capture use the same fixed-size CUDA sequence.
  absl::Status EnqueueDeposit() {
    const cudaStream_t stream = stream_.handle();
    if (auto status = particles_.Upload(stream); !status.ok()) {
      return status;
    }
    if (auto status = trial_accumulators_.ZeroDevice(stream); !status.ok()) {
      return status;
    }
    if (auto status = trial_state_.ZeroDevice(stream); !status.ok()) {
      return status;
    }
    const auto launch_error = static_cast<cudaError_t>(LaunchDeposit(
        particles_.device_data(),
        kParticleCapacity,
        kDepositScales,
        trial_accumulators_.device_data(),
        kFaceCount,
        trial_state_.device_data(),
        stream
    ));
    if (auto status = CheckCuda(launch_error, "LaunchDeposit"); !status.ok()) {
      return status;
    }
    if (auto status = trial_accumulators_.Download(stream); !status.ok()) {
      return status;
    }
    return trial_state_.Download(stream);
  }

  DepositParticle neutral_particle_{};
  DepositTotals accepted_totals_{};
  int accepted_compiled_architecture_ = 0;
  PinnedDeviceBuffer<DepositParticle, kParticleCapacity> particles_;
  PinnedDeviceBuffer<DepositFaceAccumulator, kFaceCount> trial_accumulators_;
  PinnedDeviceBuffer<DepositKernelState, 1> trial_state_;
  CudaGraphReplay replay_;
  CudaStream stream_;
};

absl::Status PrepareActiveParticle(Vec2 particle_world_position, double mass, DepositParticle& particle) {
  particle = {
      .stencil = {},
      .mass = mass,
      .velocity = {2.0, -3.0},
      .velocity_fit_slopes = {},
      .reference_volume = 2.0,
      .kirchhoff_stress = {.xx = 4.0, .xy = 0.0, .yx = 0.0, .yy = 0.0},
  };
  return PrepareTransferStencil(kMacGrid, particle_world_position, particle.stencil);
}

void ExpectDepositTotals(const DepositTotals& actual_totals, const DepositTotals& expected_totals) {
  EXPECT_EQ(actual_totals.mass_quanta_sums, expected_totals.mass_quanta_sums);
  EXPECT_EQ(actual_totals.pure_momentum_quanta_sums, expected_totals.pure_momentum_quanta_sums);
  EXPECT_EQ(actual_totals.stress_force_quanta_sums, expected_totals.stress_force_quanta_sums);
}

TEST(CudaGraphReplay, DepositMatchesDirectAndClearsInactiveInputs) {
  DepositReplayFixture fixture;
  auto status = fixture.Initialize();
  ASSERT_TRUE(status.ok()) << status;

  std::array<DepositParticle, 2> live_particles;
  status = PrepareActiveParticle(kFirstParticleWorldPosition, 1.0, live_particles[0]);
  ASSERT_TRUE(status.ok()) << status;
  status = PrepareActiveParticle({2.5, 2.5}, 2.0, live_particles[1]);
  ASSERT_TRUE(status.ok()) << status;
  const auto one_live_particle = std::span(live_particles).first<1>();

  status = fixture.Deposit(one_live_particle, DepositSubmission::kDirect);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  const auto one_particle_direct_totals = fixture.accepted_totals();
  // Face (1, 1.5) has weight 3/8 and x gradient -3/4. The force is -2 * 4 * (-3/4).
  EXPECT_EQ(one_particle_direct_totals.mass_quanta_sums[kFirstParticleXNormalFaceIndex], 393216);
  EXPECT_EQ(one_particle_direct_totals.pure_momentum_quanta_sums[kFirstParticleXNormalFaceIndex], 786432);
  EXPECT_EQ(one_particle_direct_totals.stress_force_quanta_sums[kFirstParticleXNormalFaceIndex], 6291456);

  status = fixture.Capture();
  ASSERT_TRUE(status.ok()) << status;
  status = fixture.Deposit(one_live_particle, DepositSubmission::kReplay);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  ExpectDepositTotals(fixture.accepted_totals(), one_particle_direct_totals);

  // Replay changed inputs before direct submission can write them to the device.
  status = fixture.Deposit(live_particles, DepositSubmission::kReplay);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  const auto two_particle_replay_totals = fixture.accepted_totals();
  // At face (2, 2.5), the two masses contribute 1/16 and 3/4 before scaling.
  constexpr std::size_t kSecondParticleXNormalFaceIndex = 12;
  EXPECT_EQ(two_particle_replay_totals.mass_quanta_sums[kSecondParticleXNormalFaceIndex], 851968);
  status = fixture.Deposit(live_particles, DepositSubmission::kDirect);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  ExpectDepositTotals(fixture.accepted_totals(), two_particle_replay_totals);

  std::ranges::reverse(live_particles);
  for (int replay_index = 0; replay_index < 2; ++replay_index) {
    SCOPED_TRACE(replay_index);
    status = fixture.Deposit(live_particles, DepositSubmission::kReplay);
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
    ExpectDepositTotals(fixture.accepted_totals(), two_particle_replay_totals);
  }
  status = fixture.Deposit(live_particles, DepositSubmission::kDirect);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  ExpectDepositTotals(fixture.accepted_totals(), two_particle_replay_totals);

  status = fixture.Deposit({}, DepositSubmission::kReplay);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  const auto empty_replay_totals = fixture.accepted_totals();
  ExpectDepositTotals(empty_replay_totals, DepositTotals{});
  status = fixture.Deposit({}, DepositSubmission::kDirect);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  ExpectDepositTotals(fixture.accepted_totals(), empty_replay_totals);
}

TEST(CudaGraphReplay, RejectedDepositPreservesAcceptedOutputs) {
  DepositReplayFixture fixture;
  auto status = fixture.Initialize();
  ASSERT_TRUE(status.ok()) << status;
  status = fixture.Capture();
  ASSERT_TRUE(status.ok()) << status;

  DepositParticle live_particle;
  status = PrepareActiveParticle(kFirstParticleWorldPosition, 1.0, live_particle);
  ASSERT_TRUE(status.ok()) << status;
  const auto one_live_particle = std::span(&live_particle, 1);
  status = fixture.Deposit(one_live_particle, DepositSubmission::kReplay);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  const auto accepted_totals = fixture.accepted_totals();
  EXPECT_EQ(accepted_totals.mass_quanta_sums[kFirstParticleXNormalFaceIndex], 393216);

  std::array<DepositParticle, kParticleCapacity + 1> excess_live_particles;
  excess_live_particles.fill(live_particle);
  status = fixture.Deposit(excess_live_particles, DepositSubmission::kReplay);
  EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted) << status;
  ExpectDepositTotals(fixture.accepted_totals(), accepted_totals);

  live_particle.kirchhoff_stress.xx = 0x1p60;
  status = fixture.Deposit(one_live_particle, DepositSubmission::kReplay);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange) << status;
  ExpectDepositTotals(fixture.accepted_totals(), accepted_totals);

  live_particle.kirchhoff_stress.xx = 4.0;
  live_particle.mass = 2.0;
  status = fixture.Deposit(one_live_particle, DepositSubmission::kReplay);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  const auto recovered_replay_totals = fixture.accepted_totals();
  EXPECT_EQ(recovered_replay_totals.mass_quanta_sums[kFirstParticleXNormalFaceIndex], 786432);
  status = fixture.Deposit(one_live_particle, DepositSubmission::kDirect);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fixture.accepted_compiled_architecture(), kExpectedCompiledArchitecture);
  ExpectDepositTotals(fixture.accepted_totals(), recovered_replay_totals);
}

TEST(CudaGraphReplay, FailedCaptureCanBeRetried) {
  PinnedDeviceBuffer<int, 1> memset_target;
  CudaGraphReplay replay;
  CudaStream stream;
  auto status = memset_target.Allocate();
  ASSERT_TRUE(status.ok()) << status;
  status = stream.Initialize();
  ASSERT_TRUE(status.ok()) << status;

  // Seed a nonzero device value so a recovered graph must actually run its memset.
  memset_target.host_staging().front() = 123;
  status = memset_target.Upload(stream.handle());
  ASSERT_TRUE(status.ok()) << status;
  status = stream.Synchronize();
  ASSERT_TRUE(status.ok()) << status;
  const auto callback_failure = absl::FailedPreconditionError("test callback rejected capture after memset");
  status = replay.Capture(stream.handle(), [&] {
    if (auto memset_status = memset_target.ZeroDevice(stream.handle()); !memset_status.ok()) {
      return memset_status;
    }
    return callback_failure;
  });
  ASSERT_EQ(status, callback_failure);

  cudaStreamCaptureStatus stream_capture_state = cudaStreamCaptureStatusActive;
  status = CheckCuda(cudaStreamIsCapturing(stream.handle(), &stream_capture_state), "cudaStreamIsCapturing");
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(stream_capture_state, cudaStreamCaptureStatusNone);
  status = replay.Launch(stream.handle());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition) << status;

  status = replay.Capture(stream.handle(), [&] {
    if (auto memset_status = memset_target.ZeroDevice(stream.handle()); !memset_status.ok()) {
      return memset_status;
    }
    return memset_target.Download(stream.handle());
  });
  ASSERT_TRUE(status.ok()) << status;
  status = replay.Launch(stream.handle());
  ASSERT_TRUE(status.ok()) << status;
  status = stream.Synchronize();
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(memset_target.host_staging().front(), 0);
}

}  // namespace
}  // namespace fluidity::graph_spike
