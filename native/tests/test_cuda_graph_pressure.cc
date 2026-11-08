#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <ratio>
#include <sstream>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cuda_graph_replay_support.h"
#include "gtest/gtest.h"
#include "pressure_bridge.h"

namespace fluidity::graph_spike {
namespace {

using cuda_internal::PreparedPressureInputs;
using cuda_internal::PressureKernelError;
using cuda_internal::PressureKernelResult;

constexpr std::size_t kCellCount = 2;
constexpr std::size_t kFaceCount = 7;
constexpr std::size_t kInteriorFaceIndex = 1;
constexpr int kExpectedCompiledArchitecture = 1200;
constexpr int kWarmupSequenceCount = 4;
constexpr int kMeasuredSequenceCount = 64;

enum class PressureSubmission {
  kDirect,
  kReplay,
};

struct PressureSequenceMeasurement {
  double submit_us_per_sequence;
  double stream_us_per_sequence;
  PressureKernelResult pressure_result;
};

class CudaTimingEvent {
 public:
  CudaTimingEvent() = default;
  CudaTimingEvent(const CudaTimingEvent&) = delete;
  CudaTimingEvent& operator=(const CudaTimingEvent&) = delete;

  ~CudaTimingEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  absl::Status Initialize() {
    return CheckCuda(cudaEventCreate(&event_), "cudaEventCreate (pressure timing)");
  }

  cudaEvent_t handle() const {
    return event_;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

class PressureReplayFixture {
 public:
  absl::Status Initialize() {
    if (auto status = pressure_inputs_buffer_.Allocate(); !status.ok()) {
      return status;
    }
    if (auto status = pressure_result_buffer_.Allocate(); !status.ok()) {
      return status;
    }
    if (auto status = batch_start_event_.Initialize(); !status.ok()) {
      return status;
    }
    if (auto status = batch_end_event_.Initialize(); !status.ok()) {
      return status;
    }
    return stream_.Initialize();
  }

  absl::Status Capture() {
    if (auto status = stream_.Synchronize(); !status.ok()) {
      return status;
    }
    return replay_.Capture(stream_.handle(), [this] {
      return EnqueuePressureSequence();
    });
  }

  absl::StatusOr<PressureKernelResult>
  Solve(const PreparedPressureInputs& pressure_inputs, PressureSubmission submission) {
    if (auto status = StagePressureSequence(pressure_inputs); !status.ok()) {
      return status;
    }
    if (auto status = SubmitPressureSequence(submission); !status.ok()) {
      return status;
    }
    if (auto status = stream_.Synchronize(); !status.ok()) {
      return status;
    }
    return AcceptPressureResult();
  }

  absl::StatusOr<PressureSequenceMeasurement>
  Measure(const PreparedPressureInputs& pressure_inputs, PressureSubmission submission) {
    if (auto status = StagePressureSequence(pressure_inputs); !status.ok()) {
      return status;
    }
    if (auto status =
            CheckCuda(cudaEventRecord(batch_start_event_.handle(), stream_.handle()), "cudaEventRecord (start)");
        !status.ok()) {
      return status;
    }

    // The CUDA events bracket the complete batch on the stream. The host clock
    // measures only the enqueue loop. Stream order serializes reuse of the fixed
    // buffers, and the host leaves both pinned staging buffers alone until the fence.
    const auto submit_start = std::chrono::steady_clock::now();
    for (int sequence_index = 0; sequence_index < kMeasuredSequenceCount; ++sequence_index) {
      if (auto status = SubmitPressureSequence(submission); !status.ok()) {
        return status;
      }
    }
    const auto submit_end = std::chrono::steady_clock::now();

    if (auto status = CheckCuda(cudaEventRecord(batch_end_event_.handle(), stream_.handle()), "cudaEventRecord (end)");
        !status.ok()) {
      return status;
    }
    if (auto status = stream_.Synchronize(); !status.ok()) {
      return status;
    }
    float batch_stream_ms = 0.0F;
    if (auto status = CheckCuda(
            cudaEventElapsedTime(&batch_stream_ms, batch_start_event_.handle(), batch_end_event_.handle()),
            "cudaEventElapsedTime (pressure batch)"
        );
        !status.ok()) {
      return status;
    }
    const auto pressure_result = AcceptPressureResult();
    if (!pressure_result.ok()) {
      return pressure_result.status();
    }
    constexpr double kMicrosecondsPerMillisecond = 1000.0;
    const double batch_submit_us = std::chrono::duration<double, std::micro>(submit_end - submit_start).count();
    const double batch_stream_us = static_cast<double>(batch_stream_ms) * kMicrosecondsPerMillisecond;
    return PressureSequenceMeasurement{
        .submit_us_per_sequence = batch_submit_us / kMeasuredSequenceCount,
        .stream_us_per_sequence = batch_stream_us / kMeasuredSequenceCount,
        .pressure_result = *pressure_result,
    };
  }

  double capture_us() const {
    return replay_.capture_us();
  }

 private:
  absl::Status StagePressureSequence(const PreparedPressureInputs& pressure_inputs) {
    if (auto status = stream_.Synchronize(); !status.ok()) {
      return status;
    }
    pressure_inputs_buffer_.host_staging().front() = pressure_inputs;
    pressure_result_buffer_.host_staging().front() = {};
    return absl::OkStatus();
  }

  // Both paths use the same addresses and operation order. Clear the device
  // result on every sequence because assembly adds to the original matrix.
  absl::Status EnqueuePressureSequence() const {
    if (auto status = pressure_inputs_buffer_.Upload(stream_.handle()); !status.ok()) {
      return status;
    }
    if (auto status = pressure_result_buffer_.ZeroDevice(stream_.handle()); !status.ok()) {
      return status;
    }
    const auto launch_error = static_cast<cudaError_t>(cuda_internal::LaunchPressure(
        pressure_inputs_buffer_.device_data(),
        pressure_result_buffer_.device_data(),
        stream_.handle()
    ));
    if (auto status = CheckCuda(launch_error, "CUDA pressure launch"); !status.ok()) {
      return status;
    }
    return pressure_result_buffer_.Download(stream_.handle());
  }

  absl::Status SubmitPressureSequence(PressureSubmission submission) const {
    if (submission == PressureSubmission::kReplay) {
      return replay_.Launch(stream_.handle());
    }
    return EnqueuePressureSequence();
  }

  // Read staging only after the caller has fenced the stream.
  absl::StatusOr<PressureKernelResult> AcceptPressureResult() const {
    const auto& staged_pressure_result = pressure_result_buffer_.host_staging().front();
    if (staged_pressure_result.error != PressureKernelError::kNone) {
      return absl::InternalError("CUDA pressure kernel rejected the staged inputs");
    }
    return staged_pressure_result;
  }

  // Teardown fences the stream before destroying events or the executable.
  // Both captured addresses remain live until after graph destruction.
  PinnedDeviceBuffer<PreparedPressureInputs, 1> pressure_inputs_buffer_;
  PinnedDeviceBuffer<PressureKernelResult, 1> pressure_result_buffer_;
  CudaGraphReplay replay_;
  CudaTimingEvent batch_start_event_;
  CudaTimingEvent batch_end_event_;
  CudaStream stream_;
};

// The three x-normal faces precede the four y-normal faces on a 2x1 unit MAC grid.
PreparedPressureInputs MakeTwoCellPressureInputs() {
  constexpr std::array<std::int64_t, kFaceCount> kTailCellIndices{-1, 0, 1, -1, -1, 0, 1};
  constexpr std::array<std::int64_t, kFaceCount> kHeadCellIndices{0, 1, -1, 0, 1, -1, -1};
  PreparedPressureInputs pressure_inputs{
      .cell_count = kCellCount,
      .face_count = kFaceCount,
      .step_duration = 1.0,
  };
  for (std::size_t face_index = 0; face_index < kFaceCount; ++face_index) {
    pressure_inputs.faces[face_index] = {
        .tail_cell_index = kTailCellIndices[face_index],
        .head_cell_index = kHeadCellIndices[face_index],
        .face_area = 1.0,
        .inverse_density = 1.0,
        .pressure_gradient_distance = 1.0,
    };
  }
  pressure_inputs.faces[kInteriorFaceIndex].open_area_fraction = 1.0;
  for (std::size_t cell_index = 0; cell_index < kCellCount; ++cell_index) {
    pressure_inputs.cells[cell_index] = {
        .fluid_volume = 1.0,
        .bulk_modulus = std::numeric_limits<double>::infinity(),
    };
  }
  return pressure_inputs;
}

template <std::size_t UsedEntryCount, typename Entry, std::size_t BufferExtent>
void ExpectPressureEntries(
    const Entry (&actual_entries)[BufferExtent],
    const Entry (&expected_entries)[BufferExtent],
    const char* field_name
) {
  static_assert(UsedEntryCount <= BufferExtent);
  SCOPED_TRACE(field_name);
  for (std::size_t entry_index = 0; entry_index < UsedEntryCount; ++entry_index) {
    EXPECT_EQ(actual_entries[entry_index], expected_entries[entry_index]) << "entry " << entry_index;
  }
}

void ExpectPressureResult(const PressureKernelResult& actual, const PressureKernelResult& expected) {
  EXPECT_EQ(actual.compiled_architecture, expected.compiled_architecture);
  EXPECT_EQ(actual.error, expected.error);
  EXPECT_EQ(actual.gauge_count, expected.gauge_count);
  ExpectPressureEntries<1>(actual.gauge_cell_indices, expected.gauge_cell_indices, "gauge cells");
  ExpectPressureEntries<kCellCount>(actual.cell_pressures, expected.cell_pressures, "cell pressures");
  ExpectPressureEntries<kFaceCount>(actual.corrected_face_velocities, expected.corrected_face_velocities, "velocities");
  ExpectPressureEntries<kCellCount * kCellCount>(actual.original_matrix, expected.original_matrix, "original matrix");
  ExpectPressureEntries<kCellCount>(actual.original_rhs, expected.original_rhs, "original right-hand side");
  ExpectPressureEntries<kCellCount>(actual.original_residuals, expected.original_residuals, "original residuals");
  ExpectPressureEntries<kCellCount>(actual.cell_constraint_defects, expected.cell_constraint_defects, "defects");
}

TEST(CudaGraphReplay, PressureMatchesDirectAndMeasuresReplay) {
  PressureReplayFixture fixture;
  auto status = fixture.Initialize();
  ASSERT_TRUE(status.ok()) << status;

  auto pressure_inputs = MakeTwoCellPressureInputs();
  auto& interior_face = pressure_inputs.faces[kInteriorFaceIndex];
  interior_face.predicted_face_velocity = 2.0;
  interior_face.oriented_pressure_jump = 3.0;

  // The single open face contributes unit conductance. The tail RHS is -(u + jump)
  // and the head RHS is its opposite. Gauge cell 0 leaves the head at u + jump.
  PressureKernelResult expected_pressure_result{
      .compiled_architecture = kExpectedCompiledArchitecture,
      .error = PressureKernelError::kNone,
      .gauge_count = 1,
      .gauge_cell_indices = {0},
      .cell_pressures = {0.0, 5.0},
      .corrected_face_velocities = {},
      .original_matrix = {1.0, -1.0, -1.0, 1.0},
      .original_rhs = {-5.0, 5.0},
      .original_residuals = {},
      .cell_constraint_defects = {},
  };

  const auto initial_direct_result = fixture.Solve(pressure_inputs, PressureSubmission::kDirect);
  ASSERT_TRUE(initial_direct_result.ok()) << initial_direct_result.status();
  ExpectPressureResult(*initial_direct_result, expected_pressure_result);

  status = fixture.Capture();
  ASSERT_TRUE(status.ok()) << status;
  const auto initial_replay_result = fixture.Solve(pressure_inputs, PressureSubmission::kReplay);
  ASSERT_TRUE(initial_replay_result.ok()) << initial_replay_result.status();
  ExpectPressureResult(*initial_replay_result, *initial_direct_result);

  interior_face.predicted_face_velocity = -1.0;
  interior_face.oriented_pressure_jump = 2.0;
  expected_pressure_result.cell_pressures[1] = 1.0;
  expected_pressure_result.original_rhs[0] = -1.0;
  expected_pressure_result.original_rhs[1] = 1.0;

  // Replay first so direct submission cannot populate the changed device inputs.
  const auto changed_replay_result = fixture.Solve(pressure_inputs, PressureSubmission::kReplay);
  ASSERT_TRUE(changed_replay_result.ok()) << changed_replay_result.status();
  ExpectPressureResult(*changed_replay_result, expected_pressure_result);
  const auto changed_direct_result = fixture.Solve(pressure_inputs, PressureSubmission::kDirect);
  ASSERT_TRUE(changed_direct_result.ok()) << changed_direct_result.status();
  ExpectPressureResult(*changed_direct_result, *changed_replay_result);

  for (int warmup_sequence_index = 0; warmup_sequence_index < kWarmupSequenceCount; ++warmup_sequence_index) {
    SCOPED_TRACE(warmup_sequence_index);
    const auto direct_warmup_result = fixture.Solve(pressure_inputs, PressureSubmission::kDirect);
    ASSERT_TRUE(direct_warmup_result.ok()) << direct_warmup_result.status();
    const auto replay_warmup_result = fixture.Solve(pressure_inputs, PressureSubmission::kReplay);
    ASSERT_TRUE(replay_warmup_result.ok()) << replay_warmup_result.status();
  }

  const auto direct_measurement = fixture.Measure(pressure_inputs, PressureSubmission::kDirect);
  ASSERT_TRUE(direct_measurement.ok()) << direct_measurement.status();
  ExpectPressureResult(direct_measurement->pressure_result, expected_pressure_result);

  const auto replay_measurement = fixture.Measure(pressure_inputs, PressureSubmission::kReplay);
  ASSERT_TRUE(replay_measurement.ok()) << replay_measurement.status();
  ExpectPressureResult(replay_measurement->pressure_result, direct_measurement->pressure_result);

  const double capture_us = fixture.capture_us();
  for (double duration_us : {
           capture_us,
           direct_measurement->submit_us_per_sequence,
           replay_measurement->submit_us_per_sequence,
           direct_measurement->stream_us_per_sequence,
           replay_measurement->stream_us_per_sequence,
       }) {
    ASSERT_TRUE(std::isfinite(duration_us));
    ASSERT_GE(duration_us, 0.0);
  }
  ASSERT_FALSE(HasFailure());

  std::ostringstream receipt;
  receipt.imbue(std::locale::classic());
  receipt << std::fixed << std::setprecision(3)
          << "graph_replay compiled_architecture=" << replay_measurement->pressure_result.compiled_architecture
          << " iterations=" << kMeasuredSequenceCount << " capture_us=" << capture_us
          << " direct_submit_us=" << direct_measurement->submit_us_per_sequence
          << " replay_submit_us=" << replay_measurement->submit_us_per_sequence
          << " direct_stream_us=" << direct_measurement->stream_us_per_sequence
          << " replay_stream_us=" << replay_measurement->stream_us_per_sequence;
  std::cout << receipt.str() << '\n';
}

}  // namespace
}  // namespace fluidity::graph_spike
