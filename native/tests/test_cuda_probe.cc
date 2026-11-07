#include <array>
#include <iostream>

#include "fluidity/host/cuda_probe.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(CudaProbe, SharedVec2Math) {
  // Independently expanded values for the 17 samples followed by the sentinel.
  constexpr std::array<Vec2, 18> expected{{
      {2.5, -1.25},
      {4.0, -2.75},
      {5.5, -4.25},
      {7.0, -5.75},
      {8.5, -7.25},
      {10.0, -8.75},
      {11.5, -10.25},
      {13.0, -11.75},
      {14.5, -13.25},
      {16.0, -14.75},
      {17.5, -16.25},
      {19.0, -17.75},
      {20.5, -19.25},
      {22.0, -20.75},
      {23.5, -22.25},
      {25.0, -23.75},
      {26.5, -25.25},
      {0.0, 0.0},
  }};

  ProbeReport report;
  const absl::Status status = RunCudaProbe(0, &report);
  ASSERT_TRUE(status.ok()) << status;

  EXPECT_EQ(report.element_count, 17);
  EXPECT_EQ(report.launch_result, 0);
  EXPECT_EQ(report.copy_result, 0);
  EXPECT_EQ(report.synchronize_result, 0);
  EXPECT_FALSE(report.device_name.empty());
#if defined(FLUIDITY_QUALIFIED_CUDA_DEVICE)
  EXPECT_EQ(report.device_name, FLUIDITY_QUALIFIED_CUDA_DEVICE);
#endif
  EXPECT_GT(report.compute_major, 0);
  EXPECT_EQ(report.compiled_architecture, report.compute_major * 100 + report.compute_minor * 10);
  EXPECT_GT(report.driver_version, 0);
  EXPECT_GT(report.runtime_version, 0);
  ASSERT_EQ(report.values.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(report.values[index].x, expected[index].x);
    EXPECT_EQ(report.values[index].y, expected[index].y);
  }

  std::cout << "device=" << report.device_name << " capability=" << report.compute_major << '.' << report.compute_minor
            << " compiled_architecture=" << report.compiled_architecture << " driver_api=" << report.driver_version
            << " runtime_api=" << report.runtime_version << " launch=" << report.launch_result
            << " copy=" << report.copy_result << " synchronize=" << report.synchronize_result
            << " elements=" << report.element_count << " sentinel=(" << report.values.back().x << ','
            << report.values.back().y << ")\n";
}

}  // namespace
}  // namespace fluidity
