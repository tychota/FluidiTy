#include <gtest/gtest.h>

#include "../cli/simulate_cuda_compatibility.h"
#include "../src/liquid_cache_files_3d_internal.h"

namespace fluidity::simulate {
namespace {
constexpr CudaExecutionIdentityInputs kIdentity{
    .uuid = {'\0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'},
    .compute_major = 12,
    .compute_minor = 0,
    .driver_release = "596.36",
    .driver_api_version = 13040,
    .runtime_api_version = 13030,
    .compiled_runtime_version = 13020,
};

TEST(SimulateCudaIdentity, KeepsCompatibilityBytesAndSeparatesSceneIdentity) {
  constexpr LiquidDigest3D first_fingerprint{0xab};
  constexpr LiquidDigest3D second_fingerprint{0xcd};
  const auto first = BuildCudaExecutionIdentity(kIdentity, first_fingerprint);
  const auto second = BuildCudaExecutionIdentity(kIdentity, second_fingerprint);
  // Historical FLCUDA1 preimage, including the binary UUID's leading zero byte.
  constexpr char legacy_preimage[] =
      "fluidity.simulate.cuda-compatibility.v1\n\0"
      "123456789abcdef\n12\n0\n6\n596.36\n13040\n13030\n13020\n"
      "ab00000000000000000000000000000000000000000000000000000000000000";
  const auto legacy_bytes =
      liquid_cache_files_internal::Bytes(std::string_view(legacy_preimage, sizeof(legacy_preimage) - 1));
  const auto legacy_digest = LiquidContentDigest3D(legacy_bytes);
  constexpr char device_preimage[] =
      "fluidity.simulate.cuda-execution-device.v1\n\0"
      "123456789abcdef\n12\n0\n6\n596.36\n13040\n13030\n13020\n";
  const auto device_bytes =
      liquid_cache_files_internal::Bytes(std::string_view(device_preimage, sizeof(device_preimage) - 1));
  const auto device_digest = LiquidContentDigest3D(device_bytes);
  EXPECT_EQ(first.compatibility_record, std::string("FLCUDA1\n") + LiquidDigestHex3D(legacy_digest).data() + '\n');
  EXPECT_NE(first.compatibility_record, second.compatibility_record);
  EXPECT_EQ(first.device_digest, LiquidDigestHex3D(device_digest).data());
  EXPECT_EQ(first.device_digest, second.device_digest);
  EXPECT_EQ(first.device_digest.size(), 64u);
  EXPECT_EQ(first.device_digest.find_first_not_of("0123456789abcdef"), std::string::npos);
}

TEST(SimulateCudaIdentity, BindsEveryDeviceAndRuntimeField) {
  constexpr LiquidDigest3D fingerprint{};
  const auto original = BuildCudaExecutionIdentity(kIdentity, fingerprint);
  const auto expect_changed = [&](const CudaExecutionIdentityInputs& different) {
    const auto changed = BuildCudaExecutionIdentity(different, fingerprint);
    EXPECT_NE(original.device_digest, changed.device_digest);
    EXPECT_NE(original.compatibility_record, changed.compatibility_record);
  };

  auto different = kIdentity;
  different.uuid[0] = '0';
  expect_changed(different);
  different = kIdentity;
  ++different.compute_major;
  expect_changed(different);
  different = kIdentity;
  ++different.compute_minor;
  expect_changed(different);
  different = kIdentity;
  different.driver_release = "596.37";
  expect_changed(different);
  different = kIdentity;
  ++different.driver_api_version;
  expect_changed(different);
  different = kIdentity;
  ++different.runtime_api_version;
  expect_changed(different);
  different = kIdentity;
  ++different.compiled_runtime_version;
  expect_changed(different);
}
}  // namespace
}  // namespace fluidity::simulate
