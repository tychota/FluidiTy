#include "../src/liquid_cache_files_3d_internal.h"
#include "simulate_cuda_compatibility.h"

namespace fluidity::simulate {
CudaExecutionIdentity
BuildCudaExecutionIdentity(const CudaExecutionIdentityInputs& inputs, const LiquidDigest3D& resolved_fingerprint) {
  // The UUID is a fixed 16-byte prefix after each domain. The remaining
  // fields are length-delimited or decimal lines in the historical order.
  std::string device_fields(inputs.uuid.data(), inputs.uuid.size());
  device_fields += '\n';
  device_fields += std::to_string(inputs.compute_major) + '\n';
  device_fields += std::to_string(inputs.compute_minor) + '\n';
  device_fields += std::to_string(inputs.driver_release.size()) + '\n';
  device_fields += inputs.driver_release;
  device_fields += '\n';
  device_fields += std::to_string(inputs.driver_api_version) + '\n';
  device_fields += std::to_string(inputs.runtime_api_version) + '\n';
  device_fields += std::to_string(inputs.compiled_runtime_version) + '\n';

  const std::string compatibility_preimage =
      "fluidity.simulate.cuda-compatibility.v1\n" + device_fields + LiquidDigestHex3D(resolved_fingerprint).data();
  const std::string device_preimage = "fluidity.simulate.cuda-execution-device.v1\n" + device_fields;
  const auto compatibility_digest = LiquidContentDigest3D(liquid_cache_files_internal::Bytes(compatibility_preimage));
  const auto device_digest = LiquidContentDigest3D(liquid_cache_files_internal::Bytes(device_preimage));
  return {
      .compatibility_record =
          std::string(kCudaCompatibilityRecordPrefix) + LiquidDigestHex3D(compatibility_digest).data() + '\n',
      .device_digest = LiquidDigestHex3D(device_digest).data(),
  };
}
}  // namespace fluidity::simulate
