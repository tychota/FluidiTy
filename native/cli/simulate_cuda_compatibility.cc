#include "simulate_cuda_compatibility.h"

#include <nvml.h>

#include <algorithm>
#include <array>
#include <bit>

#include "../cuda/liquid_device_3d_internal.h"
#include "../src/liquid_cache_files_3d_internal.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace fluidity::simulate {
namespace {
namespace files = liquid_cache_files_internal;

constexpr char kCompatibilityFile[] = ".simulate-cuda-compatibility";
constexpr std::string_view kRecordPrefix = "FLCUDA1\n";
constexpr std::size_t kHexDigestLength = 64;
constexpr std::size_t kRecordLength = kRecordPrefix.size() + kHexDigestLength + 1;
constexpr LiquidCacheLimits3D kRecordLimits{.maximum_bytes = kRecordLength};

// The NVIDIA management library is loaded only during CUDA execution admission.
class NvmlLibrary {
 public:
  NvmlLibrary() = default;
  NvmlLibrary(const NvmlLibrary&) = delete;
  NvmlLibrary& operator=(const NvmlLibrary&) = delete;
  ~NvmlLibrary() {
    if (handle_ != nullptr) {
#if defined(_WIN32)
      FreeLibrary(handle_);
#else
      dlclose(handle_);
#endif
    }
  }

  template <typename FunctionPointer>
  FunctionPointer Find(const char* name) const {
    if (handle_ == nullptr) {
      return nullptr;
    }
#if defined(_WIN32)
    return std::bit_cast<FunctionPointer>(GetProcAddress(handle_, name));
#else
    return std::bit_cast<FunctionPointer>(dlsym(handle_, name));
#endif
  }

 private:
#if defined(_WIN32)
  HMODULE handle_ = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
  void* handle_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
};

absl::StatusOr<std::string> NvidiaDriverRelease() {
  const NvmlLibrary library;
  const auto initialize = library.Find<decltype(&nvmlInit_v2)>("nvmlInit_v2");
  const auto get_driver_release = library.Find<decltype(&nvmlSystemGetDriverVersion)>("nvmlSystemGetDriverVersion");
  const auto shutdown = library.Find<decltype(&nvmlShutdown)>("nvmlShutdown");
  if (initialize == nullptr || get_driver_release == nullptr || shutdown == nullptr) {
    return absl::UnavailableError("");
  }
  if (initialize() != NVML_SUCCESS) {
    return absl::UnavailableError("");
  }
  std::array<char, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE> release{};
  const auto query_status = get_driver_release(release.data(), static_cast<unsigned int>(release.size()));
  const auto shutdown_status = shutdown();
  if (query_status != NVML_SUCCESS || shutdown_status != NVML_SUCCESS) {
    return absl::UnavailableError("");
  }
  const auto end = std::ranges::find(release, '\0');
  if (end == release.begin() || end == release.end()) {
    return absl::UnavailableError("");
  }
  return std::string(release.begin(), end);
}

bool IsLowercaseHex(char character) {
  return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
}
}  // namespace

absl::StatusOr<std::string> CudaCompatibilityRecord(int device_ordinal, const LiquidDigest3D& resolved_fingerprint) {
  auto properties = cuda_internal::QueryLiquidCudaDevice3D(device_ordinal);
  if (!properties.ok()) {
    return properties.status();
  }
  if (std::ranges::all_of(properties->uuid.bytes, [](char value) {
        return value == 0;
      })) {
    return absl::UnavailableError("");
  }
  int driver_api_version = 0;
  auto status = cuda_internal::LiquidCudaDeviceSetupStatus3D(cudaDriverGetVersion(&driver_api_version));
  if (!status.ok()) {
    return status;
  }
  int runtime_api_version = 0;
  status = cuda_internal::LiquidCudaDeviceSetupStatus3D(cudaRuntimeGetVersion(&runtime_api_version));
  if (!status.ok()) {
    return status;
  }
  if (driver_api_version <= 0 || runtime_api_version <= 0) {
    return absl::UnavailableError("");
  }
  auto driver_release = NvidiaDriverRelease();
  if (!driver_release.ok()) {
    return driver_release.status();
  }

  // The UUID is a fixed 16-byte prefix after the domain. Remaining fields
  // are length-delimited or decimal lines. The Windows hybrid runtime comes
  // from the display driver; its exact release and reported API are both bound.
  std::string identity = "fluidity.simulate.cuda-compatibility.v1\n";
  identity.append(properties->uuid.bytes, sizeof(properties->uuid.bytes));
  identity += '\n';
  identity += std::to_string(properties->major) + '\n';
  identity += std::to_string(properties->minor) + '\n';
  identity += std::to_string(driver_release->size()) + '\n';
  identity += *driver_release + '\n';
  identity += std::to_string(driver_api_version) + '\n';
  identity += std::to_string(runtime_api_version) + '\n';
  identity += std::to_string(CUDART_VERSION) + '\n';
  identity += LiquidDigestHex3D(resolved_fingerprint).data();
  const auto digest = LiquidContentDigest3D(files::Bytes(identity));
  return std::string(kRecordPrefix) + LiquidDigestHex3D(digest).data() + '\n';
}

absl::StatusOr<std::string> ReadCudaCompatibilityRecord(const std::filesystem::path& verified_cache_directory) {
  auto bytes = files::ReadBytes(verified_cache_directory / kCompatibilityFile, kRecordLimits, kRecordLength);
  if (absl::IsNotFound(bytes.status())) {
    return absl::FailedPreconditionError("");
  }
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto values = bytes->values();
  const std::string_view record(reinterpret_cast<const char*>(values.data()), values.size());
  if (!record.starts_with(kRecordPrefix) || record.back() != '\n'
      || !std::ranges::all_of(record.substr(kRecordPrefix.size(), kHexDigestLength), IsLowercaseHex)) {
    return absl::DataLossError("");
  }
  return std::string(record);
}

absl::Status
PublishCudaCompatibilityRecord(const std::filesystem::path& verified_cache_directory, std::string_view record) {
  return files::PublishBytes(
      verified_cache_directory / kCompatibilityFile,
      files::Bytes(record),
      kRecordLimits,
      {},
      absl::StatusCode::kFailedPrecondition
  );
}

}  // namespace fluidity::simulate
