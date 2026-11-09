#include <filesystem>
#include <iostream>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/liquid_cache_3d.h"
int Produce(int argc, const std::filesystem::path& path, bool digest) {
  if (argc != 3) {
    return 1;
  }
  if (digest) {
    fluidity::LiquidCacheLimits3D limits;
    limits.maximum_bytes = 1024 * 1024 * 1024;
    auto bytes = fluidity::liquid_cache_files_internal::ReadBytes(path, limits);
    if (!bytes.ok()) {
      return 2;
    }
    std::cout << fluidity::LiquidDigestHex3D(fluidity::LiquidContentDigest3D(bytes->values())).data() << '\n';
    return 0;
  }
  fluidity::LiquidResolvedSceneView3D scene;
  auto cache = fluidity::LiquidCache3D::Create(path, scene);
  if (!cache.ok()) {
    return 3;
  }
  auto fingerprint = fluidity::ResolvedSceneFingerprint(scene);
  if (!fingerprint.ok()) {
    return 4;
  }
  fluidity::LiquidResumeMetadata3D resume;
  resume.resolved_fingerprint = *fingerprint;
  resume.execution_id = fluidity::LiquidExecutionId3D(*fingerprint);
  fluidity::LiquidState3D state;
  fluidity::LiquidPointFrame3D points;
  return cache->PublishFrame(points, &state, &resume).ok() ? 0 : 5;
}
#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
  return argc == 3 ? Produce(argc, argv[2], std::wstring_view(argv[1]) == L"digest") : 1;
}
#else
int main(int argc, char** argv) {
  return argc == 3 ? Produce(argc, argv[2], std::string_view(argv[1]) == "digest") : 1;
}
#endif
