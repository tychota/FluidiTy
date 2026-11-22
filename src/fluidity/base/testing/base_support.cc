#include "fluidity/base/testing/base_support.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fluidity {

std::uint64_t ToBits(double value) {
  return std::bit_cast<std::uint64_t>(value);
}

std::vector<std::uint64_t> ToBits(std::span<const double> values) {
  std::vector<std::uint64_t> bits;
  bits.reserve(values.size());
  for (const double value : values) {
    bits.push_back(ToBits(value));
  }
  return bits;
}

::testing::AssertionResult CheckWithinTolerance(double reference, double compared, Tolerance tolerance) {
  const double difference = std::abs(reference - compared);
  const double allowed = tolerance.absolute_floor + tolerance.relative_term * std::abs(reference);
  // An infinity on either side agrees only with the same infinity. The tolerance cannot decide: two equal
  // infinities differ by NaN, and an infinite reference makes the allowed difference infinite.
  const bool either_is_infinite = std::isinf(reference) || std::isinf(compared);
  if (either_is_infinite) {
    if (reference == compared) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << std::hexfloat << "reference " << reference << " and compared " << compared
                                         << " are not the same infinity";
  }
  // Written so that a NaN difference fails: every comparison with NaN is false.
  if (difference <= allowed) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << std::hexfloat << "reference " << reference << " and compared " << compared
                                       << " differ by " << difference << ", more than the allowed " << allowed;
}

::testing::AssertionResult CheckSameBitsOrBothNaN(double expected, double actual) {
  if (std::isnan(expected) && std::isnan(actual)) {
    return ::testing::AssertionSuccess();
  }
  if (ToBits(expected) == ToBits(actual)) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << std::hexfloat << "expected " << expected << " but got " << actual
                                       << ", which has other bits";
}

std::vector<double> BuildMixedValues(std::size_t count) {
  constexpr std::array kScales = {1e-3, 1.0, 7.0, 1e6, 1e12};
  std::vector<double> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto spread = static_cast<double>((index * 7919) % 2003) - 1001.0;
    values[index] = spread / 13.0 * kScales[index % kScales.size()];
  }
  return values;
}

std::vector<std::uint64_t> BuildKeysWithManyTies(std::size_t count) {
  std::vector<std::uint64_t> keys(count);
  for (std::size_t index = 0; index < count; ++index) {
    keys[index] = std::uint64_t{(index * 7919) % 1009} * 0x0001000100010001ULL;
  }
  return keys;
}

std::vector<std::uint64_t> BuildKeysDifferingInTopByte(std::size_t count) {
  std::vector<std::uint64_t> keys(count);
  for (std::size_t index = 0; index < count; ++index) {
    keys[index] = std::uint64_t{(index * 31) % 7} << 56U;
  }
  return keys;
}

std::uint64_t ComputeResultDigest(std::span<const std::byte> bytes) {
  // FNV-1a, 64 bits: start from its fixed starting value, then for each byte xor it in and multiply by the prime.
  constexpr std::uint64_t kStartingValue = 0xcbf29ce484222325ULL;
  constexpr std::uint64_t kPrime = 0x100000001b3ULL;
  std::uint64_t digest = kStartingValue;
  for (const std::byte byte : bytes) {
    digest ^= std::to_integer<std::uint64_t>(byte);
    digest *= kPrime;
  }
  return digest;
}

void WriteResultDigestToFile(
    const std::filesystem::path& path,
    std::string_view name,
    std::span<const std::byte> bytes
) {
  constexpr int kHexDigitCount = 16;
  std::string hex_digits(kHexDigitCount, '0');
  std::uint64_t digest = ComputeResultDigest(bytes);
  for (int position = kHexDigitCount - 1; position >= 0; --position) {
    hex_digits[position] = "0123456789abcdef"[digest & 0xFU];
    digest >>= 4U;
  }
  // One write per line, so two test programs that append in turn never mix their lines.
  const std::string line = std::string(name) + " " + hex_digits + "\n";
  std::ofstream file(path, std::ios::app | std::ios::binary);
  file << line;
  file.flush();
  EXPECT_TRUE(file.good()) << "cannot append to the digest file " << path.string();
}

void WriteResultNoteToFile(const std::filesystem::path& path, std::string_view line) {
  // One write per line, as for the digests.
  const std::string text = std::string(line) + "\n";
  std::ofstream file(path, std::ios::app | std::ios::binary);
  file << text;
  file.flush();
  EXPECT_TRUE(file.good()) << "cannot append to the notes file " << path.string();
}

void WriteResultNote(std::string_view line) {
  const std::string path = FindEnvironmentVariable(kResultNotesFileVariable);
  if (path.empty()) {
    return;
  }
  WriteResultNoteToFile(path, line);
}

void WriteResultTimingToFile(const std::filesystem::path& path, std::string_view name, double nanoseconds_per_item) {
  std::ostringstream line;
  line << name << " " << std::setprecision(4) << nanoseconds_per_item << " ns per item";
  WriteResultNoteToFile(path, line.str());
}

void WriteResultTiming(std::string_view name, double nanoseconds_per_item) {
  const std::string path = FindEnvironmentVariable(kResultNotesFileVariable);
  if (path.empty()) {
    return;
  }
  WriteResultTimingToFile(path, name, nanoseconds_per_item);
}

std::string FindResultDigestFile() {
  return FindEnvironmentVariable(kResultDigestFileVariable);
}

std::string FindEnvironmentVariable(std::string_view name) {
  const std::string variable(name);
#if defined(_MSC_VER)
  // MSVC deprecates getenv (warning C4996, an error under /WX); _dupenv_s returns a copy the caller frees.
  char* path = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&path, &length, variable.c_str()) != 0 || path == nullptr) {
    return {};
  }
  std::string copy(path);
  std::free(path);
  return copy;
#else
  // Tests read the environment from one thread, so getenv's shared buffer is safe here.
  const char* const path = std::getenv(variable.c_str());
  return path != nullptr ? std::string(path) : std::string();
#endif
}

}  // namespace fluidity
