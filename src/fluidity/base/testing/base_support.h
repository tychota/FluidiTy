// Functions that the tests of base/ share, so no test file keeps its own copy.
//
// Purpose: give every test one way to compare doubles bit for bit, one form of the method paper's tolerance, one
//   check for "same bits, or both NaN", one way to fill and read a buffer, and one record of GPU results for the
//   repeat run.
//
// Idea: a test that demands equal bits compares the 64 bits of two doubles (ToBits), because == treats +0 and -0
//   as equal and a NaN as unequal to itself. CheckSameBitsOrBothNaN accepts two NaNs whatever their bits, since IEEE
//   754 leaves the bits of a NaN open. CheckWithinTolerance is the comparison of a CPU and a GPU result that the
//   method paper states, with the CPU result as the reference: |reference - compared| <= absolute floor + relative
//   term * |reference|.
//   BuildBufferHolding and ReadAll fill a buffer from CPU memory and read it back, failing the calling test when
//   a call fails. WriteResultDigest appends a named digest of result bytes to the file that the environment
//   variable FLUIDITY_RESULT_DIGEST_FILE names, and does nothing when it is unset: tools/run_gpu_tests.ps1 sets it
//   for each of its two runs of the gpu tests and demands identical files, so "repeat run: PASS" means the GPU
//   wrote the same bytes twice. WriteResultNote and WriteResultTiming append a line (a finding the owner must read, or
//   a measured time per item) to the notes file that FLUIDITY_RESULT_NOTES_FILE names, kept apart from the digests
//   because times differ between runs and notes are read, not compared.
//   Example: CheckWithinTolerance(1.0, 1.0 + 2e-16, {.absolute_floor = 1e-15, .relative_term = 1e-15}) passes.
//
// Words:
//   digest  a 64-bit number computed from bytes (FNV-1a), equal for equal bytes and almost never for different ones
// Owns:    ToBits, Tolerance, CheckWithinTolerance, CheckSameBitsOrBothNaN, BuildBufferHolding, ReadAll,
//          ComputeResultDigest, WriteResultDigestToFile, WriteResultDigest, FindResultDigestFile,
//          kResultDigestFileVariable, kResultNotesFileVariable, WriteResultNoteToFile, WriteResultNote,
//          WriteResultTimingToFile, WriteResultTiming,
//          FindEnvironmentVariable, BuildMixedValues, BuildKeysWithManyTies, BuildKeysDifferingInTopByte
// Uses:    Buffer, BuildBuffer, WriteBuffer, ReadBuffer (base/buffer.h), RunOn (base/run_on.h), CPU-only guard
//          (base/cpu_only.h), GoogleTest
// Used by: the tests of base/ (the library fluidity_base_testing)
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"

namespace fluidity {

// Converts a double to its 64 bits, so +0.0 and -0.0 differ and every bit is compared.
std::uint64_t ToBits(double value);

// Converts every double of `values` to its 64 bits, in order, so two arrays compare bit for bit with ==.
std::vector<std::uint64_t> ToBits(std::span<const double> values);

// The allowed difference between a reference result (the CPU's) and a compared one (the GPU's): absolute_floor +
// relative_term * |reference|.
struct Tolerance {
  double absolute_floor = 0.0;  // Same unit as the values; the allowed difference near zero.
  double relative_term = 0.0;  // No unit; the allowed difference per unit of |reference|.
};

// Checks |reference - compared| <= tolerance.absolute_floor + tolerance.relative_term * |reference| (method paper,
// <sec-run-on>). Two equal infinities of the same sign pass; a NaN on either side fails. The failure message gives
// both values and the difference.
// Example: reference 2.0, compared 2.0 + 1e-15, {1e-15, 1e-15}: allowed 3e-15, passes.
::testing::AssertionResult CheckWithinTolerance(double reference, double compared, Tolerance tolerance);

// Checks that two doubles have the same 64 bits, or are both NaN (IEEE 754 leaves the bits of a NaN open).
// Example: (+0.0, -0.0) fails; (NaN, -NaN) passes; (1.0, 1.0) passes.
::testing::AssertionResult CheckSameBitsOrBothNaN(double expected, double actual);

// Builds a buffer on `run_on` holding `values` (a std::vector, a std::array or a span). Fails the calling test, and
// returns an empty CPU buffer, when building or writing fails.
// Example: BuildBufferHolding(RunOn::kCpu, std::array{1.0, 2.0}) is a CPU buffer of 2 doubles.
template <std::ranges::contiguous_range Values>
Buffer<std::ranges::range_value_t<Values>> BuildBufferHolding(RunOn run_on, const Values& values) {
  using Value = std::ranges::range_value_t<Values>;
  const std::span<const Value> value_span(std::ranges::data(values), std::ranges::size(values));
  std::expected<Buffer<Value>, BufferError> buffer = BuildBuffer<Value>(run_on, value_span.size());
  EXPECT_TRUE(buffer.has_value()) << ToName(buffer.error());
  if (!buffer.has_value()) {
    return {};
  }
  const std::expected<void, BufferError> written = WriteBuffer(value_span, *buffer);
  EXPECT_TRUE(written.has_value()) << ToName(written.error());
  return std::move(*buffer);
}

// Reads every value of a buffer into CPU memory. Fails the calling test when reading fails.
template <BufferValue Value>
std::vector<Value> ReadAll(const Buffer<Value>& buffer) {
  std::vector<Value> values(buffer.size());
  const std::expected<void, BufferError> read = ReadBuffer(buffer, std::span<Value>(values));
  EXPECT_TRUE(read.has_value()) << ToName(read.error());
  return values;
}

// `count` doubles from a fixed integer formula, with magnitudes from 1e-3 to 1e12 and both signs, so the order of
// the additions changes the rounding of a sum. The same on every run and in every test that compares a CPU and a GPU
// sum.
// Example: the first value is (0 - 1001) / 13 * 1e-3 = -0.077.
std::vector<double> BuildMixedValues(std::size_t count);

// `count` keys with many ties: 1,009 distinct values from a fixed integer formula, each repeated in all four 16-bit
// parts of the key, so every byte of the key varies and the CPU skips no pass of the stable order.
// Example: key 1 is 7919 % 1009 = 856 times 0x0001000100010001.
std::vector<std::uint64_t> BuildKeysWithManyTies(std::size_t count);

// `count` keys that differ only in the top byte: seven values, so the CPU skips seven passes of the stable order.
// Example: key 1 is 31 % 7 = 3, shifted left by 56 bits.
std::vector<std::uint64_t> BuildKeysDifferingInTopByte(std::size_t count);

// The environment variable that names the file WriteResultDigest appends to.
inline constexpr std::string_view kResultDigestFileVariable = "FLUIDITY_RESULT_DIGEST_FILE";

// Computes the 64-bit FNV-1a digest of `bytes`. Equal bytes give equal digests on every machine.
// Example: no bytes give 0xcbf29ce484222325, the FNV-1a starting value.
std::uint64_t ComputeResultDigest(std::span<const std::byte> bytes);

// Appends the line `<name> <digest as 16 hex digits>` to the file at `path`. Fails the calling test when the file
// cannot be written.
void WriteResultDigestToFile(
    const std::filesystem::path& path,
    std::string_view name,
    std::span<const std::byte> bytes
);

// Appends the digest line of `values` to the file named by FLUIDITY_RESULT_DIGEST_FILE; does nothing when that
// variable is unset or empty. `name` names the result, unique among the gpu tests.
// Example: WriteResultDigest("ComputeTreeSum mixed values", std::span<const double>(&sum, 1)).
template <typename Value>
void WriteResultDigest(std::string_view name, std::span<const Value> values);

// The file named by FLUIDITY_RESULT_DIGEST_FILE, or an empty text when the variable is unset.
std::string FindResultDigestFile();

// The environment variable that names the notes file WriteResultNote and WriteResultTiming append to.
inline constexpr std::string_view kResultNotesFileVariable = "FLUIDITY_RESULT_NOTES_FILE";

// Appends `line` and a line end to the file at `path`. Fails the calling test when the file cannot be written.
// Example: ("discrete mismatch, catalogue, case 7 TurningBoxNearFloor: record.face: cpu 1, gpu 5") appends that line.
void WriteResultNoteToFile(const std::filesystem::path& path, std::string_view line);

// Appends `line` to the notes file named by FLUIDITY_RESULT_NOTES_FILE; does nothing when that variable is unset or
// empty. For a finding of a gpu test that does not fail it but that the owner must see and send back.
void WriteResultNote(std::string_view line);

// Appends the line `<name> <value, 4 significant digits> ns per item` to the file at `path`. Fails the calling test
// when the file cannot be written.
// Example: ("MoveParticles on the GPU, 1 solid", 12.345678) appends "MoveParticles on the GPU, 1 solid 12.35 ns per
//   item".
void WriteResultTimingToFile(const std::filesystem::path& path, std::string_view name, double nanoseconds_per_item);

// Appends the timing line of a measured GPU result to the notes file (FLUIDITY_RESULT_NOTES_FILE); does nothing
// when that variable is unset or empty. Timings change from run to run, so they never go to the digest file, whose two
// runs must be equal. `name` names the measurement, unique among the gpu tests.
// Example: WriteResultTiming("MoveParticles on the GPU, 1,000,000 particles, 8 solids", 41.7).
void WriteResultTiming(std::string_view name, double nanoseconds_per_item);

// The value of the environment variable `name`, or an empty text when it is unset.
std::string FindEnvironmentVariable(std::string_view name);

template <typename Value>
void WriteResultDigest(std::string_view name, std::span<const Value> values) {
  const std::string path = FindResultDigestFile();
  if (path.empty()) {
    return;
  }
  WriteResultDigestToFile(path, name, std::as_bytes(values));
}

}  // namespace fluidity
