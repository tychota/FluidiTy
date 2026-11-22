// Test-only: replaces every global operator new and operator delete of the program and counts each operator new. The
// memory comes from std::malloc; an aligned form asks for more and keeps, just before the bytes it gives, where the
// std::malloc block starts, so its delete frees that block. Each delete matches the new that gave its memory, as the
// language requires.

#include "fluidity/grid/testing/allocation_counter.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace fluidity::grid {
namespace {

// Calls of every replaced operator new. Constant-initialized, so a call made while other files' statics are built
// already finds it; atomic, since a test program may allocate from several threads.
constinit std::atomic<std::uint64_t> allocation_count{0};

// Bytes of std::malloc for one plain form, or null. Size 0 asks for 1 byte, so every call gives a distinct address.
void* BuildCountedBytes(std::size_t byte_count) noexcept {
  allocation_count.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(byte_count == 0 ? 1 : byte_count);
}

// Bytes aligned to `alignment` for one aligned form, or null. The std::malloc block is asked for the bytes, the
// alignment and room for one pointer; the first aligned address after that pointer is given, and the pointer just
// before it holds the block's start. A request so large that this sum would pass SIZE_MAX gives null: the sum would
// wrap to a small block that the caller then writes past.
void* BuildCountedAlignedBytes(std::size_t byte_count, std::align_val_t alignment) noexcept {
  allocation_count.fetch_add(1, std::memory_order_relaxed);
  const auto alignment_bytes = static_cast<std::size_t>(alignment);
  const std::size_t largest_byte_count = std::numeric_limits<std::size_t>::max() - alignment_bytes - sizeof(void*);
  if (byte_count > largest_byte_count) {
    return nullptr;
  }
  char* const block = static_cast<char*>(std::malloc(byte_count + alignment_bytes + sizeof(void*)));
  if (block == nullptr) {
    return nullptr;
  }
  char* const earliest = block + sizeof(void*);
  const std::size_t misalignment = reinterpret_cast<std::uintptr_t>(earliest) % alignment_bytes;
  char* const aligned = earliest + (misalignment == 0 ? 0 : alignment_bytes - misalignment);
  std::memcpy(aligned - sizeof(void*), static_cast<const void*>(&block), sizeof(block));
  return aligned;
}

void FreeCountedBytes(void* memory) noexcept {
  std::free(memory);
}

void FreeCountedAlignedBytes(void* memory) noexcept {
  if (memory == nullptr) {
    return;
  }
  char* block = nullptr;
  std::memcpy(static_cast<void*>(&block), static_cast<char*>(memory) - sizeof(void*), sizeof(block));
  std::free(block);
}

// The throwing forms: memory, or std::bad_alloc when the machine has none.
void* BuildCountedBytesOrThrow(std::size_t byte_count) {
  void* const memory = BuildCountedBytes(byte_count);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  return memory;
}

void* BuildCountedAlignedBytesOrThrow(std::size_t byte_count, std::align_val_t alignment) {
  void* const memory = BuildCountedAlignedBytes(byte_count, alignment);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  return memory;
}

}  // namespace

std::uint64_t ComputeAllocationCount() {
  return allocation_count.load(std::memory_order_relaxed);
}

}  // namespace fluidity::grid

// The replaceable forms of [new.delete.single] and [new.delete.array]. The library names their parameters otherwise.
// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
void* operator new(std::size_t byte_count) {
  return fluidity::grid::BuildCountedBytesOrThrow(byte_count);
}

void* operator new[](std::size_t byte_count) {
  return fluidity::grid::BuildCountedBytesOrThrow(byte_count);
}

void* operator new(std::size_t byte_count, const std::nothrow_t& /*nothrow*/) noexcept {
  return fluidity::grid::BuildCountedBytes(byte_count);
}

void* operator new[](std::size_t byte_count, const std::nothrow_t& /*nothrow*/) noexcept {
  return fluidity::grid::BuildCountedBytes(byte_count);
}

void* operator new(std::size_t byte_count, std::align_val_t alignment) {
  return fluidity::grid::BuildCountedAlignedBytesOrThrow(byte_count, alignment);
}

void* operator new[](std::size_t byte_count, std::align_val_t alignment) {
  return fluidity::grid::BuildCountedAlignedBytesOrThrow(byte_count, alignment);
}

void* operator new(std::size_t byte_count, std::align_val_t alignment, const std::nothrow_t& /*nothrow*/) noexcept {
  return fluidity::grid::BuildCountedAlignedBytes(byte_count, alignment);
}

void* operator new[](std::size_t byte_count, std::align_val_t alignment, const std::nothrow_t& /*nothrow*/) noexcept {
  return fluidity::grid::BuildCountedAlignedBytes(byte_count, alignment);
}

void operator delete(void* memory) noexcept {
  fluidity::grid::FreeCountedBytes(memory);
}

void operator delete[](void* memory) noexcept {
  fluidity::grid::FreeCountedBytes(memory);
}

void operator delete(void* memory, std::size_t /*byte_count*/) noexcept {
  fluidity::grid::FreeCountedBytes(memory);
}

void operator delete[](void* memory, std::size_t /*byte_count*/) noexcept {
  fluidity::grid::FreeCountedBytes(memory);
}

void operator delete(void* memory, const std::nothrow_t& /*nothrow*/) noexcept {
  fluidity::grid::FreeCountedBytes(memory);
}

void operator delete[](void* memory, const std::nothrow_t& /*nothrow*/) noexcept {
  fluidity::grid::FreeCountedBytes(memory);
}

void operator delete(void* memory, std::align_val_t /*alignment*/) noexcept {
  fluidity::grid::FreeCountedAlignedBytes(memory);
}

void operator delete[](void* memory, std::align_val_t /*alignment*/) noexcept {
  fluidity::grid::FreeCountedAlignedBytes(memory);
}

void operator delete(void* memory, std::size_t /*byte_count*/, std::align_val_t /*alignment*/) noexcept {
  fluidity::grid::FreeCountedAlignedBytes(memory);
}

void operator delete[](void* memory, std::size_t /*byte_count*/, std::align_val_t /*alignment*/) noexcept {
  fluidity::grid::FreeCountedAlignedBytes(memory);
}

void operator delete(void* memory, std::align_val_t /*alignment*/, const std::nothrow_t& /*nothrow*/) noexcept {
  fluidity::grid::FreeCountedAlignedBytes(memory);
}

void operator delete[](void* memory, std::align_val_t /*alignment*/, const std::nothrow_t& /*nothrow*/) noexcept {
  fluidity::grid::FreeCountedAlignedBytes(memory);
}
// NOLINTEND(readability-inconsistent-declaration-parameter-name)
