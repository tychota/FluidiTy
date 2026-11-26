#include "fluidity/base/run_on_call_error.h"

#include <cstddef>

namespace fluidity {
namespace {

// The last failed call of the calling thread.
RunOnCallError& FindThreadError() {
  static thread_local RunOnCallError thread_error;
  return thread_error;
}

// Copies `characters` into `destination`, cut to kMaxRunOnCallErrorChars characters, and ends it with the end mark.
// Null `characters` gives an empty text.
void ApplyCutCopy(const char* characters, char* destination) {
  std::size_t length = 0;
  if (characters != nullptr) {
    while (length < kMaxRunOnCallErrorChars && characters[length] != '\0') {
      destination[length] = characters[length];
      ++length;
    }
  }
  destination[length] = '\0';
}

}  // namespace

void ApplyLastRunOnCallError(const char* name, const char* text) {
  RunOnCallError& thread_error = FindThreadError();
  thread_error.has_error = true;
  ApplyCutCopy(name, thread_error.name);
  ApplyCutCopy(text, thread_error.text);
}

void ApplyNoRunOnCallError() {
  FindThreadError() = RunOnCallError{};
}

RunOnCallError FindLastRunOnCallError() {
  return FindThreadError();
}

}  // namespace fluidity
