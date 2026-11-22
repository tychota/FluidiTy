// Reading the processor's model out of the text of /proc/cpuinfo, private to base/.
//
// Purpose: keep the one piece of text parsing of processor_description_cpu.cc where a test can call it on any
//   machine, not only on Linux.
//
// Idea: Linux describes each processor in /proc/cpuinfo as lines of "key : value". x86 machines have a
//   "model name" line with the brand string. ARM machines have none. This file only reads text it is given.
//   Callers use base/processor_description.h.
//
// Owns:    FindCpuModelInText
// Uses:    CPU-only guard (base/cpu_only.h)
// Used by: base/processor_description_cpu.cc, which defines it, and its test
// Read:    docs/method-paper/fluidity-method.typ <sec-memory-estimate>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <string>
#include <string_view>

namespace fluidity {

// Finds the value of the first "model name" line in the text of /proc/cpuinfo, without leading or trailing spaces.
//
// Example: "processor : 0\nmodel name\t: AMD Ryzen 9 7950X  \n" gives "AMD Ryzen 9 7950X".
// Fails: never. Returns an empty string when there is no "model name" line, or when its value is empty or only
//        spaces. The caller then uses another name for the processor.
std::string FindCpuModelInText(std::string_view cpuinfo_text);

}  // namespace fluidity
