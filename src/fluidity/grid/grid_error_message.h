// The plain-language text and the stable code of each grid error.
//
// Purpose: turn a GridError into a sentence for a person and a code for a program.
//
// Idea: inner grid code returns the enum and never builds text. The outer boundary, the command line and the Blender
//   add-on, calls these two functions. The message says what the user can do. The name is the enumerator without the
//   k, in snake case, such as "too_many_cell_blocks"; programs match on it, so a name in use never changes.
//
// Owns:    ToMessage(GridError), ToName(GridError)
// Uses:    CPU-only guard (base/cpu_only.h), GridError (grid/grid_error.h)
// Used by: the command line and the Blender add-on, through the step's reports
// Read:    docs/method-paper/fluidity-method.typ <sec-grid>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <string_view>

#include "fluidity/grid/grid_error.h"

namespace fluidity::grid {

// Converts an error to a plain sentence that says what the user can do. Never empty; each error has its own text.
// An out-of-range value gives the text of an unknown error.
std::string_view ToMessage(GridError error);

// Converts an error to its code: the enumerator without the k, in snake_case, such as "too_many_cell_blocks". The
// command line and the Blender add-on use it, so a code never changes once it is in use. An out-of-range value gives
// "unknown".
std::string_view ToName(GridError error);

}  // namespace fluidity::grid
