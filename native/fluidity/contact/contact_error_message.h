// The plain-language text and the stable code of each contact error.
//
// Purpose: turn a ContactError or a ContactCloseCall into a sentence for a person and a code for a program.
//
// Idea: inner contact code returns the enum and never builds text. The outer boundary, the command line and the
//   Blender add-on, calls these two functions. The message says what the user can do. The name is the enumerator
//   without the k, in snake case, such as "solid_too_thin"; programs match on it, so a name in use never changes.
//
// Owns:    ToMessage(ContactError), ToMessage(ContactCloseCall), ToName(ContactError), ToName(ContactCloseCall)
// Uses:    CPU-only guard (base/cpu_only.h), ContactError and ContactCloseCall (contact/contact_error.h)
// Used by: the command line and the Blender add-on, through the step's reports
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <string_view>

#include "fluidity/contact/contact_error.h"

namespace fluidity::contact {

// Converts an error to a plain sentence that says what the user can do. Never empty; each error has its own text.
// An out-of-range value gives the text of an unknown error.
std::string_view ToMessage(ContactError error);

// Converts a close call to a plain sentence that says what the user can do. Never empty; each close call has its own
// text. An out-of-range value gives the text of an unknown close call.
std::string_view ToMessage(ContactCloseCall close_call);

// Converts an error to its code: the enumerator without the k, in snake_case, such as "solid_too_thin". The command
// line and the Blender add-on use it, so a code never changes once it is in use. An out-of-range value gives "unknown".
std::string_view ToName(ContactError error);

// Converts a close call to its code, in the same form: "squeezed_between_solids". An out-of-range value gives
// "unknown".
std::string_view ToName(ContactCloseCall close_call);

}  // namespace fluidity::contact
