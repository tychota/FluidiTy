#pragma once

#include "fluidity/host/simulate_cli.h"

namespace fluidity::simulate {

// Identifies a resolved gas input or a cache initialized with one.
bool IsGasCommand(const Command& command);

// Executes the fixed-grid gas run, read-only check, or completed-frame replay
// with the same process protocol and atomic publication primitives as liquids.
absl::Status ExecuteGasCommand(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic);

}  // namespace fluidity::simulate
