#pragma once

#include "fluidity/host/simulate_cli.h"

namespace fluidity::simulate {
// Runs a host derived stage from immutable Simulate accepted sources.
// Uses a separate writer lease and cooperative cancellation for publication;
// replay reads/verifies a complete requested prefix without any filesystem writes.
// Emits progress and a terminal result. Errors retain native status and stage
// diagnostic context; committed earlier frames survive failures.
absl::Status ExecuteEffects(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic);
}  // namespace fluidity::simulate
