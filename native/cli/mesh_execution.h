#pragma once

#include "fluidity/host/simulate_cli.h"

namespace fluidity::simulate {
// Runs the host geometric Mesh stage from immutable Simulate accepted sources.
// Uses a separate writer lease and cooperative cancellation for publication;
// replay reads/verifies a complete requested prefix without any filesystem writes.
// Emits progress and a terminal result. Errors retain native status and Mesh
// diagnostic context; committed earlier frames survive failures.
absl::Status ExecuteMesh(const Command& command, ProtocolWriter& writer, Diagnostic* diagnostic);
}  // namespace fluidity::simulate
