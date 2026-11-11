#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace fluidity::simulate {

// Translates UTF-8 arguments excluding argv[0] for the shared simulation parser.
// Help is written to stdout and returns an empty optional. Syntax errors return
// InvalidArgument. Failed help writes return Unavailable and parser setup errors return Internal.
absl::StatusOr<std::optional<std::vector<std::string>>> ParsePublicCli(std::span<const std::string> arguments);

}  // namespace fluidity::simulate
