#pragma once

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace fluidity::simulate {

// Translates the public CLI into arguments for the shared simulation parser.
// Help is written to stdout and returns an empty optional. Syntax errors return
// InvalidArgument, and failure to write help returns Unavailable.
absl::StatusOr<std::optional<std::vector<std::string>>> ParsePublicCli(std::vector<std::string>& arguments);

}  // namespace fluidity::simulate
