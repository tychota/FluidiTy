#include "public_cli.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string_view>
#include <utility>

#include "CLI/CLI.hpp"

namespace fluidity::simulate {
namespace {

enum CommandMask : unsigned {
  kSimulate = 1 << 0,
  kValidate = 1 << 1,
  kInspect = 1 << 2,
  kStep = 1 << 3,
  kReplay = 1 << 4,
  kDevices = 1 << 5,
  kSchema = 1 << 6,
};

constexpr unsigned kExecution = kSimulate | kStep | kReplay;

struct CommandDefinition {
  CommandMask mask;
  const char* name;
  const char* description;
  const char* legacy_name;
  const char* positional;
};

constexpr std::array kCommands{
    CommandDefinition{kSimulate, "simulate", "Run a simulation from a resolved FLRS scene.", "run", "scene"},
    CommandDefinition{kValidate, "validate", "Verify a resolved FLRS scene.", "check", "scene"},
    CommandDefinition{kInspect, "inspect", "Inspect a cache or committed frame.", "check", "cache"},
    CommandDefinition{kStep, "step", "Compute one accepted substep without publishing.", "step", "scene"},
    CommandDefinition{kReplay, "replay", "Verify and replay a cached frame and substep.", "replay", "cache"},
    CommandDefinition{kDevices, "devices", "Report CPU support or probe a CUDA device.", "device-info", nullptr},
    CommandDefinition{kSchema, "schema", "Print the resolved scene schema and defaults.", "schema", nullptr},
};

struct OptionDefinition {
  unsigned commands;
  const char* name;
  const char* value_name;
  const char* description;
  bool required = false;
  bool is_path = false;
};

constexpr std::array kOptions{
    OptionDefinition{
        .commands = kSimulate | kStep,
        .name = "cache",
        .value_name = "directory",
        .description = "Simulation cache directory.",
        .required = true,
        .is_path = true,
    },
    OptionDefinition{kReplay, "scene", "file", "Optional resolved FLRS scene.", false, true},
    OptionDefinition{kExecution, "initial-state", "file", "Raw FLST initial state.", false, true},
    OptionDefinition{kExecution | kDevices, "backend", "cpu|cuda", "Select the compute backend."},
    OptionDefinition{kExecution | kDevices, "device", "index", "CUDA device index (requires --backend cuda)."},
    OptionDefinition{kInspect, "frame", "ordinal", "Verify a committed frame."},
    OptionDefinition{kReplay, "frame", "ordinal", "Frame ordinal to replay.", true},
    OptionDefinition{kReplay, "substep", "ordinal", "Substep ordinal to replay.", true},
    OptionDefinition{kSimulate, "frame-start", "ordinal", "First output frame, inclusive (default 0)."},
    OptionDefinition{kSimulate, "frame-end", "ordinal", "Last output frame, inclusive (default 1)."},
    OptionDefinition{kSimulate, "resume", nullptr, "Resume the existing simulation cache."},
    OptionDefinition{kSimulate, "preview", nullptr, "Run without publishing."},
    OptionDefinition{kExecution, "trace", nullptr, "Emit accepted-step and attempt traces."},
    OptionDefinition{kSimulate, "trace-geometry", nullptr, "Include geometry details in traces."},
    OptionDefinition{kExecution, "measure-job", nullptr, "Emit terminal timing and resource measurements."},
};

// CLI11 treats an empty attached value as absent and accepts some explicit flag
// values even when overrides are disabled. Preserve the public token syntax.
absl::Status CheckAttachedValues(std::span<const std::string> option_arguments) {
  for (const std::string_view argument : option_arguments) {
    const auto separator = argument.find('=');
    if (separator == std::string_view::npos) {
      continue;
    }
    const auto option_name = argument.substr(0, separator);
    if (!option_name.starts_with("--")) {
      continue;
    }
    const auto definition = std::ranges::find(kOptions, option_name.substr(2), &OptionDefinition::name);
    if (definition == kOptions.end()) {
      continue;
    }
    if (definition->value_name == nullptr) {
      return absl::InvalidArgumentError(std::string(option_name) + " takes no value");
    }
    if (separator + 1 == argument.size()) {
      return absl::InvalidArgumentError(std::string(option_name) + " requires a nonempty value");
    }
  }
  return absl::OkStatus();
}

class PublicCommand {
 public:
  PublicCommand(CLI::App& parser, const CommandDefinition& definition)
      : definition_(definition), command_(parser.add_subcommand(definition.name, definition.description)) {
    if (definition.positional != nullptr) {
      const char* description = std::string_view(definition.positional) == "scene" ? "Resolved FLRS scene file."
                                                                                   : "Simulation cache directory.";
      positional_ = command_->add_option(definition.positional)->description(description)->required();
    }
    for (const auto& option : kOptions) {
      if ((option.commands & definition.mask) == 0) {
        continue;
      }
      std::string flag = std::string("--") + option.name;
      CLI::Option* binding = nullptr;
      if (option.value_name == nullptr) {
        binding = command_->add_flag(flag, option.description);
        binding->disable_flag_override()->multi_option_policy(CLI::MultiOptionPolicy::Throw);
      } else {
        binding = command_->add_option(flag)->description(option.description)->type_name(option.value_name);
        binding->required(option.required);
      }
      options_.push_back({
          .flag = std::move(flag),
          .option = binding,
          .takes_value = option.value_name != nullptr,
          .is_path = option.is_path,
      });
    }
  }

  bool Selected() const {
    return command_->parsed();
  }

  std::vector<std::string> LegacyArguments(std::span<const std::string> public_arguments) const {
    std::vector<std::string> arguments{definition_.legacy_name};
    if (definition_.mask == kSimulate) {
      arguments.insert(arguments.end(), {"--stage", "simulate"});
    }
    if (positional_ != nullptr) {
      arguments.push_back(std::string("--") + definition_.positional);
      const auto& path = positional_->results().front();
      arguments.push_back(path.starts_with("--") ? "./" + path : path);
    }
    for (const auto& binding : options_) {
      if (binding.option->empty()) {
        continue;
      }
      arguments.push_back(binding.flag);
      if (binding.takes_value) {
        const auto& value = binding.option->results().front();
        // An attached value distinguishes a dash-leading path from a missing option value.
        const bool attached_dash_path = binding.is_path && value.starts_with("--")
            && std::ranges::find(public_arguments, binding.flag + "=" + value) != public_arguments.end();
        arguments.push_back(attached_dash_path ? "./" + value : value);
      }
    }
    return arguments;
  }

 private:
  struct OptionBinding {
    std::string flag;
    const CLI::Option* option;
    bool takes_value;
    bool is_path;
  };

  const CommandDefinition& definition_;
  CLI::App* command_;
  CLI::Option* positional_ = nullptr;
  std::vector<OptionBinding> options_;
};

// CLI11 reports help as a parse result via an exception. Keep stream failures
// visible in the same status path as other process output.
absl::StatusOr<std::optional<std::vector<std::string>>> WriteHelp(const CLI::App& parser) {
  const std::string text = parser.help();
  if (std::fwrite(text.data(), 1, text.size(), stdout) != text.size() || std::fflush(stdout) != 0) {
    return absl::UnavailableError("could not write CLI help");
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<std::optional<std::vector<std::string>>> ParsePublicCli(std::span<const std::string> arguments) try {
  const auto delimiter = std::ranges::find(arguments, "--");
  const std::span option_arguments(arguments.begin(), delimiter);
  const auto attached_status = CheckAttachedValues(option_arguments);
  if (!attached_status.ok()) {
    return attached_status;
  }
  CLI::App parser("FluidiTy simulation engine", "fluidity");
  parser.require_subcommand(1);
  std::vector<PublicCommand> commands;
  commands.reserve(kCommands.size());
  for (const auto& definition : kCommands) {
    commands.emplace_back(parser, definition);
  }
  try {
    // CLI11 consumes this overload's argument vector from the back.
    std::vector<std::string> parse_arguments(arguments.rbegin(), arguments.rend());
    parser.parse(parse_arguments);
  } catch (const CLI::CallForHelp&) {
    if (delimiter != arguments.end() && std::ranges::find(option_arguments, "--help") == option_arguments.end()
        && std::ranges::find(option_arguments, "-h") == option_arguments.end()) {
      return absl::InvalidArgumentError("unexpected argument after --");
    }
    return WriteHelp(parser);
  } catch (const CLI::ParseError& error) {
    return absl::InvalidArgumentError(error.what());
  }
  for (const auto& command : commands) {
    if (command.Selected()) {
      return command.LegacyArguments(arguments);
    }
  }
  return absl::InternalError("CLI11 did not select a command");
} catch (const CLI::Error& error) {
  return absl::InternalError(error.what());
}

}  // namespace fluidity::simulate
