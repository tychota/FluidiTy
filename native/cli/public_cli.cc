#include "public_cli.h"

#include <args.hxx>
#include <array>
#include <cstdio>
#include <memory>
#include <string_view>
#include <utility>

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
};

constexpr std::array kOptions{
    OptionDefinition{kSimulate | kStep, "cache", "directory", "Simulation cache directory.", true},
    OptionDefinition{kReplay, "scene", "file", "Optional resolved FLRS scene."},
    OptionDefinition{kExecution, "initial-state", "file", "Raw FLST initial state."},
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

class PublicCommand {
 public:
  PublicCommand(args::ArgumentParser& parser, const CommandDefinition& definition)
      : definition_(definition), command_(parser, definition.name, definition.description) {
    if (definition.positional != nullptr) {
      const char* description = std::string_view(definition.positional) == "scene" ? "Resolved FLRS scene file."
                                                                                   : "Simulation cache directory.";
      positional_ = std::make_unique<args::Positional<std::string>>(
          command_,
          definition.positional,
          description,
          args::Options::Required
      );
    }
    for (const auto& option : kOptions) {
      if ((option.commands & definition.mask) == 0) {
        continue;
      }
      std::string flag = std::string("--") + option.name;
      if (option.value_name == nullptr) {
        auto binding = std::make_unique<args::Flag>(
            command_,
            option.name,
            option.description,
            args::Matcher{option.name},
            args::Options::Single
        );
        flags_.emplace_back(std::move(flag), std::move(binding));
      } else {
        auto properties = args::Options::Single;
        if (option.required) {
          properties = properties | args::Options::Required;
        }
        auto binding = std::make_unique<args::ValueFlag<std::string>>(
            command_,
            option.value_name,
            option.description,
            args::Matcher{option.name},
            properties
        );
        values_.emplace_back(std::move(flag), std::move(binding));
      }
    }
  }

  bool Selected() const {
    return command_.Matched();
  }

  std::vector<std::string> LegacyArguments() {
    std::vector<std::string> arguments{definition_.legacy_name};
    if (definition_.mask == kSimulate) {
      arguments.insert(arguments.end(), {"--stage", "simulate"});
    }
    if (positional_) {
      arguments.push_back(std::string("--") + definition_.positional);
      arguments.push_back(positional_->Get());
    }
    for (auto& [flag, binding] : flags_) {
      if (binding->Matched()) {
        arguments.push_back(flag);
      }
    }
    for (auto& [flag, binding] : values_) {
      if (binding->Matched()) {
        arguments.push_back(flag);
        arguments.push_back(binding->Get());
      }
    }
    return arguments;
  }

 private:
  const CommandDefinition& definition_;
  args::Command command_;
  std::unique_ptr<args::Positional<std::string>> positional_;
  std::vector<std::pair<std::string, std::unique_ptr<args::Flag>>> flags_;
  std::vector<std::pair<std::string, std::unique_ptr<args::ValueFlag<std::string>>>> values_;
};

}  // namespace

absl::StatusOr<std::optional<std::vector<std::string>>> ParsePublicCli(std::vector<std::string>& arguments) {
  args::ArgumentParser parser("FluidiTy simulation engine");
  parser.Prog("fluidity");
  args::HelpFlag help(parser, "help", "Show command help.", {'h', "help"}, args::Options::Global);
  std::array<std::unique_ptr<PublicCommand>, kCommands.size()> commands;
  for (std::size_t index = 0; index < kCommands.size(); ++index) {
    commands[index] = std::make_unique<PublicCommand>(parser, kCommands[index]);
  }
  parser.ParseArgs(arguments.begin(), arguments.end());
  // In exception-free mode the parser can report a missing command after matching help.
  if (help.Matched()) {
    const std::string text = parser.Help();
    if (std::fwrite(text.data(), 1, text.size(), stdout) != text.size() || std::fflush(stdout) != 0) {
      return absl::UnavailableError("could not write CLI help");
    }
    return std::nullopt;
  }
  if (parser.GetError() != args::Error::None) {
    return absl::InvalidArgumentError(parser.GetErrorMsg());
  }
  for (const auto& command : commands) {
    if (command->Selected()) {
      return command->LegacyArguments();
    }
  }
  return absl::InvalidArgumentError("a command is required; see fluidity --help");
}

}  // namespace fluidity::simulate
