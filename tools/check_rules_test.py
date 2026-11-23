"""Tests for tools/check_rules.py. Each test builds a small tree and checks the findings."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

import check_rules

GLOSSARY_HEADER = "| Say | Plain meaning | Never say |\n| --- | --- | --- |\n"

FULL_HEADER = (
    "// A thing.\n"
    "//\n"
    "// Purpose: do it.\n"
    "// Idea: plain.\n"
    "// Owns:    one thing\n"
    "// Uses:    nothing\n"
    "// Used by: tests\n"
    "// Read:    docs/x\n"
    "\n"
    "#pragma once\n"
)


@pytest.fixture
def glossary(tmp_path: Path) -> Path:
    path = tmp_path / "glossary.md"
    path.write_text(
        "# Glossary\n\n"
        + GLOSSARY_HEADER
        + "| `NearbyFaceWeights` | The faces near a particle. | stencil, component (for this), narrow band |\n"
        + "| `floor` | A side. | floor (for a side: math::Floor keeps the word) |\n"
        + "| `solid` | Cannot enter. | wall |\n"
        + "| `max_...` | A limit. | reserve, lower, epsilon |\n"
        + "| (nothing) | Not used. | quantum |\n"
        + "| `SplitCell` | Split one cell. | refine, bin, qualify, leaf, halo |\n",
        encoding="utf-8",
    )
    return path


def write(root: Path, name: str, text: str, newline: str = "\n") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return path


def run(root: Path, glossary: Path, names: Path | None = None):
    kwargs = {} if names is None else {"allowed_names_path": names}
    return check_rules.check_paths([root], glossary, **kwargs)


def rules(findings, severity=None):
    return [f.rule_id for f in findings if severity is None or f.severity == severity]


def test_header_without_purpose_fails(tmp_path, glossary):
    write(tmp_path, "a.h", "#pragma once\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["file-header"]
    assert findings[0].severity == "FAIL"


def test_header_with_all_labels_passes(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER)
    assert run(tmp_path, glossary) == []


def test_labels_after_first_code_line_do_not_count(tmp_path, glossary):
    write(tmp_path, "a.h", "#pragma once\n" + FULL_HEADER)
    assert rules(run(tmp_path, glossary)) == ["file-header"]


def test_banned_word_in_identifier_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "int stencil_count;\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["banned-word"]
    assert findings[0].severity == "FAIL"
    assert findings[0].line == 1


def test_banned_word_in_comment_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "// the stencil\n")
    assert rules(run(tmp_path, glossary)) == ["banned-word"]


def test_banned_word_in_string_literal_is_ignored(tmp_path, glossary):
    write(tmp_path, "a.cc", 'const char* message = "stencil";\n')
    assert run(tmp_path, glossary) == []


def test_one_sense_word_is_a_note(tmp_path, glossary):
    write(tmp_path, "a.cc", "int component;\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["one-sense-word"]
    assert findings[0].severity == "NOTE"
    assert rules(findings, "FAIL") == []


def test_qualifier_with_colon_is_one_sense(tmp_path, glossary):
    write(tmp_path, "a.cc", "int floor;\n")
    findings = run(tmp_path, glossary)
    assert [(f.severity, f.rule_id) for f in findings] == [("NOTE", "one-sense-word")]


def test_multi_word_entry_matches_consecutive_tokens(tmp_path, glossary):
    write(tmp_path, "a.cc", "int narrow_band_width;\nint narrow_width_band;\n")
    findings = run(tmp_path, glossary)
    assert [(f.line, f.rule_id) for f in findings] == [(1, "banned-word")]


def test_word_inside_longer_token_does_not_match(tmp_path, glossary):
    write(tmp_path, "a.cc", "int quantumness;\nint rewall;\n")
    assert run(tmp_path, glossary) == []


@pytest.mark.parametrize(
    "line",
    [
        "// Two leafs.",
        "// It refines the cell.",
        "// Refining the cell is not done.",
        "// The cell was refined twice.",
        "// Particles are binned.",
        "// Two bins.",
        "// The value qualifies.",
        "// It qualifies.",
        "// Two halos and two haloes.",
        "int RefinedCellCount;",
        "int binning_pass;",
        "// The narrow bands of the liquid.",
    ],
)
def test_inflected_banned_word_fails(tmp_path, glossary, line):
    write(tmp_path, "a.cc", line + "\n")
    assert rules(run(tmp_path, glossary)) == ["banned-word"]


@pytest.mark.parametrize(
    "line",
    [
        "// A refinement of the solver.",
        "// A binary file.",
        "// The leaves of the tree.",
        "int combined;",
        "// The narrows band.",
        "// A qualifier of the type.",
        "// The namespace-qualified name.",
        "int qualified_name;",
    ],
)
def test_word_that_only_looks_like_an_inflection_passes(tmp_path, glossary, line):
    write(tmp_path, "a.cc", line + "\n")
    assert rules(run(tmp_path, glossary)) == []


def test_inflection_of_one_sense_word_is_a_note(tmp_path, glossary):
    write(tmp_path, "a.cc", "// Two floors.\n")
    assert rules(run(tmp_path, glossary)) == ["one-sense-word"]


@pytest.mark.parametrize(
    ("word", "forms"),
    [
        ("refuse", {"refuse", "refuses", "refused", "refusing"}),
        ("qualify", {"qualify", "qualifies", "qualified", "qualifying"}),
        ("make", {"make", "makes", "maked", "making"}),
        ("bin", {"bin", "bins", "bined", "bining", "binned", "binning"}),
        ("attempt", {"attempt", "attempts", "attempted", "attempting"}),
        ("halo", {"halo", "halos", "haloes", "haloed", "haloing"}),
        ("basis", {"basis", "basises", "basised", "basising", "basissed", "basissing"}),
    ],
)
def test_inflected_forms_follow_the_spelling_rules(word, forms):
    assert check_rules.inflected_forms(word) == forms


def test_real_glossary_catches_an_inflected_banned_word(tmp_path):
    write(tmp_path, "a.cc", "// A scene the tolerance cannot protect is refused.\n")
    findings = check_rules.check_paths([tmp_path], check_rules.DEFAULT_GLOSSARY_PATH)
    assert rules(findings) == ["banned-word"]


def test_camel_case_identifier_is_split(tmp_path, glossary):
    write(tmp_path, "a.cc", "int StencilCount;\n")
    assert rules(run(tmp_path, glossary)) == ["banned-word"]


def test_literature_word_line_is_exempt(tmp_path, glossary):
    write(tmp_path, "a.cc", "// Also called stencil, Harlow and Welch 1965.\n// Papers call this a stencil.\n")
    assert run(tmp_path, glossary) == []


def test_abbreviation_in_literature_comment_is_exempt(tmp_path, glossary):
    write(tmp_path, "a.cc", "// Also called MAC grid, Harlow and Welch 1965.\n")
    assert run(tmp_path, glossary) == []


def test_banned_word_elsewhere_in_same_file_still_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "// Also called stencil, Harlow and Welch 1965.\n// The stencil is large.\nint wall;\n")
    findings = run(tmp_path, glossary)
    assert [(f.line, f.rule_id) for f in findings] == [(2, "banned-word"), (3, "banned-word")]


def test_dimension_suffix_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "struct Grid3D {};\n")
    assert rules(run(tmp_path, glossary)) == ["dimension-suffix"]


def test_milestone_code_fails(tmp_path, glossary):
    write(tmp_path, "m5_material.h", "// certified F3 split\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["milestone-code", "file-header", "milestone-code"]
    assert all(f.severity == "FAIL" for f in findings)


def test_processor_name_after_apple_is_not_a_milestone_code(tmp_path, glossary):
    write(tmp_path, "a.cc", 'const char* brand;  // Such as "Apple M3 Pro".\n')
    assert run(tmp_path, glossary) == []


def test_bare_milestone_code_near_apple_text_still_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "// Apple M3 Pro is fine, but M3 is a milestone.\n")
    assert rules(run(tmp_path, glossary)) == ["milestone-code"]


def test_history_word_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "// the qualification run\n")
    assert rules(run(tmp_path, glossary)) == ["milestone-code"]


def test_cubic_metre_unit_in_comment_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.cc", "double volume;  // m3, volume of one cell.\n")
    assert run(tmp_path, glossary) == []


def test_abbreviation_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "double pic_velocity;\n")
    assert rules(run(tmp_path, glossary)) == ["abbreviation"]


def test_abbreviation_with_digit_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "void P2G();\n")
    assert rules(run(tmp_path, glossary)) == ["abbreviation"]


def test_bare_bool_parameter_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void ApplyForces(double value, bool bounded);\n")
    assert rules(run(tmp_path, glossary)) == ["bare-bool"]


def test_bool_field_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "struct Cell {\n  bool has_liquid = false;\n};\n")
    assert run(tmp_path, glossary) == []


def test_unlisted_verb_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "double EvaluateWeight(double value);\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["verb"]
    assert "EvaluateWeight" in findings[0].text


def test_unlisted_verb_says_where_to_list_the_name(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "double EvaluateWeight(double value);\n")
    findings = run(tmp_path, glossary)
    assert "list it with its reason in tools/check_rules_allowed_names.txt" in findings[0].text


def test_allowed_verb_must_end_at_a_word_boundary(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "double Computer(double value);\n")
    assert rules(run(tmp_path, glossary)) == ["verb"]


def test_listed_physical_name_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void MoveParticles(double value);\nvoid TryStep(double value);\n")
    assert run(tmp_path, glossary) == []


def test_math_namespace_noun_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "namespace fluidity::math {\ndouble Dot(Vec3 first, Vec3 second);\n}\n")
    assert run(tmp_path, glossary) == []


def test_member_function_is_not_a_free_function(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "struct Grid {\n  double Weight(double value) const;\n};\n")
    assert run(tmp_path, glossary) == []


def test_inline_definition_in_header_is_not_scanned_for_verbs(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "inline double ComputeTwice(double value) {\n  return Twice(value);\n}\n")
    assert run(tmp_path, glossary) == []


def test_operator_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "Vec3 operator+(Vec3 left, Vec3 right);\n")
    assert run(tmp_path, glossary) == []


def test_verb_rule_does_not_apply_to_cc_files(tmp_path, glossary):
    write(tmp_path, "a.cc", "double EvaluateWeight(double value);\n")
    assert run(tmp_path, glossary) == []


def test_to_conversion_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "std::string_view ToMessage(RunOnError error);\n")
    assert run(tmp_path, glossary) == []


def test_allowed_name_file_exempts(tmp_path, glossary):
    write(tmp_path / "src", "a.h", FULL_HEADER + "double EvaluateWeight(double value);\n")
    names = write(tmp_path, "names.txt", "EvaluateWeight  # owner approved\n")
    assert run(tmp_path / "src", glossary, names) == []


def test_role_suffix_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "struct StepOptions {};\n")
    assert rules(run(tmp_path, glossary)) == ["role-suffix"]


def test_request_role_suffix_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "struct Step" "Request {};\n")
    assert rules(run(tmp_path, glossary)) == ["role-suffix"]


def test_allowed_role_suffixes_pass(tmp_path, glossary):
    write(tmp_path, "a.cc", "struct StepInputs {};\nstruct StepSettings {};\nstruct StepResult {};\n")
    assert run(tmp_path, glossary) == []


def test_backend_word_in_header_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void WriteCudaBuffer();\n")
    assert rules(run(tmp_path, glossary)) == ["backend-name"]


@pytest.mark.parametrize(
    "folder", [("fluidity", "grid", "testing"), ("tests", "grid"), ("fluidity", "grid")]
)
def test_backend_word_is_allowed_in_test_files_only(tmp_path, glossary, folder):
    write(tmp_path.joinpath(*folder), "a.h", FULL_HEADER + "inline void CheckGpuThing();\n")
    expected = ["backend-name"] if folder == ("fluidity", "grid") else []
    assert rules(run(tmp_path, glossary)) == expected


def test_run_on_enumerators_and_cuda_keywords_are_allowed(tmp_path, glossary):
    write(
        tmp_path,
        "a.h",
        FULL_HEADER + "enum class RunOn { kCpu, kGpu };\n#define FLUIDITY_CPU_AND_GPU __host__ __device__\n",
    )
    assert run(tmp_path, glossary) == []


def test_crlf_file_gives_same_findings(tmp_path, glossary):
    write(tmp_path, "a.cc", "int stencil_count;\n", newline="\r\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["banned-word"]
    assert findings[0].line == 1


def test_path_with_space_is_checked(tmp_path, glossary):
    write(tmp_path / "my dir", "a.cc", "int stencil_count;\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["banned-word"]
    assert findings[0].path == str(tmp_path / "my dir" / "a.cc")


@pytest.mark.parametrize("skipped", [".superpowers", ".venv", "out", ".git"])
def test_scan_skips_tool_directories(tmp_path, glossary, skipped):
    write(tmp_path / skipped, "a.cc", "int stencil_count;\n")
    assert run(tmp_path, glossary) == []


def test_python_strings_are_not_scanned_but_comments_and_names_are(tmp_path, glossary):
    write(tmp_path, "a.py", 'text = "stencil"\n# the stencil\nstencil_count = 1\n')
    findings = run(tmp_path, glossary)
    assert [(f.line, f.rule_id) for f in findings] == [(2, "banned-word"), (3, "banned-word")]


def test_other_file_types_are_ignored(tmp_path, glossary):
    write(tmp_path, "a.txt", "stencil\n")
    assert run(tmp_path, glossary) == []


def test_main_exit_code_follows_failures_only(tmp_path, glossary, capsys):
    write(tmp_path, "a.cc", "int component;\n")
    assert check_rules.main([str(tmp_path), "--glossary", str(glossary)]) == 0
    assert "NOTE one-sense-word" in capsys.readouterr().out
    write(tmp_path, "b.cc", "int stencil;\n")
    assert check_rules.main([str(tmp_path), "--glossary", str(glossary)]) == 1
    assert "b.cc:1: FAIL banned-word" in capsys.readouterr().out


def test_allowed_name_without_reason_is_an_error(tmp_path, glossary):
    write(tmp_path / "src", "a.h", FULL_HEADER)
    names = write(tmp_path, "names.txt", "# header comment\nGoodName  # reason\nBadName\nEmptyReason  #   \n")
    with pytest.raises(check_rules.RulesFileError, match=r"names\.txt:3.*BadName"):
        run(tmp_path / "src", glossary, names)


def test_allowed_name_without_reason_exits_with_two(tmp_path, glossary, capsys):
    write(tmp_path / "src", "a.h", FULL_HEADER)
    names = write(tmp_path, "names.txt", "BadName\n")
    arguments = [str(tmp_path / "src"), "--glossary", str(glossary), "--allowed-names", str(names)]
    assert check_rules.main(arguments) == 2
    assert "BadName" in capsys.readouterr().err


def test_glossary_with_wrong_header_is_an_error(tmp_path):
    bad = tmp_path / "glossary.md"
    bad.write_text("| Word | Meaning |\n| --- | --- |\n| a | b |\n", encoding="utf-8")
    write(tmp_path, "a.cc", "int x;\n")
    with pytest.raises(check_rules.GlossaryError):
        check_rules.check_paths([tmp_path], bad)


def test_glossary_with_no_table_is_an_error(tmp_path):
    empty = tmp_path / "glossary.md"
    empty.write_text("# Glossary\n\nNo table here.\n", encoding="utf-8")
    write(tmp_path, "a.cc", "int x;\n")
    with pytest.raises(check_rules.GlossaryError, match="no vocabulary table"):
        check_rules.check_paths([tmp_path], empty)


def test_real_glossary_and_rules_load():
    glossary = check_rules.load_glossary(check_rules.DEFAULT_GLOSSARY_PATH)
    assert any(entry.words == ("narrow", "band") for entry in glossary)
    verbs, names = check_rules.load_allowed_verbs(check_rules.DEFAULT_AGENTS_PATH)
    assert {"Compute", "Build", "Check", "Find", "Apply", "Solve", "Read", "Write", "Copy", "Free", "Take", "To"} <= verbs
    assert "Release" not in verbs
    assert {"TryStep", "ResizeCells", "MixParticleVelocity"} <= names


# ---- Library names, listed names and member definitions ----


@pytest.mark.parametrize(
    "line",
    [
        "values.reserve(8);",
        "pointer->reserve(8);",
        "auto position = std::lower_bound(first, last, value);",
        "auto gap = std::numeric_limits<double>::epsilon();",
    ],
)
def test_library_names_are_not_checked(tmp_path, glossary, line):
    write(tmp_path, "a.cc", line + "\n")
    assert run(tmp_path, glossary) == []


@pytest.mark.parametrize("line", ["int stencil_count;", "double reserve_count;", "std::vector<Stencil> weights;"])
def test_declared_names_are_still_checked(tmp_path, glossary, line):
    write(tmp_path, "a.cc", line + "\n")
    assert rules(run(tmp_path, glossary)) == ["banned-word"]


def test_run_on_enumerator_use_is_allowed_in_any_header(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "struct Settings {\n  RunOn run_on = RunOn::kGpu;\n};\n")
    assert run(tmp_path, glossary) == []


def test_listed_name_is_exempt_from_backend_name(tmp_path, glossary):
    write(tmp_path / "src", "a.h", FULL_HEADER + "bool CheckGpuMath(int* error);\nvoid WriteGpuBuffer();\n")
    names = write(tmp_path, "names.txt", "CheckGpuMath  # private to base/\n")
    findings = run(tmp_path / "src", glossary, names)
    assert rules(findings) == ["backend-name"]
    assert "WriteGpuBuffer" in findings[0].text


def test_listed_enumerator_is_exempt_from_backend_name(tmp_path, glossary):
    write(tmp_path / "src", "a.h", FULL_HEADER + "enum class Error { kNone, kGpuNotBuilt };\n")
    names = write(tmp_path, "names.txt", "kGpuNotBuilt  # decided error value\n")
    assert run(tmp_path / "src", glossary, names) == []


def test_backend_word_is_still_a_failure_next_to_run_on(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void WriteGpuBuffer();\n")
    assert rules(run(tmp_path, glossary)) == ["backend-name"]


def test_out_of_line_member_definition_is_not_a_free_function(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "inline double Grid::CellWidth() const {\n  return 1.0;\n}\n")
    assert run(tmp_path, glossary) == []


def test_out_of_line_constructor_is_not_a_free_function(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "inline Grid::Grid(int count) : count_(count) {}\n")
    assert run(tmp_path, glossary) == []


@pytest.mark.parametrize(
    "declaration",
    [
        "inline double ComputeClamped(double value) {\n  return Clamp(static_cast<bool>(value) ? 1.0 : 2.0);\n}\n",
        "inline void ApplyAll() {\n  auto each = [](bool flag) { return flag; };\n}\n",
        "inline constexpr int kBoolSize = sizeof(bool);\n",
        "using ChooseFunction = std::function<bool(int)>;\n",
        "void CheckAll(std::function<bool(int)> accepts);\n",
    ],
)
def test_bool_that_is_not_a_parameter_is_allowed(tmp_path, glossary, declaration):
    write(tmp_path, "a.h", FULL_HEADER + declaration)
    assert run(tmp_path, glossary) == []


def test_bare_bool_on_third_line_of_declaration_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void ApplyForces(double value,\n    double second,\n    bool bounded);\n")
    assert rules(run(tmp_path, glossary)) == ["bare-bool"]


def test_bare_bool_in_member_declaration_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "class Grid {\n public:\n  void Resize(bool keep);\n};\n")
    assert rules(run(tmp_path, glossary)) == ["bare-bool"]


def test_bare_bool_in_friend_declaration_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "class Grid {\n  friend void ApplyResize(Grid& grid, bool keep);\n};\n")
    assert rules(run(tmp_path, glossary)) == ["bare-bool"]


@pytest.mark.parametrize(
    "parameters",
    ["int count = 1 < 2, bool keep", "std::span<const int> values, bool keep", "int count = (1 > 2), bool keep"],
)
def test_bare_bool_after_angle_brackets_fails(tmp_path, glossary, parameters):
    write(tmp_path, "a.h", FULL_HEADER + f"void ApplyResize({parameters});\n")
    assert rules(run(tmp_path, glossary)) == ["bare-bool"]


def test_continued_define_declares_nothing(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "#define FLUIDITY_DECLARE_RESIZE \\\n  void Resize(bool keep);\nint kept;\n")
    assert run(tmp_path, glossary) == []


@pytest.mark.parametrize(
    "declaration",
    ["struct [[nodiscard]] [[maybe_unused]] StepOptions {};", "struct alignas(8) StepOptions {};"],
)
def test_role_suffix_after_attributes_fails(tmp_path, glossary, declaration):
    write(tmp_path, "a.cc", declaration + "\n")
    assert rules(run(tmp_path, glossary)) == ["role-suffix"]


@pytest.mark.parametrize("name", ["positions_3d", "cells_2d", "Grid3DView", "Vec3D"])
def test_dimension_suffix_in_any_position_fails(tmp_path, glossary, name):
    write(tmp_path, "a.cc", f"int {name};\n")
    assert rules(run(tmp_path, glossary)) == ["dimension-suffix"]


@pytest.mark.parametrize("name", ["Vec3", "Mat3", "kLevel3", "d3", "x_axis"])
def test_names_with_a_digit_but_no_dimension_are_allowed(tmp_path, glossary, name):
    write(tmp_path, "a.cc", f"int {name};\n")
    assert run(tmp_path, glossary) == []


def test_math_dot_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "namespace fluidity::math {\ndouble Dot(Vec3 first, Vec3 second);\n}\n")
    assert run(tmp_path, glossary) == []


def test_python_attribute_assignment_target_is_checked(tmp_path, glossary):
    write(tmp_path, "a.py", "self.stencil_count = 1\nprint(self.stencil_count)\n")
    assert [(f.line, f.rule_id) for f in run(tmp_path, glossary)] == [(1, "banned-word")]


def test_python_def_default_parameter_is_checked(tmp_path, glossary):
    write(tmp_path, "a.py", "def build(stencil=3):\n    return build(stencil=4)\n")
    assert [(f.line, f.rule_id) for f in run(tmp_path, glossary)] == [(1, "banned-word")]


def test_python_docstring_is_scanned_as_a_comment(tmp_path, glossary):
    text = 'def build():\n    """First line.\n\n    The stencil is large.\n    """\n    return "stencil"\n'
    write(tmp_path, "a.py", text)
    assert [(f.line, f.rule_id) for f in run(tmp_path, glossary)] == [(4, "banned-word")]


def test_hex_literals_are_not_names(tmp_path, glossary):
    write(tmp_path, "a.cc", "constexpr unsigned kMask = 0xF3;\nconstexpr unsigned kOther = 0x3D;\n")
    assert run(tmp_path, glossary) == []


def test_digit_separator_does_not_hide_the_rest_of_the_line(tmp_path, glossary):
    write(tmp_path, "a.cc", "int big = 1'000; int stencil_count;\n")
    assert rules(run(tmp_path, glossary)) == ["banned-word"]


def test_default_scan_path_is_relative_to_the_repository(tmp_path, glossary, monkeypatch, capsys):
    write(tmp_path / "src", "a.cc", "int stencil_count;\n")
    write(tmp_path / "tests", "b.cc", "int fine;\n")
    monkeypatch.setattr(check_rules, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path / "src")
    assert check_rules.main(["--glossary", str(glossary)]) == 1
    assert "FAIL banned-word" in capsys.readouterr().out


def test_nothing_to_scan_is_an_error(tmp_path, glossary, capsys):
    write(tmp_path, "notes.txt", "hello\n")
    assert check_rules.main([str(tmp_path), "--glossary", str(glossary)]) == 2
    assert "nothing was scanned" in capsys.readouterr().err


def test_shared_function_taking_vec3_by_reference_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "FLUIDITY_CPU_AND_GPU double ComputeSize(const Vec3& vector) noexcept;\n")
    assert rules(run(tmp_path, glossary), "FAIL") == ["shared-by-reference"]


def test_shared_function_taking_mat3_by_reference_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "FLUIDITY_CPU_AND_GPU double ComputeSize(const Mat3 &matrix) noexcept;\n")
    assert rules(run(tmp_path, glossary), "FAIL") == ["shared-by-reference"]


@pytest.mark.parametrize(
    "parameter",
    [
        "Vec3 const& vector",
        "Mat3 const &matrix",
        "const fluidity::Vec3& vector",
        "const ::fluidity::Mat3 &matrix",
        "fluidity::Vec3 const& vector",
    ],
)
def test_shared_function_taking_any_spelling_of_a_const_reference_fails(tmp_path, glossary, parameter):
    write(tmp_path, "a.h", FULL_HEADER + f"FLUIDITY_CPU_AND_GPU double ComputeSize({parameter}) noexcept;\n")
    assert rules(run(tmp_path, glossary), "FAIL") == ["shared-by-reference"]


def test_shared_function_taking_a_qualified_vec3_by_value_passes(tmp_path, glossary):
    declaration = "FLUIDITY_CPU_AND_GPU double ComputeSize(const fluidity::Vec3 vector) noexcept;\n"
    write(tmp_path, "a.h", FULL_HEADER + declaration)
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_shared_function_taking_vec3_by_value_passes(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "FLUIDITY_CPU_AND_GPU double ComputeSize(Vec3 vector) noexcept;\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_reference_in_file_without_shared_functions_passes(tmp_path, glossary):
    write(tmp_path, "a.test.cc", "void CheckNear(const Mat3& actual, const Vec3& expected) {}\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_has_gpu_build_macro_is_allowed_in_a_header(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "#if defined(FLUIDITY_HAS_GPU)\nvoid ComputeThing();\n#endif\n")
    assert run(tmp_path, glossary) == []


def test_copy_is_an_allowed_verb(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void CopyBuffer(int from, int to);\n")
    assert run(tmp_path, glossary) == []


def test_private_gpu_header_may_name_the_gpu_and_use_any_verb(tmp_path, glossary):
    write(tmp_path / "base", "buffer_gpu.h", FULL_HEADER + "bool AllocateGpuBytes(int byte_count);\n")
    assert run(tmp_path, glossary) == []


def test_private_gpu_source_may_name_the_gpu(tmp_path, glossary):
    write(tmp_path / "base", "buffer_gpu.cu", "bool AllocateGpuBytes(int byte_count) {\n  return true;\n}\n")
    assert run(tmp_path, glossary) == []


def test_header_whose_stem_only_contains_gpu_is_still_checked(tmp_path, glossary):
    write(tmp_path / "base", "run_gpu_thing.h", FULL_HEADER + "bool AllocateGpuBytes(int byte_count);\n")
    assert sorted(rules(run(tmp_path, glossary))) == ["backend-name", "verb"]


def test_private_gpu_header_still_gets_the_other_rules(tmp_path, glossary):
    write(tmp_path / "base", "buffer_gpu.h", FULL_HEADER + "bool AllocateGpuBytes(bool zero_filled);\n")
    assert rules(run(tmp_path, glossary)) == ["bare-bool"]


def test_private_gpu_header_included_from_same_folder_source_passes(tmp_path, glossary):
    write(tmp_path / "fluidity" / "base", "buffer_cpu.cc", '#include "fluidity/base/buffer_gpu.h"\n')
    assert run(tmp_path, glossary) == []


def test_private_gpu_header_included_from_another_folder_fails(tmp_path, glossary):
    write(tmp_path / "fluidity" / "grid", "grid_cpu.cc", '#include "fluidity/base/buffer_gpu.h"\n')
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["private-gpu-include"]
    assert "buffer_gpu.h" in findings[0].text


def test_private_gpu_header_included_from_a_public_header_fails(tmp_path, glossary):
    write(tmp_path / "fluidity" / "base", "buffer.h", FULL_HEADER + '#include "fluidity/base/buffer_gpu.h"\n')
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_private_gpu_header_included_with_angle_brackets_from_another_folder_fails(tmp_path, glossary):
    write(tmp_path / "fluidity" / "grid", "grid_cpu.cc", "#include <fluidity/base/buffer_gpu.h>\n")
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_private_gpu_cuh_header_included_from_another_folder_fails(tmp_path, glossary):
    write(tmp_path / "fluidity" / "grid", "grid_gpu.cu", '#include "fluidity/base/buffer_gpu.cuh"\n')
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_test_may_include_the_private_gpu_header_of_the_testing_folder_beside_it(tmp_path, glossary):
    write(tmp_path / "fluidity" / "grid", "cell_size_rule.gpu.test.cc", '#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"\n')
    write(tmp_path / "fluidity" / "grid" / "testing", "cell_size_rule_runner_gpu.cu", '#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"\n')
    assert run(tmp_path, glossary) == []


def test_file_that_is_not_a_test_may_not_include_a_private_gpu_header_of_the_testing_folder(tmp_path, glossary):
    write(tmp_path / "fluidity" / "grid", "grid_cpu.cc", '#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"\n')
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_test_may_not_include_the_private_gpu_header_of_the_testing_folder_of_another_module(tmp_path, glossary):
    write(tmp_path / "fluidity" / "contact", "move_particles.gpu.test.cc", '#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"\n')
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_test_of_another_folder_may_not_include_a_private_gpu_header_outside_testing(tmp_path, glossary):
    write(tmp_path / "fluidity" / "grid", "grid_tables.gpu.test.cc", '#include "fluidity/grid/other_gpu.h"\n')
    write(tmp_path / "fluidity" / "contact", "move_particles.gpu.test.cc", '#include "fluidity/grid/other_gpu.h"\n')
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["private-gpu-include"]


def test_library_header_in_angle_brackets_is_not_a_private_gpu_header(tmp_path, glossary):
    write(tmp_path / "fluidity" / "base", "buffer_gpu.cu", "#include <cuda_runtime.h>\n")
    assert run(tmp_path, glossary) == []


def test_private_gpu_header_included_from_another_private_gpu_header_passes(tmp_path, glossary):
    write(tmp_path / "fluidity" / "base", "other_gpu.h", FULL_HEADER + '#include "fluidity/base/buffer_gpu.h"\n')
    assert run(tmp_path, glossary) == []


def test_free_is_an_allowed_verb(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void FreeBufferBytes(int bytes);\n")
    assert run(tmp_path, glossary) == []


def test_take_is_an_allowed_verb(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "int TakeBuffers(int grid);\n")
    assert run(tmp_path, glossary) == []


def test_release_is_not_an_allowed_verb(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void Release" "Bytes(int bytes);\n")
    assert rules(run(tmp_path, glossary)) == ["verb"]


def test_shared_cpu_and_gpu_header_is_not_a_private_gpu_header(tmp_path, glossary):
    write(tmp_path / "fluidity" / "base", "vec3.h", FULL_HEADER + '#include "fluidity/base/cpu_and_gpu.h"\n')
    write(tmp_path / "fluidity" / "base", "cpu_and_gpu.h", FULL_HEADER + "void WriteGpuThing();\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["backend-name"]
    assert "WriteGpuThing" in findings[0].text


# ---- Notes only on the lines a commit adds ----

STAGED_DIFF = (
    "diff --git a/src/a.cc b/src/a.cc\n"
    "--- a/src/a.cc\n"
    "+++ b/src/a.cc\n"
    "@@ -2,0 +3,2 @@ int x;\n"
    "+int component;\n"
    "+int y;\n"
    "@@ -9 +11 @@\n"
    "-int old;\n"
    "+int z;\n"
    "diff --git a/src/gone.cc b/src/gone.cc\n"
    "--- a/src/gone.cc\n"
    "+++ /dev/null\n"
    "@@ -1 +0,0 @@\n"
    "-int component;\n"
)


def test_added_lines_are_read_from_a_zero_context_diff():
    assert check_rules.parse_added_lines(STAGED_DIFF) == {"src/a.cc": {3, 4, 11}}


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def test_notes_on_staged_lines_keeps_notes_of_added_lines_only(tmp_path, glossary, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    write(repo, "a.cc", "int component;\n")
    git(repo, "add", "a.cc")
    git(repo, "commit", "-q", "-m", "first")
    write(repo, "a.cc", "int component;\nint component_count;\nint stencil;\n")
    git(repo, "add", "a.cc")
    arguments = [str(repo / "a.cc"), "--glossary", str(glossary), "--notes-on-staged-lines", "--repository", str(repo)]
    assert check_rules.main(arguments) == 1
    output = capsys.readouterr().out
    assert "a.cc:1: NOTE" not in output
    assert "a.cc:2: NOTE one-sense-word" in output
    assert "a.cc:3: FAIL banned-word" in output


def test_words_check_reports_the_same_findings_in_the_same_order_as_trying_every_entry():
    """_check_words tries only the entries whose last word is in the line; the findings must be those of a loop over
    every entry, in glossary order."""
    entries = [
        check_rules.NeverSayEntry(("old", "stencil"), "old stencil", None),
        check_rules.NeverSayEntry(("stencil",), "stencil", "for a pattern"),
        check_rules.NeverSayEntry(("wall",), "wall", None),
        check_rules.NeverSayEntry(("refine",), "refine", None),
    ]
    text = "// An old stencil near the walls, and a stencil that refined.\nint wall_stencil = 0;\n"
    lines = check_rules.split_lines(Path("x.h"), text)

    fast = check_rules._Reporter(Path("x.h"))
    check_rules._check_words(lines, entries, fast)

    slow = check_rules._Reporter(Path("x.h"))
    for line in lines:
        sequences = [(check_rules.split_name_words(name), "name", False) for name in check_rules._identifiers(line.code)]
        sequences += [(segment, "comment", False) for segment in check_rules.comment_segments(line.comment)]
        joined_comment = check_rules._HYPHEN_BETWEEN_WORDS.sub(" ", line.comment)
        sequences += [(segment, "comment", True) for segment in check_rules.comment_segments(joined_comment)]
        for tokens, where, is_hyphen_joined in sequences:
            for entry in entries:
                if where == "comment" and is_hyphen_joined == (entry.qualifier is None):
                    continue
                if not check_rules.contains_phrase(tokens, entry.words):
                    continue
                if entry.qualifier is None:
                    text_of_finding = f"'{entry.spelled}' is banned (docs/glossary.md); found in a {where}"
                    slow.add(line.number, check_rules.FAIL, "banned-word", text_of_finding, key=entry.spelled)
                else:
                    text_of_finding = f"'{entry.spelled}' is banned in one sense only: {entry.qualifier}"
                    slow.add(line.number, check_rules.NOTE, "one-sense-word", text_of_finding, key=entry.spelled)
    assert fast.findings() == slow.findings()
    assert len(fast.findings()) >= 4


def test_findings_of_a_line_follow_the_glossary_order_even_when_a_set_would_not():
    """The candidate positions go through a set. Positions 8 (eight) and 1 (one) sit in a small set in the order 8, 1 (8 shares the
    first slot of an 8-slot table), so only the sort in _check_words puts the findings in glossary order."""
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    entries = [check_rules.NeverSayEntry((word,), word, None) for word in words]
    lines = check_rules.split_lines(Path("x.h"), "// eight and one\n")
    reporter = check_rules._Reporter(Path("x.h"))
    check_rules._check_words(lines, entries, reporter)
    assert [finding.text for finding in reporter.findings()] == [
        "'one' is banned (docs/glossary.md); found in a comment",
        "'eight' is banned (docs/glossary.md); found in a comment",
    ]


@pytest.mark.parametrize(
    "macro", ["__clang__", "__GNUC__", "_MSC_VER", "__apple_build_version__", "__clang_analyzer__"]
)
def test_include_under_a_condition_on_a_compiler_macro_fails(tmp_path, glossary, macro):
    write(tmp_path, "a.cc", f'#if defined({macro})\n#include "fluidity/base/x.h"\n#endif\n')
    assert rules(run(tmp_path, glossary)) == ["compiler-include"]


def test_include_under_a_has_include_condition_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", "#if __has_include(<span>)\n#include <span>\n#endif\n")
    assert rules(run(tmp_path, glossary)) == ["compiler-include"]


def test_include_in_the_else_or_elif_branch_of_a_compiler_condition_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", '#ifdef __clang__\n#else\n#include "x.h"\n#endif\n')
    write(tmp_path, "b.cc", '#if FOO\n#elif defined(_MSC_VER) && 1\n#include "x.h"\n#endif\n')
    assert rules(run(tmp_path, glossary)) == ["compiler-include", "compiler-include"]


def test_include_in_a_block_nested_in_a_compiler_condition_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", '#if __GNUC__\n#if FOO\n#include "x.h"\n#endif\n#endif\n')
    assert rules(run(tmp_path, glossary)) == ["compiler-include"]


def test_include_after_a_compiler_condition_ends_or_under_another_condition_passes(tmp_path, glossary):
    write(
        tmp_path,
        "a.cc",
        '#if defined(__clang__)\nint f();\n#endif\n#include "x.h"\n#if FOO\n#include "y.h"\n#endif\n',
    )
    assert run(tmp_path, glossary) == []


def test_test_may_not_include_a_private_gpu_header_of_the_testing_folder_that_is_not_a_runner(tmp_path, glossary):
    write(tmp_path / "fluidity" / "grid", "cell_size_rule.gpu.test.cc", '#include "fluidity/grid/testing/other_gpu.h"\n')
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_header_with_test_in_its_name_may_not_include_the_runner_header_of_the_testing_folder(tmp_path, glossary):
    write(tmp_path / "fluidity" / "grid", "cell_size_rule.test.h", '#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"\n')
    assert "private-gpu-include" in rules(run(tmp_path, glossary))


def test_test_under_tests_may_include_the_private_gpu_header_of_its_module(tmp_path, glossary):
    write(tmp_path / "tests" / "grid", "build_grid_on_gpu.test.cc", '#include "fluidity/grid/staggered_grid_gpu.h"\n')
    write(tmp_path / "tests" / "grid", "cell_size_rule_on_gpu.test.cc", '#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"\n')
    assert run(tmp_path, glossary) == []


def test_test_under_tests_may_not_include_the_private_gpu_header_of_another_module(tmp_path, glossary):
    write(tmp_path / "tests" / "contact", "move_particles_on_gpu.test.cc", '#include "fluidity/grid/staggered_grid_gpu.h"\n')
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_file_under_tests_that_is_not_a_test_may_not_include_a_private_gpu_header(tmp_path, glossary):
    write(tmp_path / "tests" / "grid", "helper.cc", '#include "fluidity/grid/staggered_grid_gpu.h"\n')
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_test_under_tests_may_not_include_a_private_gpu_header_of_the_testing_folder_that_is_not_a_runner(tmp_path, glossary):
    write(tmp_path / "tests" / "grid", "x_on_gpu.test.cc", '#include "fluidity/grid/testing/other_gpu.h"\n')
    assert rules(run(tmp_path, glossary)) == ["private-gpu-include"]


def test_default_scan_covers_tests_as_it_covers_src(tmp_path, glossary, monkeypatch, capsys):
    write(tmp_path / "src", "a.cc", "int fine;\n")
    write(tmp_path / "tests", "b.cc", "int stencil_count;\n")
    monkeypatch.setattr(check_rules, "REPO_ROOT", tmp_path)
    assert check_rules.main(["--glossary", str(glossary)]) == 1
    output = capsys.readouterr().out
    assert "FAIL banned-word" in output and "b.cc" in output


# one-word-comment ----------------------------------------------------------------------------------


@pytest.mark.parametrize("comment", ["// Done", "// ok.", "  // Later;"])
def test_one_word_comment_fails(tmp_path, glossary, comment):
    write(tmp_path / "src", "a.cc", f"int fine;\n{comment}\nint other;\n")
    assert rules(run(tmp_path, glossary), "FAIL") == ["one-word-comment"]


@pytest.mark.parametrize(
    "text",
    [
        "// Two words\n",
        "// First line of\n// Done\n",  # A line beside another comment line continues it.
        "// NOLINT\n",
        "// NOLINTNEXTLINE(readability-identifier-length)\n",
        "}  // namespace\n",
        "int fine;  // Done\n",
        "//\n",
    ],
)
def test_comment_that_is_not_one_word_on_its_own_passes(tmp_path, glossary, text):
    write(tmp_path / "src", "a.cc", text)
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_one_word_comment_outside_src_and_tests_is_not_checked(tmp_path, glossary):
    write(tmp_path / "other", "a.cc", "// Done\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


# test-helper-verb ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.test.cc", "testing/support.cc", "../tests/grid/a.cc"])
def test_test_helper_without_a_verb_fails(tmp_path, glossary, name):
    write(tmp_path / "src", name, "void ExpectEqual(int a, int b) {}\n")
    assert rules(run(tmp_path, glossary), "FAIL") == ["test-helper-verb"]


def test_test_helper_with_an_allowed_verb_passes(tmp_path, glossary):
    write(tmp_path / "src", "a.test.cc", "void CheckEqual(int a, int b) {}\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_test_macros_main_and_hooks_are_not_helpers(tmp_path, glossary):
    text = (
        "TEST(Suite, Name) {}\nTEST_F(Fixture, Name) {}\nINSTANTIATE_TEST_SUITE_P(All, Suite, Values(1));\n"
        "int main(int argc, char** argv) { return 0; }\nvoid PrintTo(const Thing& thing, std::ostream* out) {}\n"
        "bool operator==(const Thing& a, const Thing& b) { return true; }\n"
    )
    write(tmp_path / "src", "a.test.cc", text)
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_function_without_a_verb_in_a_non_test_source_file_is_not_a_test_helper(tmp_path, glossary):
    write(tmp_path / "src", "a.cc", "void ExpectEqual(int a, int b) {}\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_test_helper_listed_in_the_allowed_names_passes(tmp_path, glossary):
    names = tmp_path / "names.txt"
    names.write_text("ExpectEqual  # a reason\n", encoding="utf-8")
    write(tmp_path / "src", "a.test.cc", "void ExpectEqual(int a, int b) {}\n")
    assert rules(run(tmp_path, glossary, names), "FAIL") == []


# paper-line-length ---------------------------------------------------------------------------------


def test_paper_line_over_the_limit_fails(tmp_path, glossary):
    write(tmp_path / "docs" / "method-paper", "a.typ", "word " * 25 + "\n")
    assert rules(run(tmp_path, glossary), "FAIL") == ["paper-line-length"]


def test_paper_line_at_the_limit_passes(tmp_path, glossary):
    write(tmp_path / "docs" / "method-paper", "a.typ", "x" * 120 + "\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_paper_line_counts_columns_not_bytes(tmp_path, glossary):
    write(tmp_path / "docs" / "method-paper", "a.typ", "é" * 120 + "\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_listed_unbreakable_paper_line_passes(tmp_path, glossary):
    marker = check_rules.PAPER_LONG_LINE_EXEMPTIONS[0][1]
    name = check_rules.PAPER_LONG_LINE_EXEMPTIONS[0][0]
    write(tmp_path / "docs" / "method-paper", name, f'#link("https://example.org/{marker}")[A' + "x" * 120 + "\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_banned_word_in_a_paper_file_is_not_this_tools_business(tmp_path, glossary):
    write(tmp_path / "docs" / "method-paper", "a.typ", "a stencil\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


# The rule table --------------------------------------------------------------------------------------


def test_every_rule_the_tool_reports_is_in_the_rule_table():
    source = Path(check_rules.__file__).read_text(encoding="utf-8")
    reported = set(re.findall(r'reporter\.add\([^"]*?(?:FAIL|NOTE), "([a-z-]+)"', source))
    reported |= {"test-helper-verb"}
    assert reported <= {rule.rule_id for rule in check_rules.RULE_TABLE}


def test_rule_table_ids_are_unique_and_levels_are_known():
    ids = [rule.rule_id for rule in check_rules.RULE_TABLE]
    assert len(ids) == len(set(ids))
    assert {rule.level for rule in check_rules.RULE_TABLE} <= {check_rules.FAIL, check_rules.NOTE}


# NOTE rules --------------------------------------------------------------------------------------------


def notes(tmp_path, glossary, name, text, folder="src"):
    write(tmp_path / folder, name, text)
    return rules(run(tmp_path, glossary), "NOTE")


def test_and_in_a_test_name_is_a_note(tmp_path, glossary):
    assert notes(tmp_path, glossary, "a.test.cc", "TEST(Grid, BuildsAndChecks_Works) { EXPECT_TRUE(true); }\n") == [
        "test-name-and"
    ]


@pytest.mark.parametrize("name", ["BlockAndroid_Works", "Android_Works", "Standard_Works"])
def test_and_inside_a_longer_word_of_a_test_name_is_quiet(tmp_path, glossary, name):
    assert notes(tmp_path, glossary, "a.test.cc", f"TEST(Grid, {name}) {{ EXPECT_TRUE(true); }}\n") == []


def test_function_with_more_than_four_inputs_in_a_test_file_is_a_note(tmp_path, glossary):
    text = "void CheckAll(int first, int second, int third, int fourth, int fifth) {}\n"
    assert notes(tmp_path, glossary, "a.test.cc", text) == ["many-inputs"]


def test_function_with_four_inputs_in_a_test_file_is_quiet(tmp_path, glossary):
    assert notes(tmp_path, glossary, "a.test.cc", "void CheckAll(int first, int second, int third, int fourth) {}\n") == []


def test_private_helper_with_many_inputs_is_a_note_and_a_public_one_is_quiet(tmp_path, glossary):
    text = "int ComputeAll(int first, int second, int third, int fourth, int fifth);\n"
    definition = text.replace(";", " { return 0; }")
    assert notes(tmp_path / "public", glossary, "a.h", text) == []
    assert notes(tmp_path / "anonymous", glossary, "b.cc", "namespace {\n" + definition + "}\n") == ["many-inputs"]
    assert notes(tmp_path / "static", glossary, "c.cc", "static " + definition) == ["many-inputs"]
    assert notes(tmp_path / "plain", glossary, "d.cc", definition) == []


@pytest.mark.parametrize(
    "body",
    ["  const double x = 1.0;\n", "  for (int i = 0; i < 3; ++i) {\n  }\n", "  auto& v = values;\n", "  std::size_t k{3};\n"],
)
def test_one_letter_local_is_a_note(tmp_path, glossary, body):
    assert notes(tmp_path, glossary, "a.cc", "void ApplyIt() {\n" + body + "}\n") == ["short-local"]


def test_one_letter_parameter_is_a_note(tmp_path, glossary):
    assert notes(tmp_path, glossary, "a.cc", "double ComputeIt(double x) { return 0.0; }\n") == ["short-local"]


@pytest.mark.parametrize(
    "text",
    [
        "void ApplyIt() {\n  const double width = 1.0;\n  value.x = 2.0;\n  const Vec3 point = {.x = 1, .y = 2};\n}\n",
        "template <typename T>\nT ComputeIt(T value) { return value; }\n",
        "struct Vec2 {\n  double x;\n  double y;\n};\n",
        "double ComputeIt(double) { return 0.0; }\n",
    ],
)
def test_names_that_are_not_one_letter_locals_are_quiet(tmp_path, glossary, text):
    assert notes(tmp_path, glossary, "a.cc", text) == []


@pytest.mark.parametrize("word", ["Brief", "review", "reviewer", "plan", "ruling"])
def test_process_word_in_a_comment_is_a_note(tmp_path, glossary, word):
    assert notes(tmp_path, glossary, "a.cc", f"// As the {word} said.\nint fine;\n") == ["process-word"]


@pytest.mark.parametrize(
    "text", ["// Task 12 decided it.\n", "// Ruling 5 holds.\n"]
)
def test_numbered_process_word_in_a_comment_is_a_note(tmp_path, glossary, text):
    assert notes(tmp_path, glossary, "a.cc", text) == ["process-word"]


@pytest.mark.parametrize("text", ["// The round of the rounding.\n", "// A VS Code task builds it.\n", "// The task list.\n"])
def test_round_and_an_unnumbered_task_are_ordinary_words(tmp_path, glossary, text):
    assert notes(tmp_path, glossary, "a.cc", text) == []


def test_process_word_in_code_or_inside_a_longer_word_is_quiet(tmp_path, glossary):
    assert notes(tmp_path, glossary, "a.cc", "int task_count;\n// Rounding and planes and reviews.\n") == []


def test_test_without_a_check_is_a_note(tmp_path, glossary):
    text = "TEST(Grid, Builds) {\n  Build();\n}\n"
    assert notes(tmp_path, glossary, "a.test.cc", text) == ["test-without-check"]


@pytest.mark.parametrize(
    "check", ["EXPECT_EQ(a, b);", "ASSERT_TRUE(ok);", "CheckGrid(grid);", "SUCCEED();", "FAIL();", "GTEST_SKIP();"]
)
def test_test_with_a_check_is_quiet(tmp_path, glossary, check):
    text = f"TEST(Grid, Builds) {{\n  Build();\n  {check}\n}}\n"
    assert notes(tmp_path, glossary, "a.test.cc", text) == []


def test_two_tests_are_judged_one_by_one(tmp_path, glossary):
    text = "TEST(Grid, A) {\n  EXPECT_TRUE(true);\n}\n\nTEST(Grid, B) {\n  Build();\n}\n"
    write(tmp_path / "src", "a.test.cc", text)
    findings = [f for f in run(tmp_path, glossary) if f.rule_id == "test-without-check"]
    assert [f.line for f in findings] == [5]


# --notes-since and --json -------------------------------------------------------------------------------


def git(repository: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repository, check=True, stdout=subprocess.DEVNULL)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "a@b.c")
    git(tmp_path, "config", "user.name", "A")
    git(tmp_path, "config", "commit.gpgsign", "false")
    write(tmp_path / "src", "a.cc", "// The plan.\nint fine;\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "First")
    return tmp_path


def test_notes_since_writes_the_notes_of_added_lines_sorted_with_their_source(repository, glossary, tmp_path_factory):
    write(repository / "src", "a.cc", "// The plan.\nint fine;\n// The ruling.\n")
    write(repository / "src", "b.cc", "// A review.\n")
    destination = tmp_path_factory.mktemp("out") / "notes.json"
    exit_code = check_rules.main(
        [str(repository / "src"), "--glossary", str(glossary), "--repository", str(repository)]
        + ["--notes-since", "HEAD", "--json", str(destination)]
    )
    records = json.loads(destination.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert [(r["path"], r["line"], r["rule"], r["source_line"]) for r in records] == [
        ("src/a.cc", 3, "process-word", "// The ruling."),
        ("src/b.cc", 1, "process-word", "// A review."),
    ]
    assert set(records[0]) == {"path", "line", "rule", "text", "source_line"}


def test_notes_since_prints_the_count(repository, glossary, tmp_path_factory, capsys):
    write(repository / "src", "b.cc", "// A review.\n")
    destination = tmp_path_factory.mktemp("out") / "notes.json"
    check_rules.main(
        [str(repository / "src"), "--glossary", str(glossary), "--repository", str(repository)]
        + ["--notes-since", "HEAD", "--json", str(destination)]
    )
    assert capsys.readouterr().out.startswith("1 doubtful lines")


def test_notes_since_a_commit_that_does_not_exist_is_an_error(repository, glossary, tmp_path_factory, capsys):
    destination = tmp_path_factory.mktemp("out") / "notes.json"
    exit_code = check_rules.main(
        [str(repository / "src"), "--glossary", str(glossary), "--repository", str(repository)]
        + ["--notes-since", "no-such-commit", "--json", str(destination)]
    )
    assert exit_code == 2
    assert not destination.exists()


def test_json_without_notes_since_is_an_error(repository, glossary, capsys):
    exit_code = check_rules.main([str(repository / "src"), "--glossary", str(glossary), "--json", "x.json"])
    assert exit_code == 2


def test_notes_since_without_json_prints_the_lines_it_adds(repository, glossary, capsys):
    write(repository / "src", "b.cc", "// A review.\n")
    check_rules.main(
        [str(repository / "src"), "--glossary", str(glossary), "--repository", str(repository)]
        + ["--notes-since", "HEAD"]
    )
    out = capsys.readouterr().out
    assert "b.cc:1: NOTE process-word" in out
    assert "a.cc" not in out


# One scan for the tree check and the doubtful lines; files checked in several processes -------------------------


def test_files_checked_in_several_processes_give_the_findings_of_a_serial_run(tmp_path, glossary, monkeypatch):
    for number in range(check_rules.PARALLEL_MIN_FILES + 4):
        text = "int stencil_count;\n" if number % 3 == 0 else "// A review.\nint fine;\n"
        write(tmp_path / "src", f"file{number:02d}.cc", text)
    in_processes = run(tmp_path, glossary)
    monkeypatch.setattr(check_rules, "PARALLEL_MIN_FILES", 10**6)
    serial = run(tmp_path, glossary)
    assert in_processes == serial
    assert [finding.path for finding in in_processes] == sorted(finding.path for finding in in_processes)
    assert "banned-word" in rules(in_processes, "FAIL")


def test_also_notes_since_writes_the_notes_and_keeps_the_usual_output(repository, glossary, tmp_path_factory, capsys):
    write(repository / "src", "b.cc", "// A review.\nint stencil_count;\n")
    destination = tmp_path_factory.mktemp("out") / "notes.json"
    exit_code = check_rules.main(
        [str(repository / "src"), "--glossary", str(glossary), "--repository", str(repository)]
        + ["--also-notes-since", "HEAD", "--also-notes-json", str(destination)]
    )
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "b.cc:2: FAIL banned-word" in out
    records = json.loads(destination.read_text(encoding="utf-8"))
    assert [(r["path"], r["line"], r["rule"]) for r in records] == [("src/b.cc", 1, "process-word")]


def test_also_notes_since_gives_the_list_that_notes_since_json_gives(repository, glossary, tmp_path_factory):
    write(repository / "src", "a.cc", "// The plan.\nint fine;\n// The ruling.\n")
    write(repository / "src", "b.cc", "// A review.\n")
    folder = tmp_path_factory.mktemp("out")
    common = [str(repository / "src"), "--glossary", str(glossary), "--repository", str(repository)]
    check_rules.main(common + ["--notes-since", "HEAD", "--json", str(folder / "alone.json")])
    check_rules.main(common + ["--also-notes-since", "HEAD", "--also-notes-json", str(folder / "also.json")])
    assert (folder / "also.json").read_text(encoding="utf-8") == (folder / "alone.json").read_text(encoding="utf-8")


def test_also_notes_flags_go_together_and_a_bad_commit_is_an_error(repository, glossary, tmp_path_factory, capsys):
    common = [str(repository / "src"), "--glossary", str(glossary), "--repository", str(repository)]
    destination = tmp_path_factory.mktemp("out") / "notes.json"
    assert check_rules.main(common + ["--also-notes-since", "HEAD"]) == 2
    assert check_rules.main(common + ["--also-notes-json", str(destination)]) == 2
    assert check_rules.main(common + ["--also-notes-since", "no-such-commit", "--also-notes-json", str(destination)]) == 2
    assert not destination.exists()


def test_a_clone_under_a_folder_named_tests_still_checks_backend_names_of_production_headers(tmp_path, glossary):
    clone = tmp_path / "tests" / "clone"
    write(clone, "src/fluidity/grid/a.h", FULL_HEADER + "void WriteCudaBuffer();\n")
    findings = check_rules.check_paths([clone / "src"], glossary, repository=clone)
    assert rules(findings) == ["backend-name"]


def test_a_test_file_of_a_clone_under_a_folder_named_tests_is_still_a_test_file(tmp_path, glossary):
    clone = tmp_path / "tests" / "clone"
    write(clone, "tests/grid/a.h", FULL_HEADER + "inline void CheckGpuThing();\n")
    assert check_rules.check_paths([clone / "tests"], glossary, repository=clone) == []


def test_a_fallthrough_comment_is_not_a_one_word_comment(tmp_path, glossary):
    write(tmp_path, "src/a.cc", "void Pick(int value) {\n  switch (value) {\n    case 1:\n      // fallthrough\n    case 2:\n      break;\n  }\n}\n")
    assert rules(run(tmp_path, glossary)) == []


# Allowed forms, library names, reference lines and names of the old code ---------------------------------------


@pytest.fixture
def forms_glossary(tmp_path: Path) -> Path:
    path = tmp_path / "forms_glossary.md"
    path.write_text(
        "# Glossary\n\n"
        + GLOSSARY_HEADER
        + "| `size_level`, `SizeLevel`, `size level` | 0 is the biggest. | level (alone) |\n"
        + "| `side_low` | A side. | floor (for a side), lower (for a side), sort (for grouping) |\n"
        + "| `piece_record` | Kept data. | Record (as a suffix), support (for this) |\n"
        + "| `GridPiece` | A piece. | stencil, basis, mac, adaptive |\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "text",
    [
        "int size_level;\n",
        "SizeLevel cell;\nint cell_size_level;\n",
        "// The size level of a cell.\n",
        "// Cells of size levels 1 to 3.\n",
        "// A size-level-2 cell.\n",
    ],
)
def test_a_one_sense_word_inside_a_form_the_glossary_says_is_quiet(tmp_path, forms_glossary, text):
    assert notes(tmp_path, forms_glossary, "a.cc", text) == []


@pytest.mark.parametrize(
    "text",
    ["int level;\n", "// The water level.\n", "// A size and a level.\n", "// Size, level 2.\n", "int level_size;\n"],
)
def test_the_nearest_use_outside_the_glossarys_form_is_still_a_note(tmp_path, forms_glossary, text):
    assert notes(tmp_path, forms_glossary, "a.cc", text) == ["one-sense-word"]


def test_a_form_in_one_place_does_not_hide_the_word_in_another_on_the_same_line(tmp_path, forms_glossary):
    assert notes(tmp_path, forms_glossary, "a.cc", "// The size level and the level.\n") == ["one-sense-word"]


def test_a_say_phrase_of_one_word_allows_nothing(tmp_path):
    glossary = tmp_path / "g.md"
    glossary.write_text(GLOSSARY_HEADER + "| `sort` | A thing. | sort (for grouping) |\n", encoding="utf-8")
    assert notes(tmp_path, glossary, "a.cc", "int sort;\n") == ["one-sense-word"]


@pytest.mark.parametrize(
    "text",
    [
        "// The coordinate p - 4 floor(p / 4).\n",
        "double q = std::floor(p);\n",
        "// It calls `::floor` of the CUDA math library.\n",
        "// Uses math::Floor.\n",
        "std::ranges::sort(items);\n",
        "// Orders them with std::sort, not by hand.\n",
        "auto at = std::lower_bound(a, b, c);\n",
        "// Found by std::ranges::upper_bound.\n",
        "RecordProperty(\"checked\", 3);\n",
        "// RecordProperty(name, value) writes it.\n",
        "cudaGetLastError();\n",
        "// target_link_libraries(a b) links them.\n",
        "// CMAKE_CUDA_ARCHITECTURES lists them.\n",
    ],
)
def test_a_library_name_at_its_call_site_is_quiet(tmp_path, forms_glossary, text):
    assert notes(tmp_path, forms_glossary, "a.cc", text) == []


@pytest.mark.parametrize(
    "text",
    [
        "// The floor of the room.\n",
        "// A floor (a flat solid).\n",
        "int floor;\n",
        "int sort;\n",
        "// Sort them by hand.\n",
        "int lower;\n",
        "int Record;\n",
        "// The record of the run.\n",
        "// The std::floor and the floor of a box.\n",
        "int lower_bound_for_a_side;\n",
    ],
)
def test_the_nearest_use_of_the_same_word_as_ours_is_still_a_note(tmp_path, forms_glossary, text):
    assert notes(tmp_path, forms_glossary, "a.cc", text) == ["one-sense-word"]


@pytest.mark.parametrize(
    "text",
    [
        "// Uses:    GridPiece (grid/grid_support.h)\n",
        "// Used by: tests/grid/grid_support.test.cc\n",
        "// Read:    docs/method-paper/support_notes.typ\n",
        "// Uses:    Vec3 (base/vec3.h),\n//          Mat3 (base/grid_support.h)\n",
    ],
)
def test_a_file_name_on_a_uses_used_by_or_read_line_is_not_prose(tmp_path, forms_glossary, text):
    assert notes(tmp_path, forms_glossary, "a.h", text) == []


@pytest.mark.parametrize(
    "text",
    [
        "// Uses:    the support of the weights\n",
        "// Purpose: read grid_support.h\n",
        "// Uses:    Vec3 (base/vec3.h)\n// The support of grid_support.h.\n",
        "// Idea: x\n//   as in grid_support.h\n",
    ],
)
def test_a_word_on_those_lines_or_a_file_name_elsewhere_is_still_a_note(tmp_path, forms_glossary, text):
    assert notes(tmp_path, forms_glossary, "a.h", text) == ["one-sense-word"]


def test_an_include_line_is_not_prose(tmp_path, forms_glossary):
    assert notes(tmp_path, forms_glossary, "a.cc", '#include "fluidity/grid/grid_support.h"\n#include <support.h>\n') == []
    assert notes(tmp_path, forms_glossary, "b.cc", "int support;\n") == ["one-sense-word"]


@pytest.mark.parametrize(
    "text",
    [
        "// The old file is `main:native/src/sparse_mac_grid_3d.cc`.\n",
        "// See `main:native/src/stencil_basis.cc` and `main:adaptive_grid`.\n",
    ],
)
def test_a_name_of_the_old_code_in_backticks_after_main_is_allowed(tmp_path, forms_glossary, text):
    write(tmp_path / "src", "a.cc", text)
    assert rules(run(tmp_path, forms_glossary)) == []


@pytest.mark.parametrize(
    "text",
    [
        "// The old file is `native/src/sparse_mac_grid_3d.cc`.\n",
        "// The old file is main:native/src/sparse_mac_grid_3d.cc.\n",
        "// The old file is 'main:native/src/stencil.cc'.\n",
        "// The `main:x.cc` file has a stencil.\n",
        "// The `main: x.cc` file and `mac`.\n",
    ],
)
def test_the_same_words_without_the_main_prefix_in_backticks_still_fail(tmp_path, forms_glossary, text):
    write(tmp_path / "src", "a.cc", text)
    assert "FAIL" in {finding.severity for finding in run(tmp_path, forms_glossary)}


def test_a_message_may_name_a_file_of_the_old_code_after_main(forms_glossary):
    import check_message

    glossary = check_rules.load_glossary(forms_glossary)
    assert check_message.check_message("Compare\n\nWith `main:native/src/stencil_grid.cc`.\n", glossary) == []
    assert check_message.check_message("Compare\n\nWith `native/src/stencil_grid.cc`.\n", glossary) != []


def test_no_allowed_name_names_something_that_no_longer_exists():
    names = check_rules.load_allowed_names(check_rules.DEFAULT_ALLOWED_NAMES_PATH)
    scanned = [check_rules.REPO_ROOT / folder for folder in ("src", "tests")]
    assert check_rules.find_stale_allowed_names(names, scanned) == []


def test_an_allowed_name_that_no_scanned_file_uses_is_stale(tmp_path):
    write(tmp_path / "src", "a.h", "inline int MoveParticle(int x);\n// RenamedHelper is only in a comment.\n")
    write(tmp_path / "src", "b.cc", 'const char* text = "GoneHelper";\n')
    names = {"MoveParticle", "RenamedHelper", "GoneHelper", "NeverWritten"}
    stale = check_rules.find_stale_allowed_names(names, [tmp_path])
    assert stale == ["GoneHelper", "NeverWritten", "RenamedHelper"]


def test_an_allowed_name_used_in_any_scanned_file_is_not_stale(tmp_path):
    write(tmp_path / "src", "a.h", "int A();\n")
    write(tmp_path / "tests", "b.test.cc", "void B() { CheckVec3IsNear(1); }\n")
    assert check_rules.find_stale_allowed_names({"A", "CheckVec3IsNear"}, [tmp_path]) == []


def test_the_real_glossary_allows_size_level_for_level_from_its_say_column():
    entries = check_rules.load_glossary(check_rules.DEFAULT_GLOSSARY_PATH)
    (level,) = [entry for entry in entries if entry.words == ("level",)]
    assert ("size", "level") in level.allowed_phrases
    assert all(len(phrase) > 1 for phrase in level.allowed_phrases)


# compiler-include: more macros, and macros defined from them; wide characters; a killed worker -------------------


@pytest.mark.parametrize(
    "condition",
    [
        "__clang_major__ >= 19",
        "__GNUC_MINOR__ > 2",
        "defined(_LIBCPP_VERSION)",
        "defined(__GLIBCXX__)",
        "_MSC_FULL_VER > 1",
    ],
)
def test_include_under_a_condition_on_a_compiler_version_macro_fails(tmp_path, glossary, condition):
    write(tmp_path, "a.cc", f'#if {condition}\n#include "x.h"\n#endif\n')
    assert rules(run(tmp_path, glossary)) == ["compiler-include"]


@pytest.mark.parametrize("condition", ["__cplusplus >= 202302L", "FLUIDITY_HAS_GPU", "defined(CLANG_NAMED_THING)"])
def test_include_under_a_condition_on_a_language_or_project_macro_passes(tmp_path, glossary, condition):
    write(tmp_path, "a.cc", f'#if {condition}\n#include "x.h"\n#endif\n')
    assert rules(run(tmp_path, glossary)) == []


def test_include_under_a_macro_defined_under_a_compiler_condition_fails(tmp_path, glossary):
    text = '#if defined(__clang__)\n#define FLUIDITY_IS_CLANG 1\n#endif\n#if FLUIDITY_IS_CLANG\n#include "x.h"\n#endif\n'
    write(tmp_path, "a.cc", text)
    assert rules(run(tmp_path, glossary)) == ["compiler-include"]


def test_include_under_a_macro_defined_as_a_compiler_macro_fails(tmp_path, glossary):
    write(tmp_path, "a.cc", '#define FLUIDITY_COMPILER __GNUC__\n#ifdef FLUIDITY_COMPILER\n#include "x.h"\n#endif\n')
    assert rules(run(tmp_path, glossary)) == ["compiler-include"]


def test_include_under_a_macro_defined_without_a_compiler_condition_passes(tmp_path, glossary):
    write(tmp_path, "a.cc", '#define FLUIDITY_FAST 1\n#if FLUIDITY_FAST\n#include "x.h"\n#endif\n')
    assert run(tmp_path, glossary) == []


def test_paper_line_counts_a_wide_character_as_two_columns(tmp_path, glossary):
    write(tmp_path / "docs" / "method-paper", "a.typ", "\u4e2d" * 60 + "\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []
    write(tmp_path / "docs" / "method-paper", "b.typ", "\u4e2d" * 61 + "\n")
    assert rules(run(tmp_path, glossary), "FAIL") == ["paper-line-length"]


def test_display_columns():
    assert check_rules.display_columns("abc") == 3
    assert check_rules.display_columns("\u00e9") == 1
    assert check_rules.display_columns("e\u0301") == 1
    assert check_rules.display_columns("\u4e2d") == 2
    assert check_rules.display_columns("\tx") == 9
    assert check_rules.display_columns("ab\tx") == 9


def test_a_worker_that_dies_makes_the_scan_fail_instead_of_hang(tmp_path, glossary, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    for number in range(check_rules.PARALLEL_MIN_FILES + 4):
        write(tmp_path / "src", f"file{number:02d}.cc", "int fine;\n")

    def die(file_path, context):
        os._exit(1)

    monkeypatch.setattr(check_rules, "_check_file", die)
    with pytest.raises(BrokenProcessPool):
        run(tmp_path, glossary)


@pytest.mark.parametrize("name", ["AbslStringify", "AbslHashValue", "PrintTo"])
def test_a_free_function_that_abseil_or_googletest_asks_for_is_not_a_test_helper_to_rename(tmp_path, glossary, name):
    write(tmp_path / "src", "a.test.cc", f"void {name}(const int& value, int* sink);\n")
    assert rules(run(tmp_path, glossary), "FAIL") == []


def test_another_free_function_in_a_test_file_still_fails_the_verb_rule(tmp_path, glossary):
    write(tmp_path / "src", "a.test.cc", "void Frobnicate(const int& value, int* sink);\n")
    assert rules(run(tmp_path, glossary), "FAIL") == ["test-helper-verb"]
