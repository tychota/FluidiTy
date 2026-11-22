"""Tests for tools/check_rules.py. Each test builds a small tree and checks the findings."""

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
    write(tmp_path, "a.h", FULL_HEADER + "void ApplyForces(double x, bool bounded);\n")
    assert rules(run(tmp_path, glossary)) == ["bare-bool"]


def test_bool_field_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "struct Cell {\n  bool has_liquid = false;\n};\n")
    assert run(tmp_path, glossary) == []


def test_unlisted_verb_fails(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "double EvaluateWeight(double x);\n")
    findings = run(tmp_path, glossary)
    assert rules(findings) == ["verb"]
    assert "EvaluateWeight" in findings[0].text


def test_unlisted_verb_says_where_to_list_the_name(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "double EvaluateWeight(double x);\n")
    findings = run(tmp_path, glossary)
    assert "list it with its reason in tools/check_rules_allowed_names.txt" in findings[0].text


def test_allowed_verb_must_end_at_a_word_boundary(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "double Computer(double x);\n")
    assert rules(run(tmp_path, glossary)) == ["verb"]


def test_listed_physical_name_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "void MoveParticles(double x);\nvoid TryStep(double x);\n")
    assert run(tmp_path, glossary) == []


def test_math_namespace_noun_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "namespace fluidity::math {\ndouble Dot(Vec3 a, Vec3 b);\n}\n")
    assert run(tmp_path, glossary) == []


def test_member_function_is_not_a_free_function(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "struct Grid {\n  double Weight(double x) const;\n};\n")
    assert run(tmp_path, glossary) == []


def test_inline_definition_in_header_is_not_scanned_for_verbs(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "inline double ComputeTwice(double x) {\n  return Twice(x);\n}\n")
    assert run(tmp_path, glossary) == []


def test_operator_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "Vec3 operator+(Vec3 left, Vec3 right);\n")
    assert run(tmp_path, glossary) == []


def test_verb_rule_does_not_apply_to_cc_files(tmp_path, glossary):
    write(tmp_path, "a.cc", "double EvaluateWeight(double x);\n")
    assert run(tmp_path, glossary) == []


def test_to_conversion_is_allowed(tmp_path, glossary):
    write(tmp_path, "a.h", FULL_HEADER + "std::string_view ToMessage(RunOnError error);\n")
    assert run(tmp_path, glossary) == []


def test_allowed_name_file_exempts(tmp_path, glossary):
    write(tmp_path / "src", "a.h", FULL_HEADER + "double EvaluateWeight(double x);\n")
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
    write(tmp_path, "a.h", FULL_HEADER + "void ApplyForces(double x,\n    double y,\n    bool bounded);\n")
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
    write(tmp_path, "a.h", FULL_HEADER + "namespace fluidity::math {\ndouble Dot(Vec3 a, Vec3 b);\n}\n")
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
    write(tmp_path, "a.test.cc", "void ExpectNear(const Mat3& actual, const Vec3& expected) {}\n")
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
        sequences = [(check_rules.split_name_words(name), "name") for name in check_rules._identifiers(line.code)]
        sequences += [(segment, "comment") for segment in check_rules.comment_segments(line.comment)]
        for tokens, where in sequences:
            for entry in entries:
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


@pytest.mark.parametrize("macro", ["__clang__", "__GNUC__", "_MSC_VER", "__apple_build_version__"])
def test_include_under_a_condition_on_a_compiler_macro_fails(tmp_path, glossary, macro):
    write(tmp_path, "a.cc", f'#if defined({macro})\n#include "fluidity/base/x.h"\n#endif\n')
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
