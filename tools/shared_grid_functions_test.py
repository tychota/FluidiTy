"""Every function marked FLUIDITY_CPU_AND_GPU in a header of src/fluidity/grid is run by a GPU test.

A GPU test runs a shared function either by naming it in an answer function (grid_functions_answers.h, or the cell size
rule runner) or through a public function that reaches it. The second kind is listed, with the function that reaches it,
in the table at the head of grid_functions_runner_gpu.h. This test fails for a shared function of the grid headers that is
named in none of those files, so a new shared function cannot be added without choosing one or the other.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GRID_FOLDER = REPO_ROOT / "src" / "fluidity" / "grid"
TESTING_FOLDER = GRID_FOLDER / "testing"
# The files in which a shared function counts as run: the answer functions and the runners with their table.
NAMING_FILES = (
    "grid_functions_answers.h",
    "grid_functions_runner_gpu.h",
    "grid_functions_runner_gpu.cu",
    "cell_size_rule_runner_gpu.h",
    "cell_size_rule_runner_gpu.cu",
)
MARK = "FLUIDITY_CPU_AND_GPU"
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def remove_comments(text: str) -> str:
    """The text without `//` comments and `/* */` comments, so a mention of the mark in a comment is not a function."""
    without_blocks = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_blocks)


def find_shared_function_names(header_text: str) -> set[str]:
    """The names of the functions marked FLUIDITY_CPU_AND_GPU in a header: the identifier before the first `(` after
    each mark that is outside a comment and not the definition of the macro itself."""
    code = remove_comments(header_text)
    names = set()
    for mark in re.finditer(rf"\b{MARK}\b", code):
        if code[: mark.start()].rstrip().endswith("#define") or code[mark.end() :].lstrip().startswith("#"):
            continue
        rest = code[mark.end() :]
        parenthesis = rest.find("(")
        if parenthesis < 0:
            continue
        identifiers = IDENTIFIER.findall(rest[:parenthesis])
        if identifiers:
            names.add(identifiers[-1])
    return names


def find_names_missing(grid_folder: Path, naming_texts: list[str]) -> list[str]:
    """The shared functions of the headers directly in `grid_folder` that no text of `naming_texts` names."""
    named = set()
    for text in naming_texts:
        named.update(IDENTIFIER.findall(text))
    missing = []
    for header in sorted(grid_folder.glob("*.h")):
        for name in sorted(find_shared_function_names(header.read_text(encoding="utf-8"))):
            if name not in named:
                missing.append(f"{header.name}: {name}")
    return missing


def test_every_shared_grid_function_is_run_or_listed():
    naming_texts = [(TESTING_FOLDER / name).read_text(encoding="utf-8") for name in NAMING_FILES]
    assert find_names_missing(GRID_FOLDER, naming_texts) == []


def test_finder_reads_the_function_name_after_the_mark(tmp_path):
    (tmp_path / "a.h").write_text(
        "// FLUIDITY_CPU_AND_GPU in a comment: Ignored\n"
        "FLUIDITY_CPU_AND_GPU constexpr int ComputeOne(int x) noexcept { return x; }\n"
        "FLUIDITY_CPU_AND_GPU inline double\n"
        "ComputeTwo(double y) { return y; }\n",
        encoding="utf-8",
    )
    assert find_shared_function_names((tmp_path / "a.h").read_text(encoding="utf-8")) == {"ComputeOne", "ComputeTwo"}


def test_a_shared_function_named_nowhere_is_reported(tmp_path):
    (tmp_path / "a.h").write_text(
        "FLUIDITY_CPU_AND_GPU constexpr int ComputeOne(int x) { return x; }\n"
        "FLUIDITY_CPU_AND_GPU constexpr int ComputeTwo(int x) { return x; }\n",
        encoding="utf-8",
    )
    assert find_names_missing(tmp_path, ["the table names ComputeOne only"]) == ["a.h: ComputeTwo"]
