"""Tests for docs/method-paper/overflow_check.typ: a compile of a small document with a block outside the text area
fails, and one whose blocks fit passes. They run typst, so they are skipped where it is not installed."""

import shutil
import subprocess
from pathlib import Path

import pytest

CHECK_FILE = Path(__file__).resolve().parent.parent / "docs" / "method-paper" / "overflow_check.typ"
PAPER_FILE = CHECK_FILE.parent / "fluidity-method.typ"

pytestmark = pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")

DOCUMENT_HEAD = (
    '#import "overflow_check.typ"\n'
    "#show: overflow_check.install\n"
    '#set page(paper: "a4", margin: (x: 52mm, top: 26mm, bottom: 34mm))\n'
    "Some text.\n"
)
WIDE_EQUATION = "$ " + " + ".join(f"x_{index}" for index in range(60)) + " $\n"


def compile_document(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    shutil.copy(CHECK_FILE, tmp_path / "overflow_check.typ")
    document = tmp_path / "document.typ"
    document.write_text(DOCUMENT_HEAD + body, encoding="utf-8")
    return subprocess.run(
        ["typst", "compile", str(document), str(tmp_path / "document.pdf")],
        capture_output=True,
        text=True,
        check=False,
    )


def test_blocks_that_fit_compile(tmp_path):
    body = (
        '#figure(rect(width: 80mm, height: 20mm), caption: [A box that fits.])\n'
        "$ x = a + b $\n"
        "```\nlet x = 1\n```\n"
        "#table(columns: (30mm, 30mm), [a], [b])\n"
    )
    assert compile_document(tmp_path, body).returncode == 0


def test_a_figure_wider_than_the_text_area_fails_and_is_named(tmp_path):
    completed = compile_document(tmp_path, "#figure(rect(width: 150mm, height: 20mm), caption: [Too wide.])\n")
    assert completed.returncode != 0
    assert "page 1: figure 1 is 425 pt wide, and the text area is 300 pt" in completed.stderr


def test_a_figure_taller_than_the_text_area_fails_and_is_named(tmp_path):
    completed = compile_document(tmp_path, "#figure(rect(width: 50mm, height: 250mm), caption: [Too tall.])\n")
    assert completed.returncode != 0
    assert "figure 1 is 723 pt tall, and the text area is 672 pt" in completed.stderr


def test_a_display_equation_wider_than_the_text_area_fails_and_is_named(tmp_path):
    completed = compile_document(tmp_path, WIDE_EQUATION)
    assert completed.returncode != 0
    assert "page 1: equation is" in completed.stderr
    assert "wide, and the text area is 300 pt" in completed.stderr


def test_a_numbered_equation_is_named_by_its_number(tmp_path):
    completed = compile_document(tmp_path, "#set math.equation(numbering: \"(1)\")\n" + WIDE_EQUATION)
    assert "equation (1) is" in completed.stderr


def test_a_code_block_with_a_line_that_cannot_break_fails(tmp_path):
    completed = compile_document(tmp_path, "```\n" + "a_name_without_any_space_" * 12 + "\n```\n")
    assert completed.returncode != 0
    assert "code block is" in completed.stderr


def test_a_table_with_fixed_columns_wider_than_the_text_area_fails(tmp_path):
    completed = compile_document(tmp_path, "#table(columns: (80mm, 80mm), [a], [b])\n")
    assert completed.returncode != 0
    assert "table is" in completed.stderr


def test_a_table_whose_cells_wrap_to_fit_compiles(tmp_path):
    cell = "a long cell of many words that wraps inside its column " * 4
    body = f"#table(columns: (1fr, 1fr), [{cell}], [{cell}])\n"
    assert compile_document(tmp_path, body).returncode == 0


def test_every_finding_of_a_document_is_listed(tmp_path):
    body = "#figure(rect(width: 150mm, height: 20mm), caption: [One.])\n" + WIDE_EQUATION
    completed = compile_document(tmp_path, body)
    assert completed.stderr.count("the text area is 300 pt") == 2


def test_a_procedure_kept_in_the_text_may_be_taller_than_a_page(tmp_path):
    body = '#figure(rect(width: 50mm, height: 250mm), kind: "procedure", supplement: [Procedure], caption: [Long.])\n'
    assert compile_document(tmp_path, body).returncode == 0


def test_the_method_paper_installs_the_check():
    assert "#show: overflow_check.install" in PAPER_FILE.read_text(encoding="utf-8")
