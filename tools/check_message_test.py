"""Tests for tools/check_message.py."""

import subprocess
from pathlib import Path

import pytest

import check_message
import check_rules

GLOSSARY = [
    check_rules.NeverSayEntry(words=("stencil",), spelled="stencil", qualifier=None),
    check_rules.NeverSayEntry(words=("wall",), spelled="wall", qualifier="in a flow"),
]


def problems(text: str) -> list[str]:
    return check_message.check_message(text, GLOSSARY)


def test_plain_message_passes():
    assert problems("Add the cell size rule\n\nIt checks that two sizes differ by at most one level.\n") == []


def test_subject_over_72_columns_fails():
    assert len(problems("x" * 73 + "\n")) == 1
    assert problems("x" * 72 + "\n") == []


def test_missing_blank_line_after_the_subject_fails():
    assert problems("Subject\nbody starts here\n") == ["the line after the subject is not blank"]


def test_body_line_over_72_columns_fails_and_counts_once():
    text = "Subject\n\n" + "word " * 20 + "\n" + "word " * 20 + "\n"
    assert problems(text) == ["2 body lines are over 72 columns, the first is line 3"]


@pytest.mark.parametrize("line", ["https://example.org/" + "a" * 80, "    " + "x" * 90, "\t" + "x" * 90])
def test_long_url_and_code_lines_are_exempt(line):
    assert problems(f"Subject\n\n{line}\n") == []


@pytest.mark.parametrize(
    "trailer",
    ["Co-Authored-By: A <a@b.c>", "Signed-off-by: A", "Assisted-by: A", "Generated-by: A", "Reviewed-by: A", "Tested-by: A"],
)
def test_trailer_in_the_last_paragraph_fails(trailer):
    result = problems(f"Subject\n\nBody text.\n\n{trailer}\n")
    assert len(result) == 1 and "trailer line" in result[0]


def test_session_trailer_fails():
    assert any("trailer" in found for found in problems("Subject\n\nBody.\n\nSession-Id: x\nClaude-Session: x\n"))


def test_key_value_text_that_is_not_a_trailer_passes():
    assert problems("Subject\n\nNote: this is prose, not a trailer.\n") == []


def test_gpu_line_is_the_one_allowed_last_line():
    assert problems("Subject\n\nBody.\n\nGPU: compiled, not run\n") == []


def test_gpu_line_that_is_not_last_in_its_paragraph_still_passes_but_a_trailer_beside_it_fails():
    result = problems("Subject\n\nBody.\n\nCo-Authored-By: A\nGPU: compiled, not run\n")
    assert len(result) == 1 and "Co-Authored-By" in result[0]


@pytest.mark.parametrize("word", ["Claude", "anthropic", "SONNET", "opus", "haiku", "fable", "Copilot", "ChatGPT", "gpt", "codex", "cursor"])
def test_tool_or_model_name_fails(word):
    result = problems(f"Subject\n\nWritten with {word} today.\n")
    assert len(result) == 1 and "tool or a model" in result[0]


@pytest.mark.parametrize("word", ["Brief", "review", "reviewer", "fixup", "subagent", "agent"])
def test_process_word_fails(word):
    result = problems(f"Subject\n\nThe {word} said so.\n")
    assert len(result) == 1 and "process word" in result[0]


@pytest.mark.parametrize("word", ["round", "Round", "task", "tasks"])
def test_round_and_task_are_ordinary_words(word):
    assert problems(f"Subject\n\nThe {word} of a number, or a VS Code {word}.\n") == []


def test_fixup_subject_fails():
    assert any("fixup" in found for found in problems("fixup! Add the rule\n"))


def test_process_word_inside_a_longer_word_passes():
    assert problems("Subject\n\nRounding and tasking and reviews and agentic, cursors.\n") == []


def test_banned_glossary_word_fails_but_one_sense_word_passes():
    assert problems("Subject\n\nA stencil here.\n") == ["'stencil' is banned (docs/glossary.md)"]
    assert problems("Subject\n\nA wall here.\n") == []


def test_empty_message_fails():
    assert problems("\n") == ["the message is empty"]


def git(repository: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repository, check=True, stdout=subprocess.DEVNULL)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "a@b.c")
    git(tmp_path, "config", "user.name", "A")
    git(tmp_path, "config", "commit.gpgsign", "false")
    for message in ("First good message", "Second\n\nBody.\n\nSigned-off-by: A"):
        git(tmp_path, "commit", "-q", "--allow-empty", "-m", message)
    return tmp_path


def test_main_checks_head_by_default(repository, capsys):
    assert check_message.main([], repository) == 1
    assert "trailer line" in capsys.readouterr().out


def test_main_checks_one_commit_and_a_range(repository, capsys):
    assert check_message.main(["HEAD~1"], repository) == 0
    assert check_message.main(["--range", "HEAD~1..HEAD"], repository) == 1
    assert check_message.main(["--range", "HEAD~1..HEAD~1"], repository) == 0
    capsys.readouterr()


def test_main_reads_a_file(tmp_path, capsys):
    path = tmp_path / "message.txt"
    path.write_text("Subject\n\nBody.\n", encoding="utf-8")
    assert check_message.main(["--file", str(path)]) == 0
    path.write_text("Subject\n\nA review.\n", encoding="utf-8")
    assert check_message.main(["--file", str(path)]) == 1
    capsys.readouterr()


def test_main_reports_an_unreadable_input(repository, capsys):
    assert check_message.main(["no-such-commit"], repository) == 2
    assert check_message.main(["--file", str(repository / "missing.txt")]) == 2
    assert "error:" in capsys.readouterr().err
