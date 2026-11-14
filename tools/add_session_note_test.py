"""Tests for tools/add_session_note.py. A test that writes a note uses a throwaway repository in a temporary folder."""

import subprocess
from pathlib import Path

import pytest

import add_session_note

SESSION = "https://claude.ai/code/session_01EXAMPLE"


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], text=True, check=True, capture_output=True
    ).stdout


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "first")
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "second")
    return tmp_path


def read_note(repository: Path, commit: str = "HEAD") -> str:
    return git(repository, "notes", "--ref=claude-session", "show", commit)


def test_note_on_head_has_the_two_lines(repository):
    code = add_session_note.main(
        ["--session", SESSION, "--models", "claude-opus-5-5", "--repository", str(repository)]
    )
    assert code == 0
    assert read_note(repository) == f"session: {SESSION}\nmodels: claude-opus-5-5\n"


def test_models_are_deduplicated_and_in_the_fixed_order(repository):
    models = "claude-fable-5-2, claude-opus-5-5,claude-sonnet-5-5,claude-opus-5-5"
    add_session_note.main(["--session", SESSION, "--models", models, "--repository", str(repository)])
    assert read_note(repository).splitlines()[1] == "models: claude-opus-5-5, claude-sonnet-5-5, claude-fable-5-2"


def test_unknown_family_comes_after_the_known_ones():
    ordered = add_session_note.order_models(["other-model-1", "claude-fable-5-2", "claude-haiku-4-5"])
    assert ordered == ["claude-haiku-4-5", "claude-fable-5-2", "other-model-1"]


def test_second_note_on_the_same_commit_replaces_the_first(repository):
    arguments = ["--session", SESSION, "--repository", str(repository)]
    add_session_note.main([*arguments, "--models", "claude-opus-5-5"])
    add_session_note.main([*arguments, "--models", "claude-sonnet-5-5"])
    assert read_note(repository) == f"session: {SESSION}\nmodels: claude-sonnet-5-5\n"


def test_named_commit_gets_the_note_and_head_does_not(repository):
    add_session_note.main(
        ["--session", SESSION, "--models", "claude-opus-5-5", "--repository", str(repository), "HEAD~1"]
    )
    assert read_note(repository, "HEAD~1").startswith("session: ")
    listed = git(repository, "notes", "--ref=claude-session", "list").splitlines()
    assert len(listed) == 1


def test_empty_model_list_is_refused_and_writes_nothing(repository, capsys):
    code = add_session_note.main(["--session", SESSION, "--models", " , ", "--repository", str(repository)])
    assert code == 2
    assert "model list is empty" in capsys.readouterr().err
    assert git(repository, "notes", "--ref=claude-session", "list") == ""


@pytest.mark.parametrize("session", ["http://claude.ai/code/session_1", "claude.ai/code/session_1", "https://"])
def test_session_that_is_not_an_https_url_is_refused(repository, capsys, session):
    code = add_session_note.main(["--session", session, "--models", "claude-opus-5-5", "--repository", str(repository)])
    assert code == 2
    assert "https URL" in capsys.readouterr().err
    assert git(repository, "notes", "--ref=claude-session", "list") == ""


def test_unknown_commit_fails_with_the_git_message(repository, capsys):
    code = add_session_note.main(
        ["--session", SESSION, "--models", "claude-opus-5-5", "--repository", str(repository), "no-such-commit"]
    )
    assert code == 1
    assert "git notes failed" in capsys.readouterr().err
