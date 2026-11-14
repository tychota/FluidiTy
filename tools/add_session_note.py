"""Writes the session note of a commit: which Claude Code session and models worked on it.

Commit messages carry no AI trailer (AGENTS.md "Commit rule"). Instead each commit has a git note in
refs/notes/claude-session with exactly two lines:

    session: https://claude.ai/code/session_...
    models: claude-opus-5-5, claude-sonnet-5-5, claude-fable-5-2

Usage (from the repository root):
    uv run python tools/add_session_note.py --session <url> --models <id>[,<id>...] [<commit>]

<commit> defaults to HEAD. An existing note on that commit is replaced. The models are de-duplicated and written in a
fixed order: opus, sonnet, haiku, fable, then any other id in alphabetical order. The command refuses an empty model
list and a session link that is not an https URL. Notes stay in the local repository unless they are pushed on
purpose (AGENTS.md "Commands").
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

NOTES_REF = "refs/notes/claude-session"
# The order of model families in the models line. An id names its family as its second dash-separated word
# (claude-opus-5-5); an id of no listed family comes after them.
FAMILY_ORDER = ["opus", "sonnet", "haiku", "fable"]


class NoteError(ValueError):
    """Raised when the session link or the model list cannot be written as a note."""


def order_models(models: Sequence[str]) -> list[str]:
    """Returns the model ids without blanks and duplicates, in the fixed order of FAMILY_ORDER, then alphabetically."""
    unique = sorted({model.strip() for model in models if model.strip()})
    if not unique:
        raise NoteError("the model list is empty; give at least one model id, such as claude-opus-5-5")

    def rank(model: str) -> tuple[int, str]:
        words = model.split("-")
        family = words[1] if len(words) > 1 else ""
        position = FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)
        return position, model

    return sorted(unique, key=rank)


def build_note(session: str, models: Sequence[str]) -> str:
    """Returns the two-line note for `session` and `models`. Raises NoteError for a link that is not https."""
    link = urlparse(session.strip())
    if link.scheme != "https" or not link.netloc:
        raise NoteError(f"the session link must be an https URL, not {session!r}")
    return f"session: {session.strip()}\nmodels: {', '.join(order_models(models))}\n"


def write_note(repository: Path, commit: str, note: str) -> None:
    """Writes `note` on `commit` in NOTES_REF of `repository`, replacing a note already there."""
    subprocess.run(
        ["git", "-C", str(repository), "notes", f"--ref={NOTES_REF}", "add", "--force", "--file=-", commit],
        input=note,
        text=True,
        check=True,
        capture_output=True,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the session note of a commit (refs/notes/claude-session).")
    parser.add_argument("--session", required=True, help="the Claude Code session link (https)")
    parser.add_argument("--models", required=True, help="model ids, comma separated")
    parser.add_argument("commit", nargs="?", default="HEAD", help="the commit to annotate (default HEAD)")
    parser.add_argument("--repository", default=".", help="the repository (default: the current folder)")
    options = parser.parse_args(arguments)
    try:
        note = build_note(options.session, options.models.split(","))
    except NoteError as error:
        print(f"add_session_note: {error}", file=sys.stderr)
        return 2
    try:
        write_note(Path(options.repository), options.commit, note)
    except subprocess.CalledProcessError as error:
        print(f"add_session_note: git notes failed: {error.stderr.strip()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
