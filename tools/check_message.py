"""Checks the message of a commit against the rules of AGENTS.md "Commit rule".

Usage: uv run python tools/check_message.py [<commit> | --range A..B | --file PATH]   (default: HEAD)

A message fails when:
- the subject is longer than 72 columns, the line after it is not blank, or a body line is longer than 72 columns (a
  line that is only a URL, and an indented code line, are exempt);
- the last paragraph has a trailer line, `Key: value` with Key one of TRAILER_KEYS or ending in `-by`; the one
  allowed trailer-like line is the literal last line `GPU: compiled, not run`;
- it names a tool or a model (TOOL_WORDS) as a whole word, in any case;
- it uses a process word (PROCESS_WORDS) as a whole word, in any case; a `fixup!` subject fails too, so a fixup commit
  cannot be pushed unfolded;
- it uses a word the glossary bans in every sense, found as check_rules.py finds it in a comment.

Output is one line per problem, `<commit>: <problem>` (`message:` for --file). The exit code is 0 when every message
passes, 1 when one fails, 2 when the input cannot be read.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import check_rules

SUBJECT_LIMIT = 72
BODY_LIMIT = 72
TRAILER_KEYS = ("Co-Authored-By", "Signed-off-by", "Assisted-by", "Generated-by", "Claude-Session", "Reviewed-by")
ALLOWED_LAST_LINE = "GPU: compiled, not run"
TOOL_WORDS = (
    "claude",
    "anthropic",
    "sonnet",
    "opus",
    "haiku",
    "fable",
    "copilot",
    "chatgpt",
    "gpt",
    "codex",
    "cursor",
)
PROCESS_WORDS = ("task", "brief", "review", "reviewer", "round", "fixup", "subagent", "agent")

_TRAILER_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):\s+\S")
_URL_ONLY_LINE = re.compile(r"^\s*<?https?://\S+>?\s*$")
_TOOL_WORD = re.compile(r"\b(" + "|".join(TOOL_WORDS) + r")\b", re.IGNORECASE)
_PROCESS_WORD = re.compile(r"\b(" + "|".join(PROCESS_WORDS) + r")\b", re.IGNORECASE)


def is_trailer_key(key: str) -> bool:
    return key.lower() in {known.lower() for known in TRAILER_KEYS} or key.lower().endswith("-by")


def find_layout_problems(lines: Sequence[str]) -> list[str]:
    problems = []
    if len(lines[0]) > SUBJECT_LIMIT:
        problems.append(f"the subject has {len(lines[0])} columns; the limit is {SUBJECT_LIMIT}")
    if len(lines) > 1 and lines[1].strip():
        problems.append("the line after the subject is not blank")
    too_long = [
        number
        for number, line in enumerate(lines[1:], start=2)
        if len(line) > BODY_LIMIT and not line.startswith(("    ", "\t")) and not _URL_ONLY_LINE.match(line)
    ]
    if too_long:
        problems.append(f"{len(too_long)} body lines are over {BODY_LIMIT} columns, the first is line {too_long[0]}")
    return problems


def find_trailer_problems(lines: Sequence[str]) -> list[str]:
    last_paragraph: list[str] = []
    for line in reversed(lines):
        if not line.strip():
            break
        last_paragraph.append(line)
    problems = []
    for index, line in enumerate(reversed(last_paragraph)):
        is_last_line = index == len(last_paragraph) - 1
        match = _TRAILER_LINE.match(line)
        if match and is_trailer_key(match.group(1)) and not (is_last_line and line == ALLOWED_LAST_LINE):
            problems.append(f"'{match.group(1)}:' is a trailer line; a commit message carries none")
    return problems


def find_word_problems(text: str, glossary: Sequence[check_rules.NeverSayEntry]) -> list[str]:
    problems = []
    for word in dict.fromkeys(match.lower() for match in _TOOL_WORD.findall(text)):
        problems.append(f"'{word}' names a tool or a model; a message does not")
    for word in dict.fromkeys(match.lower() for match in _PROCESS_WORD.findall(text)):
        problems.append(f"'{word}' is a process word; say what the change does, not how it was made")
    for word in check_rules.find_banned_words(text, glossary):
        problems.append(f"'{word}' is banned (docs/glossary.md)")
    return problems


def check_message(text: str, glossary: Sequence[check_rules.NeverSayEntry]) -> list[str]:
    """The problems of one commit message, empty when it passes."""
    lines = text.rstrip().splitlines()
    if not lines:
        return ["the message is empty"]
    return find_layout_problems(lines) + find_trailer_problems(lines) + find_word_problems(text, glossary)


def read_commit_messages(commits: Sequence[str], repository: Path) -> dict[str, str]:
    """The message of each revision, by the short hash. Raises ValueError when git cannot read one."""
    messages = {}
    for revision in commits:
        completed = subprocess.run(
            ["git", "log", "-1", "--format=%h%n%B", revision],
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout:
            raise ValueError(f"git could not read the commit {revision}")
        short_hash, _, message = completed.stdout.partition("\n")
        messages[short_hash] = message
    return messages


def list_range(revision_range: str, repository: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "rev-list", "--reverse", revision_range],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"git could not list the range {revision_range}")
    return completed.stdout.split()


def main(argv: Sequence[str] | None = None, repository: Path = check_rules.REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description="Check the message of a commit.")
    parser.add_argument("commit", nargs="?", default="HEAD", help="the commit to check (default: HEAD)")
    parser.add_argument("--range", dest="revision_range", help="check every commit of A..B")
    parser.add_argument("--file", help="check the message in this file")
    arguments = parser.parse_args(argv)
    try:
        glossary = check_rules.load_glossary(check_rules.DEFAULT_GLOSSARY_PATH)
        if arguments.file:
            messages = {"message": Path(arguments.file).read_text(encoding="utf-8")}
        elif arguments.revision_range:
            messages = read_commit_messages(list_range(arguments.revision_range, repository), repository)
        else:
            messages = read_commit_messages([arguments.commit], repository)
    except (OSError, ValueError, check_rules.GlossaryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    failed = False
    for name, message in messages.items():
        for problem in check_message(message, glossary):
            print(f"{name}: {problem}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
