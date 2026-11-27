"""Applies the mechanical items of the AGENTS.md checklist to files.

Usage: uv run python tools/check_rules.py [PATH ...] [--glossary PATH] [--allowed-names PATH]
                                         [--notes-on-staged-lines | --notes-since COMMIT [--json PATH]]
                                         [--also-notes-since COMMIT --also-notes-json PATH]
                                         [--repository PATH]   (default paths: src, tests and docs/method-paper)

Output lines are `path:line: FAIL <rule-id> <text>` and `path:line: NOTE <rule-id> <text>`.
The exit code is 1 when any FAIL was printed, 2 when an input is missing or malformed, otherwise 0.
Notes never change the exit code. With --notes-on-staged-lines a NOTE is printed only for a line that the staged
changes of the repository add (`git diff --cached`), so a commit shows the notes of its own lines; FAIL lines are
always printed. With --notes-since COMMIT the lines are those the working tree adds to COMMIT, committed or not; with
--json PATH as well, the notes of those lines are written to PATH as a JSON list of `{path, line, rule, text,
source_line}`, sorted by path, line and rule, and their count is printed.
With --also-notes-since COMMIT and --also-notes-json PATH one run does both jobs: it prints what it prints without
them (the FAIL lines and the notes of the chosen mode) and also writes the notes of the lines the working tree adds
to COMMIT to PATH, in the same form as --json. The files are read and checked once, so the gate gets the tree check
and the list of doubtful lines for the price of one scan.
Files are checked one process each at a time when there are PARALLEL_MIN_FILES or more of them and the system can
fork; the findings come back in file order, so the output is the same as for a serial run.

The vocabulary comes from docs/glossary.md (only the "Never say" column is read) and the allowed verbs
come from the "Allowed verbs" section of AGENTS.md. Words are matched on tokens, so `cut` does not match
`execute`. The last word of a banned entry also matches its regular inflections (`inflected_forms`), so a banned
verb catches its -s, -ed and -ing forms and a banned noun its plural; a longer word that only starts with it, an
irregular form and the few inflected forms that are words of their own here (`WORDS_OF_THEIR_OWN`) do not match.
String literals are never scanned; identifiers, comments and Python docstrings are.

What a note for a word banned in one sense leaves out (the nearest other use is still noted): the word inside a longer
form that the glossary itself says, that is, a phrase of the "Say" column of any row that holds the word and more
(`size_level`, `SizeLevel` and `size level` for `level`; a hyphen between words joins them, `size-level-2`); the word
inside the name of a library function or macro where it is written (`_LIBRARY_NAMES`: `floor(`, `std::floor`,
`math::Floor`, `std::ranges::sort`, `std::lower_bound`, `RecordProperty(`, CUDA runtime names and qualifiers, CMake
commands and variables); and a file name on a `Uses:`, `Used by:` or `Read:` line of a header and on the lines that
continue it. A path or identifier of the old proof-of-concept code, written in backticks after the literal prefix
`main:` (`main:native/src/sparse_mac_grid_3d.cc`), is not read at all, by the notes or by the banned words, in a
comment or in a commit message.

A word banned in one sense can carry clauses after a `;` in its parentheses in the glossary (`noted only beside`,
`quiet beside`, `allowed`, `quiet in files`, `noted only in names`, `noted only as the last word of a name`; the
glossary's "How to read the tables" explains each). A use that the clauses place outside the banned sense is not
noted. The clauses of the glossary are the only place where the senses are told apart; this tool has no special case
for a word.

Which rules apply to which files (a deliberate choice, written down here). The folders `src`, `tests` and `testing`
are read from the path below the repository (--repository), so a clone inside a folder named `tests` changes nothing:
- `file-header`, `bare-bool` and `verb` apply to `.h` and `.cuh` files, where the public declarations are.
- `backend-name` applies to `.h` files only, because a `.cuh` file is allowed to name its backend, and not to test
  files (`.test.` in the name, or a `testing` or `tests` folder): a test may name the backend it tests, as in
  `CheckGpuHalfSameAsCpuLoop`. Names listed in the allowed-names file are exempt from it, as from `verb`.
- `role-suffix` applies to every C++ file; the word rules and `milestone-code` apply to every scanned file.
- `shared-by-reference` applies to C++ files that contain FLUIDITY_CPU_AND_GPU: they take Vec3 and Mat3 by value,
  because GPU code cannot bind a reference to a namespace-scope constexpr constant such as kIdentity.
- A file whose stem ends in `_gpu` (`buffer_gpu.h`, `buffer_gpu.cu`), but not in `_and_gpu`, is private to its
  folder: `backend-name` and `verb` do not apply to it, because its names wrap GPU calls. `private-gpu-include`
  applies to every C++ file: an `#include` of a `*_gpu.h` or `*_gpu.cuh` header, written with quotes or angle
  brackets, fails from another folder, or from a header whose stem does not end in `_gpu`. One exception: a test
  (a file with `.test.` in its name) may include a private GPU header of the `testing/` folder of its own folder, as
  `fluidity/grid/testing/cell_size_rule_runner_gpu.h` from `fluidity/grid/cell_size_rule.gpu.test.cc`. A test in
  `tests/<module>/` counts as a file of the folder `fluidity/<module>`: it may include that folder's private GPU
  headers, as a test beside them could.
- `compiler-include` applies to every C++ file: an `#include` inside an `#if`, `#ifdef`, `#ifndef` or `#elif` block
  whose condition names a `__clang...__` or `__GNUC...__` macro (`__clang__`, `__clang_analyzer__`, `__clang_major__`,
  `__GNUC_MINOR__`), `_MSC_...`, `__apple_build_version__`, `_LIBCPP_VERSION`, `__GLIBCXX__` or `__has_include`, or a
  macro that the file defines under such a condition, fails, in any branch of the block. tools/run_tidy.py lists the headers of a file with the build compiler and keys its cache on that list, so
  it must be the list that clang-tidy sees.
- `one-word-comment` applies to C++ files under src and tests. `test-helper-verb` applies to the functions defined in
  test files (`.test.` in the name, or a `testing` or `tests` folder): it is `verb` for them, and the same
  allowed-names file exempts a name. `paper-line-length` applies to the Typst files of docs/method-paper.
RULE_TABLE below lists every rule with its level, the files it applies to and its reason; a rule added to the tool
is one entry there and one pair of tests (it fires, it stays quiet on the nearest innocent text).
"""

from __future__ import annotations

import argparse
import ast
import functools
import io
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tokenize
import unicodedata
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOSSARY_PATH = REPO_ROOT / "docs" / "glossary.md"
DEFAULT_AGENTS_PATH = REPO_ROOT / "AGENTS.md"
DEFAULT_ALLOWED_NAMES_PATH = Path(__file__).resolve().parent / "check_rules_allowed_names.txt"
DEFAULT_SCAN_PATHS = ("src", "tests", "docs/method-paper")
# Below this many files a run checks them in this process: starting workers costs more than it saves.
PARALLEL_MIN_FILES = 16

SCANNED_SUFFIXES = {".h", ".cuh", ".cc", ".cu", ".py", ".lean", ".typ"}
CPP_SUFFIXES = {".h", ".cuh", ".cc", ".cu"}
HEADER_SUFFIXES = {".h", ".cuh"}
SKIPPED_DIRECTORY_NAMES = {".superpowers", ".venv", "out", ".git", "__pycache__"}

GLOSSARY_HEADER_CELLS = ["Say", "Plain meaning", "Never say"]
HEADER_LABELS = ["Purpose", "Idea", "Owns", "Uses", "Used by", "Read"]
LITERATURE_MARKERS = ["papers call this", "papers:", "textbook:", "also called"]
ABBREVIATIONS = [("mac",), ("pic",), ("apic",), ("p", "2", "g"), ("g", "2", "p"), ("eos",)]
ROLE_SUFFIXES = ["Input", "Options", "Parameters", "Params", "Config", "Results", "Outcome", "Request"]
BACKEND_WORDS = {"cuda", "gpu", "device", "host"}
# Build macros, not names a caller sees.
BACKEND_EXEMPT_IDENTIFIERS = {"FLUIDITY_CPU_AND_GPU", "FLUIDITY_HAS_GPU"}
PRIVATE_GPU_STEM_SUFFIX = "_gpu"
# The folder of a module that holds the code its tests share; a test may include its private GPU headers.
TEST_CODE_FOLDER = "testing"
# The folder of the functional tests, one folder per module: tests/<module>/.
TESTS_FOLDER = "tests"
# `cpu_and_gpu.h` ends in `_gpu` but is the shared mark every shared header includes, not a private GPU header.
SHARED_STEM_SUFFIX = "_and_gpu"


def display_columns(text: str) -> int:
    """The columns `text` takes on a terminal: a wide character (East Asian width W or F) takes 2, a combining mark 0,
    a tab goes to the next multiple of 8, any other character 1. `len` counts code points, which is not the same."""
    columns = 0
    for character in text:
        if character == "\t":
            columns += 8 - columns % 8
        elif unicodedata.combining(character):
            continue
        elif unicodedata.east_asian_width(character) in ("W", "F"):
            columns += 2
        else:
            columns += 1
    return columns


def is_private_gpu_stem(stem: str) -> bool:
    """True for `buffer_gpu`, false for `cpu_and_gpu` and for `gpu_launch`."""
    return stem.endswith(PRIVATE_GPU_STEM_SUFFIX) and not stem.endswith(SHARED_STEM_SUFFIX)


# Lowercase `m2` and `m3` in a comment are units (square and cubic metres), not milestone codes.
UNIT_TOKENS_IN_COMMENTS = {"m2", "m3"}

FAIL = "FAIL"
NOTE = "NOTE"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    level: str  # FAIL: no false positive, fails the gate. NOTE: a detector that can be wrong, a reader decides.
    applies_to: str
    reason: str


RULE_TABLE = (
    Rule("file-header", FAIL, ".h and .cuh files", "a header says its purpose, idea, owner, users and reading"),
    Rule("banned-word", FAIL, "every scanned file", "the glossary's Never say words"),
    Rule("one-sense-word", NOTE, "every scanned file", "a word banned in one sense only; a reader decides the sense"),
    Rule("abbreviation", FAIL, "every scanned file", "an abbreviation of the glossary is written out"),
    Rule("milestone-code", FAIL, "every scanned file", "a milestone code says nothing to a reader of the code"),
    Rule("dimension-suffix", FAIL, "every scanned file", "a name says Grid3D by its type, not by a 2D or 3D suffix"),
    Rule("role-suffix", FAIL, "C++ files", "a type is named for what it is, not for the role Input or Options"),
    Rule("backend-name", FAIL, ".h files", "a shared header names no backend"),
    Rule("shared-by-reference", FAIL, "C++ files with FLUIDITY_CPU_AND_GPU", "GPU code cannot bind a constant"),
    Rule("verb", FAIL, ".h and .cuh files", "a function starts with a verb AGENTS.md allows"),
    Rule("bare-bool", FAIL, ".h and .cuh files", "a bool parameter says nothing at the call"),
    Rule("private-gpu-include", FAIL, "C++ files", "a private GPU header is for its own folder"),
    Rule("compiler-include", FAIL, "C++ files", "the tidy cache lists the headers the build compiler sees"),
    Rule("one-word-comment", FAIL, "C++ files under src and tests", "a one-word comment says nothing the code does not"),
    Rule("test-helper-verb", FAIL, "functions defined in test files", "a test helper starts with an allowed verb"),
    Rule("paper-line-length", FAIL, "docs/method-paper/*.typ", "a paper line is at most 120 columns (wide characters count 2)"),
    Rule("test-name-and", NOTE, "TEST names in C++ files", "a test that checks two things should be two tests"),
    Rule("many-inputs", NOTE, "functions of test files and private helpers", "more than four inputs hide the call"),
    Rule("short-local", NOTE, "C++ files", "a one-letter variable or parameter says nothing; the project uses none"),
    Rule("process-word", NOTE, "comments of every scanned file", "a comment says what the code does, not how it was made"),
    Rule("test-without-check", NOTE, "TEST bodies in C++ files", "a test with no check passes whatever the code does"),
)


class GlossaryError(Exception):
    """The glossary does not have the shape this tool reads."""


class RulesFileError(Exception):
    """AGENTS.md does not have the section this tool reads."""


class Finding(NamedTuple):
    path: str
    line: int
    severity: str
    rule_id: str
    text: str


@dataclass(frozen=True)
class NeverSayEntry:
    words: tuple[str, ...]
    spelled: str
    qualifier: str | None  # None: banned in every sense. Text: banned in that one sense only.
    # Longer phrases that the glossary itself says (the "Say" column of any row) and that contain `words`, as `size
    # level` holds `level`. A use of a one-sense word inside one of them is the allowed form, not a hit.
    allowed_phrases: tuple[tuple[str, ...], ...] = ()
    # Where the glossary says the banned sense is, or is not (the clauses after `;` in the parentheses of the entry, see
    # `_parse_sense_clauses`). Each word is lowercase; a name piece such as `SizeLevel` is two words.
    noted_only_beside: frozenset[str] = frozenset()  # Noted only when one of these words is in the same stretch.
    quiet_beside: frozenset[str] = frozenset()  # Not noted when one of these words is in the same stretch.
    quiet_in_files: tuple[str, ...] = ()  # Not noted in a file whose path contains one of these.
    is_noted_only_in_names: bool = False  # Not noted in a comment.
    is_noted_only_as_name_suffix: bool = False  # Noted only as the last word of a name.


# ---------------------------------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------------------------------

_RUN = re.compile(r"[A-Za-z0-9_]+")
_WORD_PIECE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_CAMEL_BREAK = re.compile(r"(?<=[a-z])(?=[A-Z])")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MILESTONE = re.compile(r"^[mMfF][0-9]+$")
# A processor name such as "Apple M3 Pro" is not a milestone code.
_PROCESSOR_NAME = re.compile(r"\bApple [mM][0-9]+\b")


def split_name_pieces(identifier: str) -> list[str]:
    """Splits at `_` and where a small letter meets a capital. Digits stay with their letters: `Grid3D` is one token."""
    tokens: list[str] = []
    for part in identifier.split("_"):
        tokens.extend(piece for piece in _CAMEL_BREAK.split(part) if piece)
    return tokens


def split_name_words(text: str) -> list[str]:
    """Lowercase words of a name: splits like split_name_pieces and also at every letter and digit change."""
    words: list[str] = []
    for piece in split_name_pieces(text):
        words.extend(match.lower() for match in _WORD_PIECE.findall(piece))
    return words


_VOWELS = frozenset("aeiou")
# Inflected forms of a banned word that are words of their own in this code base, with the sense they carry. They are
# never matched as an inflection; the banned word itself still is.
WORDS_OF_THEIR_OWN = {
    "qualified": "C++: a name written with its namespace, as in a namespace-qualified name",
}


@functools.cache
def inflected_forms(word: str) -> frozenset[str]:
    """The word and its regular inflections, by the English spelling rules: -s, or -es after s, x, z, ch and sh, and
    both after o (plural or third person: `walks`, `passes`, `echoes`); -ed and -ing, a final silent e dropped (`make`:
    `maked`, `making`); a final y after a consonant becomes -ies and -ied (`carry`: `carries`, `carried`, `carrying`);
    a final consonant after one short vowel is doubled as well as kept single (`stop`: `stopped` and `stoped`). Forms
    that are not English words are harmless: no code writes them. Irregular forms are not made (`mouse` gives no
    `mice`), so a word only spelled like an inflection of another is not caught, and a derived word (`walker`) is
    another word."""
    forms = {word}
    if len(word) >= 2 and word.endswith("y") and word[-2] not in _VOWELS:
        forms |= {word[:-1] + "ies", word[:-1] + "ied", word + "ing"}
        return frozenset(forms)
    forms.add(word + "es" if word.endswith(("s", "x", "z", "ch", "sh")) else word + "s")
    if word.endswith("o"):
        forms.add(word + "es")
    if word.endswith("e"):
        forms |= {word + "d", word[:-1] + "ing"}
        return frozenset(forms)
    forms |= {word + "ed", word + "ing"}
    ends_consonant_vowel_consonant = (
        len(word) >= 3
        and word[-1] not in _VOWELS
        and word[-1] not in "wxy"
        and word[-2] in _VOWELS
        and word[-3] not in _VOWELS
    )
    if ends_consonant_vowel_consonant:
        forms |= {word + word[-1] + "ed", word + word[-1] + "ing"}
    return frozenset(forms)


@functools.cache
def matching_last_forms(last_word: str) -> frozenset[str]:
    """The forms of the last word of a phrase that count as the phrase: the word itself, and each inflection that is
    not a word of its own."""
    return frozenset(form for form in inflected_forms(last_word) if form == last_word or form not in WORDS_OF_THEIR_OWN)


def contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    """True when `phrase` appears as consecutive items of `tokens`, its last word in any of its inflected forms."""
    width = len(phrase)
    if width == 0:
        return False
    last_forms = matching_last_forms(phrase[-1])
    for start in range(len(tokens) - width + 1):
        leading_match = tuple(tokens[start : start + width - 1]) == tuple(phrase[:-1])
        if leading_match and tokens[start + width - 1] in last_forms:
            return True
    return False


def contains_run(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    """True when `phrase` appears as consecutive items of `tokens`."""
    width = len(phrase)
    if width == 0:
        return False
    for start in range(len(tokens) - width + 1):
        if tuple(tokens[start : start + width]) == tuple(phrase):
            return True
    return False


# A path or an identifier of the old proof-of-concept code, written in backticks after the literal prefix `main:` as in
# `main:native/src/sparse_mac_grid_3d.cc`. Its words are the old code's own, so they are not read as ours.
_OLD_CODE_NAME = re.compile(r"`main:[^`\n]*`")


def comment_segments(comment_text: str) -> list[list[str]]:
    """Word lists of a comment, one list per stretch without punctuation, so phrases do not span sentences. A name of
    the old code (`main:...` in backticks) is left out."""
    segments = []
    for segment in re.split(r"[^A-Za-z0-9_\s]", _OLD_CODE_NAME.sub(" ", comment_text)):
        tokens: list[str] = []
        for run in _RUN.findall(segment):
            tokens.extend(split_name_words(run))
        if tokens:
            segments.append(tokens)
    return segments


# ---------------------------------------------------------------------------------------------------
# Glossary and AGENTS.md
# ---------------------------------------------------------------------------------------------------


def _split_cells(table_line: str) -> list[str]:
    inner = table_line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def _split_at_outer_commas(cell: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for position, character in enumerate(cell):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(depth - 1, 0)
        elif character == "," and depth == 0:
            parts.append(cell[start:position])
            start = position + 1
    parts.append(cell[start:])
    return [part.strip() for part in parts if part.strip()]


_NOTED_ONLY_IN_NAMES = "noted only in names"
_NOTED_ONLY_AS_NAME_SUFFIX = "noted only as the last word of a name"


def _word_list(text: str) -> list[str]:
    """The lowercase words of a comma list: `side, sides` is two words, `size level` is the two words of one."""
    return [word for item in text.split(",") for word in split_name_words(item)]


def _parse_sense_clauses(qualifier: str, spelled: str) -> dict[str, object]:
    """Splits the text in the parentheses of a one-sense entry into the sense and its clauses.

    `sense; allowed: a b, c d; noted only beside: x, y; quiet beside: z; quiet in files: f, g; noted only in names;
    noted only as the last word of a name`. The sense comes first and is what a note says. Every clause is optional,
    and an unknown clause is an error, so a typo cannot silently allow a word."""
    sense, *clauses = [part.strip() for part in qualifier.split(";")]
    fields: dict[str, object] = {"qualifier": sense}
    allowed: list[tuple[str, ...]] = []
    for clause in clauses:
        if clause == _NOTED_ONLY_IN_NAMES:
            fields["is_noted_only_in_names"] = True
        elif clause == _NOTED_ONLY_AS_NAME_SUFFIX:
            fields["is_noted_only_as_name_suffix"] = True
        elif clause.startswith("allowed:"):
            allowed.extend(tuple(split_name_words(phrase)) for phrase in clause[len("allowed:") :].split(","))
        elif clause.startswith("noted only beside:"):
            fields["noted_only_beside"] = frozenset(_word_list(clause[len("noted only beside:") :]))
        elif clause.startswith("quiet beside:"):
            fields["quiet_beside"] = frozenset(_word_list(clause[len("quiet beside:") :]))
        elif clause.startswith("quiet in files:"):
            fields["quiet_in_files"] = tuple(item.strip() for item in clause[len("quiet in files:") :].split(","))
        else:
            raise GlossaryError(f"'{spelled}' has an unknown clause '{clause}' in its parentheses")
    fields["allowed_phrases"] = tuple(phrase for phrase in allowed if phrase)
    return fields


def _parse_never_say_cell(cell: str) -> list[NeverSayEntry]:
    entries = []
    for raw in _split_at_outer_commas(cell):
        qualifier = None
        spelled = raw
        match = re.match(r"^(.*?)\s*\((.*)\)\s*$", raw)
        if match and match.group(1).strip():
            spelled, qualifier = match.group(1), match.group(2).strip()
        spelled = spelled.replace("`", "").strip()
        words = tuple(split_name_words(spelled))
        if not words:
            continue
        if qualifier is None:
            entries.append(NeverSayEntry(words=words, spelled=spelled, qualifier=None))
        else:
            entries.append(NeverSayEntry(words=words, spelled=spelled, **_parse_sense_clauses(qualifier, spelled)))  # type: ignore[arg-type]
    return entries


def _say_phrases(cell: str) -> list[tuple[str, ...]]:
    """The words of each phrase of a "Say" cell, `size_level` and `SizeLevel` both as (size, level). A phrase of one
    word is kept; the caller keeps only those longer than the word they allow."""
    phrases = []
    for raw in _split_at_outer_commas(cell):
        words = tuple(split_name_words(raw.replace("`", "")))
        if words:
            phrases.append(words)
    return phrases


def _with_allowed_phrases(
    entries: Sequence[NeverSayEntry], say_phrases: Sequence[tuple[str, ...]]
) -> list[NeverSayEntry]:
    """Gives each one-sense entry the Say phrases of the glossary that are longer than it and contain it."""
    completed = []
    for entry in entries:
        if entry.qualifier is None:
            completed.append(entry)
            continue
        allowed = {
            phrase for phrase in say_phrases if len(phrase) > len(entry.words) and contains_run(phrase, entry.words)
        }
        allowed.update(entry.allowed_phrases)
        completed.append(replace(entry, allowed_phrases=tuple(sorted(allowed))))
    return completed


def load_glossary(glossary_path: Path) -> list[NeverSayEntry]:
    """Reads the "Never say" column of every table. Every table must have the exact header, and there must be one.

    A glossary with no table would ban nothing, so every banned word would pass unseen: it is an error."""
    lines = glossary_path.read_text(encoding="utf-8").splitlines()
    entries: list[NeverSayEntry] = []
    say_phrases: list[tuple[str, ...]] = []
    table_count = 0
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        table_count += 1
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        table = lines[start:index]
        if _split_cells(table[0]) != GLOSSARY_HEADER_CELLS:
            raise GlossaryError(
                f"{glossary_path}:{start + 1}: table header must be '| Say | Plain meaning | Never say |',"
                f" got {table[0]!r}"
            )
        for line_number, table_line in enumerate(table[2:], start=start + 3):
            cells = _split_cells(table_line)
            if len(cells) != 3:
                raise GlossaryError(f"{glossary_path}:{line_number}: expected 3 cells, found {len(cells)}")
            entries.extend(_parse_never_say_cell(cells[2]))
            say_phrases.extend(_say_phrases(cells[0]))
    if table_count == 0:
        raise GlossaryError(f"{glossary_path}: no vocabulary table ('| Say | Plain meaning | Never say |') was found")
    return _with_allowed_phrases(entries, say_phrases)


def load_allowed_verbs(agents_path: Path) -> tuple[set[str], set[str]]:
    """Returns (verbs, exact names) from the "Allowed verbs" section of AGENTS.md.

    A function name must start with a verb at a word boundary, or equal an exact name.
    """
    text = agents_path.read_text(encoding="utf-8")
    match = re.search(r"^## Allowed verbs\s*$(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise RulesFileError(f"{agents_path}: no '## Allowed verbs' section")
    section = match.group(1)
    verbs = set(re.findall(r"^- `([A-Z][A-Za-z]*)(?:<[^>]*>)?`:", section, re.MULTILINE))
    names = set(re.findall(r"`([A-Z][A-Za-z0-9]*)`", section))
    if not verbs:
        raise RulesFileError(f"{agents_path}: the 'Allowed verbs' section lists no verbs")
    return verbs, names


def load_allowed_names(allowed_names_path: Path | None) -> set[str]:
    """Reads owner-approved names, one `Name  # reason` per line; a name without a reason is an error."""
    if allowed_names_path is None or not allowed_names_path.exists():
        return set()
    names = set()
    for number, line in enumerate(allowed_names_path.read_text(encoding="utf-8").splitlines(), start=1):
        name, _, reason = line.partition("#")
        name = name.strip()
        if not name:
            continue
        if not reason.strip():
            raise RulesFileError(f"{allowed_names_path}:{number}: '{name}' has no reason after '#'")
        names.add(name)
    return names


def find_stale_allowed_names(allowed_names: Iterable[str], paths: Iterable[Path | str]) -> list[str]:
    """The allowed names, sorted, that no C++ file under `paths` uses as an identifier, as a name declared or called
    (a name in a comment or a string does not count). An exemption for a name that was renamed or deleted would stay
    unseen and allow the next function that takes the name, so a test fails on each one."""
    remaining = set(allowed_names)
    for file_path in iter_files(Path(path) for path in paths):
        if not remaining:
            break
        if file_path.suffix not in CPP_SUFFIXES:
            continue
        text = file_path.read_bytes().decode("utf-8", errors="replace")
        for line in split_lines(file_path, text):
            remaining.difference_update(_identifiers(line.code))
    return sorted(remaining)


# ---------------------------------------------------------------------------------------------------
# Splitting a file into code and comments
# ---------------------------------------------------------------------------------------------------


@dataclass
class ScannedLine:
    number: int
    code: str  # Comments removed, string and character literal contents blanked.
    comment: str  # Text of the comment on this line, without the comment marker.


def _split_cpp(text: str) -> list[ScannedLine]:
    lines = []
    in_block_comment = False
    for number, raw in enumerate(text.splitlines(), start=1):
        code: list[str] = []
        comment: list[str] = []
        position = 0
        while position < len(raw):
            two = raw[position : position + 2]
            character = raw[position]
            if in_block_comment:
                if two == "*/":
                    in_block_comment = False
                    position += 2
                else:
                    comment.append(character)
                    position += 1
            elif two == "//":
                comment.append(raw[position + 2 :])
                break
            elif two == "/*":
                in_block_comment = True
                code.append(" ")
                position += 2
            elif character == "'" and re.search(r"(?<![A-Za-z0-9_'])[0-9][0-9A-Fa-f']*$", raw[:position]):
                code.append(character)  # A digit separator, as in 1'000, not a character literal.
                position += 1
            elif character in "\"'":
                end = position + 1
                while end < len(raw) and raw[end] != character:
                    end += 2 if raw[end] == "\\" else 1
                code.append(character + " " * max(end - position - 1, 0) + character)
                position = end + 1
            else:
                code.append(character)
                position += 1
        lines.append(ScannedLine(number, "".join(code), "".join(comment)))
    return lines


def _is_operator(token: tokenize.TokenInfo | None, text: str) -> bool:
    return token is not None and token.type == tokenize.OP and token.string == text


_LINE_START_TOKENS = {
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
    tokenize.ENCODING,
}


def _docstring_text(token: tokenize.TokenInfo) -> str:
    try:
        value = ast.literal_eval(token.string)
    except (ValueError, SyntaxError):
        return token.string
    return value if isinstance(value, str) else ""


def _split_python(text: str) -> list[ScannedLine]:
    """Names, comments and docstrings. Strings are never names, so only a string that makes up
    a whole statement (a docstring) is read, as a comment.

    Skipped as library names: an attribute that is only read (`value.method`) and a keyword argument at a
    call (`call(name=1)`). Still scanned: an attribute that is assigned (`self.cells = 1`) and the default-valued
    parameters of a `def`."""
    by_number = {number: ScannedLine(number, "", "") for number in range(1, len(text.splitlines()) + 1)}
    tokens: list[tokenize.TokenInfo] = []
    try:
        tokens.extend(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # Report what was read before the error. The Python parser reports syntax errors itself.
    depth = 0
    pending_def = False
    def_depth: int | None = None  # Bracket depth inside the parameter list of the current `def`.
    for index, token in enumerate(tokens):
        line_index = token.start[0]
        previous = tokens[index - 1] if index > 0 else None
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if line_index not in by_number:
            continue
        if token.type == tokenize.OP and token.string in ("(", "[", "{"):
            depth += 1
            if pending_def and token.string == "(":
                def_depth = depth
                pending_def = False
        elif token.type == tokenize.OP and token.string in (")", "]", "}"):
            if def_depth == depth:
                def_depth = None
            depth = max(depth - 1, 0)
        elif token.type == tokenize.COMMENT:
            by_number[line_index].comment = token.string.lstrip("#")
        elif token.type == tokenize.STRING:
            starts_statement = previous is None or previous.type in _LINE_START_TOKENS
            ends_statement = following is not None and following.type == tokenize.NEWLINE
            if starts_statement and ends_statement:
                for offset, doc_line in enumerate(_docstring_text(token).splitlines()):
                    target = by_number.get(line_index + offset)
                    if target is not None:
                        target.comment += doc_line
        elif token.type == tokenize.NAME:
            if token.string == "def":
                pending_def = True
            after_dot = _is_operator(previous, ".")
            assigned = _is_operator(following, "=")
            in_signature = def_depth is not None and depth >= def_depth
            read_attribute = after_dot and not assigned
            keyword_argument = depth > 0 and assigned and not in_signature and not after_dot
            if not read_attribute and not keyword_argument:
                by_number[line_index].code += " " + token.string
    return [by_number[number] for number in sorted(by_number)]


def _split_lean(text: str) -> list[ScannedLine]:
    lines = []
    for number, raw in enumerate(text.splitlines(), start=1):
        code, _, comment = raw.partition("--")
        lines.append(ScannedLine(number, re.sub(r'"[^"]*"', '""', code), comment))
    return lines


def split_lines(path: Path, text: str) -> list[ScannedLine]:
    if path.suffix == ".py":
        return _split_python(text)
    if path.suffix == ".lean":
        return _split_lean(text)
    return _split_cpp(text)


# ---------------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------------


_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])(?:0[xX][0-9A-Fa-f']+|0[bB][01']+|[0-9][0-9']*(?:\.[0-9']*)?(?:[eE][+-]?[0-9]+)?)[A-Za-z_]*"
)
_MEMBER_ACCESS = re.compile(r"(?<!\.)(?:\.|->)\s*([A-Za-z_]\w*)")


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _blank_standard_library_chains(code: str) -> str:
    """Blanks `std` and every name in a `std::a<...>::b` chain. Template arguments stay: they may be ours."""
    for match in list(re.finditer(r"\bstd\s*::", code)):
        code = _blank(code, match.start(), match.start() + 3)
        position = match.end()
        while True:
            name = re.compile(r"\s*([A-Za-z_]\w*)").match(code, position)
            if name is None:
                break
            code = _blank(code, name.start(1), name.end(1))
            position = name.end()
            opening = re.compile(r"\s*<").match(code, position)
            if opening is not None:
                depth = 0
                for index in range(opening.end() - 1, len(code)):
                    depth += {"<": 1, ">": -1}.get(code[index], 0)
                    if depth == 0:
                        position = index + 1
                        break
                else:
                    break
            scope = re.compile(r"\s*::").match(code, position)
            if scope is None:
                break
            position = scope.end()
    return code


def _identifiers(code: str) -> list[str]:
    """Names declared or used by us. Numbers, `.name` and `->name` member names, and `std::` chains are
    library names and are left out."""
    code = _NUMBER.sub(lambda match: " " * len(match.group(0)), code)
    for match in list(_MEMBER_ACCESS.finditer(code)):
        code = _blank(code, match.start(1), match.end(1))
    return _IDENTIFIER.findall(_blank_standard_library_chains(code))


def _is_literature_line(comment: str) -> bool:
    lowered = comment.lower()
    return any(marker in lowered for marker in LITERATURE_MARKERS)


def _is_include_line(code: str) -> bool:
    return re.match(r"\s*#\s*include\b", code) is not None


class _Reporter:
    """Collects findings and keeps the most severe one for the same line, rule group and word."""

    def __init__(self, path: Path) -> None:
        self._path = str(path)
        self.path = Path(path).as_posix()
        self._seen: dict[tuple[int, str, str], Finding] = {}
        self._order: list[tuple[int, str, str]] = []

    def add(self, line: int, severity: str, rule_id: str, text: str, key: str = "") -> None:
        group = "banned-word" if rule_id == "one-sense-word" else rule_id
        dedupe_key = (line, group, key or text)
        finding = Finding(self._path, line, severity, rule_id, text)
        existing = self._seen.get(dedupe_key)
        if existing is None:
            self._order.append(dedupe_key)
            self._seen[dedupe_key] = finding
        elif existing.severity == NOTE and severity == FAIL:
            self._seen[dedupe_key] = finding

    def findings(self) -> list[Finding]:
        return [self._seen[key] for key in self._order]


def _check_file_header(lines: list[ScannedLine], reporter: _Reporter) -> None:
    header_text = []
    for line in lines:
        if line.code.strip():
            break
        if not line.comment.strip():
            if header_text or line.comment:
                header_text.append("")
            continue
        header_text.append(line.comment.strip())
    missing = [label for label in HEADER_LABELS if not any(text.startswith(label + ":") for text in header_text)]
    if missing:
        reporter.add(1, FAIL, "file-header", "file header lacks " + ", ".join(f"'{label}:'" for label in missing))


def _entries_by_last_form(glossary: Sequence[NeverSayEntry]) -> dict[str, list[int]]:
    """For each form of the last word that counts as a phrase, the positions in `glossary` of the entries it ends.
    An entry can match a token list only when one of these forms is in it, so the others are never tried."""
    by_form: dict[str, list[int]] = {}
    for position, entry in enumerate(glossary):
        if not entry.words or entry.words in ABBREVIATIONS:
            continue
        for form in matching_last_forms(entry.words[-1]):
            by_form.setdefault(form, []).append(position)
    return by_form


# Names of library functions and macros, matched where they are written (a call, or the qualified name), so that the
# word in them is not read as ours: the mathematical floor, the standard sorts and bounds, GoogleTest's
# RecordProperty, the CUDA runtime and its qualifiers, CMake's commands and variables, and `math::Floor`, which
# the glossary keeps (the "floor" row). The same word anywhere else is still read.
_LIBRARY_NAMES = re.compile(
    "|".join(
        [
            r"(?:\bstd)?::floor\b|\bfloor\(|\bmath::Floor\b",
            r"\bstd::(?:ranges::)?(?:stable_)?sort\b|\b(?:ranges::)?(?:stable_)?sort\(",
            r"\bstd::(?:ranges::)?(?:lower|upper)_bound\b|\b(?:lower|upper)_bound\(",
            r"\bRecordProperty\(",
            r"\bcuda[A-Z]\w*|\b__(?:global|device|host|shared|constant|managed|launch_bounds|syncthreads)__",
            r"\bCMAKE_[A-Z0-9_]+\b|\btarget_\w+\(|\badd_(?:library|executable|test|custom_\w+)\(",
        ]
    )
)
# A header line that lists files: its label, and the lines under it that continue the list (indented further).
_REFERENCE_LABEL = re.compile(r"^\s*(?:Uses|Used by|Read):")
_REFERENCE_CONTINUATION = re.compile(r"^\s{4,}\S")
_HYPHEN_BETWEEN_WORDS = re.compile(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])")
_FILE_NAME = re.compile(r"[\w./+-]*\.(?:h|cuh|cc|cu|typ|md|py|txt|json|lean|cmake)\b")


def _blank_matches(pattern: re.Pattern[str], text: str) -> str:
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _has_use_outside_allowed_phrases(tokens: Sequence[str], entry: NeverSayEntry) -> bool:
    """True when `entry` occurs in `tokens` somewhere that is not inside one of its allowed phrases (the glossary's own
    longer forms, as `size level` for `level`)."""
    covered: set[int] = set()
    for phrase in entry.allowed_phrases:
        width = len(phrase)
        last_forms = matching_last_forms(phrase[-1])
        for start in range(len(tokens) - width + 1):
            if tuple(tokens[start : start + width - 1]) == phrase[:-1] and tokens[start + width - 1] in last_forms:
                covered.update(range(start, start + width))
    width = len(entry.words)
    last_forms = matching_last_forms(entry.words[-1])
    for start in range(len(tokens) - width + 1):
        if tuple(tokens[start : start + width - 1]) == entry.words[:-1] and tokens[start + width - 1] in last_forms:
            if not covered.issuperset(range(start, start + width)):
                return True
    return False


def _is_outside_the_banned_sense(entry: NeverSayEntry, tokens: Sequence[str], where: str, path_text: str) -> bool:
    """True when the glossary's clauses for a one-sense `entry` say that this use, in `tokens` (the words of one name
    or of one stretch of comment without punctuation) of a file at `path_text`, is not the banned sense."""
    if entry.is_noted_only_in_names and where != "name":
        return True
    if entry.is_noted_only_as_name_suffix:
        ends_the_name = len(tokens) > len(entry.words) and tokens[-1] in matching_last_forms(entry.words[-1])
        if where != "name" or not ends_the_name:
            return True
    if entry.noted_only_beside and entry.noted_only_beside.isdisjoint(tokens):
        return True
    if entry.quiet_beside and not entry.quiet_beside.isdisjoint(tokens):
        return True
    return any(part in path_text for part in entry.quiet_in_files)


def _check_words(lines: list[ScannedLine], glossary: Sequence[NeverSayEntry], reporter: _Reporter) -> None:
    by_last_form = _entries_by_last_form(glossary)
    in_reference_list = False
    for line in lines:
        is_comment_only = not line.code.strip()
        starts_reference_list = _REFERENCE_LABEL.match(line.comment) is not None
        in_reference_list = starts_reference_list or (
            in_reference_list and is_comment_only and _REFERENCE_CONTINUATION.match(line.comment) is not None
        )
        comment = line.comment
        if in_reference_list:
            comment = _blank_matches(_FILE_NAME, comment)
        # Each sequence says where its words stand and whether a hyphen between two words joined them: `size-level-0`
        # is read as `size level 0` for a word banned in one sense (an allowed form), and as three stretches for a word
        # banned in every sense, as before.
        sequences: list[tuple[list[str], str, bool]] = []
        if not _is_include_line(line.code):
            for identifier in _identifiers(_blank_matches(_LIBRARY_NAMES, line.code)):
                sequences.append((split_name_words(identifier), "name", False))
        if not _is_literature_line(comment):
            comment_without_library_names = _blank_matches(_LIBRARY_NAMES, comment)
            for segment in comment_segments(comment_without_library_names):
                sequences.append((segment, "comment", False))
            for segment in comment_segments(_HYPHEN_BETWEEN_WORDS.sub(" ", comment_without_library_names)):
                sequences.append((segment, "comment", True))
        for tokens, where, is_hyphen_joined in sequences:
            # The glossary order is kept, so the findings come in the same order as when every entry was tried.
            candidates = sorted({position for token in set(tokens) for position in by_last_form.get(token, ())})
            for position in candidates:
                entry = glossary[position]
                if where == "comment" and is_hyphen_joined == (entry.qualifier is None):
                    continue
                if not contains_phrase(tokens, entry.words):
                    continue
                if entry.allowed_phrases and not _has_use_outside_allowed_phrases(tokens, entry):
                    continue
                if entry.qualifier is not None and _is_outside_the_banned_sense(entry, tokens, where, reporter.path):
                    continue
                key = entry.spelled.lower()
                if entry.qualifier is None:
                    text = f"'{entry.spelled}' is banned (docs/glossary.md); found in a {where}"
                    reporter.add(line.number, FAIL, "banned-word", text, key=key)
                else:
                    text = f"'{entry.spelled}' is banned in one sense only: {entry.qualifier}"
                    reporter.add(line.number, NOTE, "one-sense-word", text, key=key)
            for abbreviation in ABBREVIATIONS:
                if not is_hyphen_joined and contains_run(tokens, abbreviation):
                    spelled = "".join(abbreviation).upper()
                    reporter.add(
                        line.number, FAIL, "abbreviation", f"'{spelled}' is an abbreviation; spell it out", key=spelled
                    )


def find_banned_words(text: str, glossary: Sequence[NeverSayEntry]) -> list[str]:
    """The glossary words banned in every sense that `text` contains, spelled as the glossary spells them, in
    glossary order. Prose is read as `_check_words` reads a comment: a phrase does not span punctuation. A word banned
    in one sense only is a NOTE for code, so it is not reported here."""
    by_last_form = _entries_by_last_form(glossary)
    found: dict[str, None] = {}
    for line in text.splitlines():
        for tokens in comment_segments(line):
            candidates = sorted({position for token in set(tokens) for position in by_last_form.get(token, ())})
            for position in candidates:
                entry = glossary[position]
                if entry.qualifier is None and contains_phrase(tokens, entry.words):
                    found.setdefault(entry.spelled, None)
    return list(found)


def _check_milestones(path: Path, lines: list[ScannedLine], reporter: _Reporter) -> None:
    for token in split_name_pieces(path.stem):
        if _MILESTONE.match(token):
            reporter.add(0, FAIL, "milestone-code", f"file name contains the milestone code '{token}'")
    for line in lines:
        names = [token for identifier in _identifiers(line.code) for token in split_name_pieces(identifier)]
        comment_without_processor_names = _PROCESSOR_NAME.sub("", line.comment)
        words = [token for run in _RUN.findall(comment_without_processor_names) for token in split_name_pieces(run)]
        for token in names + [word for word in words if word not in UNIT_TOKENS_IN_COMMENTS]:
            if _MILESTONE.match(token):
                reporter.add(line.number, FAIL, "milestone-code", f"'{token}' is a milestone code", key=token)
        mentions = [split_name_words(identifier) for identifier in _identifiers(line.code)]
        mentions.extend(comment_segments(line.comment))
        if any("qualification" in tokens for tokens in mentions):
            reporter.add(line.number, FAIL, "milestone-code", "'qualification' is project history", key="word")


_DIMENSION_WORD = re.compile(r"(?<![0-9])[23][dD](?![a-z0-9])")


def _check_dimension_suffix(lines: list[ScannedLine], reporter: _Reporter) -> None:
    for line in lines:
        for identifier in _identifiers(line.code):
            if any(_DIMENSION_WORD.search(piece) for piece in split_name_pieces(identifier)):
                text = f"'{identifier}' carries 2D or 3D; a namespace carries the context"
                reporter.add(line.number, FAIL, "dimension-suffix", text, key=identifier)


# Attributes may stand between the keyword and the name: `struct [[nodiscard]] alignas(8) Name`.
_ROLE_TYPE = re.compile(
    r"\b(?:struct|class|union|enum(?:\s+(?:class|struct))?)\s+(?:(?:\[\[.*?\]\]|alignas\s*\([^()]*\))\s*)*"
    r"([A-Za-z_]\w*)"
    r"|\busing\s+([A-Za-z_]\w*)\s*="
)


def _check_role_suffix(lines: list[ScannedLine], reporter: _Reporter) -> None:
    for line in lines:
        for match in _ROLE_TYPE.finditer(line.code):
            name = match.group(1) or match.group(2)
            suffix = next((suffix for suffix in ROLE_SUFFIXES if name.endswith(suffix)), None)
            if suffix:
                text = f"type '{name}' ends in '{suffix}'; use Inputs, Settings or Result, or name the content"
                reporter.add(line.number, FAIL, "role-suffix", text, key=name)


def _check_backend_names(lines: list[ScannedLine], allowed_names: set[str], reporter: _Reporter) -> None:
    code_text = "\n".join(line.code for line in lines)
    exempt = set(BACKEND_EXEMPT_IDENTIFIERS) | allowed_names
    for match in re.finditer(r"\benum\s+(?:class\s+|struct\s+)?RunOn\b[^{;]*\{([^}]*)\}", code_text):
        exempt.update(_identifiers(match.group(1)))
    for line in lines:
        if _is_include_line(line.code):
            continue
        code = re.sub(r"\bRunOn\s*::\s*k(?:Cpu|Gpu)\b", "RunOn", line.code)
        for identifier in _identifiers(code):
            is_cuda_keyword = identifier.startswith("__") and identifier.endswith("__")
            if identifier in exempt or is_cuda_keyword:
                continue
            words = BACKEND_WORDS.intersection(split_name_words(identifier))
            if words:
                text = f"'{identifier}' names the backend ({', '.join(sorted(words))}); use RunOn"
                reporter.add(line.number, FAIL, "backend-name", text, key=identifier)


# `const Vec3&`, `Vec3 const&` and the namespace-qualified forms such as `const fluidity::Mat3 &`.
_QUALIFIED_PREFIX = r"(?:(?:::)?\s*[A-Za-z_]\w*\s*::\s*)*(?:::\s*)?"
_SHARED_BY_REFERENCE = re.compile(
    rf"\bconst\s+{_QUALIFIED_PREFIX}(Vec3|Mat3)\s*&|(?<![\w:]){_QUALIFIED_PREFIX}(Vec3|Mat3)\s+const\s*&"
)


def _check_shared_by_reference(lines: list[ScannedLine], reporter: _Reporter) -> None:
    if not any("FLUIDITY_CPU_AND_GPU" in line.code for line in lines):
        return
    for line in lines:
        for match in _SHARED_BY_REFERENCE.finditer(line.code):
            type_name = match.group(1) or match.group(2)
            text = f"'const {type_name}&' in a file with shared functions; take {type_name} by value"
            reporter.add(line.number, FAIL, "shared-by-reference", text, key=type_name)


def _starts_with_verb(name: str, verbs: set[str]) -> bool:
    return any(name == verb or (name.startswith(verb) and name[len(verb)].isupper()) for verb in verbs)


def _drop_template_prefix(text: str) -> str | None:
    """Removes leading `template<...>` clauses. Returns None when the clause is unbalanced."""
    while text.startswith("template"):
        depth = 0
        end = None
        for position, character in enumerate(text):
            if character == "<":
                depth += 1
            elif character == ">":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
        if end is None:
            return None
        text = text[end:].strip()
    return text


@dataclass(frozen=True)
class _Declared:
    name: str  # `operator` for any operator.
    parameters: list[str]  # Text of each parameter, empty for an operator.
    is_member_definition: bool  # The name follows `::`, as in `double Grid::CellWidth() const`.


def _split_parameters(text: str) -> list[str]:
    """Parameter texts of the parenthesis that opens `text` (its first character is not included).

    `<` opens a template argument list only right after a name (`std::span<`), so a comparison in a default
    argument (`count = 1 < 2`) does not hide the commas that follow it.
    """
    open_brackets: list[str] = []
    parameters: list[str] = []
    current = ""
    previous = ""
    for character in text:
        if character in "([{":
            open_brackets.append(character)
        elif character == "<" and (previous.isalnum() or previous == "_"):
            open_brackets.append(character)
        elif character == ">" and open_brackets and open_brackets[-1] == "<":
            open_brackets.pop()
        elif character in ")]}":
            # A `<` still open here was a comparison, not a template argument list.
            while open_brackets and open_brackets[-1] == "<":
                open_brackets.pop()
            if not open_brackets and character == ")":
                break
            if open_brackets:
                open_brackets.pop()
        if character == "," and not open_brackets:
            parameters.append(current)
            current = ""
        else:
            current += character
        previous = character
    parameters.append(current)
    return [parameter.strip() for parameter in parameters if parameter.strip()]


def _declared_function(statement: str) -> _Declared | None:
    """What `statement` declares when it declares a function or a constructor, otherwise None."""
    text = re.sub(r"\[\[[^\]]*\]\]", " ", statement).strip()
    text = re.sub(r"^(?:(?:public|private|protected)\s*:\s*)+", "", text)
    text = _drop_template_prefix(text)
    # A friend function is still a function declaration, so its parameters are checked too.
    text = re.sub(r"^friend\s+", "", text or "")
    if not text or text.startswith(("using ", "typedef ", "static_assert")):
        return None
    open_paren = text.find("(")
    if open_paren < 0:
        return None
    before = text[:open_paren]
    if "=" in before or re.search(r"\b(struct|class|union|enum|namespace)\b", before):
        return None
    if re.search(r"\boperator\b", before):
        return _Declared("operator", [], False)
    names = _identifiers(before)
    if not names:
        return None
    is_member_definition = re.search(r"::\s*~?\s*[A-Za-z_]\w*\s*$", before) is not None
    return _Declared(names[-1], _split_parameters(text[open_paren + 1 :]), is_member_definition)


# A macro call at namespace scope (`TEST(Suite, Name)`, `INSTANTIATE_TEST_SUITE_P(...)`) is not a function.
_MACRO_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
# Names a test file must spell as GoogleTest, Abseil and the C runtime ask.
TEST_FRAMEWORK_NAMES = {"main", "SetUp", "TearDown", "PrintTo", "AbslStringify", "AbslHashValue"}
_BARE_BOOL_PARAMETER = re.compile(r"^(?:const\s+)?bool\b(?!\s*[&*(<])")


@dataclass(frozen=True)
class _Scope:
    # "namespace"; "class" (the braces of a struct, class or union); "other" (function definition, enum, initializer).
    kind: str
    name: str = ""


def _brace_kind(statement: str) -> _Scope:
    text = _drop_template_prefix(re.sub(r"^(?:(?:public|private|protected)\s*:\s*)+", "", statement.strip())) or ""
    namespace = re.match(r"^(?:inline\s+)?namespace\b\s*([\w:]*)\s*$", text)
    if namespace:
        return _Scope("namespace", namespace.group(1))
    if re.search(r"\benum\b", text) or "=" in text.split("(")[0]:
        return _Scope("other")
    if re.search(r"\b(struct|class|union)\b", text):
        return _Scope("class")
    return _Scope("other")


MAX_INPUTS = 4
# A declaration of a variable or a parameter with a one-letter name: a type, then the name, then what follows a name.
_TYPE_BEFORE_NAME = (
    r"(?:auto|bool|char|int|long|short|float|double|unsigned|size_t|u?int(?:8|16|32|64)_t"
    r"|std::[A-Za-z_]\w*(?:<[^;(){}]*>)?|[A-Z][A-Za-z0-9_]*(?:<[^;(){}]*>)?)"
)
_SHORT_LOCAL = re.compile(
    r"(?<![\w:.>])" + _TYPE_BEFORE_NAME + r"(?:\s+const)?\s*[&*]*\s+([A-Za-z])\s*(?=[=;,){:\[({])"
)


def _is_private_helper(statement: str, stack: list[_Scope]) -> bool:
    """True for a function in an anonymous namespace or declared `static`."""
    in_anonymous_namespace = any(scope.kind == "namespace" and not scope.name for scope in stack)
    return in_anonymous_namespace or re.match(r"(?:(?:inline|constexpr)\s+)*static\b", statement) is not None


def _note_short_parameters(declared: _Declared, line: int, reporter: _Reporter) -> None:
    for parameter in declared.parameters:
        match = re.search(r"\b([A-Za-z])\s*(?:=.*)?$", parameter)
        if match and re.search(r"[\w>&*]\s*[&*]*\s+[A-Za-z]\b", parameter.split("=")[0]):
            text = f"the parameter '{match.group(1)}' of '{declared.name}' has a one-letter name"
            reporter.add(line, NOTE, "short-local", text, key=f"{declared.name}:{match.group(1)}")


def _note_short_locals(line: ScannedLine, reporter: _Reporter) -> None:
    for match in _SHORT_LOCAL.finditer(line.code):
        reporter.add(line.number, NOTE, "short-local", f"'{match.group(1)}' is a one-letter name", key=match.group(1))


def _check_declarations(
    lines: list[ScannedLine],
    verbs: set[str] | None,
    exact_names: set[str],
    allowed_names: set[str],
    reporter: _Reporter,
    is_test_file: bool = False,
    checks_bare_bool: bool = True,
    declarations_are_private: bool = False,
) -> None:
    """Applies `verb` to free functions and `bare-bool` to every function declared at declaration scope.

    `verbs` None skips the verb rule (a private `_gpu` header) and keeps bare-bool. In a test file the verb rule is
    `test-helper-verb`, and a macro call such as `TEST(...)`, `main` and the GoogleTest hooks are not helpers.
    `many-inputs` and `short-local` are NOTEs: the first for a private free function (every one when
    `declarations_are_private`, else one in an anonymous namespace or `static`), the second for a one-letter parameter
    and, in a function body, a one-letter local variable.
    A brace-stack scan assembles each statement. Declaration scope is a namespace or the braces of a class; function
    definitions, enums and initializers are skipped. A free function is declared where every scope is a namespace.
    """
    stack: list[_Scope] = []
    statement = ""
    statement_line = 0

    def check_statement() -> None:
        declared = _declared_function(statement.strip())
        if declared is None or declared.name == "operator":
            return
        for index, parameter in enumerate(declared.parameters):
            if checks_bare_bool and _BARE_BOOL_PARAMETER.match(parameter):
                text = f"'{declared.name}' has a bool parameter; use an enum or two functions"
                reporter.add(statement_line, FAIL, "bare-bool", text, key=f"{declared.name}:{index}")
        is_free = all(scope.kind == "namespace" for scope in stack)
        _note_short_parameters(declared, statement_line, reporter)
        is_private = declarations_are_private or _is_private_helper(statement.strip(), stack)
        if is_free and is_private and not declared.is_member_definition and len(declared.parameters) > MAX_INPUTS:
            text = f"'{declared.name}' takes {len(declared.parameters)} inputs; a group of them could be a struct"
            reporter.add(statement_line, NOTE, "many-inputs", text, key=declared.name)
        in_math = any(scope.name.split("::")[-1] == "math" for scope in stack if scope.kind == "namespace")
        name = declared.name
        if verbs is None or not is_free or declared.is_member_definition or in_math:
            return
        if name in exact_names or name in allowed_names or _starts_with_verb(name, verbs):
            return
        if is_test_file and (_MACRO_NAME.fullmatch(name) or name in TEST_FRAMEWORK_NAMES):
            return
        text = (
            f"function '{name}' does not start with an allowed verb (AGENTS.md); "
            "list it with its reason in tools/check_rules_allowed_names.txt"
        )
        reporter.add(statement_line, FAIL, "test-helper-verb" if is_test_file else "verb", text, key=name)

    continues_preprocessor_line = False
    for line in lines:
        # A preprocessor line, and every line it continues with a trailing backslash, declares nothing.
        is_preprocessor_line = continues_preprocessor_line or line.code.lstrip().startswith("#")
        continues_preprocessor_line = is_preprocessor_line and line.code.rstrip().endswith("\\")
        if is_preprocessor_line:
            continue
        if any(scope.kind == "other" for scope in stack):
            _note_short_locals(line, reporter)
        for character in line.code:
            at_declaration_scope = all(scope.kind != "other" for scope in stack)
            if character == ";":
                if at_declaration_scope:
                    check_statement()
                statement = ""
            elif character == "{":
                scope = _brace_kind(statement)
                if scope.kind == "other" and at_declaration_scope:
                    check_statement()
                stack.append(scope)
                statement = ""
            elif character == "}":
                if stack:
                    stack.pop()
                statement = ""
            else:
                if not statement.strip() and not character.isspace():
                    statement_line = line.number
                statement += character
        statement += " "


# The path of an include, written with quotes or with angle brackets.
_INCLUDED_PATH = re.compile(r'^\s*#\s*include\s*(?:"([^"]+)"|<([^>]+)>)')


def _is_test_including_own_testing_folder(path: Path, including_folder: str, included: Path) -> bool:
    """True when `path` is a test (`.test.` in its name, a .cc or .cu file) and `included` is a
    `testing/<name>_runner_gpu` header in the `testing/` folder directly under the folder of `path`, such as
    `fluidity/grid/testing` for a test in `.../fluidity/grid`."""
    if ".test." not in path.name or path.suffix not in {".cc", ".cu"}:
        return False
    if not included.stem.endswith("_runner_gpu"):
        return False
    included_folder = included.parent.as_posix()
    parent_of_included, _, last_part = included_folder.rpartition("/")
    if last_part != TEST_CODE_FOLDER or parent_of_included == "":
        return False
    return including_folder.endswith("/" + parent_of_included)


def _is_test_of_module_under_tests(path: Path) -> bool:
    """True for a test (`.test.` in its name, a .cc or .cu file) in `tests/<module>/`."""
    if ".test." not in path.name or path.suffix not in {".cc", ".cu"}:
        return False
    return path.resolve().parent.parent.name == TESTS_FOLDER


def _check_private_gpu_includes(path: Path, text: str, lines: list[ScannedLine], reporter: _Reporter) -> None:
    """Fails an include of a `*_gpu.h` or `*_gpu.cuh` header from another folder, or from a header that is not
    itself `_gpu`.

    ScannedLine.code blanks string contents, so the include path is read from the raw line of an include line.
    """
    raw_lines = text.splitlines()
    including_is_public_header = path.suffix in HEADER_SUFFIXES and not is_private_gpu_stem(path.stem)
    including_folder = path.resolve().parent.as_posix()
    if _is_test_of_module_under_tests(path):
        # The folder of the module's headers: a test of tests/<module>/ stands where a test beside them would.
        including_folder = f"/fluidity/{path.resolve().parent.name}"
    for line in lines:
        if not _is_include_line(line.code) or line.number > len(raw_lines):
            continue
        match = _INCLUDED_PATH.match(raw_lines[line.number - 1])
        if not match:
            continue
        included = Path(match.group(1) or match.group(2))
        if included.suffix not in HEADER_SUFFIXES or not is_private_gpu_stem(included.stem):
            continue
        included_folder = included.parent.as_posix()
        same_folder = included_folder in ("", ".") or including_folder.endswith("/" + included_folder)
        if not same_folder and _is_test_including_own_testing_folder(path, including_folder, included):
            same_folder = True
        if not same_folder:
            text = f"'{included}' is private to its folder; only files of that folder include it"
            reporter.add(line.number, FAIL, "private-gpu-include", text, key=str(included))
        elif including_is_public_header:
            text = f"'{included}' is private; a header that is not itself _gpu must not include it"
            reporter.add(line.number, FAIL, "private-gpu-include", text, key=str(included))


# The predefined macros and operators that differ between the build compiler and the clang-tidy preprocessor:
# `__clang_analyzer__` is defined by clang-tidy alone, and `__has_include` can answer differently for a standard header.
_COMPILER_MACROS = re.compile(
    r"\b(?:__clang\w*__|__GNUC\w*__|_MSC_\w+|__apple_build_version__|__has_include(?:_next)?|_LIBCPP_VERSION|__GLIBCXX__)\b"
)
_CONDITION_OPENER = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef)\b")
_CONDITION_BRANCH = re.compile(r"^\s*#\s*(?:elif|elifdef|elifndef)\b")
_CONDITION_ELSE = re.compile(r"^\s*#\s*else\b")
_CONDITION_CLOSER = re.compile(r"^\s*#\s*endif\b")


_DEFINED_MACRO = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)(.*)$")


def _check_compiler_includes(lines: list[ScannedLine], reporter: _Reporter) -> None:
    """Fails an `#include` inside a conditional block whose condition names a compiler macro, or a macro that the file
    defines under such a condition or from one (`#define FLUIDITY_IS_CLANG 1` inside `#if defined(__clang__)`).

    Each open block is one flag: True once a condition of the block (the `#if` or any `#elif`) names a compiler
    macro. The `#else` branch of such a block depends on the compiler as well, so the flag stays on for it."""
    open_blocks: list[bool] = []
    dependent_macros: set[str] = set()

    def names_compiler(code: str) -> bool:
        if _COMPILER_MACROS.search(code):
            return True
        return any(re.search(rf"\b{re.escape(name)}\b", code) for name in dependent_macros)

    for line in lines:
        code = line.code
        defined = _DEFINED_MACRO.match(code)
        if defined and (any(open_blocks) or names_compiler(defined.group(2))):
            dependent_macros.add(defined.group(1))
        if _CONDITION_OPENER.match(code):
            open_blocks.append(names_compiler(code))
        elif _CONDITION_BRANCH.match(code):
            if open_blocks and names_compiler(code):
                open_blocks[-1] = True
        elif _CONDITION_CLOSER.match(code):
            if open_blocks:
                open_blocks.pop()
        elif _is_include_line(code) and any(open_blocks):
            text = "an #include under a condition on a compiler macro; the header list of the tidy cache would differ"
            reporter.add(line.number, FAIL, "compiler-include", text)


# A comment line whose whole text is one word, as `// Done` or `// ok.`.
_ONE_WORD_COMMENT = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*[.!:;,?]*")
# The one word of such a line that is not a comment about the code: the end of a namespace, a NOLINT marker, and
# `// fallthrough`, which GCC's -Wimplicit-fallthrough reads.
_ONE_WORD_COMMENT_EXEMPT_PREFIXES = ("NOLINT", "namespace", "fallthrough")


def _check_one_word_comments(lines: list[ScannedLine], reporter: _Reporter) -> None:
    """Fails a `//` comment line of one word. A line beside another comment line continues a sentence or a list, so it
    is left alone; `NOLINT...` and the end of a namespace are exempt."""

    def is_comment_only(index: int) -> bool:
        return 0 <= index < len(lines) and not lines[index].code.strip() and bool(lines[index].comment.strip())

    for index, line in enumerate(lines):
        if not is_comment_only(index) or is_comment_only(index - 1) or is_comment_only(index + 1):
            continue
        word = line.comment.strip()
        if _ONE_WORD_COMMENT.fullmatch(word) and not word.startswith(_ONE_WORD_COMMENT_EXEMPT_PREFIXES):
            reporter.add(line.number, FAIL, "one-word-comment", f"the comment '{word}' is one word; say more or delete it")


PAPER_LINE_LIMIT = 120
# Paper lines over the limit that cannot be rewrapped, each with its reason: a Typst content block `[...]` must
# start right after the `)` of its call, so a long URL and the first word of the block share a line.
PAPER_LONG_LINE_EXEMPTIONS = (
    ("references.typ", "Huang_Neural_Kernel_Surface_Reconstruction_CVPR_2023_paper.pdf", "one URL longer than the limit"),
)


def _check_paper_line_length(path: Path, text: str, reporter: _Reporter) -> None:
    """Fails a line of a method-paper Typst file over PAPER_LINE_LIMIT columns, unless PAPER_LONG_LINE_EXEMPTIONS
    lists it."""
    for number, line in enumerate(text.splitlines(), start=1):
        columns = display_columns(line)
        if columns <= PAPER_LINE_LIMIT:
            continue
        if any(path.name == name and marker in line for name, marker, _reason in PAPER_LONG_LINE_EXEMPTIONS):
            continue
        reporter.add(number, FAIL, "paper-line-length", f"the line has {columns} columns; the limit is {PAPER_LINE_LIMIT}")


# Words of a comment that name the process that made the code, not what the code does.
# `round` (rounding a number) is an ordinary word here, as in the message check; `task` is one too (a VS Code task), so
# it counts only when a number follows (`Task 12`).
PROCESS_COMMENT_WORDS = frozenset({"task", "brief", "review", "reviewer", "plan", "ruling"})
_NUMBERED_TASK = re.compile(r"\btask\s+[0-9]", re.IGNORECASE)
_COMMENT_WORD = re.compile(r"[A-Za-z]+")


def _note_process_words(lines: list[ScannedLine], reporter: _Reporter) -> None:
    """Notes a process word in a comment; `Task 12` and `Ruling 5` are caught by the word itself."""
    for line in lines:
        for word in dict.fromkeys(word.lower() for word in _COMMENT_WORD.findall(line.comment)):
            if word == "task" and not _NUMBERED_TASK.search(line.comment):
                continue
            if word in PROCESS_COMMENT_WORDS:
                text = f"'{word}' in a comment names how the work was done; say what the code does"
                reporter.add(line.number, NOTE, "process-word", text, key=word)


_TEST_MACRO_START = re.compile(r"^\s*(?:TEST|TEST_F|TEST_P)\s*\(\s*\w+\s*,\s*(\w+)")
_TEST_CHECK = re.compile(
    r"\b(?:EXPECT|ASSERT)_\w+|\b(?:Check|Expect)\w*\s*\(|\bSUCCEED\b|\bFAIL\b|\bGTEST_SKIP\b|\bADD_FAILURE\w*"
)


def _note_test_shape(lines: list[ScannedLine], reporter: _Reporter) -> None:
    """Notes `And` in the name of a test, and a test body with no check. A body ends where its first brace closes."""
    index = 0
    while index < len(lines):
        start = _TEST_MACRO_START.match(lines[index].code)
        if not start:
            index += 1
            continue
        if "And" in split_name_pieces(start.group(1)):
            reporter.add(lines[index].number, NOTE, "test-name-and", f"the test name '{start.group(1)}' says And")
        depth = 0
        opened = False
        body = ""
        end = index
        while end < len(lines):
            for character in lines[end].code:
                depth += character == "{"
                depth -= character == "}"
                opened = opened or character == "{"
            body += lines[end].code + "\n"
            if opened and depth <= 0:
                break
            end += 1
        if opened and not _TEST_CHECK.search(body):
            reporter.add(lines[index].number, NOTE, "test-without-check", f"the test '{start.group(1)}' has no check")
        index = max(end, index) + 1


def _parts_in_repository(path: Path, repository: Path) -> tuple[str, ...]:
    """The folders and the name of `path` below `repository`, so that the place of a clone (a folder named `tests` above
    it) decides nothing. A path outside `repository` keeps all its parts."""
    for candidate, root in ((path, repository), (path.resolve(), repository.resolve())):
        try:
            return candidate.relative_to(root).parts
        except ValueError:
            continue
    return path.parts


def _is_test_file(path: Path, parts: tuple[str, ...]) -> bool:
    """A test file has `.test.` in its name, or lies in a `testing` folder or a `tests` folder; `parts` are the parts of
    `path` below the repository (see _parts_in_repository)."""
    return ".test." in path.name or TEST_CODE_FOLDER in parts[:-1] or TESTS_FOLDER in parts[:-1]


def _is_under_source_folders(parts: tuple[str, ...]) -> bool:
    return "src" in parts or TESTS_FOLDER in parts


def check_text(
    path: Path,
    text: str,
    glossary: Sequence[NeverSayEntry],
    verbs: set[str],
    exact_names: set[str],
    allowed_names: set[str],
    repository: Path = REPO_ROOT,
) -> list[Finding]:
    """Returns the findings for one file, ordered by line. `repository` is the root that the folders `src`, `tests` and
    `testing` are read from: a folder of that name above it does not count."""
    lines = split_lines(path, text)
    parts = _parts_in_repository(path, repository)
    reporter = _Reporter(path)
    suffix = path.suffix
    if suffix == ".typ":
        _check_paper_line_length(path, text, reporter)
        return sorted(reporter.findings(), key=lambda finding: (finding.line, finding.rule_id))
    # Scope of the rules (see the module docstring): file-header, bare-bool and verb look at public
    # declarations, so they apply to .h and .cuh only. backend-name applies to .h only, where names are public
    # and shared; a .cuh file may name its backend, and so may a test file.
    if suffix in HEADER_SUFFIXES:
        _check_file_header(lines, reporter)
    _check_words(lines, glossary, reporter)
    _note_process_words(lines, reporter)
    _check_milestones(path, lines, reporter)
    _check_dimension_suffix(lines, reporter)
    is_private_gpu_file = is_private_gpu_stem(path.stem)
    if suffix in CPP_SUFFIXES:
        _check_role_suffix(lines, reporter)
        _check_shared_by_reference(lines, reporter)
        _check_private_gpu_includes(path, text, lines, reporter)
        _check_compiler_includes(lines, reporter)
        _note_test_shape(lines, reporter)
        if _is_under_source_folders(parts):
            _check_one_word_comments(lines, reporter)
    is_test_file = _is_test_file(path, parts)
    if suffix in HEADER_SUFFIXES:
        _check_declarations(
            lines,
            None if is_private_gpu_file else verbs,
            exact_names,
            allowed_names,
            reporter,
            is_test_file,
            declarations_are_private=is_test_file or is_private_gpu_file,
        )
    elif suffix in CPP_SUFFIXES:
        _check_declarations(
            lines,
            verbs if is_test_file and not is_private_gpu_file else None,
            exact_names,
            allowed_names,
            reporter,
            is_test_file,
            checks_bare_bool=False,
            declarations_are_private=is_test_file or is_private_gpu_file,
        )
    if suffix == ".h" and not is_private_gpu_file and not is_test_file:
        _check_backend_names(lines, allowed_names, reporter)
    return sorted(reporter.findings(), key=lambda finding: (finding.line, finding.rule_id))


# ---------------------------------------------------------------------------------------------------
# Walking and the command line
# ---------------------------------------------------------------------------------------------------


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yields scanned files under `paths` in sorted order, never entering SKIPPED_DIRECTORY_NAMES."""
    for root in paths:
        if root.is_file():
            if root.suffix in SCANNED_SUFFIXES:
                yield root
            continue
        stack = [root]
        while stack:
            directory = stack.pop()
            entries = sorted(directory.iterdir())
            for child in entries:
                if child.is_file() and child.suffix in SCANNED_SUFFIXES:
                    yield child
            for child in reversed(entries):
                if child.is_dir() and child.name not in SKIPPED_DIRECTORY_NAMES:
                    stack.append(child)


def check_paths(
    paths: Iterable[Path | str],
    glossary_path: Path | str = DEFAULT_GLOSSARY_PATH,
    allowed_names_path: Path | str | None = DEFAULT_ALLOWED_NAMES_PATH,
    agents_path: Path | str = DEFAULT_AGENTS_PATH,
    repository: Path | str = REPO_ROOT,
) -> list[Finding]:
    """Checks every scanned file under `paths` and returns the findings in file and line order. `repository` is the
    root that the folders of a file are read from (see check_text)."""
    glossary = load_glossary(Path(glossary_path))
    verbs, exact_names = load_allowed_verbs(Path(agents_path))
    allowed_names = load_allowed_names(Path(allowed_names_path) if allowed_names_path else None)
    file_paths = list(iter_files(Path(path) for path in paths))
    context = (glossary, verbs, exact_names, allowed_names, Path(repository))
    if len(file_paths) >= PARALLEL_MIN_FILES and "fork" in multiprocessing.get_all_start_methods():
        return _check_files_in_processes(file_paths, context)
    findings: list[Finding] = []
    for file_path in file_paths:
        findings.extend(_check_file(file_path, context))
    return findings


# What a worker process needs to check a file: the glossary, the allowed verbs, the exact names, the allowed names
# and the repository root. A forked worker inherits it, so nothing is sent through a pipe but the file path and its findings.
_WorkerContext = tuple[Sequence[NeverSayEntry], set[str], set[str], set[str], Path]
_worker_context: _WorkerContext | None = None


def _check_file(file_path: Path, context: _WorkerContext) -> list[Finding]:
    """The findings for one file. Each file is checked on its own text, so the order of checking changes nothing."""
    glossary, verbs, exact_names, allowed_names, repository = context
    text = file_path.read_bytes().decode("utf-8", errors="replace")
    return check_text(file_path, text, glossary, verbs, exact_names, allowed_names, repository)


def _check_file_in_worker(file_path: Path) -> list[Finding]:
    assert _worker_context is not None
    return _check_file(file_path, _worker_context)


def _check_files_in_processes(file_paths: Sequence[Path], context: _WorkerContext) -> list[Finding]:
    """Checks the files in one forked process per core and returns the findings in the order of `file_paths`."""
    global _worker_context
    _worker_context = context
    try:
        workers = min(len(file_paths), os.cpu_count() or 1)
        # A worker killed by a signal or by the out-of-memory killer raises BrokenProcessPool here; a Pool would wait for
        # it forever.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as executor:
            per_file = list(executor.map(_check_file_in_worker, file_paths, chunksize=4))
    finally:
        _worker_context = None
    return [finding for findings in per_file for finding in findings]


# A hunk header of a diff with no context lines: `@@ -<old> +<first>[,<count>] @@`.
_HUNK_HEADER = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def parse_added_lines(diff_text: str) -> dict[str, set[int]]:
    """The lines each file gains in a `git diff --unified=0` text, by repository-relative path, 1-based."""
    added: dict[str, set[int]] = {}
    current: set[int] | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            current = added.setdefault(target[2:], set()) if target.startswith("b/") else None
            continue
        match = _HUNK_HEADER.match(line)
        if match and current is not None:
            first = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            current.update(range(first, first + count))
    return {path: lines for path, lines in added.items() if lines}


def staged_added_lines(repository: Path) -> dict[str, set[int]] | None:
    """The lines the staged changes of `repository` add, or None when git fails."""
    try:
        completed = subprocess.run(
            ["git", "diff", "--cached", "--unified=0", "--no-color", "--no-ext-diff"],
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return parse_added_lines(completed.stdout)


def added_lines_since(repository: Path, commit: str) -> dict[str, set[int]] | None:
    """The lines the working tree of `repository` adds to `commit`: committed, staged and unstaged changes, and every
    line of an untracked file. None when git fails, for example when `commit` does not exist."""

    def run_git(arguments: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        return completed.stdout if completed.returncode == 0 else None

    diff = run_git(["diff", "--unified=0", "--no-color", "--no-ext-diff", commit])
    untracked = run_git(["ls-files", "--others", "--exclude-standard", "-z"])
    if diff is None or untracked is None:
        return None
    added = parse_added_lines(diff)
    for relative in filter(None, untracked.split("\0")):
        try:
            count = len((repository / relative).read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
        if count:
            added[relative] = set(range(1, count + 1))
    return added


def notes_as_records(findings: list[Finding], repository: Path) -> list[dict[str, object]]:
    """The NOTE findings as `{path, line, rule, text, source_line}`, path relative to `repository`, sorted by path,
    line and rule. `source_line` is the text of the line in the file as it is now."""
    root = repository.resolve()
    records = []
    for finding in findings:
        if finding.severity != NOTE:
            continue
        absolute = Path(finding.path).resolve()
        try:
            relative = absolute.relative_to(root).as_posix()
            source_lines = absolute.read_text(encoding="utf-8", errors="replace").splitlines()
        except (ValueError, OSError):
            continue
        source_line = source_lines[finding.line - 1] if 0 < finding.line <= len(source_lines) else ""
        records.append(
            {
                "path": relative,
                "line": finding.line,
                "rule": finding.rule_id,
                "text": finding.text,
                "source_line": source_line,
            }
        )
    return sorted(records, key=lambda record: (record["path"], record["line"], record["rule"]))


def keep_notes_on_lines(findings: list[Finding], added_lines: dict[str, set[int]], repository: Path) -> list[Finding]:
    """Keeps every FAIL, and a NOTE only when its line is in `added_lines` (repository-relative paths).

    A note asks a person to look at a word; on a line the commit does not touch it was already looked at, so printing
    it again only buries the new ones."""
    root = repository.resolve()
    kept = []
    for finding in findings:
        if finding.severity != NOTE:
            kept.append(finding)
            continue
        try:
            relative = Path(finding.path).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if finding.line in added_lines.get(relative, set()):
            kept.append(finding)
    return kept


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the mechanical AGENTS.md checks to files.")
    parser.add_argument(
        "paths", nargs="*", help=f"files or directories (default: {' and '.join(DEFAULT_SCAN_PATHS)} in the repository root)"
    )
    parser.add_argument("--glossary", default=str(DEFAULT_GLOSSARY_PATH), help="path of docs/glossary.md")
    parser.add_argument(
        "--allowed-names",
        default=str(DEFAULT_ALLOWED_NAMES_PATH),
        help="path of the owner-approved names (tools/check_rules_allowed_names.txt)",
    )
    parser.add_argument(
        "--notes-on-staged-lines",
        action="store_true",
        help="print a NOTE only for a line that the staged changes add (FAIL lines are always printed)",
    )
    parser.add_argument(
        "--notes-since",
        metavar="COMMIT",
        help="print a NOTE only for a line that the working tree adds to COMMIT (committed or not)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="with --notes-since: write the notes as a JSON list of {path, line, rule, text, source_line} and print "
        "their count; FAIL lines are not printed and do not change the exit code",
    )
    parser.add_argument(
        "--also-notes-since",
        metavar="COMMIT",
        help="with --also-notes-json: also write the notes of the lines the working tree adds to COMMIT, from the same scan",
    )
    parser.add_argument(
        "--also-notes-json",
        metavar="PATH",
        help="with --also-notes-since: where to write those notes (the form of --json); the usual output is unchanged",
    )
    parser.add_argument(
        "--repository", default=str(REPO_ROOT), help="the git repository whose changes are read"
    )
    arguments = parser.parse_args(argv)
    default_paths = [REPO_ROOT / name for name in DEFAULT_SCAN_PATHS if (REPO_ROOT / name).exists()]
    paths = [Path(path) for path in arguments.paths] or default_paths
    missing = [path for path in paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"{path}: no such file or directory", file=sys.stderr)
        return 2
    if not any(True for _ in iter_files(paths)):
        listed = ", ".join(str(path) for path in paths)
        print(f"nothing was scanned: no {' '.join(sorted(SCANNED_SUFFIXES))} files under {listed}", file=sys.stderr)
        return 2
    try:
        findings = check_paths(
            paths, Path(arguments.glossary), Path(arguments.allowed_names), repository=Path(arguments.repository)
        )
    except (GlossaryError, RulesFileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if arguments.json and not arguments.notes_since:
        print("error: --json needs --notes-since", file=sys.stderr)
        return 2
    if bool(arguments.also_notes_since) != bool(arguments.also_notes_json):
        print("error: --also-notes-since and --also-notes-json go together", file=sys.stderr)
        return 2
    if arguments.also_notes_since:
        also_repository = Path(arguments.repository)
        also_added_lines = added_lines_since(also_repository, arguments.also_notes_since)
        if also_added_lines is None:
            print(
                f"error: git could not read the changes of {also_repository} since {arguments.also_notes_since}",
                file=sys.stderr,
            )
            return 2
        also_records = notes_as_records(keep_notes_on_lines(findings, also_added_lines, also_repository), also_repository)
        Path(arguments.also_notes_json).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.also_notes_json).write_text(json.dumps(also_records, indent=2) + "\n", encoding="utf-8")
    if arguments.notes_since:
        repository = Path(arguments.repository)
        added_lines = added_lines_since(repository, arguments.notes_since)
        if added_lines is None:
            print(f"error: git could not read the changes of {repository} since {arguments.notes_since}", file=sys.stderr)
            return 2
        findings = keep_notes_on_lines(findings, added_lines, repository)
        if arguments.json:
            records = notes_as_records(findings, repository)
            Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
            Path(arguments.json).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            print(f"{len(records)} doubtful lines, written to {arguments.json}")
            return 0
    elif arguments.notes_on_staged_lines:
        repository = Path(arguments.repository)
        added_lines = staged_added_lines(repository)
        if added_lines is None:
            print(f"error: git could not read the staged changes of {repository}", file=sys.stderr)
            return 2
        findings = keep_notes_on_lines(findings, added_lines, repository)
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.severity} {finding.rule_id} {finding.text}")
    return 1 if any(finding.severity == FAIL for finding in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
