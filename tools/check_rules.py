"""Applies the mechanical items of the AGENTS.md checklist to files.

Usage: uv run python tools/check_rules.py [PATH ...] [--glossary PATH] [--allowed-names PATH]
                                         [--notes-on-staged-lines [--repository PATH]]   (default path: src)

Output lines are `path:line: FAIL <rule-id> <text>` and `path:line: NOTE <rule-id> <text>`.
The exit code is 1 when any FAIL was printed, 2 when an input is missing or malformed, otherwise 0.
Notes never change the exit code. With --notes-on-staged-lines a NOTE is printed only for a line that the staged
changes of the repository add (`git diff --cached`), so a commit shows the notes of its own lines; FAIL lines are
always printed.

The vocabulary comes from docs/glossary.md (only the "Never say" column is read) and the allowed verbs
come from the "Allowed verbs" section of AGENTS.md. Words are matched on tokens, so `cut` does not match
`execute`. The last word of a banned entry also matches its regular inflections (`inflected_forms`), so a banned
verb catches its -s, -ed and -ing forms and a banned noun its plural; a longer word that only starts with it, an
irregular form and the few inflected forms that are words of their own here (`WORDS_OF_THEIR_OWN`) do not match.
String literals are never scanned; identifiers, comments and Python docstrings are.

Which rules apply to which files (a deliberate choice, written down here):
- `file-header`, `bare-bool` and `verb` apply to `.h` and `.cuh` files, where the public declarations are.
- `backend-name` applies to `.h` files only, because a `.cuh` file is allowed to name its backend.
  Names listed in the allowed-names file are exempt from it, as from `verb`.
- `role-suffix` applies to every C++ file; the word rules and `milestone-code` apply to every scanned file.
- `shared-by-reference` applies to C++ files that contain FLUIDITY_CPU_AND_GPU: they take Vec3 and Mat3 by value,
  because GPU code cannot bind a reference to a namespace-scope constexpr constant such as kIdentity.
- A file whose stem ends in `_gpu` (`buffer_gpu.h`, `buffer_gpu.cu`), but not in `_and_gpu`, is private to its
  folder: `backend-name` and `verb` do not apply to it, because its names wrap GPU calls. `private-gpu-include`
  applies to every C++ file: an `#include` of a `*_gpu.h` or `*_gpu.cuh` header, written with quotes or angle
  brackets, fails from another folder, or from a header whose stem does not end in `_gpu`. One exception: a test
  (a file with `.test.` in its name) may include a private GPU header of the `testing/` folder of its own folder, as
  `fluidity/grid/testing/cell_size_rule_runner_gpu.h` from `fluidity/grid/cell_size_rule.gpu.test.cc`.
"""

from __future__ import annotations

import argparse
import ast
import functools
import io
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOSSARY_PATH = REPO_ROOT / "docs" / "glossary.md"
DEFAULT_AGENTS_PATH = REPO_ROOT / "AGENTS.md"
DEFAULT_ALLOWED_NAMES_PATH = Path(__file__).resolve().parent / "check_rules_allowed_names.txt"
DEFAULT_SCAN_PATH = "src"

SCANNED_SUFFIXES = {".h", ".cuh", ".cc", ".cu", ".py", ".lean"}
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
# `cpu_and_gpu.h` ends in `_gpu` but is the shared mark every shared header includes, not a private GPU header.
SHARED_STEM_SUFFIX = "_and_gpu"


def is_private_gpu_stem(stem: str) -> bool:
    """True for `buffer_gpu`, false for `cpu_and_gpu` and for `gpu_launch`."""
    return stem.endswith(PRIVATE_GPU_STEM_SUFFIX) and not stem.endswith(SHARED_STEM_SUFFIX)


# Lowercase `m2` and `m3` in a comment are units (square and cubic metres), not milestone codes.
UNIT_TOKENS_IN_COMMENTS = {"m2", "m3"}

FAIL = "FAIL"
NOTE = "NOTE"


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


def comment_segments(comment_text: str) -> list[list[str]]:
    """Word lists of a comment, one list per stretch without punctuation, so phrases do not span sentences."""
    segments = []
    for segment in re.split(r"[^A-Za-z0-9_\s]", comment_text):
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
        if words:
            entries.append(NeverSayEntry(words=words, spelled=spelled, qualifier=qualifier))
    return entries


def load_glossary(glossary_path: Path) -> list[NeverSayEntry]:
    """Reads the "Never say" column of every table. Every table must have the exact header, and there must be one.

    A glossary with no table would ban nothing, so every banned word would pass unseen: it is an error."""
    lines = glossary_path.read_text(encoding="utf-8").splitlines()
    entries: list[NeverSayEntry] = []
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
    if table_count == 0:
        raise GlossaryError(f"{glossary_path}: no vocabulary table ('| Say | Plain meaning | Never say |') was found")
    return entries


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


def _check_words(lines: list[ScannedLine], glossary: Sequence[NeverSayEntry], reporter: _Reporter) -> None:
    by_last_form = _entries_by_last_form(glossary)
    for line in lines:
        sequences: list[tuple[list[str], str]] = []
        if not _is_include_line(line.code):
            for identifier in _identifiers(line.code):
                sequences.append((split_name_words(identifier), "name"))
        if not _is_literature_line(line.comment):
            for segment in comment_segments(line.comment):
                sequences.append((segment, "comment"))
        for tokens, where in sequences:
            # The glossary order is kept, so the findings come in the same order as when every entry was tried.
            candidates = sorted({position for token in set(tokens) for position in by_last_form.get(token, ())})
            for position in candidates:
                entry = glossary[position]
                if not contains_phrase(tokens, entry.words):
                    continue
                key = entry.spelled.lower()
                if entry.qualifier is None:
                    text = f"'{entry.spelled}' is banned (docs/glossary.md); found in a {where}"
                    reporter.add(line.number, FAIL, "banned-word", text, key=key)
                else:
                    text = f"'{entry.spelled}' is banned in one sense only: {entry.qualifier}"
                    reporter.add(line.number, NOTE, "one-sense-word", text, key=key)
            for abbreviation in ABBREVIATIONS:
                if contains_run(tokens, abbreviation):
                    spelled = "".join(abbreviation).upper()
                    reporter.add(
                        line.number, FAIL, "abbreviation", f"'{spelled}' is an abbreviation; spell it out", key=spelled
                    )


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


def _check_declarations(
    lines: list[ScannedLine],
    verbs: set[str] | None,
    exact_names: set[str],
    allowed_names: set[str],
    reporter: _Reporter,
) -> None:
    """Applies `verb` to free functions and `bare-bool` to every function declared at declaration scope.

    `verbs` None skips the verb rule (a private `_gpu` header) and keeps bare-bool.
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
            if _BARE_BOOL_PARAMETER.match(parameter):
                text = f"'{declared.name}' has a bool parameter; use an enum or two functions"
                reporter.add(statement_line, FAIL, "bare-bool", text, key=f"{declared.name}:{index}")
        is_free = all(scope.kind == "namespace" for scope in stack)
        in_math = any(scope.name.split("::")[-1] == "math" for scope in stack if scope.kind == "namespace")
        name = declared.name
        if verbs is None or not is_free or declared.is_member_definition or in_math:
            return
        if name in exact_names or name in allowed_names or _starts_with_verb(name, verbs):
            return
        text = (
            f"function '{name}' does not start with an allowed verb (AGENTS.md); "
            "list it with its reason in tools/check_rules_allowed_names.txt"
        )
        reporter.add(statement_line, FAIL, "verb", text, key=name)

    continues_preprocessor_line = False
    for line in lines:
        # A preprocessor line, and every line it continues with a trailing backslash, declares nothing.
        is_preprocessor_line = continues_preprocessor_line or line.code.lstrip().startswith("#")
        continues_preprocessor_line = is_preprocessor_line and line.code.rstrip().endswith("\\")
        if is_preprocessor_line:
            continue
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


def _is_test_including_own_testing_folder(path: Path, including_folder: str, included_folder: str) -> bool:
    """True when `path` is a test (`.test.` in its name) and `included_folder` is the `testing/` folder directly under
    the folder of `path`, such as `fluidity/grid/testing` for a test in `.../fluidity/grid`."""
    if ".test." not in path.name:
        return False
    parent_of_included, _, last_part = included_folder.rpartition("/")
    if last_part != TEST_CODE_FOLDER or parent_of_included == "":
        return False
    return including_folder.endswith("/" + parent_of_included)


def _check_private_gpu_includes(path: Path, text: str, lines: list[ScannedLine], reporter: _Reporter) -> None:
    """Fails an include of a `*_gpu.h` or `*_gpu.cuh` header from another folder, or from a header that is not
    itself `_gpu`.

    ScannedLine.code blanks string contents, so the include path is read from the raw line of an include line.
    """
    raw_lines = text.splitlines()
    including_is_public_header = path.suffix in HEADER_SUFFIXES and not is_private_gpu_stem(path.stem)
    including_folder = path.resolve().parent.as_posix()
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
        if not same_folder and _is_test_including_own_testing_folder(path, including_folder, included_folder):
            same_folder = True
        if not same_folder:
            text = f"'{included}' is private to its folder; only files of that folder include it"
            reporter.add(line.number, FAIL, "private-gpu-include", text, key=str(included))
        elif including_is_public_header:
            text = f"'{included}' is private; a header that is not itself _gpu must not include it"
            reporter.add(line.number, FAIL, "private-gpu-include", text, key=str(included))


def check_text(
    path: Path,
    text: str,
    glossary: Sequence[NeverSayEntry],
    verbs: set[str],
    exact_names: set[str],
    allowed_names: set[str],
) -> list[Finding]:
    """Returns the findings for one file, ordered by line."""
    lines = split_lines(path, text)
    reporter = _Reporter(path)
    suffix = path.suffix
    # Scope of the rules (see the module docstring): file-header, bare-bool and verb look at public
    # declarations, so they apply to .h and .cuh only. backend-name applies to .h only, where names are public
    # and shared; a .cuh file may name its backend.
    if suffix in HEADER_SUFFIXES:
        _check_file_header(lines, reporter)
    _check_words(lines, glossary, reporter)
    _check_milestones(path, lines, reporter)
    _check_dimension_suffix(lines, reporter)
    is_private_gpu_file = is_private_gpu_stem(path.stem)
    if suffix in CPP_SUFFIXES:
        _check_role_suffix(lines, reporter)
        _check_shared_by_reference(lines, reporter)
        _check_private_gpu_includes(path, text, lines, reporter)
    if suffix in HEADER_SUFFIXES:
        _check_declarations(lines, None if is_private_gpu_file else verbs, exact_names, allowed_names, reporter)
    if suffix == ".h" and not is_private_gpu_file:
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
) -> list[Finding]:
    """Checks every scanned file under `paths` and returns the findings in file and line order."""
    glossary = load_glossary(Path(glossary_path))
    verbs, exact_names = load_allowed_verbs(Path(agents_path))
    allowed_names = load_allowed_names(Path(allowed_names_path) if allowed_names_path else None)
    findings: list[Finding] = []
    for file_path in iter_files(Path(path) for path in paths):
        text = file_path.read_bytes().decode("utf-8", errors="replace")
        findings.extend(check_text(file_path, text, glossary, verbs, exact_names, allowed_names))
    return findings


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
        "paths", nargs="*", help=f"files or directories (default: {DEFAULT_SCAN_PATH} in the repository root)"
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
        "--repository", default=str(REPO_ROOT), help="the git repository whose staged changes are read"
    )
    arguments = parser.parse_args(argv)
    paths = [Path(path) for path in arguments.paths] or [REPO_ROOT / DEFAULT_SCAN_PATH]
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
        findings = check_paths(paths, Path(arguments.glossary), Path(arguments.allowed_names))
    except (GlossaryError, RulesFileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if arguments.notes_on_staged_lines:
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
