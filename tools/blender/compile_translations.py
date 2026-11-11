"""Compile the extension's singular gettext catalogs deterministically, without dependencies.

Usage: python tools/blender/compile_translations.py
Blender owns plural/formatting of dynamic diagnostic data; these UI catalogs
contain singular labels and complete messages. Unsupported plural entries fail.
"""

import ast
import struct
from pathlib import Path


def compile_catalog(source, destination):
    messages = {}
    entry = {}
    field = None
    fuzzy = False

    def finish():
        if "msgid" in entry and not fuzzy:
            key = entry["msgid"]
            if "msgctxt" in entry:
                key = entry["msgctxt"] + "\x04" + key
            value = entry.get("msgstr", "")
            if value or not key:
                if key in messages:
                    raise ValueError(f"Duplicate message: {key!r}")
                messages[key] = value

    for line in source.read_text(encoding="utf-8").splitlines() + [""]:
        line = line.strip()
        if not line:
            finish()
            entry, field, fuzzy = {}, None, False
        elif line.startswith("#, "):
            fuzzy = fuzzy or "fuzzy" in {flag.strip() for flag in line[2:].split(",")}
        elif line.startswith("#"):
            continue
        elif line.startswith(("msgid_plural", "msgstr[")):
            raise ValueError("Plural entries are not supported by the Blender UI catalog compiler")
        elif line.startswith(("msgctxt ", "msgid ", "msgstr ")):
            field, value = line.split(" ", 1)
            entry[field] = ast.literal_eval(value)
        elif line.startswith('"') and field:
            entry[field] += ast.literal_eval(line)
        else:
            raise ValueError(f"Invalid PO line: {line}")
    pairs = sorted((key.encode("utf-8"), value.encode("utf-8")) for key, value in messages.items())
    count = len(pairs)
    offset = 28 + count * 16
    originals = b""
    translations = b""
    original_table = []
    translation_table = []
    for key, _ in pairs:
        original_table.append((len(key), offset + len(originals)))
        originals += key + b"\0"
    for _, value in pairs:
        translation_table.append((len(value), offset + len(originals) + len(translations)))
        translations += value + b"\0"
    header = struct.pack("<7I", 0x950412DE, 0, count, 28, 28 + count * 8, 0, 0)
    tables = b"".join(struct.pack("<2I", *item) for item in original_table + translation_table)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + tables + originals + translations)


def main():
    locale = Path(__file__).resolve().parents[2] / "blender/fluidity/locale"
    for source in sorted(locale.glob("*/LC_MESSAGES/fluidity.po")):
        compile_catalog(source, source.with_suffix(".mo"))
        print(source.with_suffix(".mo"))


if __name__ == "__main__":
    main()
