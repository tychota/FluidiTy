"""Verify the shipped gettext compiler with independent stdlib MO readback."""

import gettext
import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "compile_translations", Path(__file__).parents[1] / "blender/compile_translations.py"
)
compiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler)


class CatalogTest(unittest.TestCase):
    def test_unicode_context_and_fuzzy_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.po"
            output = source.with_suffix(".mo")
            source.write_text(
                'msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=UTF-8\\n"\n\n'
                'msgctxt "FluidiTy"\nmsgid "Preview"\nmsgstr "Aperçu"\n\n'
                '#, fuzzy\nmsgid "Bake"\nmsgstr "Incorrect"\n\n',
                encoding="utf-8",
            )
            compiler.compile_catalog(source, output)
            first = output.read_bytes()
            compiler.compile_catalog(source, output)
            self.assertEqual(first, output.read_bytes())
            with output.open("rb") as stream:
                catalog = gettext.GNUTranslations(stream)
            self.assertEqual(catalog.pgettext("FluidiTy", "Preview"), "Aperçu")
            self.assertEqual(catalog.gettext("Bake"), "Bake")

    def test_plural_entries_fail_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.po"
            source.write_text('msgid "frame"\nmsgid_plural "frames"\nmsgstr[0] "image"\n', encoding="utf-8")
            output = source.with_suffix(".mo")
            with self.assertRaisesRegex(ValueError, "Plural"):
                compiler.compile_catalog(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
