"""Load compiled gettext catalogs into Blender's translation registry."""

import gettext
from pathlib import Path


def load_catalogs():
    translations = {}
    for path in sorted((Path(__file__).parent / "locale").glob("*/LC_MESSAGES/fluidity.mo")):
        with path.open("rb") as stream:
            catalog = gettext.GNUTranslations(stream)
        messages = {}
        for key, value in catalog._catalog.items():
            if isinstance(key, str) and key and isinstance(value, str):
                if "\x04" in key:
                    context, message = key.split("\x04", 1)
                    messages[context, message] = value
                else:
                    for context in ("*", "Operator", "FluidiTy"):
                        messages[context, key] = value
        translations[path.parents[1].name] = messages
    return translations


TRANSLATIONS = load_catalogs()
