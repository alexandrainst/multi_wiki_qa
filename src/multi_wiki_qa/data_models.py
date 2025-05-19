"""Data models used in the project."""

from dataclasses import dataclass


@dataclass
class Language:
    """A language.

    Attributes:
        code:
            The ISO 639-1 language code of the language.
        name:
            The name of the language.
    """

    code: str
    name: str
