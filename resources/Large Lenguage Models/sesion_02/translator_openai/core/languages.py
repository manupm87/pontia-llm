from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    name: str
    display_name: str
    native_name: str
    code: str

    @property
    def label(self) -> str:
        if self.code == "auto":
            return self.display_name
        return f"{self.display_name} ({self.native_name})"


AUTO_DETECT_LANGUAGE = Language(
    "Auto-detect",
    "Detectar idioma de origen",
    "Automático",
    "auto",
)

SUPPORTED_LANGUAGES: tuple[Language, ...] = (
    Language("English", "Inglés", "English", "en"),
    Language("Spanish", "Español", "Español", "es"),
    Language("Italian", "Italiano", "Italiano", "it"),
    Language("French", "Francés", "Français", "fr"),
    Language("German", "Alemán", "Deutsch", "de"),
    Language("Portuguese", "Portugués", "Português", "pt"),
    Language("Catalan", "Catalán", "Català", "ca"),
    Language("Galician", "Gallego", "Galego", "gl"),
    Language("Basque", "Euskera", "Euskara", "eu"),
    Language("Dutch", "Neerlandés", "Nederlands", "nl"),
    Language("Chinese", "Chino", "中文", "zh"),
    Language("Japanese", "Japonés", "日本語", "ja"),
    Language("Korean", "Coreano", "한국어", "ko"),
    Language("Arabic", "Árabe", "العربية", "ar"),
)


def labels() -> list[str]:
    return [language.label for language in SUPPORTED_LANGUAGES]


def source_labels() -> list[str]:
    return [AUTO_DETECT_LANGUAGE.label, *labels()]


def target_labels() -> list[str]:
    return labels()


def get_by_label(label: str) -> Language:
    if label == AUTO_DETECT_LANGUAGE.label:
        return AUTO_DETECT_LANGUAGE
    for language in SUPPORTED_LANGUAGES:
        if language.label == label:
            return language
    raise ValueError(f"Unsupported language label: {label}")


def get_by_name(name: str) -> Language | None:
    normalized = name.strip().casefold()
    for language in SUPPORTED_LANGUAGES:
        if language.name.casefold() == normalized:
            return language
        if language.display_name.casefold() == normalized:
            return language
        if language.native_name.casefold() == normalized:
            return language
    return None


def default_source_index() -> int:
    return source_labels().index(AUTO_DETECT_LANGUAGE.label)


def default_target_index() -> int:
    return target_labels().index("Italiano (Italiano)")
