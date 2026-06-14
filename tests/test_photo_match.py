"""Tests del emparejado de fotos con sus menciones en el texto (``photo_match``)."""

from __future__ import annotations

from core.photo_match import place_tokens, plan_inline_images


def _img(path: str, caption: str) -> dict:
    return {"path": path, "caption": caption}


def test_place_tokens_drops_generic_and_short_words() -> None:
    # "playa" y "de" se descartan; queda el nombre distintivo.
    assert place_tokens("Playa de Las Teresitas") == ["teresitas"]


def test_image_is_placed_after_its_mention() -> None:
    text = "Te recomiendo Las Teresitas.\nTambién está bien el Teide."
    images = [_img("/t/teide.jpg", "Teide"), _img("/t/teresitas.jpg", "Playa de Las Teresitas")]
    plan = plan_inline_images(text, images)

    # La foto de Teresitas va tras la primera línea; la del Teide tras la segunda.
    assert plan[0] == ("text", "Te recomiendo Las Teresitas.")
    assert plan[1] == ("images", [images[1]])
    assert plan[2] == ("text", "También está bien el Teide.")
    assert plan[3] == ("images", [images[0]])


def test_unmentioned_images_go_to_the_end() -> None:
    text = "Un día perfecto en la costa."
    images = [_img("/t/anaga.jpg", "Bosque de Anaga")]
    plan = plan_inline_images(text, images)
    assert plan[-1] == ("images", [images[0]])


def test_match_is_accent_and_case_insensitive() -> None:
    text = "Sube al teide al amanecer."
    images = [_img("/t/teide.jpg", "El Teide")]
    plan = plan_inline_images(text, images)
    assert ("images", [images[0]]) in plan


def test_token_does_not_match_inside_longer_word() -> None:
    # "anaga" no debe encajar dentro de "anagaza" (palabra mayor no relacionada):
    # el emparejado es por palabra completa, no por subcadena. Sin mención real,
    # la foto se agrupa al final.
    text = "Cuidado con la anagaza del vendedor."
    images = [_img("/t/anaga.jpg", "Bosque de Anaga")]
    plan = plan_inline_images(text, images)
    # No se coloca tras la línea (no hay mención de palabra completa).
    assert plan[0] == ("text", "Cuidado con la anagaza del vendedor.")
    assert plan[1] == ("images", [images[0]])  # va al final como no mencionada
    assert len([seg for seg in plan if seg[0] == "text"]) == 1


def test_each_image_placed_once() -> None:
    text = "Teide por la mañana.\nTeide por la tarde."
    images = [_img("/t/teide.jpg", "Teide")]
    plan = plan_inline_images(text, images)
    image_segments = [seg for seg in plan if seg[0] == "images"]
    assert len(image_segments) == 1
