"""Tests del almacén de fotos de la guía (``images``).

Se evita depender del PDF real o de la red: se construye un ``GuideImageStore``
con un directorio de imágenes temporal y se inyecta un manifiesto sintético
(en memoria o escrito a disco), de modo que los tests cubran la lógica de
indexado, filtrado, deduplicación y la validación de rutas del manifiesto.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.config import Settings
from core.images import GuideImageStore


def _store(tmp_path: Path, **overrides) -> GuideImageStore:
    """Crea un almacén con ``images_dir`` temporal y los ajustes indicados."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(google_api_key=None, images_dir=images_dir, **overrides)
    return GuideImageStore(settings)


def _index_from(store: GuideImageStore, items: list[dict]) -> None:
    """Rellena el índice en memoria del almacén con un manifiesto sintético."""
    store._index(items)


def test_images_for_pages_orders_dedups_and_caps(tmp_path: Path) -> None:
    store = _store(tmp_path, max_images_shown=3, min_image_size=200)
    shared = {"path": "shared.jpg", "page": 1, "width": 300, "height": 300, "caption": ""}
    _index_from(
        store,
        [
            {"path": "a.jpg", "page": 2, "width": 300, "height": 300, "caption": ""},
            shared,
            {"path": "b.jpg", "page": 1, "width": 300, "height": 300, "caption": ""},
            shared,  # duplicado en la misma página
            {"path": "c.jpg", "page": 3, "width": 300, "height": 300, "caption": ""},
            {"path": "d.jpg", "page": 3, "width": 300, "height": 300, "caption": ""},
        ],
    )

    # Orden de páginas solicitado: 1, 2, 3. El tope (3) corta antes de "d".
    result = store.images_for_pages([1, 2, 3])
    paths = [img["path"] for img in result]
    # Las rutas se entregan absolutas (dentro de images_dir) a los consumidores.
    assert [Path(p).name for p in paths] == ["shared.jpg", "b.jpg", "a.jpg"]
    assert all(Path(p).is_absolute() for p in paths)


def test_images_for_pages_dedups_across_pages(tmp_path: Path) -> None:
    store = _store(tmp_path, max_images_shown=5, min_image_size=200)
    repeated = {"path": "x.jpg", "page": None, "width": 300, "height": 300, "caption": ""}
    _index_from(
        store,
        [
            {**repeated, "page": 1},
            {**repeated, "page": 2},
        ],
    )
    # La misma ruta aparece en dos páginas distintas: solo se devuelve una vez.
    result = store.images_for_pages([1, 2])
    assert [Path(img["path"]).name for img in result] == ["x.jpg"]


def test_images_for_pages_filters_small_images(tmp_path: Path) -> None:
    store = _store(tmp_path, max_images_shown=5, min_image_size=200)
    _index_from(
        store,
        [
            {"path": "big.jpg", "page": 1, "width": 300, "height": 300, "caption": ""},
            {"path": "narrow.jpg", "page": 1, "width": 50, "height": 300, "caption": ""},
            {"path": "short.jpg", "page": 1, "width": 300, "height": 50, "caption": ""},
        ],
    )
    result = store.images_for_pages([1])
    assert [Path(img["path"]).name for img in result] == ["big.jpg"]


def test_images_for_pages_tolerates_missing_dimensions(tmp_path: Path) -> None:
    # Entradas antiguas sin width/height se aceptan (el filtro ya se aplicó al extraer).
    store = _store(tmp_path, max_images_shown=5, min_image_size=200)
    _index_from(store, [{"path": "legacy.jpg", "page": 1, "caption": ""}])
    assert [Path(img["path"]).name for img in store.images_for_pages([1])] == ["legacy.jpg"]


def _write_manifest(store: GuideImageStore, entries: list[dict]) -> None:
    store.manifest_path.write_text(json.dumps(entries), encoding="utf-8")


def test_relative_manifest_paths_resolve_to_absolute(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # La imagen debe existir en disco (si no, build() la consideraría obsoleta).
    (store.images_dir / "photo.jpg").write_bytes(b"x")
    _write_manifest(
        store, [{"path": "photo.jpg", "page": 0, "width": 300, "height": 300, "caption": ""}]
    )
    store.build()

    images = store.images_for_pages([0])
    assert len(images) == 1
    path = images[0]["path"]
    # El consumidor (app.py -> st.image) necesita una ruta absoluta y dentro del dir.
    assert Path(path).is_absolute()
    assert Path(path) == store.images_dir / "photo.jpg"


def test_index_resolves_relative_paths_to_existing_absolute(tmp_path: Path) -> None:
    # Regresión: la rama de extracción indexa rutas RELATIVAS; deben llegar a la
    # UI como absolutas y existentes (antes se entregaban relativas e invisibles).
    store = _store(tmp_path)
    (store.images_dir / "rel.jpg").write_bytes(b"x")
    store._index([{"path": "rel.jpg", "page": 0, "width": 300, "height": 300, "caption": ""}])
    images = store.images_for_pages([0])
    assert len(images) == 1
    assert Path(images[0]["path"]) == store.images_dir / "rel.jpg"
    assert Path(images[0]["path"]).is_file()


def test_build_rebuilds_when_manifest_images_missing(tmp_path: Path) -> None:
    # Manifiesto con rutas válidas pero sin archivos en disco (proyecto movido):
    # build() debe reconstruir. Aquí se fuerza el camino comprobando que un
    # manifiesto sin imágenes existentes no se da por bueno.
    store = _store(tmp_path)
    _write_manifest(
        store, [{"path": "ghost.jpg", "page": 0, "width": 300, "height": 300, "caption": ""}]
    )
    manifest = store._load_manifest()
    store._index(manifest)
    # Ninguna imagen existe en disco -> el almacén lo detecta para reconstruir.
    assert store._any_image_exists() is False


def test_corrupt_manifest_does_not_raise_and_is_treated_as_missing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.manifest_path.write_text("{bad json", encoding="utf-8")

    # No debe lanzar: se devuelve None y se reconstruiría desde el PDF.
    assert store._load_manifest() is None


def test_manifest_entry_outside_images_dir_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # La validación de rutas (path traversal) ocurre al indexar.
    store._index(
        [
            {"path": "../../../../etc/passwd", "page": 0, "caption": ""},
            {"path": "/etc/passwd", "page": 0, "caption": ""},
            {"path": "ok.jpg", "page": 0, "width": 300, "height": 300, "caption": ""},
        ]
    )
    result = store.images_for_pages([0])
    # Solo sobrevive la entrada que vive dentro del directorio de imágenes.
    paths = [Path(item["path"]) for item in result]
    assert paths == [store.images_dir / "ok.jpg"]
    for path in paths:
        assert path.is_relative_to(store.images_dir)


def test_non_list_manifest_is_treated_as_missing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.manifest_path.write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")
    assert store._load_manifest() is None
