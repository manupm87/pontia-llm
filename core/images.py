"""Extracción de las fotos de la guía y asociación por página.

La guía oficial de Tenerife (``TENERIFE.pdf``) incluye fotografías de los
lugares de interés (playas, miradores, el Teide, pueblos...). Este módulo las
extrae con PyMuPDF, las guarda en disco y construye un índice
``página -> [imágenes]`` para que el asistente pueda mostrar la foto del lugar
recuperado durante el chat.

El índice se persiste como ``manifest.json`` junto a las imágenes y se
reconstruye desde el PDF (igual que el índice FAISS), por lo que vive en
``storage/`` y no se versiona.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from .config import Settings

logger = logging.getLogger("asistente_tenerife.images")

# Etiquetas de la guía que no aportan nada como pie de foto (p. ej. enlaces).
_TAG_RE = re.compile(r"\[[^\]]*\]")
# Viñeta inicial de un punto de la guía ("• " o "o " como subnivel).
_BULLET_RE = re.compile(r"^[•▪◦·o\-–]\s+")


class GuideImageStore:
    """Almacén de las fotos de la guía, indexadas por página del PDF.

    Extrae las imágenes embebidas en el PDF, las guarda en
    ``settings.images_dir`` y mantiene un mapa ``página -> [fotos]`` (con su
    ruta y un pie de foto aproximado) que permite recuperar las imágenes
    asociadas a los fragmentos devueltos por el RAG.
    """

    def __init__(self, settings: Settings) -> None:
        """Inicializa el almacén con la configuración del proyecto."""
        self.settings = settings
        self.images_dir: Path = settings.images_dir
        self.manifest_path = self.images_dir / "manifest.json"
        # Índice en memoria: número de página (base 0) -> lista de fotos.
        self._by_page: dict[int, list[dict]] = {}

    def build(self, force: bool = False) -> None:
        """Extrae las imágenes del PDF (o recarga el manifiesto existente).

        Si el manifiesto ya existe y ``force`` es ``False``, se recarga desde
        disco. En caso contrario, recorre el PDF, guarda cada foto relevante y
        escribe el manifiesto.
        """
        if self.manifest_path.exists() and not force:
            self._load_manifest()
            return

        self.images_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._extract_images()
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._index(manifest)
        logger.info("Extraídas %d imágenes de la guía.", len(manifest))

    @property
    def total_images(self) -> int:
        """Número total de fotos indexadas (en todas las páginas)."""
        return sum(len(items) for items in self._by_page.values())

    def images_for_pages(self, pages: list[int], limit: int | None = None) -> list[dict]:
        """Devuelve las fotos asociadas a una lista de páginas (sin duplicados).

        Respeta el orden de ``pages`` (normalmente, el de relevancia de la
        recuperación) y limita el resultado a ``limit`` fotos (por defecto,
        ``settings.max_images_shown``).
        """
        if limit is None:
            limit = self.settings.max_images_shown
        seen: set[str] = set()
        images: list[dict] = []
        for page in pages:
            for item in self._by_page.get(page, []):
                if item["path"] in seen:
                    continue
                seen.add(item["path"])
                images.append(item)
                if len(images) >= limit:
                    return images
        return images

    def _extract_images(self) -> list[dict]:
        """Recorre el PDF y guarda cada foto relevante; devuelve el manifiesto."""
        manifest: list[dict] = []
        doc = fitz.open(str(self.settings.pdf_path))
        min_size = self.settings.min_image_size
        for page_index in range(doc.page_count):
            page = doc[page_index]
            for order, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                info = doc.extract_image(xref)
                # Descarta imágenes pequeñas (viñetas, iconos, decoraciones).
                if info["width"] < min_size or info["height"] < min_size:
                    continue
                filename = f"p{page_index + 1:02d}_{order}.{info['ext']}"
                path = self.images_dir / filename
                path.write_bytes(info["image"])
                manifest.append(
                    {
                        "page": page_index,
                        "path": str(path),
                        "caption": _caption_for(page, xref),
                    }
                )
        doc.close()
        return manifest

    def _load_manifest(self) -> None:
        """Recarga el índice en memoria desde el manifiesto persistido."""
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._index(manifest)

    def _index(self, manifest: list[dict]) -> None:
        """Construye el mapa ``página -> [fotos]`` a partir del manifiesto."""
        self._by_page = {}
        for item in manifest:
            self._by_page.setdefault(item["page"], []).append(item)


def _caption_for(page: "fitz.Page", xref: int) -> str:
    """Deriva un pie de foto a partir del texto situado justo encima de la imagen.

    Busca el bloque de texto cuyo borde inferior queda inmediatamente por
    encima de la foto (suele ser el nombre del lugar) y lo limpia de viñetas y
    etiquetas. Devuelve cadena vacía si no hay texto encima.
    """
    rects = page.get_image_rects(xref)
    if not rects:
        return ""
    top = rects[0].y0
    best: tuple[float, str] | None = None
    for block in page.get_text("blocks"):
        y1 = block[3]
        text = block[4].strip().replace("\n", " ")
        if not text or y1 > top + 5:
            continue
        if best is None or y1 > best[0]:
            best = (y1, text)
    return _clean_caption(best[1]) if best else ""


def _clean_caption(text: str) -> str:
    """Limpia un pie de foto: quita etiquetas, viñetas y lo acorta."""
    text = _TAG_RE.sub("", text).strip()
    text = _BULLET_RE.sub("", text).strip()
    return text[:80]
