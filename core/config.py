"""Configuración inmutable del asistente turístico de Tenerife.

Define las constantes geográficas y de rutas del proyecto, la clase
``Settings`` (parámetros del modelo y del pipeline RAG) y el helper
``load_settings`` que carga la configuración desde variables de entorno.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Coordenadas de referencia de Tenerife (Santa Cruz) para la previsión del tiempo.
TENERIFE_LATITUDE = 28.4636
TENERIFE_LONGITUDE = -16.2518

# Rutas del proyecto derivadas de la ubicación de este módulo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "TENERIFE.pdf"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "storage" / "faiss_index"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "storage" / "images"


@dataclass(frozen=True)
class Settings:
    """Parámetros inmutables del asistente (modelo, RAG y diálogo).

    Agrupa la clave de API, los ajustes del modelo de generación y de
    embeddings, las rutas del PDF y del índice FAISS, así como los
    parámetros de troceado, recuperación y gestión del historial.
    """

    google_api_key: str | None
    generation_model: str = "gemini-2.5-flash-lite"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = 0.2
    top_p: float = 0.95
    max_output_tokens: int = 1024
    # Presupuesto de "thinking" de Gemini para mostrar el razonamiento en vivo
    # (0 = desactivado, -1 = dinámico). Configurable con THINKING_BUDGET.
    thinking_budget: int = 1024
    pdf_path: Path = DEFAULT_PDF_PATH
    index_dir: Path = DEFAULT_INDEX_DIR
    images_dir: Path = DEFAULT_IMAGES_DIR
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5
    max_history_messages: int = 12
    request_timeout: float = 45.0
    # Imágenes de la guía: tamaño mínimo (px) para descartar decoraciones y
    # número máximo de fotos que se muestran junto a cada respuesta.
    min_image_size: int = 200
    max_images_shown: int = 3

    @property
    def has_api_key(self) -> bool:
        """Indica si hay una clave de API de Google configurada."""
        return bool(self.google_api_key)


def load_settings() -> Settings:
    """Carga la configuración desde el entorno (con valores por defecto).

    Lee el archivo ``.env`` mediante ``load_dotenv`` y construye un objeto
    ``Settings``. Toma del entorno la clave de API y los parámetros del
    modelo (``GENERATION_MODEL``, ``EMBEDDING_MODEL``, ``TEMPERATURE``,
    ``TOP_P``, ``MAX_OUTPUT_TOKENS``, ``THINKING_BUDGET``); el resto de campos
    usa sus defaults.

    Nota: ``load_dotenv`` deja ``GOOGLE_API_KEY`` en el entorno para que
    ``langchain-google-genai`` la detecte automáticamente.
    """
    load_dotenv()

    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        generation_model=os.getenv("GENERATION_MODEL", "gemini-2.5-flash-lite"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        top_p=float(os.getenv("TOP_P", "0.95")),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1024")),
        thinking_budget=int(os.getenv("THINKING_BUDGET", "1024")),
    )
