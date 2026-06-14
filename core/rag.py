"""Indexado del PDF y recuperación con citas (RAG).

Este módulo carga la guía oficial de Tenerife (``TENERIFE.pdf``), la divide en
fragmentos, genera embeddings con Gemini y los almacena en un índice FAISS
persistente. Expone además una recuperación que devuelve el contexto formateado
junto con las fuentes citadas (nombre de archivo, página y fragmento) y las
fotos de los lugares recuperados.
"""

from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .images import GuideImageStore


class TouristGuideRAG:
    """Motor RAG sobre la guía turística de Tenerife.

    Encapsula la carga del PDF, el troceado, los embeddings de Gemini y el
    índice FAISS persistente, ofreciendo búsqueda y recuperación con citas.
    """

    def __init__(self, settings: Settings) -> None:
        """Inicializa el motor con la configuración y los embeddings de Gemini."""
        self.settings = settings
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
        self.vector_store: FAISS | None = None
        # Almacén de fotos de la guía, indexadas por página.
        self.image_store = GuideImageStore(settings)

    def build_index(self, force: bool = False) -> None:
        """Construye o carga el índice FAISS (y las imágenes) desde disco.

        Si el índice ya existe y ``force`` es ``False``, se carga directamente.
        En caso contrario, lee el PDF, lo trocea, añade metadatos de cita a cada
        fragmento, genera el índice y lo guarda en ``settings.index_dir``.
        Además extrae (o recarga) las fotos de la guía.
        """
        # Las fotos de la guía se extraen una vez y se reutilizan.
        self.image_store.build(force=force)

        index_dir = self.settings.index_dir
        if index_dir.exists() and not force:
            self.vector_store = FAISS.load_local(
                str(index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return

        # Carga del PDF y troceado en fragmentos solapados.
        pdf_path = self.settings.pdf_path
        docs = PyPDFLoader(str(pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)

        # Metadatos de cita para poder atribuir cada fragmento a su origen.
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_name"] = pdf_path.name
            chunk.metadata["chunk_id"] = i

        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(str(index_dir))

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Devuelve los ``k`` fragmentos más similares a la consulta."""
        if k is None:
            k = self.settings.top_k
        if self.vector_store is None:
            self.build_index()
        if self.vector_store is None:  # Salvaguarda: el índice debería existir ya.
            raise RuntimeError("El índice FAISS no se pudo construir ni cargar.")
        return self.vector_store.similarity_search(query, k)

    def retrieve(self, query: str, k: int | None = None) -> dict:
        """Recupera contexto, fuentes e imágenes para una consulta.

        Devuelve un diccionario con el contexto formateado (listo para el
        prompt), la lista de fuentes citadas y las fotos de los lugares
        recuperados. Fuentes e imágenes se almacenan además en
        ``last_sources`` y ``last_images`` para que la interfaz las muestre.
        """
        docs = self.search(query, k)
        sources = [_doc_to_source(d) for d in docs]
        # Fotos de las páginas recuperadas, en orden de relevancia y sin repetir.
        pages = [d.metadata.get("page") for d in docs if isinstance(d.metadata.get("page"), int)]
        images = self.image_store.images_for_pages(pages)
        self.last_sources = sources
        self.last_images = images
        return {
            "context": self.format_context(docs),
            "sources": sources,
            "images": images,
        }

    @staticmethod
    def format_context(docs: list[Document]) -> str:
        """Une los fragmentos en un único bloque de texto con sus encabezados de cita."""
        blocks = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_label = page + 1 if isinstance(page, int) else "?"
            source_name = doc.metadata.get("source_name", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            header = (
                f"[Fuente {i}: {source_name}, página {page_label}, "
                f"fragmento {chunk_id}]"
            )
            blocks.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(blocks)


def _doc_to_source(doc: Document) -> dict:
    """Extrae los datos de cita de un fragmento recuperado."""
    page = doc.metadata.get("page")
    return {
        "source_name": doc.metadata.get("source_name", "?"),
        "page": page,
        "chunk_id": doc.metadata.get("chunk_id", "?"),
        # Fragmento completo recuperado (lo mismo que lee el modelo), para poder
        # auditar el *grounding* desde la interfaz.
        "snippet": doc.page_content,
    }
