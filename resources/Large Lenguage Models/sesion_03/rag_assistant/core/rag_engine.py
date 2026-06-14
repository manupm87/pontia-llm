from __future__ import annotations

import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from core.categories import CategoryName, categorize
from core.models import (
    AnalyzedQuery,
    ChunkSummary,
    Citation,
    IndexFingerprint,
    IngestionConfig,
    RetrievalConfig,
    RetrievedChunk,
)


REFUSAL_SENTENCE = "No tengo información suficiente en la base documental"

SYSTEM_PROMPT = (
    "Eres un agente de primer nivel de una mesa de ayuda interna. "
    "Tu trabajo es resolver consultas de empleados sobre procedimientos internos "
    "a partir de la base documental corporativa. "
    "Respondes en español con un tono claro, operativo y profesional. "
    "Usa exclusivamente la información del CONTEXTO recuperado. "
    f"Si el contexto no contiene la respuesta, di literalmente: '{REFUSAL_SENTENCE}'. "
    "No inventes datos, no completes con conocimiento general y no inventes fuentes. "
    "Cuando exista respuesta, estructura la salida con estas secciones breves: "
    "'Resolución', 'Pasos' y 'Cuándo escalar'. "
    "Si el contexto no dice cuándo escalar, escribe en esa sección que no está indicado "
    "en la base documental. "
    "No incluyas la sección de fuentes en tu respuesta: el sistema la añadirá aparte."
)

USER_PROMPT_TEMPLATE = (
    "Solicitud del empleado:\n{question}\n\n"
    "CONTEXTO recuperado (fragmentos numerados):\n{context}\n\n"
    "Redacta una respuesta lista para enviar al empleado usando solo el contexto."
)


class IndexBuildError(RuntimeError):
    """Raised when the knowledge base cannot be built."""


class GenerationError(RuntimeError):
    """Raised when the generation step fails."""


class _SearchQuery(BaseModel):
    query: str = Field(description="Consulta optimizada para buscar en la base documental")
    category: CategoryName = Field(
        description=(
            "Categoría más probable de la pregunta: 'it' para accesos y herramientas, "
            "'rrhh' para personas y procesos internos, 'seguridad' para incidentes, "
            "'general' si no aplica ninguna."
        )
    )


@dataclass(frozen=True)
class IndexHandle:
    fingerprint: IndexFingerprint
    source_label: str
    chunks: tuple[ChunkSummary, ...]
    _vector_store: InMemoryVectorStore

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def categories_distribution(self) -> dict[CategoryName, int]:
        distribution: dict[CategoryName, int] = {}
        for chunk in self.chunks:
            distribution[chunk.category] = distribution.get(chunk.category, 0) + 1
        return distribution


class RagEngine:
    def __init__(
        self,
        generation_model: str,
        embedding_model: str,
        timeout: float,
    ) -> None:
        self._generation_model = generation_model
        self._embedding_model = embedding_model
        self._timeout = timeout
        self._llm = init_chat_model(generation_model, model_provider="google_genai")
        self._embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self._query_analyzer = self._llm.with_structured_output(_SearchQuery)

    def build_index_from_path(
        self,
        pdf_path: Path,
        ingestion: IngestionConfig,
    ) -> IndexHandle:
        if not pdf_path.exists():
            raise IndexBuildError(f"No se encontró la fuente documental en {pdf_path}.")

        if pdf_path.is_dir():
            pdf_paths = tuple(sorted(pdf_path.glob("*.pdf")))
            if not pdf_paths:
                raise IndexBuildError(f"No se encontraron PDFs en {pdf_path}.")

            stats = [path.stat() for path in pdf_paths]
            fingerprint = IndexFingerprint(
                source_label=pdf_path.name,
                source_size=sum(stat.st_size for stat in stats),
                source_mtime=max(stat.st_mtime for stat in stats),
                chunk_size=ingestion.chunk_size,
                chunk_overlap=ingestion.chunk_overlap,
            )
            return self._build_index(
                pdf_paths=pdf_paths,
                source_label=f"{pdf_path.name} · {len(pdf_paths)} PDFs",
                fingerprint=fingerprint,
                ingestion=ingestion,
            )

        stat = pdf_path.stat()
        fingerprint = IndexFingerprint(
            source_label=pdf_path.name,
            source_size=stat.st_size,
            source_mtime=stat.st_mtime,
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
        )
        return self._build_index(
            pdf_paths=(pdf_path,),
            source_label=pdf_path.name,
            fingerprint=fingerprint,
            ingestion=ingestion,
        )

    def build_index_from_bytes(
        self,
        data: bytes,
        source_label: str,
        ingestion: IngestionConfig,
    ) -> IndexHandle:
        if not data:
            raise IndexBuildError("El PDF está vacío.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as buffer:
            buffer.write(data)
            tmp_path = Path(buffer.name)

        fingerprint = IndexFingerprint(
            source_label=source_label,
            source_size=len(data),
            source_mtime=tmp_path.stat().st_mtime,
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
        )
        try:
            return self._build_index(
                pdf_paths=(tmp_path,),
                source_label=source_label,
                fingerprint=fingerprint,
                ingestion=ingestion,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def analyze_query(self, question: str) -> AnalyzedQuery:
        try:
            response = self._query_analyzer.invoke(question)
        except Exception as exc:
            raise GenerationError(f"No se pudo analizar la pregunta: {exc}") from exc
        return AnalyzedQuery(query=response.query, category=response.category)

    def retrieve(
        self,
        index: IndexHandle,
        query: str,
        retrieval: RetrievalConfig,
        category_override: CategoryName | None = None,
    ) -> list[RetrievedChunk]:
        category_filter = (
            category_override if category_override is not None else retrieval.category_filter
        )
        documents = index._vector_store.similarity_search(
            query,
            k=retrieval.top_k,
            filter=_build_category_filter(category_filter),
        )
        return [_to_retrieved_chunk(doc) for doc in documents]

    def stream_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
    ) -> Iterator[str]:
        if not chunks:
            yield REFUSAL_SENTENCE + "."
            return

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT_TEMPLATE.format(
                    question=question,
                    context=_format_context(chunks),
                )
            ),
        ]
        try:
            for chunk in self._llm.stream(messages):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:
            raise GenerationError(f"Se interrumpió la generación de la respuesta: {exc}") from exc

    def _build_index(
        self,
        pdf_paths: tuple[Path, ...],
        source_label: str,
        fingerprint: IndexFingerprint,
        ingestion: IngestionConfig,
    ) -> IndexHandle:
        documents: list[Document] = []
        for pdf_path in pdf_paths:
            try:
                loaded = PyPDFLoader(str(pdf_path)).load()
            except Exception as exc:
                raise IndexBuildError(f"No se pudo leer {pdf_path.name}: {exc}") from exc

            for doc in loaded:
                doc.metadata["source_name"] = pdf_path.name
            documents.extend(loaded)

        if not documents:
            raise IndexBuildError("La fuente documental no contiene páginas legibles.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
            add_start_index=True,
        )
        splits = splitter.split_documents(documents)
        if not splits:
            raise IndexBuildError("El PDF no produjo fragmentos indexables.")

        summaries: list[ChunkSummary] = []
        for chunk_id, doc in enumerate(splits):
            category = categorize(doc.page_content)
            doc.metadata["chunk_id"] = chunk_id
            doc.metadata["source_name"] = doc.metadata.get("source_name", source_label)
            doc.metadata["category"] = category
            summaries.append(
                ChunkSummary(
                    chunk_id=chunk_id,
                    page=_safe_page(doc.metadata.get("page")),
                    category=category,
                    char_count=len(doc.page_content),
                )
            )

        try:
            vector_store = InMemoryVectorStore(self._embeddings)
            vector_store.add_documents(splits)
        except Exception as exc:
            raise IndexBuildError(f"No se pudieron generar los embeddings: {exc}") from exc

        return IndexHandle(
            fingerprint=fingerprint,
            source_label=source_label,
            chunks=tuple(summaries),
            _vector_store=vector_store,
        )


def citations_from_chunks(chunks: tuple[RetrievedChunk, ...] | list[RetrievedChunk]) -> tuple[Citation, ...]:
    seen: set[tuple[str, int | None, int]] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        key = (chunk.source_name, chunk.page, chunk.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source_name=chunk.source_name,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                category=chunk.category,
            )
        )
    return tuple(citations)


def _build_category_filter(
    category: CategoryName | None,
) -> Callable[[Document], bool] | None:
    if category is None:
        return None

    def _filter(doc: Document) -> bool:
        return doc.metadata.get("category") == category

    return _filter


def _to_retrieved_chunk(doc: Document) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=int(doc.metadata.get("chunk_id", -1)),
        page=_safe_page(doc.metadata.get("page")),
        source_name=str(doc.metadata.get("source_name", "fuente desconocida")),
        category=doc.metadata.get("category", "general"),
        content=doc.page_content,
    )


def _format_context(chunks: list[RetrievedChunk] | tuple[RetrievedChunk, ...]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        page_label = chunk.page + 1 if isinstance(chunk.page, int) else "?"
        blocks.append(
            f"[Fragmento {index} | Fuente: {chunk.source_name} | "
            f"Página: {page_label} | Chunk: {chunk.chunk_id}]\n{chunk.content}"
        )
    return "\n\n".join(blocks)


def _safe_page(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
