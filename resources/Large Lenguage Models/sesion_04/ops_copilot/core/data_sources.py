from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI


class InventoryCatalog:
    def __init__(self, path: Path) -> None:
        self._items = json.loads(path.read_text(encoding="utf-8"))

    @property
    def product_ids(self) -> list[str]:
        return [item["product_id"] for item in self._items]

    def find(self, product_id: str) -> dict[str, Any]:
        product = next((item for item in self._items if item["product_id"] == product_id), None)
        if product is None:
            return {
                "found": False,
                "product_id": product_id,
                "message": "No existe un producto con ese identificador.",
                "available_product_ids": self.product_ids,
            }
        return {"found": True, **product}

    def list_products(self, *, low_stock_only: bool) -> list[dict[str, Any]]:
        products = self._items
        if low_stock_only:
            products = [item for item in products if item["stock"] <= 5]
        return sorted(products, key=lambda item: (item["stock"], item["product_id"]))

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._items)


class BusinessDataset:
    def __init__(self, path: Path) -> None:
        self.frame = pd.read_csv(path)

    def overview(self) -> dict[str, Any]:
        latest_month = str(self.frame["date"].max())
        latest = self.frame[self.frame["date"] == latest_month]
        total_revenue = float(self.frame["revenue"].sum())
        total_margin = float((self.frame["revenue"] - self.frame["cost"]).sum())
        return {
            "latest_month": latest_month,
            "regions": sorted(self.frame["region"].unique().tolist()),
            "row_count": int(len(self.frame)),
            "total_revenue": round(total_revenue, 2),
            "total_margin": round(total_margin, 2),
            "latest_revenue": float(latest["revenue"].sum()),
            "latest_active_users": int(latest["active_users"].sum()),
            "average_nps": round(float(self.frame["nps"].mean()), 1),
        }

    def summarize_by_region(self) -> list[dict[str, Any]]:
        summary = (
            self.frame.assign(margin=lambda data: data["revenue"] - data["cost"])
            .groupby("region", as_index=False)
            .agg(
                revenue=("revenue", "sum"),
                cost=("cost", "sum"),
                margin=("margin", "sum"),
                active_users=("active_users", "max"),
                new_signups=("new_signups", "sum"),
                average_churn=("churn_rate", "mean"),
                average_nps=("nps", "mean"),
            )
            .sort_values("revenue", ascending=False)
        )
        summary["margin_rate"] = summary["margin"] / summary["revenue"]
        return _records(summary)

    def metric_trend(self, metric: str, region: str | None = None) -> list[dict[str, Any]]:
        if metric not in self.frame.columns:
            raise ValueError(f"Métrica no disponible: {metric}")
        data = self.frame.copy()
        if region:
            if region not in set(data["region"]):
                raise ValueError(f"Región no disponible: {region}")
            data = data[data["region"] == region]
        trend = data.groupby("date", as_index=False).agg(value=(metric, "sum"))
        return _records(trend)

    def dataframe(self) -> pd.DataFrame:
        return self.frame.copy()


class KnowledgeBase:
    def __init__(
        self,
        docs_dir: Path,
        client: OpenAI,
        embedding_model: str,
        *,
        chunk_size: int = 900,
        overlap: int = 150,
    ) -> None:
        self._client = client
        self._embedding_model = embedding_model
        self._chunks = _load_chunks(docs_dir, chunk_size=chunk_size, overlap=overlap)
        self._embeddings: np.ndarray | None = None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def source_count(self) -> int:
        return len({chunk["source"] for chunk in self._chunks})

    def search(self, query: str, *, k: int) -> dict[str, Any]:
        safe_k = max(1, min(k, 8))
        if not self._chunks:
            return {"query": query, "results": [], "context": ""}

        embeddings = self._ensure_embeddings()
        query_vector = self._embed_texts([query])[0]
        scores = _cosine_similarity(query_vector, embeddings)
        top_indices = np.argsort(scores)[::-1][:safe_k]
        results = []

        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": round(float(scores[int(idx)]), 4),
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                }
            )

        return {
            "query": query,
            "results": results,
            "context": _format_context(results),
        }

    def _ensure_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = self._embed_texts([chunk["text"] for chunk in self._chunks])
        return self._embeddings

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._embedding_model, input=texts)
        return np.array([item.embedding for item in response.data], dtype=np.float32)


def _load_chunks(docs_dir: Path, *, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(_chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(
                {
                    "id": f"{path.name}::chunk_{index}",
                    "source": path.name,
                    "chunk_index": index,
                    "text": chunk,
                }
            )
    return chunks


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(query_norm * matrix_norms, 1e-12)
    return (matrix @ query_vector) / denominator


def _format_context(results: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        "\n".join(
            [
                f"Fuente: {result['source']} (chunk {result['chunk_index']}, score {result['score']})",
                result["text"],
            ]
        )
        for result in results
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    return json.loads(json.dumps(records, default=_json_default))


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
