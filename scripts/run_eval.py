"""Ejecuta la evaluación del asistente y guarda un informe (tabla + gráfico).

Uso (requiere GOOGLE_API_KEY en el entorno o en .env):

    python -m scripts.run_eval

Construye el RAG y el asistente reales, activa el juez de fidelidad y el
clasificador de tema (guardarraíles LLM) y evalúa el conjunto por defecto.
Imprime las métricas, guarda ``storage/eval_results.csv`` y
``storage/eval_summary.png``.
"""

from __future__ import annotations

from pathlib import Path

from core.assistant import TouristAssistant
from core.config import load_settings
from core.evaluation import (
    default_dataset,
    plot_summary,
    run_evaluation,
    summarize,
    to_dataframe,
)
from core.guardrails import build_llm_guardrails
from core.rag import TouristGuideRAG


def main() -> None:
    """Punto de entrada de la evaluación."""
    settings = load_settings()
    if not settings.has_api_key:
        raise SystemExit("Define GOOGLE_API_KEY (en .env) para ejecutar la evaluación.")

    rag = TouristGuideRAG(settings)
    rag.build_index()

    assistant = TouristAssistant(settings, rag)
    # Guardarraíles LLM: clasificador de tema (rechazos) + juez de fidelidad.
    guardrails = build_llm_guardrails(assistant.llm)
    assistant.guardrails = guardrails

    cases = default_dataset()
    results = run_evaluation(rag, assistant, cases, grounding_judge=guardrails._grounding_judge)

    summary = summarize(results)
    print("\n=== Métricas ===")
    for key, value in summary.items():
        print(f"{key:28s}: {value}")

    out_dir = Path(settings.index_dir).parent  # storage/
    out_dir.mkdir(parents=True, exist_ok=True)

    df = to_dataframe(results)
    csv_path = out_dir / "eval_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTabla guardada en {csv_path}")

    fig = plot_summary(summary)
    png_path = out_dir / "eval_summary.png"
    fig.savefig(png_path, dpi=120)
    print(f"Gráfico guardado en {png_path}")


if __name__ == "__main__":
    main()
