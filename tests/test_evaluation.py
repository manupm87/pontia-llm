"""Tests de la lógica pura de evaluación (``core.evaluation``)."""

from __future__ import annotations

from core.evaluation import (
    EvalCase,
    EvalResult,
    default_dataset,
    is_refusal,
    retrieval_hit,
    run_evaluation,
    score_correct,
    summarize,
)


def test_is_refusal() -> None:
    assert is_refusal("Solo puedo ayudarte con turismo en Tenerife.") is True
    assert is_refusal("No tengo información sobre eso.") is True
    assert is_refusal("Te recomiendo la playa de Las Teresitas.") is False


def test_retrieval_hit_is_accent_insensitive() -> None:
    assert retrieval_hit("Subida al Teide por el sur", ("teide",)) is True
    assert retrieval_hit("Texto sin el lugar", ("anaga",)) is False
    # Sin palabras esperadas, siempre acierta.
    assert retrieval_hit("cualquier cosa", ()) is True


def test_score_correct_in_scope() -> None:
    case = EvalCase("q", "in_scope", ("teide",))
    assert score_correct(case, True, True, False) is True
    assert score_correct(case, True, False, False) is False  # no fiel
    assert score_correct(case, False, True, False) is False  # no recupera
    assert score_correct(case, True, None, True) is False  # rechazó
    assert score_correct(case, True, None, False) is True  # sin juez, ok


def test_score_correct_out_of_scope_requires_refusal() -> None:
    case = EvalCase("q", "out_of_scope")
    assert score_correct(case, False, None, True) is True
    assert score_correct(case, True, None, False) is False


def test_summarize_aggregates() -> None:
    results = [
        EvalResult(EvalCase("a", "in_scope", ("x",)), "ans", retrieval_hit=True, faithful=True, correct=True),
        EvalResult(EvalCase("b", "in_scope", ("y",)), "ans", retrieval_hit=False, faithful=None, correct=False),
        EvalResult(EvalCase("c", "out_of_scope"), "Solo puedo ayudarte con turismo", refused=True, correct=True),
    ]
    summary = summarize(results)
    assert summary["n_cases"] == 3.0
    assert summary["accuracy"] == round(2 / 3, 3)
    assert summary["retrieval_hit_rate"] == 0.5  # 1 de 2 in_scope
    assert summary["faithfulness_rate"] == 1.0  # 1 juzgado, fiel
    assert summary["refusal_rate_out_of_scope"] == 1.0


def test_run_evaluation_with_fakes() -> None:
    class _FakeRag:
        def retrieve(self, q: str) -> dict:
            return {"context": "El Teide es el pico más alto."}

    class _FakeAssistant:
        def reset(self) -> None: ...

        def chat(self, q: str) -> dict:
            return {"answer": "Sube al Teide al amanecer.", "tool_calls": []}

    cases = [EvalCase("¿Cómo subir al Teide?", "in_scope", ("teide",))]
    results = run_evaluation(_FakeRag(), _FakeAssistant(), cases, grounding_judge=lambda a, c: True)
    assert len(results) == 1
    assert results[0].correct is True
    assert results[0].retrieval_hit is True
    assert results[0].faithful is True


def test_default_dataset_has_both_kinds() -> None:
    kinds = {case.kind for case in default_dataset()}
    assert kinds == {"in_scope", "out_of_scope"}
