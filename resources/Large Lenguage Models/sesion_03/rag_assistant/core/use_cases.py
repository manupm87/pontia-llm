from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from core.categories import CategoryName
from core.models import RetrievedChunk


@dataclass(frozen=True)
class SuggestedCase:
    title: str
    area: str
    question: str
    impact: str


@dataclass(frozen=True)
class EvaluationCase:
    title: str
    question: str
    expected_category: CategoryName | None
    expected_terms: tuple[str, ...]
    should_answer: bool


@dataclass(frozen=True)
class EvaluationResult:
    case: EvaluationCase
    retrieved_count: int
    category_hit: bool
    matched_terms: tuple[str, ...]
    answer: str
    answer_refused: bool
    answer_matched_terms: tuple[str, ...]

    @property
    def term_coverage(self) -> float:
        if not self.case.expected_terms:
            return 1.0
        return len(self.matched_terms) / len(self.case.expected_terms)

    @property
    def missing_terms(self) -> tuple[str, ...]:
        return tuple(
            term for term in self.case.expected_terms if term not in self.matched_terms
        )

    @property
    def answer_term_coverage(self) -> float:
        if not self.case.expected_terms:
            return 1.0
        return len(self.answer_matched_terms) / len(self.case.expected_terms)

    @property
    def retrieval_passed(self) -> bool:
        if not self.case.should_answer:
            return len(self.matched_terms) == 0
        has_terms = self.term_coverage >= 0.5
        return self.retrieved_count > 0 and has_terms and self.category_hit

    @property
    def answer_passed(self) -> bool:
        if not self.case.should_answer:
            return self.answer_refused
        has_expected_signals = self.answer_term_coverage >= 0.5
        return bool(self.answer.strip()) and not self.answer_refused and has_expected_signals

    @property
    def passed(self) -> bool:
        return self.retrieval_passed and self.answer_passed


SUGGESTED_CASES: tuple[SuggestedCase, ...] = (
    SuggestedCase(
        title="Acceso desde casa",
        area="IT",
        question="¿Cómo puedo acceder al portal de empleados desde casa?",
        impact="Empleado bloqueado fuera de oficina",
    ),
    SuggestedCase(
        title="Clave del correo",
        area="IT",
        question="He olvidado la contraseña del correo corporativo. ¿Qué tengo que hacer?",
        impact="No puede trabajar con email",
    ),
    SuggestedCase(
        title="Vacaciones",
        area="RRHH",
        question="Quiero pedir unos días libres. ¿Dónde tengo que hacerlo y quién lo aprueba?",
        impact="Solicitud y aprobaciones",
    ),
    SuggestedCase(
        title="Seguridad en oficina",
        area="Seguridad",
        question="¿Cómo reporto un problema de seguridad en la oficina?",
        impact="Incidente físico o activo perdido",
    ),
    SuggestedCase(
        title="Gastos",
        area="Finanzas",
        question="¿Cómo registro un gasto reembolsable y qué justificante necesito?",
        impact="Liquidación de gastos",
    ),
    SuggestedCase(
        title="Incidencia cliente",
        area="Operaciones",
        question="¿Cómo debo registrar y escalar una incidencia SEV2 de cliente?",
        impact="Continuidad de servicio",
    ),
)


EVALUATION_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        title="Portal de empleados",
        question="¿Cómo accedo al portal de empleados desde casa?",
        expected_category="it",
        expected_terms=("portal", "vpn"),
        should_answer=True,
    ),
    EvaluationCase(
        title="Contraseña del correo",
        question="No recuerdo mi clave del email de la empresa. ¿Qué hago?",
        expected_category="it",
        expected_terms=("contraseña", "mfa"),
        should_answer=True,
    ),
    EvaluationCase(
        title="Solicitud de vacaciones",
        question="¿Cómo puedo solicitar vacaciones?",
        expected_category="rrhh",
        expected_terms=("vacaciones", "manager"),
        should_answer=True,
    ),
    EvaluationCase(
        title="Evaluación de desempeño",
        question="¿En qué consiste la evaluación de desempeño?",
        expected_category="rrhh",
        expected_terms=("desempeño", "objetivos"),
        should_answer=True,
    ),
    EvaluationCase(
        title="Incidencia de seguridad",
        question="¿Cómo reporto un problema de seguridad?",
        expected_category="seguridad",
        expected_terms=("seguridad", "incidente"),
        should_answer=True,
    ),
    EvaluationCase(
        title="Gasto reembolsable",
        question="¿Cómo registro un gasto reembolsable?",
        expected_category="general",
        expected_terms=("gasto", "factura"),
        should_answer=True,
    ),
    EvaluationCase(
        title="Incidencia SEV2",
        question="¿Cada cuánto tengo que actualizar una incidencia SEV2?",
        expected_category="general",
        expected_terms=("sev2", "60 minutos"),
        should_answer=True,
    ),
    EvaluationCase(
        title="Dato no documentado",
        question="¿Cuál es el IBAN de la cuenta bancaria de la empresa?",
        expected_category=None,
        expected_terms=("iban",),
        should_answer=False,
    ),
)


def evaluate_retrieved_chunks(
    case: EvaluationCase,
    chunks: tuple[RetrievedChunk, ...],
    answer: str,
    refusal_sentence: str,
) -> EvaluationResult:
    haystack = _normalize(" ".join(chunk.content for chunk in chunks))
    matched_terms = tuple(
        term for term in case.expected_terms if _normalize(term) in haystack
    )
    normalized_answer = _normalize(answer)
    answer_matched_terms = tuple(
        term for term in case.expected_terms if _normalize(term) in normalized_answer
    )
    category_hit = (
        True
        if case.expected_category is None
        else any(chunk.category == case.expected_category for chunk in chunks)
    )
    return EvaluationResult(
        case=case,
        retrieved_count=len(chunks),
        category_hit=category_hit,
        matched_terms=matched_terms,
        answer=answer.strip(),
        answer_refused=_normalize(refusal_sentence) in normalized_answer,
        answer_matched_terms=answer_matched_terms,
    )


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
