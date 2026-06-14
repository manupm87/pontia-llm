from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal


ToolChoice = Literal["auto", "required", "none"]
DraftStatus = Literal["requires_human_confirmation", "confirmed", "rejected"]


@dataclass(frozen=True)
class ToolSpec:
    schema: dict[str, Any]
    function: Callable[..., Any]

    @property
    def name(self) -> str:
        return self.schema["name"]

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.schema.get("parameters", {"type": "object", "properties": {}})


@dataclass(frozen=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    ok: bool
    output: Any
    elapsed_seconds: float


@dataclass
class ToolRunResult:
    final_response: Any
    conversation: list[Any]
    executions: list[ToolExecution] = field(default_factory=list)

    @property
    def output_text(self) -> str:
        return self.final_response.output_text

    @property
    def tool_names(self) -> list[str]:
        return [execution.name for execution in self.executions]


@dataclass(frozen=True)
class StructuredAnswer:
    answer: str
    used_tools: list[str]
    sources: list[str]
    confidence: Literal["low", "medium", "high"]
    next_step: str | None


@dataclass(frozen=True)
class ChatTurn:
    question: str
    answer: StructuredAnswer
    executions: list[ToolExecution]
    created_at: datetime
    raw_output: str


@dataclass
class ExpenseDraft:
    draft_id: str
    vendor: str
    amount_eur: float
    concept: str
    cost_center: str
    requires_prior_approval: bool
    status: DraftStatus
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "vendor": self.vendor,
            "amount_eur": self.amount_eur,
            "concept": self.concept,
            "cost_center": self.cost_center,
            "requires_prior_approval": self.requires_prior_approval,
            "status": self.status,
            "created_at": self.created_at,
        }
