from __future__ import annotations

import json
import time
from typing import Any

from jsonschema import Draft202012Validator
from openai import OpenAI

from core.models import ToolExecution, ToolRunResult, ToolSpec


class ToolExecutionError(RuntimeError):
    pass


def validate_tool_arguments(tool: ToolSpec, arguments: dict[str, Any]) -> None:
    validator = Draft202012Validator(tool.parameters_schema)
    errors = sorted(validator.iter_errors(arguments), key=lambda error: error.path)
    if errors:
        messages = [error.message for error in errors]
        raise ValueError("Argumentos inválidos: " + "; ".join(messages))


def execute_tool_call(call: Any, registry: dict[str, ToolSpec]) -> tuple[dict[str, Any], ToolExecution]:
    start = time.perf_counter()
    name = getattr(call, "name", "")
    arguments: dict[str, Any] = {}

    try:
        parsed_arguments = json.loads(call.arguments or "{}")
        if not isinstance(parsed_arguments, dict):
            raise ValueError("Los argumentos de herramienta deben ser un objeto JSON.")
        arguments = parsed_arguments

        if name not in registry:
            raise ValueError(f"Herramienta no registrada: {name}")

        tool = registry[name]
        validate_tool_arguments(tool, arguments)
        output = tool.function(**arguments)
        ok = True
        payload = {"ok": True, "data": output}
    except Exception as exc:
        output = {"error_type": type(exc).__name__, "message": str(exc)}
        ok = False
        payload = {"ok": False, "error": output}

    execution = ToolExecution(
        name=name,
        arguments=arguments,
        ok=ok,
        output=output,
        elapsed_seconds=time.perf_counter() - start,
    )
    return payload, execution


def run_llm_with_tools(
    client: OpenAI,
    user_input: str,
    *,
    tools: list[ToolSpec],
    instructions: str,
    model: str,
    conversation_context: list[dict[str, str]] | None = None,
    max_tool_rounds: int = 5,
    tool_choice: str | dict[str, Any] = "auto",
    parallel_tool_calls: bool = True,
) -> ToolRunResult:
    registry = {tool.name: tool for tool in tools}
    conversation: list[Any] = list(conversation_context or [])
    conversation.append({"role": "user", "content": user_input})
    executions: list[ToolExecution] = []

    for round_index in range(max_tool_rounds):
        current_tool_choice = tool_choice if round_index == 0 else "auto"
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=conversation,
            tools=[tool.schema for tool in tools],
            tool_choice=current_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        function_calls = [item for item in response.output if item.type == "function_call"]
        conversation.extend(response.output)
        if not function_calls:
            return ToolRunResult(
                final_response=response,
                conversation=conversation,
                executions=executions,
            )

        for call in function_calls:
            payload, execution = execute_tool_call(call, registry)
            executions.append(execution)
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(payload, ensure_ascii=False),
                }
            )

    raise ToolExecutionError(f"Se alcanzó el límite de {max_tool_rounds} rondas de herramientas.")
