import inspect
from typing import Any, Callable
from pydantic import create_model

from agentkit.tools.base import ToolDefinition


def _generate_schema_from_func(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Fonksiyonun imzasını analiz ederek Pydantic tabanlı JSON Schema üretir.
    """
    sig = inspect.signature(func)
    fields = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any

        if param.default != inspect.Parameter.empty:
            fields[param_name] = (param_type, param.default)
        else:
            fields[param_name] = (param_type, ...)

    DynamicModel = create_model(f"{func.__name__}_Schema", **fields)  # type: ignore
    schema = DynamicModel.model_json_schema()

    if "title" in schema:
        del schema["title"]

    from typing import cast

    return cast(dict[str, Any], schema)


def tool(func: Callable[..., Any]) -> ToolDefinition:
    """
    Python fonksiyonlarını LLM'in anlayabileceği Tool formatına çeviren decorator.
    """
    description = inspect.getdoc(func) or f"{func.__name__} fonksiyonu."
    description = description.strip()

    parameters_schema = _generate_schema_from_func(func)

    tool_def = ToolDefinition(
        name=func.__name__, description=description, parameters=parameters_schema, func=func
    )

    return tool_def
