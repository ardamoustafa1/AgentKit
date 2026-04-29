"""Tool, ToolRegistry ve execute_tool fonksiyonlarının kapsamlı testleri."""

import pytest
from typing import Any
from agentkit.tools import execute_tool
from agentkit.tools.base import ToolDefinition
from agentkit.tools.decorator import tool


def test_tool_decorator_schema(sample_tool: Any) -> None:
    """@tool decorator'ının JSON Schema'yı doğru üretip üretmediği test edilir."""
    assert sample_tool.name == "topla"
    assert sample_tool.description == "İki sayıyı toplar"
    props = sample_tool.parameters["properties"]
    assert "a" in props
    assert "b" in props
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"


@pytest.mark.asyncio
async def test_tool_execution_success(sample_tool: Any) -> None:
    """Tool'un başarıyla çalışması test edilir."""
    result = await execute_tool(sample_tool, a=5, b=3)
    assert result == "8"


@pytest.mark.asyncio
async def test_tool_execution_error(error_tool: Any) -> None:
    """Tool içerisindeki exception'ların güvenli yakalandığı test edilir."""
    result = await execute_tool(error_tool)
    assert "Hata oluştu" in result
    assert "Kasıtlı Hata" in result


@pytest.mark.asyncio
async def test_tool_registry(tool_registry: Any, sample_tool: Any, error_tool: Any) -> None:
    """ToolRegistry kayıt ve getirme işlemleri."""
    tool_registry.register(sample_tool)
    tool_registry.register(error_tool)
    assert tool_registry.get_tool("topla") is not None
    assert tool_registry.get_tool("olmayan_tool") is None
    schemas = tool_registry.get_all_schemas()
    assert len(schemas) == 2
    assert schemas[0]["function"]["name"] == "topla"


def test_tool_decorator_with_defaults() -> None:
    """Varsayılan parametre değeri olan fonksiyonların schema üretimi."""

    @tool
    def selamla(isim: str, dil: str = "tr") -> str:
        """Kişiye seçilen dilde selam verir."""
        return f"Merhaba {isim}" if dil == "tr" else f"Hello {isim}"

    assert selamla.name == "selamla"
    props = selamla.parameters["properties"]
    assert "isim" in props
    assert "dil" in props
    # Varsayılan değeri olduğu için 'required' listesinde isim olmalı ama dil olmamalı
    required = selamla.parameters.get("required", [])
    assert "isim" in required
    # dil opsiyonel olduğu için required'da olmamalı
    assert "dil" not in required


@pytest.mark.asyncio
async def test_async_tool_execution() -> None:
    """Asenkron fonksiyonların tool olarak çalışması."""

    @tool
    async def async_hesapla(x: int) -> int:
        """Asenkron hesaplama yapar."""
        return x * 2

    result = await execute_tool(async_hesapla, x=5)
    assert result == "10"


def test_tool_definition_model() -> None:
    """ToolDefinition pydantic modelinin çalışması."""
    td = ToolDefinition(
        name="test_tool",
        description="Test aracı",
        parameters={"type": "object", "properties": {}},
        func=lambda: "ok",
    )
    assert td.name == "test_tool"
    assert td.func is not None
