import inspect
import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel
from loguru import logger


class ToolDefinition(BaseModel):
    """
    LLM'e gönderilecek Tool (Function Calling) şemasının standart temsili.
    """

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema formatında parametreler

    # Gerçek fonksiyon referansı
    func: Any = None

    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """Sistemdeki mevcut tool'ları kaydeden ve isme göre bulmayı sağlayan kayıt defteri."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Tool kaydedildi: {tool.name}")

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def get_all_schemas(self) -> list[Dict[str, Any]]:
        """LLM'e gönderilecek formatta tüm araçları listeler."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]


async def execute_tool(tool_def: ToolDefinition, **kwargs: Any) -> str:
    """
    Seçilen tool'u verilen parametrelerle çalıştırır, hataları yakalar ve loglar.
    Sync ve Async fonksiyonları otomatik yönetir.
    """
    logger.info(f"[Tool Execution] Çalıştırılıyor: {tool_def.name} | Parametreler: {kwargs}")
    try:
        if inspect.iscoroutinefunction(tool_def.func):
            result = await tool_def.func(**kwargs)
        else:
            result = await asyncio.to_thread(tool_def.func, **kwargs)

        str_result = str(result)
        logger.success(
            f"[Tool Execution] Başarılı: {tool_def.name} | Sonuç uzunluğu: {len(str_result)}"
        )
        return str_result
    except Exception as e:
        error_msg = f"Hata oluştu ({tool_def.name}): {str(e)}"
        logger.error(f"[Tool Execution] {error_msg}")
        return error_msg
