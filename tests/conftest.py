import pytest
from typing import List, Optional, Any, AsyncGenerator

from agentkit.llm.base import BaseLLM
from agentkit.types.schemas import Message, LLMResponse, TokenUsage
from agentkit.tools import tool, ToolRegistry


class MockLLM(BaseLLM):
    """
    Testler sırasında gerçek API çağrıları yapmak yerine,
    önceden tanımlanmış (mocked) cevapları dönen sanal LLM sınıfı.
    """

    def __init__(self, responses: Optional[List[str]] = None):
        super().__init__(model_name="mock-model")
        # LLM'e her çağrı yapıldığında bu listeden sıradaki cevabı döneceğiz.
        self.responses = responses or ["Mocked Response"]
        self.call_count = 0
        self.last_messages: List[Message] = []

    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> LLMResponse:
        self.last_messages = messages
        if self.call_count < len(self.responses):
            content = self.responses[self.call_count]
            self.call_count += 1
        else:
            content = self.responses[-1] if self.responses else "Default Mock Response"

        return LLMResponse(
            content=content, usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
        )

    async def generate_stream_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        response = await self.generate_async(messages, tools)

        # Sadece basit kelime kelime bölme simülasyonu
        words = response.content.split(" ")
        for i, word in enumerate(words):
            usage = TokenUsage()
            if i == len(words) - 1:
                usage = response.usage
            yield LLMResponse(content=word + (" " if i < len(words) - 1 else ""), usage=usage)


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def sample_tool() -> Any:
    @tool
    def topla(a: int, b: int) -> int:
        """İki sayıyı toplar"""
        return a + b

    return topla


@pytest.fixture
def error_tool() -> Any:
    @tool
    def hata_ver() -> str:
        """Kasten hata fırlatır"""
        raise ValueError("Kasıtlı Hata")

    return hata_ver
