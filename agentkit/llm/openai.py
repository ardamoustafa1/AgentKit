import os
from typing import List, AsyncGenerator, Any, Optional
from openai import AsyncOpenAI

from agentkit.llm.base import BaseLLM, retry_with_backoff
from agentkit.types.schemas import Message, LLMResponse, TokenUsage


class OpenAILLM(BaseLLM):
    """
    OpenAI (GPT) modelleri için implementasyon.
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature)
        self.client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @retry_with_backoff(max_retries=3)
    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> LLMResponse:
        formatted_msgs = [{"role": m.role, "content": m.content} for m in messages]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_msgs,  # type: ignore
            temperature=self.temperature,
        )

        usage = TokenUsage(
            input_tokens=getattr(response.usage, "prompt_tokens", 0),
            output_tokens=getattr(response.usage, "completion_tokens", 0),
            total_tokens=getattr(response.usage, "total_tokens", 0),
        )

        return LLMResponse(content=response.choices[0].message.content or "", usage=usage)

    async def generate_stream_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        formatted_msgs = [{"role": m.role, "content": m.content} for m in messages]

        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_msgs,  # type: ignore
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True},  # Streaming sonunda token bilgisi almak için
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else ""

            # OpenAI streaming'de kullanım bilgisi son chunk'ta gelir
            usage_obj = getattr(chunk, "usage", None)
            usage = TokenUsage()
            if usage_obj:
                usage = TokenUsage(
                    input_tokens=usage_obj.prompt_tokens,
                    output_tokens=usage_obj.completion_tokens,
                    total_tokens=usage_obj.total_tokens,
                )

            yield LLMResponse(content=content or "", usage=usage)
