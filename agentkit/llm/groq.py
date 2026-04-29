import os
from typing import List, AsyncGenerator, Any, Optional
from groq import AsyncGroq

from agentkit.llm.base import BaseLLM, retry_with_backoff
from agentkit.types.schemas import Message, LLMResponse, TokenUsage


class GroqLLM(BaseLLM):
    """
    Groq Cloud API için LLM implementasyonu.
    """

    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature)
        self.client = AsyncGroq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

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
        )

        async for chunk in stream:  # type: ignore
            content = chunk.choices[0].delta.content if chunk.choices else ""

            # Groq streaming'de kullanım bilgisi genelde son chunk'ta x_groq altında gelir
            # Basitlik için burada TokenUsage boş geçilebilir veya chunk'tan çekilebilir
            usage_obj = getattr(chunk, "usage", None)
            usage = TokenUsage()
            if usage_obj:
                usage = TokenUsage(
                    input_tokens=getattr(usage_obj, "prompt_tokens", 0),
                    output_tokens=getattr(usage_obj, "completion_tokens", 0),
                    total_tokens=getattr(usage_obj, "total_tokens", 0),
                )

            yield LLMResponse(content=content or "", usage=usage)
