import os
from typing import List, AsyncGenerator, Any, Optional
from anthropic import AsyncAnthropic

from agentkit.llm.base import BaseLLM, retry_with_backoff
from agentkit.types.schemas import Message, LLMResponse, TokenUsage


class AnthropicLLM(BaseLLM):
    """
    Anthropic (Claude) modelleri için implementasyon.
    """

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature)
        self.client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @retry_with_backoff(max_retries=3)
    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> LLMResponse:
        # Anthropic 'system' rolünü parametre olarak alır, mesaj dizisi içinde istemez.
        system_msg = next((m.content for m in messages if m.role == "system"), "")
        formatted_msgs = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]

        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            temperature=self.temperature,
            system=system_msg,
            messages=formatted_msgs,  # type: ignore
        )

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(content=response.content[0].text, usage=usage)

    async def generate_stream_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        system_msg = next((m.content for m in messages if m.role == "system"), "")
        formatted_msgs = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]

        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=4096,
            temperature=self.temperature,
            system=system_msg,
            messages=formatted_msgs,  # type: ignore
        ) as stream:
            input_tokens = 0

            async for event in stream:
                if event.type == "message_start":
                    input_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_delta":
                    yield LLMResponse(
                        content=event.delta.text, usage=TokenUsage(input_tokens=input_tokens)
                    )
                elif event.type == "message_delta":
                    output_tokens = event.usage.output_tokens
                    yield LLMResponse(
                        content="",
                        usage=TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                        ),
                    )
