import httpx
import json
from typing import List, AsyncGenerator, Any, Optional

from agentkit.llm.base import BaseLLM, retry_with_backoff
from agentkit.types.schemas import Message, LLMResponse, TokenUsage


class OllamaLLM(BaseLLM):
    """
    Lokal Ollama modelleri için implementasyon (Llama3, Mistral vs.).
    """

    def __init__(
        self,
        model_name: str = "llama3",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(model_name, temperature)
        self.base_url = base_url

    @retry_with_backoff(max_retries=3)
    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> LLMResponse:
        formatted_msgs = [{"role": m.role, "content": m.content} for m in messages]
        payload = {
            "model": self.model_name,
            "messages": formatted_msgs,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()

            usage = TokenUsage(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            )

            return LLMResponse(content=data["message"]["content"], usage=usage)

    async def generate_stream_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        formatted_msgs = [{"role": m.role, "content": m.content} for m in messages]
        payload = {
            "model": self.model_name,
            "messages": formatted_msgs,
            "stream": True,
            "options": {"temperature": self.temperature},
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload, timeout=120.0
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")

                    # Ollama'da streaming sırasında token kullanım metrikleri done=True olduğunda en son chunk ile gelir
                    usage = TokenUsage()
                    if data.get("done"):
                        usage = TokenUsage(
                            input_tokens=data.get("prompt_eval_count", 0),
                            output_tokens=data.get("eval_count", 0),
                            total_tokens=data.get("prompt_eval_count", 0)
                            + data.get("eval_count", 0),
                        )

                    yield LLMResponse(content=content, usage=usage)
