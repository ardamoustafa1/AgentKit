import abc
import asyncio
from typing import List, AsyncGenerator, Callable, Any, Optional
from functools import wraps
from loguru import logger

from agentkit.types.schemas import Message, LLMResponse


def retry_with_backoff(
    max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0
) -> Callable[..., Callable[..., Any]]:
    """
    LLM API çağrılarında yaşanacak anlık kesintiler (Rate limit, 500 hataları vs.)
    için exponential backoff uygulayan decorator.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"[{func.__name__}] {max_retries} deneme sonrası başarısız: {e}"
                        )
                        raise
                    logger.warning(
                        f"[{func.__name__}] Deneme {attempt + 1} başarısız: {e}. {delay}s sonra tekrar deneniyor..."
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor

        return wrapper

    return decorator


class BaseLLM(abc.ABC):
    """
    Tüm LLM sağlayıcıları için temel sınıf.
    Alt sınıflar 'generate_async' ve 'generate_stream_async' metotlarını implemente etmek zorundadır.
    """

    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    @abc.abstractmethod
    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> LLMResponse:
        """Asenkron olarak LLM'den tam yanıt (non-streaming) üretir."""
        pass

    @abc.abstractmethod
    def generate_stream_async(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> AsyncGenerator[LLMResponse, None]:
        """Asenkron olarak LLM yanıtını parça parça (streaming) üretir."""
        pass

    def generate(self, messages: List[Message], tools: Optional[List[Any]] = None) -> LLMResponse:
        """Senkron kod bloklarında (sync) çalışabilmek için Async-to-Sync wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.generate_async(messages, tools))
        finally:
            loop.close()

    def generate_stream(self, messages: List[Message], tools: Optional[List[Any]] = None) -> Any:
        """Senkron ortamlarda streaming yanıtları almak için Generator wrapper."""
        loop = asyncio.new_event_loop()
        async_gen = self.generate_stream_async(messages, tools)
        try:
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
