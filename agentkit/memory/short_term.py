import tiktoken
from typing import List, Optional
from loguru import logger
from agentkit.types.schemas import Message


class ShortTermMemory:
    """
    Konuşma geçmişini (Message listesi) tutan ve sliding window
    yaklaşımıyla maksimum token limitini yöneten kısa vadeli bellek.
    """

    def __init__(self, max_tokens: int = 4000, model_for_token_counting: str = "gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        self.system_prompt: Optional[Message] = None
        self.messages: List[Message] = []
        try:
            self.encoding = tiktoken.encoding_for_model(model_for_token_counting)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def set_system_prompt(self, content: str) -> None:
        """Sistem komutunu ayrı bir değişkende saklar, pencere dışına itilmesini engeller."""
        self.system_prompt = Message(role="system", content=content)

    def add_message(self, message: Message) -> None:
        """Yeni bir mesaj ekler ve token limitini aşarsak en eski mesajları atar."""
        if message.role == "system":
            self.system_prompt = message
        else:
            self.messages.append(message)
            self._apply_sliding_window()

    def get_messages(self) -> List[Message]:
        """LLM'e gönderilecek tam mesaj listesini döndürür."""
        result = []
        if self.system_prompt:
            result.append(self.system_prompt)
        result.extend(self.messages)
        return result

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _apply_sliding_window(self) -> None:
        """Maksimum token sınırı aşılırsa, en eski (ilk) kullanıcı/asistan mesajlarını siler."""
        system_tokens = self._count_tokens(self.system_prompt.content) if self.system_prompt else 0

        while self.messages:
            total_tokens = system_tokens + sum(self._count_tokens(m.content) for m in self.messages)
            if total_tokens <= self.max_tokens:
                break

            # En eski mesajı sil
            removed = self.messages.pop(0)
            logger.debug(
                f"[ShortTermMemory] Token limiti aşıldı. Eski mesaj silindi: {removed.content[:30]}..."
            )
