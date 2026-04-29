from pydantic import BaseModel
from typing import Optional, Literal


class TokenUsage(BaseModel):
    """Her LLM yanıtında dönecek standart token kullanım özeti."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class Message(BaseModel):
    """Sistem içinde dolaşacak standart mesaj şeması."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class LLMResponse(BaseModel):
    """LLM'den dönen standartlaştırılmış yanıt şeması."""

    content: str
    role: Literal["assistant"] = "assistant"
    usage: TokenUsage
