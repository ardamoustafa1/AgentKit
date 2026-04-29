import json
from typing import Dict, Any
from loguru import logger
from agentkit.llm.base import BaseLLM
from agentkit.types.schemas import Message


class EntityMemory:
    """
    Konuşma geçmişinden belirli varlıkları (kullanıcı bilgileri, tarihler, tercihler)
    LLM yardımıyla otomatik çıkarıp Key-Value store olarak saklayan bellek.
    """

    def __init__(self, llm: BaseLLM):
        self.entities: Dict[str, Any] = {}
        self.llm = llm

    async def extract_and_store(self, conversation_text: str) -> None:
        """Konuşma metnini analiz eder ve yeni bilgileri mevcut bilgi deposuna yazar."""
        prompt = f"""
Aşağıdaki konuşma metninden kişi adları, yerler, tarihler, meslekler veya tercihler gibi önemli varlıkları (entities) çıkar.
Bana SADECE JSON formatında bir key-value sözlüğü döndür. Örnek:
{{"Kullanıcı Adı": "Ahmet", "Şehir": "İstanbul", "Meslek": "Mühendis"}}
Eğer çıkarılacak yeni bir bilgi yoksa, sadece boş bir JSON {{}} döndür.

Konuşma:
{conversation_text}
"""
        messages = [
            Message(
                role="system",
                content="Sen bilgi çıkaran bir asistansın. Yalnızca geçerli bir JSON döndür. Başka metin yazma.",
            ),
            Message(role="user", content=prompt),
        ]

        response = await self.llm.generate_async(messages)
        content = response.content.strip()

        # Olası Markdown kod bloklarını temizle
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        try:
            extracted_data = json.loads(content.strip())
            if isinstance(extracted_data, dict) and extracted_data:
                self.entities.update(extracted_data)
                logger.info(f"[EntityMemory] Yeni bilgiler eklendi/güncellendi: {extracted_data}")
        except json.JSONDecodeError:
            logger.warning(f"[EntityMemory] JSON parse hatası. LLM Çıktısı: {response.content}")

    def get_all(self) -> Dict[str, Any]:
        """Kaydedilmiş tüm entity'leri döndürür."""
        return self.entities

    def get_context_string(self) -> str:
        """Sistem prompt'una eklenebilecek formatlı string döner."""
        if not self.entities:
            return ""
        context = "Bilinen Kullanıcı Varlıkları ve Tercihleri:\n"
        for k, v in self.entities.items():
            context += f"- {k}: {v}\n"
        return context
