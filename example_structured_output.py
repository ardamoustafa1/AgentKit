import asyncio
from pydantic import BaseModel, Field
from typing import List

from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.tools import ToolRegistry, web_search
from agentkit.memory.short_term import ShortTermMemory

# 1. Pydantic Şemanızı (İstenilen Çıktı Formatı) Tanımlayın
class KullaniciProfili(BaseModel):
    isim: str = Field(..., description="Kişinin tam adı")
    meslek: str = Field(..., description="Kişinin yaptığı iş veya mesleği")
    yetenekler: List[str] = Field(..., description="Kişinin sahip olduğu yetenekler veya ilgi alanları")
    yas: int = Field(..., description="Eğer yaş bulunamazsa tahmini veya 0 yazın.")


async def main():
    llm = OpenAILLM(model_name="gpt-4o-mini")
    tools = ToolRegistry()
    tools.register(web_search)

    agent = Agent(
        llm=llm,
        tools=tools,
        memory=ShortTermMemory(),
        system_prompt="Sen çok zeki bir veri ayıklama asistanısın. Gerekirse web'de arama yapabilirsin."
    )

    task = "Linus Torvalds kimdir? Mesleği nedir ve hangi programlama dillerini/yetenekleri bilir? Benim için bu bilgileri bul ve derle."

    print(f"🤖 Ajan '{task}' görevini çalıştırıyor...\n")

    # 2. agent.run fonksiyonuna 'response_model' parametresini geçiyoruz.
    # AgentResponse[KullaniciProfili] dönecek ve IDE'niz otomatik tamamlama sunacak!
    response = await agent.run(task, response_model=KullaniciProfili)

    print("\n--- ÇIKTI (Düz Metin) ---")
    print(response.final_answer)

    print("\n--- STRUCTURED OUTPUT (Pydantic Model) ---")
    
    # response.structured_output doğrudan KullaniciProfili nesnesidir!
    kullanici = response.structured_output
    if kullanici:
        print(f"İsim: {kullanici.isim}")
        print(f"Meslek: {kullanici.meslek}")
        print(f"Yetenekler: {', '.join(kullanici.yetenekler)}")
        print(f"Yaş: {kullanici.yas}")
    else:
        print("HATA: JSON şemaya dönüştürülemedi.")

    print(f"\n💸 Toplam Maliyet: ${response.estimated_usd:.5f}")


if __name__ == "__main__":
    asyncio.run(main())
