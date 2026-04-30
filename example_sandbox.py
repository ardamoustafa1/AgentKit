import asyncio
from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.tools.base import ToolRegistry
from agentkit.tools.builtins import sandbox_python_repl
from agentkit.memory.short_term import ShortTermMemory

async def main():
    llm = OpenAILLM(model_name="gpt-4o-mini")
    
    # Sadece Sandbox aracını kaydediyoruz
    registry = ToolRegistry()
    registry.register(sandbox_python_repl)
    
    agent = Agent(
        name="DataScientist",
        llm=llm,
        tools=registry,
        memory=ShortTermMemory(),
        system_prompt="Sen güvenli ve izole bir ortamda çalışan bir veri bilimcisisin. Matematiksel işlemleri 'sandbox_python_repl' aracıyla yap."
    )
    
    # 1. Görev
    task1 = "Bana 1'den 10'a kadar olan sayıların karesini hesaplayan bir Python kodu yaz ve sandbox'ta çalıştırıp sonucunu ver."
    print(f"\n👨‍💻 Kullanıcı: {task1}\n")
    response1 = await agent.run(task1)
    
    print("\n--- NİHAİ SONUÇ ---")
    print(response1.final_answer)

    # 2. Görev (Sistem Bilgisi Öğrenme)
    task2 = "Şu an çalıştığın makinenin işletim sistemi versiyonunu (platform.platform()) sandbox üzerinden çalıştırıp öğrenir misin?"
    print(f"\n👨‍💻 Kullanıcı: {task2}\n")
    response2 = await agent.run(task2)
    
    print("\n--- NİHAİ SONUÇ ---")
    print(response2.final_answer)


if __name__ == "__main__":
    # Not: Çalıştırmadan önce terminalde 'export E2B_API_KEY=your_key' ayarlamayı unutmayın!
    asyncio.run(main())
