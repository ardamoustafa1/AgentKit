import asyncio
from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.memory.short_term import ShortTermMemory
from agentkit.tools.base import ToolRegistry
from agentkit.orchestrator import Swarm

async def main():
    llm = OpenAILLM(model_name="gpt-4o-mini")
    
    # 1. Triage (Yönlendirici) Ajanı
    triage_agent = Agent(
        name="TriageAgent",
        llm=llm,
        tools=ToolRegistry(),
        memory=ShortTermMemory(),
        system_prompt="Sen bir yönlendirme asistanısın. Kullanıcının isteğini analiz et ve eğer kod yazılması gerekiyorsa CoderAgent'a, eğer şiir/makale yazılması gerekiyorsa WriterAgent'a devret. Kendi başına görev yapma, sadece doğru kişiye devret."
    )
    
    # 2. Coder Ajanı
    coder_agent = Agent(
        name="CoderAgent",
        llm=llm,
        tools=ToolRegistry(),
        memory=ShortTermMemory(),
        system_prompt="Sen usta bir yazılımcısın. İstenilen kodu yazar ve görevi bitirirsin."
    )
    
    # 3. Writer Ajanı
    writer_agent = Agent(
        name="WriterAgent",
        llm=llm,
        tools=ToolRegistry(),
        memory=ShortTermMemory(),
        system_prompt="Sen bir şairsin. İstenilen konudaki şiiri yazar ve görevi bitirirsin."
    )
    
    # Swarm (Sürü) oluştur ve ajanları ekle (TriageAgent ilk karşılayan)
    swarm = Swarm(starting_agent=triage_agent)
    swarm.add_agent(coder_agent)
    swarm.add_agent(writer_agent)
    
    # Görev 1
    task1 = "Bana gökyüzü hakkında kısa bir şiir yaz."
    print(f"\n👨‍💻 Kullanıcı: {task1}\n")
    response1 = await swarm.run(task1)
    
    print("\n--- NİHAİ SONUÇ ---")
    print(response1.final_answer)
    print("\n" + "="*50)

    # Görev 2
    task2 = "Python'da bir liste içindeki çift sayıları bulan bir fonksiyon yazar mısın?"
    print(f"\n👨‍💻 Kullanıcı: {task2}\n")
    response2 = await swarm.run(task2)
    
    print("\n--- NİHAİ SONUÇ ---")
    print(response2.final_answer)

if __name__ == "__main__":
    asyncio.run(main())
