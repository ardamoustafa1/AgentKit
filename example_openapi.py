import asyncio
from agentkit.tools.openapi import import_openapi
from agentkit.tools.base import ToolRegistry
from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.memory.short_term import ShortTermMemory

async def main():
    url = "https://petstore.swagger.io/v2/swagger.json"
    print(f"🔄 '{url}' adresinden OpenAPI şeması indiriliyor ve analiz ediliyor...")
    
    # 1. OpenAPI Şemasından dinamik araçları oluştur
    dynamic_tools = import_openapi(url)
    
    registry = ToolRegistry()
    for t in dynamic_tools:
        registry.register(t)
        
    print(f"✅ Başarılı! Toplam {len(dynamic_tools)} adet endpoint 'Tool' olarak sisteme eklendi.")
    
    # 2. Ajanı oluştur ve çalıştır
    llm = OpenAILLM(model_name="gpt-4o-mini")
    agent = Agent(
        llm=llm,
        tools=registry,
        memory=ShortTermMemory(),
        system_prompt="Sen bir API test asistanısın. Petstore API'sini kullanarak sorulara cevap ver."
    )
    
    task = "Find pets by status endpointini kullanarak 'available' olan evcil hayvanları listele. Sadece 3 tanesinin adını söyle."
    print(f"\n🤖 Ajan Görevi: {task}\n")
    
    response = await agent.run(task)
    print("\n--- SONUÇ ---")
    print(response.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
