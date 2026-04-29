import asyncio
import os
from agentkit.types.schemas import Message
from agentkit.memory import ShortTermMemory, LongTermMemory, EntityMemory
from agentkit.llm.openai import OpenAILLM

async def demo_short_term() -> None:
    print("\n--- Short Term Memory Demo (Sliding Window) ---")
    # Çok düşük token limiti veriyoruz ki eski mesajlar hemen silinsin
    memory = ShortTermMemory(max_tokens=50) 
    
    memory.set_system_prompt("Sen kısa cevaplar veren bir asistansın.")
    
    memory.add_message(Message(role="user", content="Merhaba, benim adım Arda. Havalar nasıl?"))
    memory.add_message(Message(role="assistant", content="Merhaba Arda! Havalar güneşli."))
    memory.add_message(Message(role="user", content="Bugün ne yapsam?"))
    
    print("Mevcut Mesajlar (Eski mesajlar silinmiş olabilir):")
    for m in memory.get_messages():
        print(f"[{m.role.upper()}]: {m.content[:40]}")

async def demo_long_term() -> None:
    print("\n--- Long Term Memory Demo (ChromaDB + SentenceTransformers) ---")
    # memory = LongTermMemory(persist_directory=None) # InMemory Test
    try:
        memory = LongTermMemory(persist_directory=None)
        
        print("Bilgiler ekleniyor...")
        memory.add("Python, Guido van Rossum tarafından 1991'de geliştirilmiş bir dildir.", metadata={"category": "tech"})
        memory.add("Arda'nın en sevdiği renk mavidir ve yapay zeka ile ilgilenmektedir.", metadata={"user": "arda"})
        memory.add("Kedi mırlaması stresi azaltabilir.", metadata={"category": "animals"})
        
        query = "Arda neyi seviyor?"
        print(f"\nSorgu: '{query}'")
        results = memory.search(query, k=1)
        
        for r in results:
            print(f"Bulunan: {r['document']} (Uzaklık: {r['distance']:.3f})")
    except ImportError:
        print("sentence-transformers kütüphanesi eksik, LTM testi atlanıyor.")

async def demo_entity() -> None:
    print("\n--- Entity Memory Demo (LLM Extraction) ---")
    # Örnek için LLM'e ihtiyacımız var (OpenAILLM vs)
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY bulunamadı, bu adım atlanıyor.")
        return

    llm = OpenAILLM(model_name="gpt-3.5-turbo")
    entity_memory = EntityMemory(llm=llm)
    
    text = "Selam, ben Arda. İstanbul'da yaşıyorum. Yazılım mühendisi olarak 5 yıldır çalışıyorum. Köpekleri çok severim."
    print("Konuşma:")
    print(text)
    
    print("\nBilgiler çıkarılıyor...")
    await entity_memory.extract_and_store(text)
    
    print("\nÇıkarılan Bilgiler:")
    print(entity_memory.get_context_string())

async def main() -> None:
    await demo_short_term()
    await demo_long_term()
    await demo_entity()

if __name__ == "__main__":
    asyncio.run(main())
