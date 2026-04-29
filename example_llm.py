import asyncio
import os
from agentkit.types.schemas import Message
from agentkit.llm.ollama import OllamaLLM

os.environ["OPENAI_API_KEY"] = "sk-xxxxxx"

async def main() -> None:
    # llm = OpenAILLM(model_name="gpt-4o")
    llm = OllamaLLM(model_name="llama3")

    messages = [
        Message(role="system", content="Sen kıdemli bir yazılım mimarısın. Her zaman kısa ve öz cevap verirsin."),
        Message(role="user", content="Microservice mimarisi nedir? 1 cümle ile açıkla.")
    ]

    print("--- Asenkron Streaming Örneği ---")
    async for response in llm.generate_stream_async(messages):
        print(response.content, end="", flush=True)
        if response.usage.total_tokens > 0:
            print(f"\n[Token Kullanımı: {response.usage.total_tokens}]")

    print("\n--- Asenkron Non-Streaming Örneği ---")
    full_response = await llm.generate_async(messages)
    print(full_response.content)
    print(f"Toplam Token: {full_response.usage.total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
