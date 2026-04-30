import asyncio
import os
from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.tools import ToolRegistry, web_search, local_python_repl
from agentkit.memory.short_term import ShortTermMemory

os.environ["OPENAI_API_KEY"] = "sk-xxxxxx" # Kendi anahtarını yaz

async def main() -> None:
    llm = OpenAILLM(model_name="gpt-4o")
    tools = ToolRegistry()
    tools.register(web_search)
    tools.register(local_python_repl)
    
    memory = ShortTermMemory()
    system_prompt = "Sen araştıran ve kod yazıp hesaplayan akıllı bir asistansın."
    
    # Agent'ı başlat. require_human_approval=True yaparsak tool çalışmadan onay ister.
    agent = Agent(llm=llm, tools=tools, memory=memory, system_prompt=system_prompt, require_human_approval=False)

    print("Kullanıcı: Şu anki OpenAI CEO'su kim? Ayrıca 35'in faktöriyeli nedir?\n")
    
    print("Ajan Streaming Yanıtı:")
    async for chunk in agent.arun("Şu anki OpenAI CEO'su kim? Ayrıca 35'in faktöriyeli nedir?"):
        print(chunk, end="", flush=True)

    print(f"\n\nMaliyet (USD): ${agent.cost_tracker.get_estimated_usd():.5f}")
    print(f"Toplam Token: {agent.cost_tracker.input_tokens + agent.cost_tracker.output_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
