import asyncio
from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.orchestrator import Team
from agentkit.memory import ShortTermMemory
from agentkit.tools import ToolRegistry, web_search, local_python_repl
from agentkit.tools.integrations import github_get_issue


async def main() -> None:
    # 1. LLM Kurulumu
    llm = OpenAILLM(model_name="gpt-4-turbo-preview")

    # 2. Uzman Ajanları Yarat
    # Araştırmacı Ajan: Web'de arama yapabilir
    researcher_tools = ToolRegistry()
    researcher_tools.register(web_search)
    researcher = Agent(
        llm=llm,
        tools=researcher_tools,
        memory=ShortTermMemory(),
        system_prompt="Sen usta bir araştırmacısın. Web'i tarar ve doğru bilgiyi bulursun.",
    )

    # Yazılımcı Ajan: Python kodu yazıp test edebilir ve GitHub'dan veri çekebilir
    coder_tools = ToolRegistry()
    coder_tools.register(local_python_repl)
    coder_tools.register(github_get_issue)
    coder = Agent(
        llm=llm,
        tools=coder_tools,
        memory=ShortTermMemory(),
        system_prompt="Sen kıdemli bir yazılımcısın. Python kodu yazarsın ve gerektiğinde kodu test edersin.",
    )

    # 3. Yönetici Ajanı Yarat
    manager = Agent(
        llm=llm,
        tools=ToolRegistry(),  # Kendi araçlarına ek olarak Team sınıfı delegate ekleyecek
        memory=ShortTermMemory(),
        system_prompt="Sen baş mühendissin. İhtiyacın olduğunda görevleri araştırmacıya veya yazılımcıya devredersin.",
    )

    # 4. Takımı (Team) Kur ve Ajanları Ata
    team = Team(manager=manager)
    team.add_agent("researcher", researcher)
    team.add_agent("coder", coder)

    # 5. Görevi Başlat
    task = "Lütfen önce internetten güncel Python sürümünün numarasını bul, sonra bunu yazdıran basit bir python scripti oluşturup çalıştır."
    print("🤖 Baş Mühendis görevi devraldı ve planlamaya başladı...")

    response = await team.run(task)

    print("\n--- SONUÇ ---")
    print(response.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
