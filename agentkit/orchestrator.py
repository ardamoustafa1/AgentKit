from typing import Dict
from pydantic import BaseModel, Field

from agentkit.agent import Agent, AgentResponse
from agentkit.tools.base import ToolDefinition


class DelegateToolParams(BaseModel):
    agent_name: str = Field(..., description="The name of the agent to delegate to.")
    task_description: str = Field(
        ...,
        description="The clear and comprehensive task description to send to the delegated agent.",
    )


class Team:
    """
    Team, birden fazla ajanın kaydedildiği ve birbirlerine görev devredebileceği
    (Delegate) bir multi-agent yöneticisidir.
    """

    def __init__(self, manager: Agent) -> None:
        self.manager = manager
        self.agents: Dict[str, Agent] = {}
        # Manager'in tool'ları arasına kendimizi register ediyoruz.
        self._register_delegation_tool()

    def add_agent(self, name: str, agent: Agent) -> None:
        """Yeni bir ajanı takıma ekler."""
        self.agents[name] = agent

    def _register_delegation_tool(self) -> None:
        """Manager ajana devretme aracını (tool) ekler."""

        def delegate(agent_name: str, task_description: str) -> str:
            """
            Bir görevi takımdaki diğer uzman ajanlardan birine devreder ve sonucunu bekler.
            """
            if agent_name not in self.agents:
                return f"Hata: '{agent_name}' adında bir ajan bulunamadı. Mevcut ajanlar: {list(self.agents.keys())}"

            target_agent = self.agents[agent_name]

            # Sub-agent asenkron çalışıyor olsa da, mevcut yapı senkron olarak wrap edilmeli.
            # Fakat biz async uyumlu yapıda tasarladık, bu yüzden bir async task run yapmalıyız
            # veya sync wrapper kullanmalıyız. Agent'in .run() metodunun coroutine olduğunu biliyoruz.
            # Not: Tool'lar default olarak senkron ise execute_tool üzerinden loop ile çalıştırılır.

            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                import nest_asyncio  # type: ignore[import-untyped]
                nest_asyncio.apply()

            # Agent'ı senkron olarak koştur
            coro = target_agent.run(task_description)
            response: AgentResponse = loop.run_until_complete(coro)

            return f"Agent '{agent_name}' Görevi Tamamladı.\nSonuç:\n{response.final_answer}"

        # Delegate fonksiyonunu manuel olarak tool schema'ya çevirelim.
        # Aslında @tool decorator'ı kullanabiliriz, ama self state'ine ihtiyacımız var.
        delegate_tool = ToolDefinition(
            name="delegate_to_agent",
            description="Bir görevi takımdaki diğer uzman ajanlardan birine devreder ve sonucunu bekler. Sadece gerektiğinde uzman desteği almak için kullan.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "The name of the agent to delegate to. (e.g. 'coder', 'researcher')",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "The clear and comprehensive task description to send to the delegated agent.",
                    },
                },
                "required": ["agent_name", "task_description"],
            },
            func=delegate,
        )
        self.manager.tools.register(delegate_tool)

    async def run(self, prompt: str) -> AgentResponse:
        """
        Görevi yönetici (Manager) ajana başlatır.
        """
        # Manager'a sistem promptunda takım üyelerini öğret
        agents_info = ", ".join(self.agents.keys())
        system_addon = f"\n\nSen takım yöneticisisin. Şu ajanlara görev devredebilirsin: {agents_info}. Bunu 'delegate_to_agent' aracıyla yap."

        # System promptuna ekleme yapıyoruz
        original_prompt = (
            self.manager.memory.system_prompt.content if self.manager.memory.system_prompt else ""
        )
        if "takım yöneticisisin" not in original_prompt:
            self.manager.memory.set_system_prompt(original_prompt + system_addon)

        return await self.manager.run(prompt)
