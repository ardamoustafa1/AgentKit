from typing import Dict
from pydantic import BaseModel, Field

from agentkit.agent import Agent, AgentResponse
from agentkit.tools.base import ToolDefinition, TransferException
from loguru import logger


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


class Swarm:
    """
    Swarm, ajanların bir merkezi yönetici (manager) olmadan birbiriyle iletişime geçebildiği
    (Graph-based/Non-hierarchical) modern bir multi-agent sürüsüdür.
    """

    def __init__(self, starting_agent: Agent) -> None:
        self.starting_agent = starting_agent
        self.agents: Dict[str, Agent] = {starting_agent.name: starting_agent}

    def add_agent(self, agent: Agent) -> None:
        """Swarm (sürü) ağına yeni bir ajan ekler."""
        self.agents[agent.name] = agent

    def _inject_transfer_tools(self) -> None:
        """Sürüdeki her ajana, diğer ajanlara geçiş yapabilmesi için transfer_to_agent aracını enjekte eder."""
        agent_names = list(self.agents.keys())
        for name, agent in self.agents.items():
            
            def make_transfer_func(current_agent_name=name):
                def transfer_to_agent(target_agent: str, context_message: str) -> str:
                    """Görevi sürüdeki başka bir ajana devreder. Bu aracı kullandığında kontrolü kaybedersin."""
                    if target_agent not in self.agents:
                        return f"Hata: {target_agent} bulunamadı. Mevcut ajanlar: {agent_names}"
                    
                    # Bu noktada istisna fırlatıyoruz ki execute_tool ve arun döngüsü anında kırılsın!
                    raise TransferException(target_agent=target_agent, message=context_message)
                return transfer_to_agent

            # Eğer zaten ekliyse tekrar eklememek için kontrol edebiliriz
            if not agent.tools.get_tool("transfer_to_agent"):
                transfer_tool = ToolDefinition(
                    name="transfer_to_agent",
                    description=f"Görevi başka bir ajana devreder. Kullanılabilir ajanlar: {agent_names}. Bunu çalıştırdığında kontrolü kaybedersin.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "target_agent": {"type": "string"},
                            "context_message": {"type": "string", "description": "Hedef ajana devredilirken gönderilecek bağlam veya yönerge."}
                        },
                        "required": ["target_agent", "context_message"]
                    },
                    func=make_transfer_func()
                )
                agent.tools.register(transfer_tool)

            # System prompt'u güncelle
            orig = agent.memory.system_prompt.content if agent.memory.system_prompt else ""
            if "Swarm Sürüsü" not in orig:
                addon = f"\n[Swarm Sürüsü] Sen bir ajan sürüsünün parçasısın. Kendi yeteneklerinin yetmediği yerde 'transfer_to_agent' aracını kullanarak görevi şu uzmanlara devredebilirsin: {agent_names}."
                agent.memory.set_system_prompt(orig + addon)

    async def run(self, user_message: str) -> AgentResponse:
        """Swarm döngüsünü başlatır ve bir ajan son sözü söyleyene kadar (transfer yapmayana kadar) çalışır."""
        self._inject_transfer_tools()
        
        current_agent = self.starting_agent
        current_message = user_message
        
        while True:
            logger.info(f"🐝 [Swarm] Aktif Ajan: {current_agent.name}")
            try:
                # Ajan görevini yapmaya başlar
                response = await current_agent.run(current_message)
                
                # Ajan hiçbir transfer yapmadan kendi işini bitirip döndüyse, sürü görevi bitirir.
                logger.success(f"🏁 [Swarm] Görev {current_agent.name} tarafından tamamlandı!")
                return response
                
            except TransferException as e:
                logger.warning(f"🔄 [Swarm Transfer] {current_agent.name} -> {e.target_agent} | Neden: {e.message}")
                current_agent = self.agents[e.target_agent]
                current_message = f"Bir önceki ajan sana görevi şu notla devretti:\n{e.message}"
                # Döngü yeni ajanın yeni prompt'u ile devam eder...
