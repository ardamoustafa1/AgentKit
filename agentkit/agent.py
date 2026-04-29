import re
import json
from typing import List, Dict, Any, AsyncGenerator, Optional
from pydantic import BaseModel
from loguru import logger

from agentkit.types.schemas import Message, TokenUsage
from agentkit.llm.base import BaseLLM
from agentkit.tools.base import ToolRegistry, execute_tool
from agentkit.memory.short_term import ShortTermMemory


class AgentStep(BaseModel):
    """Ajanın aldığı her bir kararı ve eylemi loglamak için kullanılan yapı."""

    step_type: str  # "thought", "action", "observation", "response"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Ajanın nihai dönüş objesi."""

    final_answer: str
    steps: List[AgentStep]
    total_tokens: int
    estimated_usd: float


class CostTracker:
    """Token kullanımlarını takip edip yaklaşık dolar maliyetini hesaplar."""

    def __init__(self, input_price_per_m: float = 5.0, output_price_per_m: float = 15.0):
        self.input_tokens = 0
        self.output_tokens = 0
        self.input_price_per_m = input_price_per_m
        self.output_price_per_m = output_price_per_m

    def add(self, usage: TokenUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def get_estimated_usd(self) -> float:
        return (self.input_tokens / 1_000_000) * self.input_price_per_m + (
            self.output_tokens / 1_000_000
        ) * self.output_price_per_m


class Agent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        memory: ShortTermMemory,
        system_prompt: str,
        max_iterations: int = 5,
        require_human_approval: bool = False,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
        self.require_human_approval = require_human_approval
        self.cost_tracker = CostTracker()

        # Tool şemalarını alıp System Prompt'a ReAct formatıyla birlikte enjekte ediyoruz.
        tool_schemas = json.dumps(self.tools.get_all_schemas(), indent=2, ensure_ascii=False)
        react_instructions = f"""
{system_prompt}

Kullanabileceğin Araçlar (Tools):
{tool_schemas}

Eğer bir araca ihtiyacın varsa KESİNLİKLE aşağıdaki formatı kullan:
Thought: <Neden bu araca ihtiyaç duyduğun>
Action: <Araç adı>
Action Input: <JSON formatında parametreler>

Eğer doğrudan yanıt vereceksen veya işlemin bittiyse sadece cevabını yaz. Thought veya Action kelimelerini kullanma.
"""
        self.memory.set_system_prompt(react_instructions.strip())

    def _parse_react_response(self, text: str) -> Optional[tuple[str, str, dict[str, Any]]]:
        """LLM çıktısından Thought, Action ve Action Input ayıklar."""
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)(?=Action Input:|$)", text, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)

        if action_match and input_match:
            thought = thought_match.group(1).strip() if thought_match else "Düşünüyorum..."
            action = action_match.group(1).strip()
            action_input_str = input_match.group(1).strip()

            try:
                # Olası Markdown json bloklarını temizle
                action_input_str = action_input_str.strip("`")
                if action_input_str.startswith("json\n"):
                    action_input_str = action_input_str[5:]

                action_input = json.loads(action_input_str)
                return thought, action, action_input
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse hatası: {action_input_str}")
                # Tool adını koruyup hatayı LLM'e observation olarak iletmek için
                # parse hatasını geri döndürüyoruz.
                return (
                    thought,
                    action,
                    {"error": f"JSON parse hatası: {str(e)}. Lütfen formatı düzelt."},
                )
        return None

    async def arun(self, user_message: str) -> AsyncGenerator[str, None]:
        """Kullanıcı mesajını alır, streaming ile yanıt üretir ve ReAct döngüsünü işletir."""
        self.memory.add_message(Message(role="user", content=user_message))
        self._current_steps = []

        for iteration in range(self.max_iterations):
            logger.info(f"--- Iteration {iteration + 1}/{self.max_iterations} ---")

            full_response = ""
            async for chunk in self.llm.generate_stream_async(self.memory.get_messages()):
                full_response += chunk.content
                if chunk.usage.total_tokens > 0:
                    self.cost_tracker.add(chunk.usage)

                # Sadece cevap dönüyorsa (Action yoksa) streaming yap.
                # Eylem planlıyorsa Thought metnini streaming yapmamak daha iyidir.
                if "Action:" not in full_response:
                    yield chunk.content

            self.memory.add_message(Message(role="assistant", content=full_response))
            parsed = self._parse_react_response(full_response)

            if parsed:
                thought, action, action_input = parsed
                self._current_steps.append(AgentStep(step_type="thought", content=thought))
                self._current_steps.append(
                    AgentStep(
                        step_type="action", content="", tool_name=action, tool_input=action_input
                    )
                )

                logger.info(f"[Agent Thought]: {thought}")
                logger.warning(f"[Agent Action]: {action}({action_input})")

                target_tool = self.tools.get_tool(action)
                if not target_tool:
                    observation = f"Hata: {action} adında bir araç bulunamadı."
                else:
                    if self.require_human_approval:
                        # Human-in-the-loop: Terminal üzerinden onay iste
                        # Gerçek projede bu bir callback/websocket event'i olabilir.
                        onay = input(
                            f"Tool {action} çalıştırılacak ({action_input}). Onaylıyor musun? (y/n): "
                        )
                        if onay.lower() != "y":
                            observation = "Kullanıcı bu aracın çalıştırılmasını reddetti."
                            self.memory.add_message(
                                Message(role="user", content=f"Observation: {observation}")
                            )
                            continue

                    observation = await execute_tool(target_tool, **action_input)

                logger.success(f"[Agent Observation]: {observation[:100]}...")
                self._current_steps.append(AgentStep(step_type="observation", content=observation))
                self.memory.add_message(Message(role="user", content=f"Observation: {observation}"))
            else:
                # Herhangi bir action bulunmadıysa, LLM kullanıcıya nihai cevabı üretmiştir.
                self._current_steps.append(AgentStep(step_type="response", content=full_response))
                break
        else:
            msg = "\n[Sistem]: Maksimum iterasyon limitine ulaşıldı."
            self._current_steps.append(AgentStep(step_type="response", content=msg))
            yield msg

    async def run(self, user_message: str) -> AgentResponse:
        """Asenkron çalışır ama streaming yapmaz. Tüm adımlar bittiğinde AgentResponse döner."""
        async for chunk in self.arun(user_message):
            pass

        steps = getattr(self, "_current_steps", [])
        final_answer = ""
        if steps and steps[-1].step_type == "response":
            final_answer = steps[-1].content
        elif steps:
            final_answer = steps[-1].content

        return AgentResponse(
            final_answer=final_answer.strip(),
            steps=steps,
            total_tokens=self.cost_tracker.input_tokens + self.cost_tracker.output_tokens,
            estimated_usd=self.cost_tracker.get_estimated_usd(),
        )
