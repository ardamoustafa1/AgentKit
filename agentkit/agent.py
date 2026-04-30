import re
import json
import asyncio
import os
from typing import List, Dict, Any, AsyncGenerator, Optional, TypeVar, Generic, Type
from pydantic import BaseModel
from loguru import logger

from agentkit.types.schemas import Message, TokenUsage
from agentkit.llm.base import BaseLLM
from agentkit.tools.base import ToolRegistry, execute_tool, TransferException
from agentkit.memory.short_term import ShortTermMemory
from agentkit.utils.observability import trace_agent


class AgentStep(BaseModel):
    """Ajanın aldığı her bir kararı ve eylemi loglamak için kullanılan yapı."""

    step_type: str  # "thought", "action", "observation", "response"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None


T = TypeVar("T", bound=BaseModel)

class AgentResponse(BaseModel, Generic[T]):
    """Ajanın nihai dönüş objesi."""

    final_answer: str
    structured_output: Optional[T] = None
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
        name: str = "Agent",
        max_iterations: int = 5,
        require_human_approval: bool = False,
    ):
        self.name = name
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

Eğer bir araca ihtiyacın varsa KESİNLİKLE aşağıdaki formatı kullan. Paralel çalıştırmak için birden fazla Action belirtebilirsin:
Thought: <Neden bu araca ihtiyaç duyduğun>
Action: <Araç adı>
Action Input: <JSON formatında parametreler>
Action: <Başka bir araç adı>
Action Input: <Parametreler>

Eğer doğrudan yanıt vereceksen veya işlemin bittiyse sadece cevabını yaz. Thought veya Action kelimelerini kullanma.
"""
        self.memory.set_system_prompt(react_instructions.strip())

    def _parse_react_response(self, text: str) -> List[tuple[str, str, dict[str, Any]]]:
        """LLM çıktısından Thought, Action ve Action Input ayıklar. Paralel çağrılar için list döndürür."""
        results = []
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "Düşünüyorum..."

        # Find all Action and Action Input pairs
        pattern = re.compile(r"Action:\s*(.*?)\n\s*Action Input:\s*(.*?)(?=\n\s*Action:|\Z)", re.DOTALL)
        matches = pattern.findall(text)

        # Fallback for single block without strict formatting
        if not matches:
            action_match = re.search(r"Action:\s*(.*?)(?=Action Input:|$)", text, re.DOTALL)
            input_match = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
            if action_match and input_match:
                matches = [(action_match.group(1), input_match.group(1))]

        for action_name, action_input_str in matches:
            action = action_name.strip()
            action_input_str = action_input_str.strip()
            try:
                action_input_str = action_input_str.strip("`")
                if action_input_str.startswith("json\n"):
                    action_input_str = action_input_str[5:]
                action_input = json.loads(action_input_str)
                results.append((thought, action, action_input))
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse hatası: {action_input_str}")
                results.append(
                    (thought, action, {"error": f"JSON parse hatası: {str(e)}. Lütfen formatı düzelt."})
                )

        return results

    @trace_agent(name="agent_arun")
    async def arun(self, user_message: str, response_model: Optional[Type[BaseModel]] = None) -> AsyncGenerator[str, None]:
        """Kullanıcı mesajını alır, streaming ile yanıt üretir ve ReAct döngüsünü işletir."""
        
        # Pydantic Model Output Injection
        if response_model:
            schema_json = json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)
            user_message += f"\n\n[SİSTEM KURALI]: Görevini tamamladığında, nihai cevabını KESİNLİKLE aşağıdaki JSON şemasına birebir uyacak şekilde saf bir JSON formatında döndürmek zorundasın:\n```json\n{schema_json}\n```\nSadece JSON döndür, başka metin ekleme."

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
                if "Action:" not in full_response:
                    yield chunk.content

            self.memory.add_message(Message(role="assistant", content=full_response))
            parsed_actions = self._parse_react_response(full_response)

            if parsed_actions:
                # Add thought once
                thought = parsed_actions[0][0]
                self._current_steps.append(AgentStep(step_type="thought", content=thought))
                logger.info(f"[Agent Thought]: {thought}")

                tasks = []
                for _, action, action_input in parsed_actions:
                    self._current_steps.append(
                        AgentStep(step_type="action", content="", tool_name=action, tool_input=action_input)
                    )
                    logger.warning(f"[Agent Action]: {action}({action_input})")

                    target_tool = self.tools.get_tool(action)
                    if not target_tool:
                        async def dummy_err(a=action):
                            return f"Hata: {a} adında bir araç bulunamadı."
                        tasks.append(dummy_err())
                        continue

                    if self.require_human_approval:
                        onay = input(
                            f"Tool {action} çalıştırılacak ({action_input}). Onaylıyor musun? (y/n): "
                        )
                        if onay.lower() != "y":
                            async def dummy_rej():
                                return "Kullanıcı bu aracın çalıştırılmasını reddetti."
                            tasks.append(dummy_rej())
                            continue

                    tasks.append(execute_tool(target_tool, **action_input))

                # Execute all tools concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                obs_parts = []
                for action_tuple, res in zip(parsed_actions, results):
                    _, action_name, _ = action_tuple
                    if isinstance(res, TransferException):
                        # Sürü mimarisinde yönetici (Swarm) yakalasın diye fırlatıyoruz
                        raise res
                    elif isinstance(res, Exception):
                        obs_str = f"Araç {action_name} çöktü: {str(res)}"
                    else:
                        obs_str = f"Araç {action_name} sonucu: {str(res)}"
                    obs_parts.append(obs_str)

                    self._current_steps.append(AgentStep(step_type="observation", content=obs_str))
                    logger.success(f"[Agent Observation]: {obs_str[:100]}...")

                final_obs = "\n".join(obs_parts)
                self.memory.add_message(Message(role="user", content=f"Observation:\n{final_obs}"))
            else:
                self._current_steps.append(AgentStep(step_type="response", content=full_response))
                break
        else:
            msg = "\n[Sistem]: Maksimum iterasyon limitine ulaşıldı."
            self._current_steps.append(AgentStep(step_type="response", content=msg))
            yield msg

    @trace_agent(name="agent_run")
    async def run(self, user_message: str, response_model: Optional[Type[T]] = None) -> AgentResponse[T]:
        """Asenkron çalışır ama streaming yapmaz. Tüm adımlar bittiğinde AgentResponse döner."""
        async for chunk in self.arun(user_message, response_model=response_model):
            pass

        steps = getattr(self, "_current_steps", [])
        final_answer = ""
        if steps and steps[-1].step_type == "response":
            final_answer = steps[-1].content
        elif steps:
            final_answer = steps[-1].content

        structured_output = None
        if response_model and final_answer:
            try:
                json_str = final_answer
                # Regex ile JSON bloğunu bulmaya çalış
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", final_answer, re.DOTALL | re.IGNORECASE)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # En dıştaki {} parantezleri ara
                    json_match = re.search(r"(\{.*\})", final_answer, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                
                structured_output = response_model.model_validate_json(json_str)
                logger.success(f"[Structured Output] Pydantic modeline başarıyla çevrildi: {response_model.__name__}")
            except Exception as e:
                logger.error(f"[Structured Output] Parse Hatası: {e}")

        return AgentResponse(
            final_answer=final_answer.strip(),
            structured_output=structured_output,
            steps=steps,
            total_tokens=self.cost_tracker.input_tokens + self.cost_tracker.output_tokens,
            estimated_usd=self.cost_tracker.get_estimated_usd(),
        )

    def save_checkpoint(self, filepath: str) -> None:
        """Ajanın güncel belleğini ve token maliyetini diske JSON olarak kaydeder."""
        data = {
            "memory": {
                "system_prompt": self.memory.system_prompt.model_dump() if self.memory.system_prompt else None,
                "messages": [m.model_dump() for m in self.memory.messages]
            },
            "cost": {
                "input_tokens": self.cost_tracker.input_tokens,
                "output_tokens": self.cost_tracker.output_tokens
            }
        }
        with open(filepath, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, filepath: str) -> None:
        """Diskteki JSON dosyasından ajan belleğini ve maliyet durumunu geri yükler."""
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint dosyası bulunamadı: {filepath}")
            return
        
        with open(filepath, "r") as f:
            data = json.load(f)
            
        mem_data = data.get("memory", {})
        if mem_data.get("system_prompt"):
            self.memory.system_prompt = Message(**mem_data["system_prompt"])
        self.memory.messages = [Message(**m) for m in mem_data.get("messages", [])]
        
        cost_data = data.get("cost", {})
        self.cost_tracker.input_tokens = cost_data.get("input_tokens", 0)
        self.cost_tracker.output_tokens = cost_data.get("output_tokens", 0)
        logger.info(f"Checkpoint başarıyla yüklendi: {filepath}")
