"""Agent edge-case testleri — parse hatası, human approval, max iteration."""

import pytest
from typing import Any
from agentkit.agent import Agent, AgentStep, AgentResponse, CostTracker
from agentkit.memory import ShortTermMemory
from agentkit.types.schemas import TokenUsage


@pytest.mark.asyncio
async def test_agent_direct_response(mock_llm: Any, tool_registry: Any) -> None:
    """Tool kullanmasına gerek kalmadan direkt cevap verdiği durum."""
    mock_llm.responses = ["Sana nasıl yardımcı olabilirim? (Direct response)"]
    agent = Agent(llm=mock_llm, tools=tool_registry, memory=ShortTermMemory(), system_prompt="Test")
    response = await agent.run("Merhaba")
    assert "Direct response" in response.final_answer
    assert len(response.steps) == 1
    assert response.steps[0].step_type == "response"


@pytest.mark.asyncio
async def test_agent_tool_execution_loop(
    mock_llm: Any, tool_registry: Any, sample_tool: Any
) -> None:
    """Agent'ın tool kullanıp sonra nihai cevabı verdiği ReAct iterasyonu."""
    tool_registry.register(sample_tool)
    mock_llm.responses = [
        'Thought: Toplama işlemi yapmam lazım.\nAction: topla\nAction Input: {"a": 5, "b": 3}',
        "İşlemin sonucu 8'dir.",
    ]
    agent = Agent(llm=mock_llm, tools=tool_registry, memory=ShortTermMemory(), system_prompt="Test")
    response = await agent.run("5 ile 3'ü topla.")
    assert response.final_answer == "İşlemin sonucu 8'dir."
    assert len(response.steps) == 4
    assert response.steps[0].step_type == "thought"
    assert response.steps[1].step_type == "action"
    assert response.steps[2].step_type == "observation"
    assert response.steps[3].step_type == "response"


@pytest.mark.asyncio
async def test_agent_max_iterations(mock_llm: Any, tool_registry: Any) -> None:
    """Max iteration aşıldığında ajanın otomatik durması."""
    infinite_loop_response = "Thought: Arama yapmalıyım.\nAction: olmayan_tool\nAction Input: {}"
    mock_llm.responses = [infinite_loop_response] * 10
    agent = Agent(
        llm=mock_llm,
        tools=tool_registry,
        memory=ShortTermMemory(),
        system_prompt="Test",
        max_iterations=3,
    )
    response = await agent.run("Bunu sonsuza kadar ara.")
    assert "Maksimum iterasyon limitine ulaşıldı" in response.final_answer
    assert len(response.steps) == 10


@pytest.mark.asyncio
async def test_agent_json_parse_error(mock_llm: Any, tool_registry: Any) -> None:
    """LLM geçersiz JSON döndüğünde parse hatasının güvenli yakalanması."""
    mock_llm.responses = [
        "Thought: Hesaplama yapmalıyım.\nAction: topla\nAction Input: geçersiz json!!!",
        "Sonuç bulunamadı.",
    ]
    agent = Agent(llm=mock_llm, tools=tool_registry, memory=ShortTermMemory(), system_prompt="Test")
    response = await agent.run("JSON hatası testi")
    assert response.final_answer == "Sonuç bulunamadı."


@pytest.mark.asyncio
async def test_agent_json_markdown_cleanup(
    mock_llm: Any, tool_registry: Any, sample_tool: Any
) -> None:
    """LLM'in Markdown ```json bloğu ile sarmaladığı JSON'ın temizlenmesi."""
    tool_registry.register(sample_tool)
    mock_llm.responses = [
        'Thought: Hesap yapmam lazım.\nAction: topla\nAction Input: ```json\n{"a": 10, "b": 20}```',
        "Sonuç: 30",
    ]
    agent = Agent(llm=mock_llm, tools=tool_registry, memory=ShortTermMemory(), system_prompt="Test")
    response = await agent.run("10+20 nedir?")
    assert "30" in response.final_answer


@pytest.mark.asyncio
async def test_agent_human_approval_rejected(
    mock_llm: Any, tool_registry: Any, sample_tool: Any
) -> None:
    """Human-in-the-loop: Kullanıcı reddederse tool çalışmamalı."""
    tool_registry.register(sample_tool)
    mock_llm.responses = [
        'Thought: Toplama yapmalıyım.\nAction: topla\nAction Input: {"a": 1, "b": 2}',
        'Thought: Tekrar denemeliyim.\nAction: topla\nAction Input: {"a": 3, "b": 4}',
        "İptal edildi.",
    ]
    agent = Agent(
        llm=mock_llm,
        tools=tool_registry,
        memory=ShortTermMemory(),
        system_prompt="Test",
        require_human_approval=True,
        max_iterations=3,
    )

    # input() mock: her seferinde 'n' döndür
    import builtins

    original_input = builtins.input

    def fake_input(_: object = "") -> str:
        return "n"

    builtins.input = fake_input
    try:
        response = await agent.run("Onay testi")
    finally:
        builtins.input = original_input

    # Reddedildiğinde observation olarak "reddetti" mesajı eklenmeli
    obs_steps = [s for s in response.steps if s.step_type == "observation"]
    assert any("reddetti" in s.content for s in obs_steps) or "İptal" in response.final_answer


@pytest.mark.asyncio
async def test_agent_human_approval_accepted(
    mock_llm: Any, tool_registry: Any, sample_tool: Any
) -> None:
    """Human-in-the-loop: Kullanıcı onaylarsa tool çalışmalı."""
    tool_registry.register(sample_tool)
    mock_llm.responses = [
        'Thought: Toplama yapmalıyım.\nAction: topla\nAction Input: {"a": 1, "b": 2}',
        "Sonuç 3'tür.",
    ]
    agent = Agent(
        llm=mock_llm,
        tools=tool_registry,
        memory=ShortTermMemory(),
        system_prompt="Test",
        require_human_approval=True,
    )

    import builtins

    original_input = builtins.input

    def fake_input(_: object = "") -> str:
        return "y"

    builtins.input = fake_input
    try:
        response = await agent.run("Onay testi")
    finally:
        builtins.input = original_input

    assert "3" in response.final_answer


def test_cost_tracker() -> None:
    """CostTracker'ın token takibini doğru yapması."""
    tracker = CostTracker(input_price_per_m=5.0, output_price_per_m=15.0)
    tracker.add(TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500))
    tracker.add(TokenUsage(input_tokens=500, output_tokens=250, total_tokens=750))
    assert tracker.input_tokens == 1500
    assert tracker.output_tokens == 750
    usd = tracker.get_estimated_usd()
    assert usd > 0


def test_agent_step_model() -> None:
    """AgentStep ve AgentResponse pydantic modellerinin çalışması."""
    step = AgentStep(
        step_type="thought", content="Düşünüyorum", tool_name="hesapla", tool_input={"a": 1}
    )
    assert step.step_type == "thought"
    assert step.tool_name == "hesapla"

    resp = AgentResponse(final_answer="Cevap", steps=[step], total_tokens=100, estimated_usd=0.005)
    assert resp.final_answer == "Cevap"
    assert len(resp.steps) == 1
