"""Multi-Agent orchestrator testleri."""

import pytest
from unittest.mock import MagicMock
from agentkit.agent import Agent, AgentResponse
from agentkit.orchestrator import Team
from agentkit.memory import ShortTermMemory


@pytest.fixture
def mock_manager() -> Agent:
    manager_llm = MagicMock()
    from agentkit.tools import ToolRegistry

    return Agent(
        llm=manager_llm, tools=ToolRegistry(), memory=ShortTermMemory(), system_prompt="Manager"
    )


@pytest.fixture
def mock_coder() -> Agent:
    coder_llm = MagicMock()
    from agentkit.tools import ToolRegistry

    return Agent(
        llm=coder_llm, tools=ToolRegistry(), memory=ShortTermMemory(), system_prompt="Coder"
    )


def test_team_initialization(mock_manager: Agent) -> None:
    """Team sınıfının manager ile düzgün başlatılması."""
    team = Team(manager=mock_manager)
    assert team.manager == mock_manager
    assert "delegate_to_agent" in team.manager.tools._tools


def test_team_add_agent(mock_manager: Agent, mock_coder: Agent) -> None:
    """Team'e ajan eklenebilmesi."""
    team = Team(manager=mock_manager)
    team.add_agent("coder", mock_coder)
    assert "coder" in team.agents


def test_delegate_tool_missing_agent(mock_manager: Agent) -> None:
    """Olmayan bir ajana delegate edilmeye çalışıldığında hata mesajı dönmesi."""
    team = Team(manager=mock_manager)
    delegate_tool = team.manager.tools.get_tool("delegate_to_agent")
    assert delegate_tool is not None

    # ToolDefinition func çağrısı senkron veya asenkron olabilir.
    result = delegate_tool.func(agent_name="olmayan", task_description="test")
    assert "bulunamadı" in result


@pytest.mark.asyncio
async def test_delegate_tool_success(mock_manager: Agent) -> None:
    """Delegate aracının diğer ajanı başarıyla çalıştırıp sonucunu dönmesi."""
    team = Team(manager=mock_manager)

    # Coder ajanı yarat ve run metodunu mockla
    mock_coder = MagicMock(spec=Agent)
    mock_response = AgentResponse(
        final_answer="Kod yazıldı", steps=[], total_tokens=10, estimated_usd=0.0
    )

    async def fake_run(task: str) -> AgentResponse:
        return mock_response

    mock_coder.run = fake_run
    team.add_agent("coder", mock_coder)

    delegate_tool = team.manager.tools.get_tool("delegate_to_agent")
    assert delegate_tool is not None
    result = delegate_tool.func(agent_name="coder", task_description="Merhaba yaz")

    assert "Kod yazıldı" in result


@pytest.mark.asyncio
async def test_team_run(mock_manager: Agent) -> None:
    """Team.run() çağrıldığında manager'a system prompt set edip onu run etmesi."""
    team = Team(manager=mock_manager)

    mock_response = AgentResponse(
        final_answer="Tamam", steps=[], total_tokens=10, estimated_usd=0.0
    )

    async def fake_run(task: str) -> AgentResponse:
        return mock_response

    mock_manager.run = fake_run  # type: ignore

    res = await team.run("Görev")
    assert res.final_answer == "Tamam"
    assert team.manager.memory.system_prompt is not None
    assert "takım yöneticisisin" in team.manager.memory.system_prompt.content
