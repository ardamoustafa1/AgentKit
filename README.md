<div align="center">

<img src="https://raw.githubusercontent.com/agentkit/agentkit/main/docs/assets/logo.svg" width="120" alt="AgentKit Logo"/>

# AgentKit

**The AI agent framework that doesn't hide anything from you.**

[![PyPI version](https://img.shields.io/pypi/v/agentkit-ai?color=6366f1&labelColor=1e1e2e&style=flat-square)](https://pypi.org/project/agentkit-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-6366f1?labelColor=1e1e2e&style=flat-square)](https://www.python.org/)
[![Tests](https://img.shields.io/github/actions/workflow/status/agentkit/agentkit/tests.yml?label=tests&color=22c55e&labelColor=1e1e2e&style=flat-square)](https://github.com/agentkit/agentkit/actions)
[![Coverage](https://img.shields.io/codecov/c/github/agentkit/agentkit?color=22c55e&labelColor=1e1e2e&style=flat-square)](https://codecov.io/gh/agentkit/agentkit)
[![License: MIT](https://img.shields.io/badge/license-MIT-6366f1?labelColor=1e1e2e&style=flat-square)](LICENSE)
[![Discord](https://img.shields.io/discord/XXXXXXX?color=6366f1&label=discord&labelColor=1e1e2e&style=flat-square)](https://discord.gg/agentkit)

<br/>

```
pip install agentkit-ai
```

<br/>

[**Quickstart**](#-quickstart) · [**Docs**](https://docs.agentkit.dev) · [**Examples**](#-examples) · [**Discord**](https://discord.gg/agentkit)

</div>

---

## Why AgentKit?

LangChain and LlamaIndex are powerful — but they're also 5 layers of abstraction deep. When something breaks, you're reading framework source code instead of building your product.

AgentKit is different:

| | AgentKit | LangChain |
|---|---|---|
| **Debug a failing tool** | Read your own function | Trace through 5 abstraction layers |
| **Understand what LLM sees** | `agent.steps` shows every thought | Good luck |
| **Cost per run** | `response.estimated_usd` | Integrate 3rd party tool |
| **Add a tool** | `@tool` decorator on any function | Subclass `BaseTool`, override methods |
| **Multi-agent setup** | `Team(manager=agent)` | Custom callbacks + chains |
| **Switch LLM provider** | Change one import | Rewrite your chains |

> **"If you can't explain what your agent is doing, you can't fix it."**

---

## ✨ Features

- 🔍 **Transparent ReAct loop** — every Thought, Action, and Observation is logged and stored
- 🧰 **`@tool` decorator** — turns any Python function into an LLM-callable tool automatically
- 🤝 **Multi-agent orchestration** — Manager agents delegate to specialists via `Team`
- 🧠 **Pluggable memory** — Short-term (sliding window), Long-term (ChromaDB RAG), Entity extraction
- 💰 **Built-in cost tracking** — exact USD cost per run, per token, per model
- 🔒 **Human-in-the-loop** — pause before any tool execution with `require_human_approval=True`
- 🌐 **Multi-provider** — OpenAI, Anthropic, Groq, Ollama — same API
- 📦 **Zero magic** — pure Python, Pydantic schemas, no hidden state

---

## 🚀 Quickstart

### 1. Install

```bash
pip install agentkit-ai
```

### 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Build your first agent

```python
import asyncio
from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.tools import ToolRegistry, tool

@tool
def get_weather(city: str) -> str:
    """Returns current weather for a city."""
    return f"It's 22°C and sunny in {city}."

async def main():
    agent = Agent(
        llm=OpenAILLM(model_name="gpt-4o"),
        tools=ToolRegistry([get_weather]),
        system_prompt="You are a helpful assistant.",
    )

    response = await agent.run("What's the weather like in Istanbul?")
    print(response.final_answer)
    print(f"Cost: ${response.estimated_usd:.4f}")

asyncio.run(main())
```

**Output:**
```
It's 22°C and sunny in Istanbul!
Cost: $0.0003
```

That's it. No chains. No callbacks. No config files.

---

## 🔍 Full Transparency

Every agent run returns an `AgentResponse` with complete execution details:

```python
response = await agent.run("Find the latest Python version and write a script that prints it.")

# See every step the agent took
for step in response.steps:
    print(f"[{step.type}] {step.content}")

# --- Output ---
# [thought]  I need to search the web for the latest Python version.
# [action]   web_search({"query": "latest Python version 2024"})
# [observation] Python 3.13.1 was released on December 3, 2024.
# [thought]  Now I'll write a Python script that prints this version.
# [action]   python_repl({"code": "print('Python 3.13.1')"})
# [observation] Python 3.13.1
# [answer]   The latest Python version is 3.13.1. I've run a script that confirms this.

print(f"Total tokens: {response.token_usage.total}")
print(f"Total cost:   ${response.estimated_usd:.6f}")
```

---

## 🧰 The `@tool` Decorator

Turn any Python function into an LLM tool. AgentKit reads your type hints and docstring — you write zero schema.

```python
from agentkit.tools import tool

@tool
def search_database(query: str, table: str, limit: int = 10) -> list[dict]:
    """
    Searches the database for records matching a query.

    Args:
        query: The search term to look for.
        table: The database table to search in (e.g. 'users', 'orders').
        limit: Maximum number of results to return. Defaults to 10.
    """
    # Your real implementation here
    return db.search(query, table, limit)
```

AgentKit automatically generates the correct JSON schema for OpenAI, Anthropic, or any other provider. The LLM sees exactly what it needs to call your function correctly.

---

## 🤝 Multi-Agent Teams

Real problems need specialists. Build a team where a Manager delegates to expert sub-agents:

```python
from agentkit.orchestrator import Team

# Specialist agents
researcher = Agent(llm=llm, tools=ToolRegistry([web_search]), 
                   system_prompt="You find accurate information on the web.")

coder = Agent(llm=llm, tools=ToolRegistry([python_repl]),
              system_prompt="You write and test Python code.")

# Manager orchestrates
manager = Agent(llm=llm, tools=ToolRegistry(),
                system_prompt="You delegate tasks to researcher and coder specialists.")

team = Team(manager=manager)
team.add_agent("researcher", researcher)
team.add_agent("coder", coder)

# The manager automatically gets a `delegate_to_agent` tool
response = await team.run(
    "Find the current EUR/USD exchange rate, then write a Python function "
    "that converts any EUR amount to USD using that rate."
)
```

The manager thinks, delegates, collects results, and synthesizes a final answer — autonomously.

---

## 🧠 Memory Strategies

```python
from agentkit.memory import ShortTermMemory, LongTermMemory, EntityMemory

# Sliding window — keeps last N tokens, prunes automatically
agent = Agent(..., memory=ShortTermMemory(max_tokens=4000))

# RAG memory — stores to ChromaDB, retrieves by semantic similarity
agent = Agent(..., memory=LongTermMemory(persist_dir="./memory"))

# Entity memory — extracts and tracks key-value facts (names, preferences, etc.)
agent = Agent(..., memory=EntityMemory())
```

---

## 🔒 Human-in-the-Loop

Never let an agent run destructive operations without your approval:

```python
agent = Agent(
    ...,
    require_human_approval=True,
    approval_tools=["execute_sql", "send_email", "delete_file"]  # only gate these
)
```

Before any gated tool runs, the agent pauses:

```
⚠️  Agent wants to run: execute_sql
    Input: {"query": "DELETE FROM users WHERE inactive = true"}
    Approve? [y/n]:
```

---

## 🌐 Supported LLM Providers

Switch providers with a single import — your tools, memory, and logic stay the same:

```python
from agentkit.llm.openai    import OpenAILLM     # GPT-4o, GPT-4 Turbo
from agentkit.llm.anthropic import AnthropicLLM  # Claude 3.5 Sonnet, Opus
from agentkit.llm.groq      import GroqLLM       # Llama 3, Mixtral (ultra-fast)
from agentkit.llm.ollama    import OllamaLLM     # Local models, free, private
```

---

## 📁 Project Structure

```
agentkit/
├── agent.py              # Agent class, ReAct loop, CostTracker
├── orchestrator.py       # Team — multi-agent delegation
├── cli.py                # Rich terminal interface (run agents from the CLI)
├── llm/
│   ├── base.py           # BaseLLM abstract class
│   ├── openai.py         # OpenAI async/streaming
│   ├── anthropic.py      # Anthropic Claude
│   ├── groq.py           # Groq (Llama 3, Mixtral)
│   └── ollama.py         # Local Ollama models
├── memory/
│   ├── short_term.py     # Sliding window context
│   ├── long_term.py      # ChromaDB vector memory + RAG
│   └── entity.py         # Key-value entity extraction
├── tools/
│   ├── base.py           # ToolRegistry, ToolDefinition
│   ├── decorator.py      # @tool — type hints → JSON schema
│   ├── builtins.py       # web_search, python_repl
│   └── integrations/
│       ├── github.py     # GitHub: issues, PRs, repos
│       └── notion.py     # Notion: pages, databases
├── types/schemas.py      # Pydantic models (Message, AgentStep, TokenUsage…)
└── utils/logging.py      # Color-coded terminal output (Loguru)
```

---

## 📦 Installation Options

```bash
# Core
pip install agentkit-ai

# With GitHub + Notion integrations
pip install agentkit-ai[integrations]

# With long-term vector memory
pip install agentkit-ai[memory]

# Everything
pip install agentkit-ai[all]
```

**Requirements:** Python 3.10+

**Development setup:**
```bash
git clone https://github.com/agentkit/agentkit.git
cd agentkit
poetry install --all-extras
poetry run pytest --cov=agentkit tests/
```

---

## 💡 Examples

| Example | Description |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | Single agent with a custom tool |
| [`examples/multi_agent.py`](examples/multi_agent.py) | Manager + researcher + coder team |
| [`examples/rag_memory.py`](examples/rag_memory.py) | Long-term memory with ChromaDB |
| [`examples/human_in_loop.py`](examples/human_in_loop.py) | Approval gates for dangerous tools |
| [`examples/local_llm.py`](examples/local_llm.py) | Fully local with Ollama |
| [`examples/github_agent.py`](examples/github_agent.py) | Agent that manages GitHub issues |

---

## 🤝 Contributing

Contributions are welcome. Please open an issue before submitting large PRs.

```bash
git clone https://github.com/agentkit/agentkit.git
cd agentkit
poetry install --all-extras
poetry run pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT © [AgentKit Contributors](https://github.com/agentkit/agentkit/graphs/contributors)

---

<div align="center">

**If AgentKit saves you time, please consider giving it a ⭐**

Built with frustration for opaque frameworks, and love for clean Python.

</div>
