<div align="center">
  <h1>🚀 AgentKit</h1>
  <p><b>A lightweight, transparent, and observable AI agent framework.</b></p>
  <p>AgentKit takes the magic out of AI agents. No black-box abstractions, no over-engineered chains—just clean Python code, Pydantic schemas, and explicit control over LLMs, tools, and memory.</p>

  <p>
    <a href="https://pypi.org/project/agentkit/"><img src="https://img.shields.io/pypi/v/agentkit?color=blue" alt="PyPI version"></a>
    <a href="https://github.com/agentkit/agentkit/actions"><img src="https://img.shields.io/github/actions/workflow/status/agentkit/agentkit/tests.yml?label=tests" alt="Tests"></a>
    <a href="https://codecov.io/gh/agentkit/agentkit"><img src="https://img.shields.io/codecov/c/github/agentkit/agentkit?color=success" alt="Coverage"></a>
    <a href="https://github.com/agentkit/agentkit/blob/main/LICENSE"><img src="https://img.shields.io/github/license/agentkit/agentkit" alt="License"></a>
  </p>
</div>

---

### 📦 Installation

```bash
pip install agentkit
```

### ⚡ Hello World (Agent in < 10 Lines)

```python
import asyncio
from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.tools import ToolRegistry, web_search
from agentkit.memory.short_term import ShortTermMemory

# 1. Register tools, 2. Init Agent, 3. Run
tools = ToolRegistry()
tools.register(web_search)

agent = Agent(llm=OpenAILLM(), tools=tools, memory=ShortTermMemory(), system_prompt="You are a helpful assistant.")

# Ask, stream and watch it use tools!
asyncio.run(agent.run("Who is the current CEO of OpenAI?"))
```

### ⚔️ AgentKit vs. LangChain

Why did we build AgentKit when LangChain exists? Because we wanted to understand our stack.

| Feature | 🦜🔗 LangChain | 🚀 AgentKit |
| :--- | :--- | :--- |
| **Abstractions** | 5-layer deep subclass hierarchies | Simple, explicit Python classes |
| **Tool Creation** | Requires specific BaseTool subclasses | Just use standard `@tool` on normal functions |
| **Observability** | Needs external heavy tools (LangSmith) | Built-in colored logging & Cost Tracking |
| **Memory** | Opaque abstractions, tricky to manage | Standard Pydantic Message arrays + DB interfaces |
| **Learning Curve**| Steep | Flat. If you know Python, you know AgentKit |

### ✨ Features

*   **🔌 Single LLM Interface:** Easily switch between OpenAI, Anthropic, or local Ollama models.
*   **🛠️ Magic-Free Tools:** The `@tool` decorator dynamically extracts parameters into Pydantic JSON schemas.
*   **🧠 Tri-Level Memory:** 
    *   *Short-Term* (Sliding Window based on token limits)
    *   *Long-Term* (RAG via ChromaDB + SentenceTransformers)
    *   *Entity* (LLM-extracted structured Key-Value memory)
*   **🔄 Robust ReAct Loop:** Built-in infinite loop prevention, parsing, and structured reasoning (Thought → Action → Observation).
*   **🧑‍💻 Human-in-the-Loop:** Optionally require human confirmation before executing sensitive tools.
*   **💸 Built-in Cost Tracking:** Never wonder how much an agent run cost. Accurate token counting and USD estimations natively.
*   **🖥️ DX & CLI Tools:** Ships with a powerful Typer/Rich CLI for instant chatting, debugging, and profiling.

### 📐 Architecture

```mermaid
graph TD
    Agent[Agent Orchestrator] --> LLM[LLM Interface]
    Agent --> Memory[Memory Module]
    Agent --> Tools[Tool Registry]
    
    LLM --> OpenAI[OpenAI API]
    LLM --> Anthropic[Anthropic API]
    LLM --> Ollama[Local Ollama]
    
    Memory --> ST[Short-Term / Sliding Window]
    Memory --> LT[Long-Term / ChromaDB]
    Memory --> Ent[Entity Extraction]
    
    Tools --> Decorator[@tool Decorator]
    Decorator --> Schemas[Pydantic JSON Schemas]
```

### 🤝 Community & Contributing

We welcome all contributions! Whether it's adding a new LLM provider, fixing a bug, or proposing a new tool:
*   Read our [Contributing Guide](CONTRIBUTING.md) to get started.
*   Join the conversation on our [Discord Community](#) (Discord Placeholder)

### 📄 License

AgentKit is open-source software licensed under the [MIT License](LICENSE).
