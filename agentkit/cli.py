import os
import typer
import asyncio
import time
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

from agentkit.utils.logging import setup_logging, console
from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.llm.ollama import OllamaLLM
from agentkit.llm.groq import GroqLLM
from agentkit.llm.base import BaseLLM
from agentkit.llm.anthropic import AnthropicLLM
from agentkit.tools import ToolRegistry, web_search, local_python_repl
from agentkit.memory.short_term import ShortTermMemory

# .env dosyasını yükle
load_dotenv()

app = typer.Typer(help="AgentKit CLI: Akıllı Ajan Yönetimi")
tools_app = typer.Typer()
app.add_typer(tools_app, name="tools", help="Araç (Tool) yönetimi")

# CLI için global Tool Registry
registry = ToolRegistry()
registry.register(web_search)
registry.register(local_python_repl)


def get_llm(model: str) -> BaseLLM:
    if model.startswith("gpt"):
        return OpenAILLM(model_name=model)
    elif model.startswith("claude"):
        return AnthropicLLM(model_name=model)
    elif model.startswith(("llama", "mixtral", "gemma")):
        # Eğer GROQ_API_KEY tanımlıysa Groq kullan, yoksa Ollama
        if os.environ.get("GROQ_API_KEY"):
            return GroqLLM(model_name=model)
    return OllamaLLM(model_name=model)


@app.command("run")
def run_command(
    query: str = typer.Argument(..., help="Ajana sormak istediğiniz soru"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Kullanılacak model"),
    stream: bool = typer.Option(True, "--stream", help="Akışlı yanıt (streaming)"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug modunu aktif et"),
) -> None:
    """Tek bir soruyu yanıtlar ve çıkar."""
    setup_logging(debug=debug)

    async def _run() -> None:
        llm = get_llm(model)
        memory = ShortTermMemory()
        agent = Agent(
            llm=llm, tools=registry, memory=memory, system_prompt="Sen yardımsever bir asistansın."
        )

        start_time = time.time()

        if stream:
            console.print(
                f"[bold cyan]AgentKit ([/bold cyan][bold yellow]{model}[/bold yellow][bold cyan]) Yanıtlıyor...[/bold cyan]\n"
            )
            async for chunk in agent.arun(query):
                print(chunk, end="", flush=True)
            print()
        else:
            with console.status(f"[bold green]Yanıt bekleniyor ({model})..."):
                response = await agent.run(query)
            console.print(Markdown(response.final_answer))

        elapsed = time.time() - start_time
        if debug:
            console.print(
                Panel(
                    f"⏱️ Süre: {elapsed:.2f}s\n"
                    f"🪙 Token: {agent.cost_tracker.input_tokens + agent.cost_tracker.output_tokens}\n"
                    f"💵 Maliyet: ${agent.cost_tracker.get_estimated_usd():.5f}",
                    title="Profiling",
                    expand=False,
                )
            )

    asyncio.run(_run())


@app.command("chat")
def chat_command(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Kullanılacak model"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug modunu aktif et"),
) -> None:
    """Sürekli etkileşimli (interactive) sohbet modunu başlatır."""
    setup_logging(debug=debug)

    async def _chat() -> None:
        llm = get_llm(model)
        memory = ShortTermMemory()
        agent = Agent(
            llm=llm, tools=registry, memory=memory, system_prompt="Sen yardımsever bir asistansın."
        )

        console.print(
            Panel.fit(
                f"[bold green]AgentKit Interactive Chat Modu ({model})[/bold green]\nÇıkmak için 'exit' veya 'quit' yazın."
            )
        )

        while True:
            try:
                user_input = console.input("\n[bold blue]Sen:[/bold blue] ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                console.print("[bold purple]AgentKit:[/bold purple] ", end="")
                async for chunk in agent.arun(user_input):
                    print(chunk, end="", flush=True)
                print()

            except KeyboardInterrupt:
                break

        console.print("\n[bold green]Sohbet sonlandırıldı.[/bold green]")

    asyncio.run(_chat())


@app.command("deploy")
def deploy_command(
    port: int = typer.Option(8000, "--port", "-p", help="Sunucu portu"),
    host: str = typer.Option("127.0.0.1", "--host", help="Sunucu hostu")
) -> None:
    """AgentKit Web UI ve API sunucusunu ayağa kaldırır (One-command deploy)."""
    import uvicorn
    console.print(Panel.fit(f"[bold green]AgentKit Server başlatılıyor...[/bold green]\n🌐 Web UI: http://{host}:{port}\n🔌 API: http://{host}:{port}/api/chat"))
    uvicorn.run("agentkit.server.app:app", host=host, port=port, reload=False)


@tools_app.command("list")
def list_tools() -> None:
    """Kayıtlı ve kullanılabilir tüm araçları listeler."""
    console.print(Panel.fit("[bold yellow]Kayıtlı Araçlar (Tools)[/bold yellow]"))
    schemas = registry.get_all_schemas()
    for s in schemas:
        func = s["function"]
        console.print(f"[bold green]- {func['name']}[/bold green]: {func['description']}")


if __name__ == "__main__":
    app()

# ==========================================
# ÖRNEK TERMİNAL ÇIKTISI (Chat & Debug)
# ==========================================
# $ python agentkit/cli.py chat --debug --model gpt-4o
#
# ╭─────────────────────────────────────────╮
# │ AgentKit Interactive Chat Modu (gpt-4o) │
# │ Çıkmak için 'exit' veya 'quit' yazın.   │
# ╰─────────────────────────────────────────╯
#
# Sen: Türkiye'nin başkenti neresidir?
# [DEBUG] --- Iteration 1/5 ---
# AgentKit: Türkiye'nin başkenti Ankara'dır.
#
# Sen: Bana 500'ün yüzde 15'ini hesapla.
# [DEBUG] --- Iteration 1/5 ---
# [DEBUG] [Agent Thought]: Bunun için matematik aracı kullanmalıyım.
# [WARNING] [Agent Action]: local_python_repl({'code': 'print(500 * 0.15)'})
# [DEBUG] [Tool Execution] Çalıştırılıyor: local_python_repl | Parametreler: {'code': 'print(500 * 0.15)'}
# [DEBUG] [Tool Execution] Başarılı: local_python_repl | Sonuç uzunluğu: 4
# [DEBUG] [Agent Observation]: 75.0
# [DEBUG] --- Iteration 2/5 ---
# AgentKit: 500'ün yüzde 15'i 75'tir.
# ==========================================
