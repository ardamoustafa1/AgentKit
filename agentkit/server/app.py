from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json

from agentkit.agent import Agent
from agentkit.llm.openai import OpenAILLM
from agentkit.memory.short_term import ShortTermMemory
from agentkit.tools import ToolRegistry, web_search, local_python_repl

app = FastAPI(title="AgentKit API")

registry = ToolRegistry()
registry.register(web_search)
registry.register(local_python_repl)

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o"

def get_agent(model_name: str) -> Agent:
    llm = OpenAILLM(model_name=model_name)
    memory = ShortTermMemory()
    return Agent(llm=llm, tools=registry, memory=memory, system_prompt="Sen yardımsever bir asistansın.")

# Create static dir if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AgentKit UI (index.html bulunamadı)</h1>"

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    agent = get_agent(req.model)
    response = await agent.run(req.message)
    return {
        "final_answer": response.final_answer,
        "steps": [s.model_dump() for s in response.steps],
        "estimated_usd": response.estimated_usd
    }

@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    agent = get_agent(req.model)
    
    async def event_generator():
        # SSE formatında arun çıktısını gönder
        # Her bir yield string döndürür, format: "data: {JSON}\n\n"
        async for chunk in agent.arun(req.message):
            data = json.dumps({"chunk": chunk})
            yield f"data: {data}\n\n"
            
        # Bittiğinde steps'i gönder
        steps_data = json.dumps({"steps": [s.model_dump() for s in agent._current_steps]})
        yield f"data: {steps_data}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")
