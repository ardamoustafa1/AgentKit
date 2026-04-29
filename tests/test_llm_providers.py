import pytest
import respx
import httpx
from unittest.mock import MagicMock
from typing import Any
from agentkit.types.schemas import Message
from agentkit.llm.openai import OpenAILLM
from agentkit.llm.anthropic import AnthropicLLM
from agentkit.llm.groq import GroqLLM
from agentkit.llm.ollama import OllamaLLM


# ────────────────────────────────────────────
# OpenAI Tests
# ────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_openai_generate() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "openai response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )
    )
    llm = OpenAILLM(api_key="test-key")
    res = await llm.generate_async([Message(role="user", content="hello")])
    assert "openai response" in res.content
    assert res.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_openai_generate_stream() -> None:
    """OpenAI streaming — mock the client directly for SSE compatibility."""
    llm = OpenAILLM(api_key="test-key")

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "hello "
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "world"
    chunk2.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    async def mock_stream(*args: Any, **kwargs: Any) -> Any:
        class FakeStream:
            def __aiter__(self) -> "FakeStream":
                return self

            async def __anext__(self) -> Any:
                raise StopAsyncIteration

        items = [chunk1, chunk2]

        class RealStream:
            def __init__(self) -> None:
                self.idx = 0

            def __aiter__(self) -> "RealStream":
                return self

            async def __anext__(self) -> Any:
                if self.idx >= len(items):
                    raise StopAsyncIteration
                item = items[self.idx]
                self.idx += 1
                return item

        return RealStream()

    llm.client.chat.completions.create = mock_stream  # type: ignore

    chunks = []
    async for c in llm.generate_stream_async([Message(role="user", content="hi")]):
        chunks.append(c.content)
    assert "hello " in chunks
    assert "world" in chunks


# ────────────────────────────────────────────
# Anthropic Tests
# ────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_generate() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "anthropic response"}],
                "model": "claude-3",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )
    )
    llm = AnthropicLLM(api_key="test-key")
    res = await llm.generate_async([Message(role="user", content="hello")])
    assert "anthropic response" in res.content
    assert res.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_anthropic_generate_stream() -> None:
    """Anthropic streaming — mock internal stream context manager."""
    llm = AnthropicLLM(api_key="test-key")

    evt_start = MagicMock()
    evt_start.type = "message_start"
    evt_start.message.usage.input_tokens = 10

    evt_delta = MagicMock()
    evt_delta.type = "content_block_delta"
    evt_delta.delta.text = "streamed"

    evt_msg_delta = MagicMock()
    evt_msg_delta.type = "message_delta"
    evt_msg_delta.usage.output_tokens = 20

    events = [evt_start, evt_delta, evt_msg_delta]

    class FakeStream:
        async def __aenter__(self) -> "FakeStream":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        def __aiter__(self) -> "FakeStream":
            return self

        def __init__(self) -> None:
            self.idx = 0

        async def __anext__(self) -> Any:
            if self.idx >= len(events):
                raise StopAsyncIteration
            evt = events[self.idx]
            self.idx += 1
            return evt

    llm.client.messages.stream = MagicMock(return_value=FakeStream())  # type: ignore

    chunks = []
    async for c in llm.generate_stream_async([Message(role="user", content="hi")]):
        chunks.append(c)
    assert len(chunks) >= 2
    assert any("streamed" in c.content for c in chunks)


# ────────────────────────────────────────────
# Groq Tests
# ────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_groq_generate() -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "llama3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "groq response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )
    )
    llm = GroqLLM(api_key="test-key")
    res = await llm.generate_async([Message(role="user", content="hello")])
    assert "groq response" in res.content
    assert res.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_groq_generate_stream() -> None:
    """Groq streaming — mock the client directly."""
    llm = GroqLLM(api_key="test-key")

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "fast "
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "response"
    chunk2.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)

    items = [chunk1, chunk2]

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        class Stream:
            def __init__(self) -> None:
                self.idx = 0

            def __aiter__(self) -> "Stream":
                return self

            async def __anext__(self) -> Any:
                if self.idx >= len(items):
                    raise StopAsyncIteration
                item = items[self.idx]
                self.idx += 1
                return item

        return Stream()

    llm.client.chat.completions.create = mock_create  # type: ignore

    chunks = []
    async for c in llm.generate_stream_async([Message(role="user", content="hi")]):
        chunks.append(c.content)
    assert "fast " in chunks
    assert "response" in chunks


# ────────────────────────────────────────────
# Ollama Tests
# ────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3",
                "message": {"role": "assistant", "content": "ollama response"},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 10,
            },
        )
    )
    llm = OllamaLLM(model_name="llama3")
    res = await llm.generate_async([Message(role="user", content="hello")])
    assert "ollama response" in res.content
    assert res.usage.input_tokens == 5
    assert res.usage.output_tokens == 10


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate_stream() -> None:
    """Ollama streaming — httpx NDJSON stream."""
    line1 = b'{"message": {"content": "ollama "}, "done": false}\n'
    line2 = b'{"message": {"content": "stream"}, "done": true, "prompt_eval_count": 5, "eval_count": 10}\n'
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content=line1 + line2)
    )
    llm = OllamaLLM(model_name="llama3")
    chunks = []
    async for c in llm.generate_stream_async([Message(role="user", content="hi")]):
        chunks.append(c.content)
    assert "ollama " in chunks
    assert "stream" in chunks
