"""Base LLM sınıfının sync wrapper'ları ve retry mekanizmasının testleri."""

import pytest
from tests.conftest import MockLLM
from agentkit.types.schemas import Message
from agentkit.llm.base import retry_with_backoff


@pytest.mark.asyncio
async def test_mock_llm_generation() -> None:
    """Mock LLM'in async metotlarını doğrular."""
    llm = MockLLM(responses=["Test cevabı"])
    response = await llm.generate_async([Message(role="user", content="selam")])
    assert response.content == "Test cevabı"
    assert response.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_mock_llm_streaming() -> None:
    llm = MockLLM(responses=["Bu bir test"])
    chunks = []
    async for chunk in llm.generate_stream_async([Message(role="user", content="selam")]):
        chunks.append(chunk.content)
    assert "".join(chunks) == "Bu bir test"


def test_sync_generate_wrapper() -> None:
    """BaseLLM.generate() sync wrapper'ını test eder."""
    llm = MockLLM(responses=["Sync cevap"])
    response = llm.generate([Message(role="user", content="test")])
    assert response.content == "Sync cevap"
    assert response.usage.total_tokens == 30


def test_sync_stream_wrapper() -> None:
    """BaseLLM.generate_stream() sync wrapper'ını test eder."""
    llm = MockLLM(responses=["Kelime kelime"])
    chunks = []
    for chunk in llm.generate_stream([Message(role="user", content="test")]):
        chunks.append(chunk.content)
    assert "".join(chunks) == "Kelime kelime"


@pytest.mark.asyncio
async def test_retry_with_backoff_success() -> None:
    """Retry decorator'ünün başarılı durumda direkt döndüğünü test eder."""
    call_count = 0

    @retry_with_backoff(max_retries=3, initial_delay=0.01)
    async def successful_func() -> str:
        nonlocal call_count
        call_count += 1
        return "success"

    result = await successful_func()
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_retries() -> None:
    """Retry decorator'ünün hata durumunda tekrar denediğini test eder."""
    call_count = 0

    @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=1.0)
    async def flaky_func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Geçici hata")
        return "eventually works"

    result = await flaky_func()
    assert result == "eventually works"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_exhausted() -> None:
    """Retry decorator'ünün max retry'dan sonra hata fırlattığını test eder."""

    @retry_with_backoff(max_retries=2, initial_delay=0.01)
    async def always_fail() -> str:
        raise ValueError("Kalıcı hata")

    with pytest.raises(ValueError, match="Kalıcı hata"):
        await always_fail()


@pytest.mark.asyncio
async def test_mock_llm_multiple_responses() -> None:
    """MockLLM'in birden fazla cevap sıralamasını test eder."""
    llm = MockLLM(responses=["ilk", "ikinci", "üçüncü"])
    r1 = await llm.generate_async([Message(role="user", content="1")])
    r2 = await llm.generate_async([Message(role="user", content="2")])
    r3 = await llm.generate_async([Message(role="user", content="3")])
    assert r1.content == "ilk"
    assert r2.content == "ikinci"
    assert r3.content == "üçüncü"

    # 4. çağrıda son cevabı tekrarlamalı
    r4 = await llm.generate_async([Message(role="user", content="4")])
    assert r4.content == "üçüncü"


@pytest.mark.asyncio
async def test_mock_llm_default_response() -> None:
    """MockLLM'in varsayılan (None) ile başlatılmasını test eder."""
    llm = MockLLM()
    r = await llm.generate_async([Message(role="user", content="test")])
    assert r.content == "Mocked Response"
