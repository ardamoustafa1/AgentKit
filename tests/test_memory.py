"""Memory modüllerinin kapsamlı testleri — ShortTerm, LongTerm, Entity."""

import pytest
from typing import Any
from agentkit.memory import ShortTermMemory, LongTermMemory, EntityMemory
from agentkit.types.schemas import Message


def test_short_term_memory_sliding_window() -> None:
    """Çok düşük limitli memory'nin eski mesajları atıp atmadığı test edilir."""
    memory = ShortTermMemory(max_tokens=30)
    memory.add_message(Message(role="user", content="Bu birinci mesaj çok uzun olmasın"))
    memory.add_message(Message(role="assistant", content="Bu da ikinci mesajımız kısa cevap"))
    memory.add_message(Message(role="user", content="Üçüncü mesaj ile birlikte limit doldu taştı"))
    msgs = memory.get_messages()
    assert len(msgs) < 3


def test_short_term_memory_system_prompt_preservation() -> None:
    """Sliding window'un System prompt'unu silmediğinden emin olunur."""
    memory = ShortTermMemory(max_tokens=20)
    memory.set_system_prompt("Sen asistansın")
    for i in range(10):
        memory.add_message(Message(role="user", content=f"Uzun mesaj {i} test token silme işlemi"))
    msgs = memory.get_messages()
    assert msgs[0].role == "system"
    assert msgs[0].content == "Sen asistansın"


def test_short_term_memory_empty() -> None:
    """Boş memory'nin get_messages'ının boş liste döndürmesi."""
    memory = ShortTermMemory()
    msgs = memory.get_messages()
    assert len(msgs) == 0


def test_short_term_memory_system_via_add() -> None:
    """add_message ile system mesajı eklendiğinde system_prompt'a atanması."""
    memory = ShortTermMemory()
    memory.add_message(Message(role="system", content="Yeni sistem komutu"))
    msgs = memory.get_messages()
    assert len(msgs) == 1
    assert msgs[0].role == "system"
    assert msgs[0].content == "Yeni sistem komutu"


def test_short_term_memory_unknown_model() -> None:
    """Bilinmeyen model adı için fallback encoding kullanılması."""
    memory = ShortTermMemory(model_for_token_counting="bilinmeyen-model-xyz")
    memory.add_message(Message(role="user", content="Test"))
    msgs = memory.get_messages()
    assert len(msgs) == 1


def test_long_term_memory_add_search() -> None:
    """LTM bellek ekleme ve vektör arama test edilir."""
    try:
        memory = LongTermMemory(persist_directory=None)
        memory.add("Python nesne yönelimli bir dildir.", metadata={"lang": "python"})
        results = memory.search("Python nasıl bir dildir?", k=1)
        assert len(results) == 1
        assert "nesne yönelimli" in results[0]["document"]
    except ImportError:
        pytest.skip("sentence-transformers kurulu değil.")


def test_long_term_memory_empty_search() -> None:
    """Boş koleksiyonda arama yapıldığında boş sonuç dönmesi."""
    try:
        memory = LongTermMemory(collection_name="empty_test_collection", persist_directory=None)
        results = memory.search("olmayan bilgi", k=3)
        # Yeni koleksiyondan arama hata vermemeli
        assert isinstance(results, list)
    except ImportError:
        pytest.skip("sentence-transformers kurulu değil.")


def test_long_term_memory_multiple_results() -> None:
    """Birden fazla sonuç döndürme testi."""
    try:
        import uuid

        unique_name = f"multi_{uuid.uuid4().hex[:8]}"
        memory = LongTermMemory(collection_name=unique_name, persist_directory=None)
        memory.add("FastAPI hızlı bir web framework'üdür.")
        memory.add("Django tam özellikli bir web framework'üdür.")
        memory.add("Flask hafif bir mikro framework'tür.")
        results = memory.search("web framework", k=2)
        assert len(results) >= 2
    except (ImportError, ValueError):
        pytest.skip("sentence-transformers kurulu değil veya ChromaDB hatası.")


@pytest.mark.asyncio
async def test_entity_memory_extraction(mock_llm: Any) -> None:
    """LLM'den Entity Memory'nin beklediği JSON cevabını simüle ederek extract testi."""
    mock_llm.responses = ['{"isim": "Ahmet", "yas": 30}']
    entity_mem = EntityMemory(llm=mock_llm)
    await entity_mem.extract_and_store("Benim adım Ahmet, 30 yaşındayım.")
    assert entity_mem.get_all() == {"isim": "Ahmet", "yas": 30}
    assert "Ahmet" in entity_mem.get_context_string()


@pytest.mark.asyncio
async def test_entity_memory_empty_extraction(mock_llm: Any) -> None:
    """LLM boş JSON döndüğünde entity'lerin değişmemesi."""
    mock_llm.responses = ["{}"]
    entity_mem = EntityMemory(llm=mock_llm)
    await entity_mem.extract_and_store("Hiçbir bilgi yok.")
    assert entity_mem.get_all() == {}
    assert entity_mem.get_context_string() == ""


@pytest.mark.asyncio
async def test_entity_memory_json_parse_error(mock_llm: Any) -> None:
    """LLM geçersiz JSON döndüğünde hata yakalanması."""
    mock_llm.responses = ["Bu JSON değil!!!"]
    entity_mem = EntityMemory(llm=mock_llm)
    await entity_mem.extract_and_store("Geçersiz cevap testi.")
    assert entity_mem.get_all() == {}


@pytest.mark.asyncio
async def test_entity_memory_markdown_cleanup(mock_llm: Any) -> None:
    """LLM'in ```json bloğu ile sardığı cevabın temizlenmesi."""
    mock_llm.responses = ['```json\n{"sehir": "Istanbul"}\n```']
    entity_mem = EntityMemory(llm=mock_llm)
    await entity_mem.extract_and_store("Istanbul'da yaşıyorum.")
    assert entity_mem.get_all() == {"sehir": "Istanbul"}


@pytest.mark.asyncio
async def test_entity_memory_markdown_triple_backtick(mock_llm: Any) -> None:
    """LLM'in sadece ``` ile sardığı cevabın temizlenmesi."""
    mock_llm.responses = ['```{"meslek": "muhendis"}```']
    entity_mem = EntityMemory(llm=mock_llm)
    await entity_mem.extract_and_store("Mühendisim.")
    assert entity_mem.get_all() == {"meslek": "muhendis"}


@pytest.mark.asyncio
async def test_entity_memory_update(mock_llm: Any) -> None:
    """Yeni bilgi geldiğinde mevcut entity'lerin güncellenmesi."""
    mock_llm.responses = ['{"isim": "Ali"}', '{"isim": "Ali", "sehir": "Ankara"}']
    entity_mem = EntityMemory(llm=mock_llm)
    await entity_mem.extract_and_store("Adım Ali.")
    assert entity_mem.get_all() == {"isim": "Ali"}

    await entity_mem.extract_and_store("Ankara'da yaşıyorum.")
    assert entity_mem.get_all() == {"isim": "Ali", "sehir": "Ankara"}
