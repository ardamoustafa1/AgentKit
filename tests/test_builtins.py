"""Builtin tool'ların (web_search, python_repl, read_file, write_file) kapsamlı testleri."""

import pytest
from typing import Any
from agentkit.tools.builtins import web_search, python_repl, read_file, write_file


def test_python_repl_safe() -> None:
    """Python REPL'in güvenli kodları çalıştırması."""
    res = python_repl.func(code="print(5 * 5)")
    assert "25" in res


def test_python_repl_no_output() -> None:
    """Çıktı üretmeyen kodun 'Çıktı yok' mesajı döndürmesi."""
    res = python_repl.func(code="x = 42")
    assert "Çıktı yok" in res


def test_python_repl_blocked_import() -> None:
    """Tehlikeli import'ların engellenmesi."""
    res = python_repl.func(code="import os")
    assert "Hata türü" in res or "ImportError" in res


def test_python_repl_runtime_error() -> None:
    """Çalışma zamanı hatasının güvenli yakalanması."""
    res = python_repl.func(code="print(1/0)")
    assert "Hata türü" in res
    assert "ZeroDivision" in res


def test_python_repl_math() -> None:
    """math modülünün sandbox'ta kullanılabilmesi."""
    res = python_repl.func(code="m = __builtins__['math']; print(m.factorial(5))")
    assert "120" in res


@pytest.mark.asyncio
async def test_read_write_file(tmp_path: Any) -> None:
    """Dosya yazma ve okuma işlemlerinin başarılı olması."""
    file_path = tmp_path / "test.txt"
    str_path = str(file_path)

    # Write
    write_res = await write_file.func(file_path=str_path, content="test content")
    assert "Başarılı" in write_res

    # Read
    read_res = await read_file.func(file_path=str_path)
    assert "test content" in read_res


@pytest.mark.asyncio
async def test_read_file_missing() -> None:
    """Olmayan dosyanın okunmasında hata mesajı dönmesi."""
    read_missing = await read_file.func(file_path="/tmp/nonexistent_agentkit_test_file.txt")
    assert "bulunamadı" in read_missing


@pytest.mark.asyncio
async def test_write_file_error() -> None:
    """Yazılamayan dosya yolunda hata mesajı dönmesi."""
    res = await write_file.func(file_path="/nonexistent_dir/impossible/file.txt", content="test")
    assert "hata" in res.lower()


def test_web_search_success(monkeypatch: Any) -> None:
    """Web aramanın başarılı sonuç döndürmesi."""

    class MockDDGS:
        def __enter__(self) -> "MockDDGS":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def text(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
            return [{"title": "Test Title", "body": "Test Body", "href": "http://test.com"}]

    monkeypatch.setattr("agentkit.tools.builtins.DDGS", MockDDGS)
    res = web_search.func(query="test", max_results=1)
    assert "Test Title" in res


def test_web_search_no_results(monkeypatch: Any) -> None:
    """Web aramada sonuç bulunamazsa uygun mesaj dönmesi."""

    class MockDDGSEmpty:
        def __enter__(self) -> "MockDDGSEmpty":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def text(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
            return []

    monkeypatch.setattr("agentkit.tools.builtins.DDGS", MockDDGSEmpty)
    res = web_search.func(query="boş arama", max_results=1)
    assert "bulunamadı" in res


def test_web_search_error(monkeypatch: Any) -> None:
    """Web aramada hata oluşursa güvenli mesaj dönmesi."""

    class MockDDGSError:
        def __enter__(self) -> "MockDDGSError":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def text(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
            raise RuntimeError("Network error")

    monkeypatch.setattr("agentkit.tools.builtins.DDGS", MockDDGSError)
    res = web_search.func(query="hata testi", max_results=1)
    assert "hata" in res.lower()
