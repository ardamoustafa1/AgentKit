import os
import contextlib
from io import StringIO
from duckduckgo_search import DDGS

from agentkit.tools.decorator import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    DuckDuckGo kullanarak internette arama yapar.
    Güncel bilgilere ve haberlere ulaşmak için bu aracı kullan.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Başlık: {r['title']}\nÖzet: {r['body']}\nLink: {r['href']}")

        if not results:
            return "Arama sonucu bulunamadı."
        return "\n\n".join(results)
    except Exception as e:
        return f"Arama sırasında hata oluştu: {str(e)}"


@tool
def python_repl(code: str) -> str:
    """
    Verilen Python kodunu çalıştırır ve stdout (print) çıktısını döndürür.
    Matematiksel hesaplamalar veya veri analizi için idealdir.
    ÖNEMLİ: Kod güvenli kabul edilir, zararlı işlemlerden kaçının.
    """
    output = StringIO()

    safe_env = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "enumerate": enumerate,
            "zip": zip,
            "math": __import__("math"),
        }
    }

    try:
        with contextlib.redirect_stdout(output):
            exec(code, safe_env, {})
        result = output.getvalue()
        return result.strip() if result else "Kod başarıyla çalıştı (Çıktı yok)."
    except Exception as e:
        return f"Hata türü: {type(e).__name__}\nDetay: {str(e)}"


@tool
async def read_file(file_path: str) -> str:
    """Verilen dosya yolundaki metin dosyasının içeriğini okur."""
    if not os.path.exists(file_path):
        return f"Hata: {file_path} dosyası bulunamadı."

    try:
        import aiofiles

        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content = await f.read()
        return content
    except Exception as e:
        return f"Dosya okunurken hata oluştu: {str(e)}"


@tool
async def write_file(file_path: str, content: str) -> str:
    """Verilen dosya yoluna içeriği yazar (üzerine yazar)."""
    try:
        import aiofiles

        async with aiofiles.open(file_path, mode="w", encoding="utf-8") as f:
            await f.write(content)
        return f"Başarılı: İçerik {file_path} dosyasına yazıldı."
    except Exception as e:
        return f"Dosya yazılırken hata oluştu: {str(e)}"
