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
def local_python_repl(code: str) -> str:
    """
    [DİKKAT: YEREL MAKİNEDE ÇALIŞIR] Verilen Python kodunu çalıştırır ve stdout (print) çıktısını döndürür.
    Matematiksel hesaplamalar veya veri analizi için idealdir.
    ÖNEMLİ: Kod güvenli kabul edilir, zararlı işlemlerden kaçının. Sadece lokal testler içindir.
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
def sandbox_python_repl(code: str) -> str:
    """
    Verilen Python kodunu E2B Code Interpreter kullanarak %100 izole bir bulut Sandbox'ında çalıştırır.
    Ana makineye zarar verme riski sıfırdır. İnternet erişimi ve dosya yazma yetkisi sadece sandbox ile sınırlıdır.
    Veri analizi, grafik çizimi ve ağır kod yürütmeleri için bunu tercih edin.
    """
    try:
        from e2b_code_interpreter import Sandbox # type: ignore
    except ImportError:
        return "E2B SDK kurulu değil. 'pip install e2b_code_interpreter' komutunu çalıştırın."

    try:
        # E2B_API_KEY environment variable olarak tanımlı olmalıdır.
        with Sandbox() as sandbox:
            execution = sandbox.run_code(code)
            
            output = ""
            if execution.logs.stdout:
                output += "\n".join(execution.logs.stdout)
            if execution.logs.stderr:
                output += "\n[STDERR]:\n" + "\n".join(execution.logs.stderr)
            if execution.error:
                output += f"\n[HATA]:\n{execution.error.name}: {execution.error.value}"
                
            return output.strip() if output else "Kod Sandbox'ta başarıyla çalıştı (Çıktı yok)."
    except Exception as e:
        return f"Sandbox Hatası: {str(e)}"


@tool
async def read_file(file_path: str) -> str:
    """Verilen dosya yolundaki metin dosyasının içeriğini okur."""
    if not os.path.exists(file_path):
        return f"Hata: {file_path} dosyası bulunamadı."

    try:
        import aiofiles  # type: ignore[import-untyped]

        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content = await f.read()
            from typing import cast
            return cast(str, content)
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
