"""
Notion entegrasyon araçları.
Kullanmak için `pip install "agentkit-ai[notion]"` veya `pip install notion-client` gereklidir.
"""

import os
from typing import Any
from agentkit.tools.decorator import tool
from loguru import logger

try:
    from notion_client import Client

    HAS_NOTION = True
except ImportError:
    HAS_NOTION = False


def _get_notion_client() -> Any:
    if not HAS_NOTION:
        raise ImportError(
            "notion-client kütüphanesi bulunamadı. Lütfen 'pip install notion-client' komutunu çalıştırın."
        )

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise ValueError("NOTION_TOKEN çevre değişkeni bulunamadı.")

    return Client(auth=token)


@tool
def notion_read_page(page_id: str) -> str:
    """
    Verilen ID'ye sahip Notion sayfasının düz metin içeriğini getirir.
    """
    try:
        notion = _get_notion_client()
        blocks = notion.blocks.children.list(block_id=page_id)

        text_content = []
        for block in blocks.get("results", []):
            block_type = block.get("type")
            if block_type and block_type in block:
                rich_texts = block[block_type].get("rich_text", [])
                for rt in rich_texts:
                    text_content.append(rt.get("plain_text", ""))

        return "\n".join(text_content) if text_content else "Sayfa boş veya okunamadı."
    except Exception as e:
        logger.error(f"[Notion Tool] Hata: {str(e)}")
        return f"Sayfa okunamadı: {str(e)}"
