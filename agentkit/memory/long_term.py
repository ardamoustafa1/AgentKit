import chromadb
from typing import List, Dict, Any, Optional
from loguru import logger
from agentkit.memory.vector_db.base import BaseVectorDB
from agentkit.memory.vector_db.chroma import ChromaVectorDB

class LongTermMemory:
    """
    RAG (Retrieval-Augmented Generation) altyapısı için uzun vadeli bellek (Vector DB).
    Varsayılan olarak ChromaDB kullanır, ancak BaseVectorDB arayüzünü uygulayan
    herhangi bir veritabanı (Pinecone, Qdrant, vb.) enjekte edilebilir.
    """

    def __init__(
        self,
        db: Optional[BaseVectorDB] = None,
        # Geriye dönük uyumluluk parametreleri (eski ChromaDB-only API)
        collection_name: str = "agent_memory",
        persist_directory: Optional[str] = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if db is not None:
            self.db = db
        else:
            # Eski API ile de çalışabilmesi için ChromaDB'yi legacy parametrelerle başlat
            self.db = ChromaVectorDB(
                collection_name=collection_name,
                persist_directory=persist_directory,
                model_name=model_name,
            )
        logger.info(f"[LongTermMemory] Başlatıldı. Sürücü: {self.db.__class__.__name__}")

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Yeni bir bilgiyi (belge/konuşma) vektör veritabanına ekler."""
        return self.db.add(text, metadata)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Verilen sorguya anlamsal olarak en yakın 'k' adet sonucu döndürür."""
        return self.db.search(query, k)
