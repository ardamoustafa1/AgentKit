import uuid
from typing import List, Dict, Any, Optional
from loguru import logger
from agentkit.memory.vector_db.base import BaseVectorDB

class QdrantVectorDB(BaseVectorDB):
    """Qdrant tabanlı vektör veritabanı implementasyonu."""
    
    def __init__(self, collection_name: str = "agent_memory", url: str = "http://localhost:6333", embedding_model: str = "all-MiniLM-L6-v2"):
        try:
            from qdrant_client import QdrantClient # type: ignore
            from qdrant_client.models import Distance, VectorParams # type: ignore
            from sentence_transformers import SentenceTransformer # type: ignore
            
            self.client = QdrantClient(url=url)
            self.collection_name = collection_name
            self.model = SentenceTransformer(embedding_model)
            
            # Koleksiyon yoksa yarat
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
                )
            logger.info(f"[QdrantVectorDB] Başlatıldı. Koleksiyon: {collection_name}")
        except ImportError:
            raise ImportError("Lütfen 'pip install qdrant-client sentence-transformers' komutunu çalıştırın.")

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        from qdrant_client.models import PointStruct # type: ignore
        doc_id = str(uuid.uuid4())
        vector = self.model.encode([text])[0].tolist()
        
        meta = metadata.copy() if metadata else {}
        meta["text_content"] = text
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=doc_id, vector=vector, payload=meta)]
        )
        logger.debug(f"[QdrantVectorDB] Bilgi eklendi. ID: {doc_id}")
        return doc_id

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.model.encode([query])[0].tolist()
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )
        
        extracted = []
        for hit in hits:
            meta = hit.payload or {}
            text = meta.pop("text_content", "")
            extracted.append({
                "id": str(hit.id),
                "document": text,
                "metadata": meta,
                "distance": hit.score
            })
        return extracted
