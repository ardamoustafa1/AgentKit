import uuid
from typing import List, Dict, Any, Optional
from loguru import logger
from agentkit.memory.vector_db.base import BaseVectorDB

class PineconeVectorDB(BaseVectorDB):
    """Pinecone tabanlı vektör veritabanı implementasyonu."""
    
    def __init__(self, api_key: str, index_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        try:
            from pinecone import Pinecone # type: ignore
            from sentence_transformers import SentenceTransformer # type: ignore
            
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(index_name)
            self.model = SentenceTransformer(embedding_model)
            logger.info(f"[PineconeVectorDB] Başlatıldı. İndeks: {index_name}")
        except ImportError:
            raise ImportError("Lütfen 'pip install pinecone-client sentence-transformers' komutunu çalıştırın.")

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        vector = self.model.encode([text])[0].tolist()
        
        meta = metadata.copy() if metadata else {}
        # Pinecone metni kendiliğinden saklamaz, metadatanın içine koyuyoruz.
        meta["text_content"] = text
        
        self.index.upsert(vectors=[{"id": doc_id, "values": vector, "metadata": meta}])
        logger.debug(f"[PineconeVectorDB] Bilgi eklendi. ID: {doc_id}")
        return doc_id

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.model.encode([query])[0].tolist()
        results = self.index.query(vector=query_vector, top_k=k, include_metadata=True)
        
        extracted = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            text = meta.pop("text_content", "")
            extracted.append({
                "id": match.get("id"),
                "document": text,
                "metadata": meta,
                "distance": match.get("score", 0.0) # Pinecone score genelde cosine similarity (1.0 = tam eşleşme)
            })
        return extracted
