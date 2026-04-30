import uuid
from typing import List, Dict, Any, Optional
from loguru import logger
from agentkit.memory.vector_db.base import BaseVectorDB

class WeaviateVectorDB(BaseVectorDB):
    """Weaviate tabanlı vektör veritabanı implementasyonu."""
    
    def __init__(self, class_name: str = "AgentMemory", url: str = "http://localhost:8080", embedding_model: str = "all-MiniLM-L6-v2"):
        try:
            import weaviate # type: ignore
            from sentence_transformers import SentenceTransformer # type: ignore
            
            self.client = weaviate.Client(url=url)
            self.class_name = class_name
            self.model = SentenceTransformer(embedding_model)
            
            # Class (Collection) yoksa yarat
            if not self.client.schema.exists(self.class_name):
                class_obj = {
                    "class": self.class_name,
                    "vectorizer": "none", # Biz external (sentence-transformers) kullanıyoruz
                }
                self.client.schema.create_class(class_obj)
                
            logger.info(f"[WeaviateVectorDB] Başlatıldı. Sınıf: {class_name}")
        except ImportError:
            raise ImportError("Lütfen 'pip install weaviate-client sentence-transformers' komutunu çalıştırın.")

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        vector = self.model.encode([text])[0].tolist()
        
        meta = metadata.copy() if metadata else {}
        meta["text_content"] = text
        
        self.client.data_object.create(
            data_object=meta,
            class_name=self.class_name,
            uuid=doc_id,
            vector=vector
        )
        logger.debug(f"[WeaviateVectorDB] Bilgi eklendi. ID: {doc_id}")
        return doc_id

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.model.encode([query])[0].tolist()
        
        # Weaviate GraphQL sorgusu
        result = (
            self.client.query
            .get(self.class_name, ["text_content"])
            .with_additional(["id", "distance"])
            .with_near_vector({
                "vector": query_vector
            })
            .with_limit(k)
            .do()
        )
        
        extracted = []
        if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
            objects = result["data"]["Get"][self.class_name]
            for obj in objects:
                additional = obj.get("_additional", {})
                text = obj.pop("text_content", "")
                
                extracted.append({
                    "id": additional.get("id"),
                    "document": text,
                    "metadata": obj, # Kalan alanlar metadatadır
                    "distance": additional.get("distance", 0.0)
                })
        return extracted
