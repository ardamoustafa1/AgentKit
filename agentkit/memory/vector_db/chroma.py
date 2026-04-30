import chromadb
import uuid
from typing import List, Dict, Any, Optional
from loguru import logger

from agentkit.memory.vector_db.base import BaseVectorDB


class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction[chromadb.Documents]):
    """Local embedding işlemi için SentenceTransformers sarmalayıcısı."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("Lütfen 'pip install sentence-transformers' komutunu çalıştırın.")

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        from typing import cast

        embeddings = self.model.encode(input).tolist()
        return cast(chromadb.Embeddings, embeddings)


class ChromaVectorDB(BaseVectorDB):
    """ChromaDB tabanlı vektör veritabanı implementasyonu."""

    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_directory: Optional[str] = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,  # type: ignore[arg-type]
        )
        logger.info(f"[ChromaVectorDB] Başlatıldı. Koleksiyon: {collection_name}")

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text], metadatas=[metadata] if metadata else [{}], ids=[doc_id]
        )
        logger.debug(f"[ChromaVectorDB] Bilgi eklendi. ID: {doc_id}")
        return doc_id

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=k)

        extracted: List[Dict[str, Any]] = []
        if not results["documents"]:
            return extracted

        for i in range(len(results["documents"][0])):
            extracted.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i]
                    if "distances" in results and results["distances"]
                    else 0.0,
                }
            )

        return extracted
