import uuid
import json
from typing import List, Dict, Any, Optional
from loguru import logger
from agentkit.memory.vector_db.base import BaseVectorDB

class PGVectorDB(BaseVectorDB):
    """PostgreSQL (pgvector) tabanlı vektör veritabanı implementasyonu."""
    
    def __init__(self, connection_string: str, table_name: str = "agent_memory", embedding_model: str = "all-MiniLM-L6-v2"):
        try:
            import psycopg2 # type: ignore
            from pgvector.psycopg2 import register_vector # type: ignore
            from sentence_transformers import SentenceTransformer # type: ignore
            
            self.conn = psycopg2.connect(connection_string)
            register_vector(self.conn)
            self.table_name = table_name
            self.model = SentenceTransformer(embedding_model)
            
            dim = self.model.get_sentence_embedding_dimension()
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id VARCHAR(36) PRIMARY KEY,
                        content TEXT,
                        metadata JSONB,
                        embedding vector({dim})
                    )
                """)
            self.conn.commit()
            logger.info(f"[PGVectorDB] Başlatıldı. Tablo: {table_name}")
        except ImportError:
            raise ImportError("Lütfen 'pip install psycopg2-binary pgvector sentence-transformers' komutunu çalıştırın.")

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        vector = self.model.encode([text])[0].tolist()
        
        with self.conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table_name} (id, content, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (doc_id, text, json.dumps(metadata or {}), vector)
            )
        self.conn.commit()
        logger.debug(f"[PGVectorDB] Bilgi eklendi. ID: {doc_id}")
        return doc_id

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.model.encode([query])[0].tolist()
        
        extracted = []
        with self.conn.cursor() as cur:
            # Cosine distance: <=>
            cur.execute(
                f"""
                SELECT id, content, metadata, embedding <=> %s AS distance
                FROM {self.table_name}
                ORDER BY distance
                LIMIT %s
                """,
                (query_vector, k)
            )
            rows = cur.fetchall()
            for row in rows:
                extracted.append({
                    "id": row[0],
                    "document": row[1],
                    "metadata": row[2] if row[2] else {},
                    "distance": float(row[3])
                })
        return extracted
