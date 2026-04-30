from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorDB(ABC):
    """
    AgentKit için Vektör Veritabanı Soyutlama Arayüzü.
    Pinecone, Qdrant, ChromaDB, Weaviate ve pgvector gibi tüm veritabanları
    bu arayüzü uygulayarak sisteme entegre edilebilir.
    """

    @abstractmethod
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Yeni bir bilgiyi (belge/konuşma) vektör veritabanına ekler.
        
        Args:
            text (str): Vektöre dönüştürülecek metin.
            metadata (Optional[Dict[str, Any]]): Metne ait ek veriler.
            
        Returns:
            str: Eklenen dokümanın benzersiz ID'si.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Verilen sorguya anlamsal olarak en yakın 'k' adet sonucu döndürür.
        
        Args:
            query (str): Arama sorgusu.
            k (int): Döndürülecek maksimum sonuç sayısı.
            
        Returns:
            List[Dict[str, Any]]: İçinde "id", "document", "metadata" ve "distance" barındıran sonuç listesi.
        """
        pass
