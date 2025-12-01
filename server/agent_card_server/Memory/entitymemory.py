from Memory.base import MemoryBank 
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.faiss import dependable_faiss_import
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple 

class EntityMemory(MemoryBank):
    @classmethod 
    def initialize_memory(cls, embedding: Embeddings=None, dim: int=1536): 
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(dim) # 1536
        pass 

    
    def add_memory(self, texts: Iterable[str], metadata: Optional[List[dict]] = None, ids: Optional[List[str]] = None,
            **kwargs: Any):
        """ extract entities from texts, check if there is an existing memory and then update it. """
        pass 
    
        
         