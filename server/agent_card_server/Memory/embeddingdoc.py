from langchain.docstore.document import Document
from typing import List, Optional

class EmbeddingDocument(Document):
    """Document with an associated embedding."""
    embedding: List[float] | None
    docstore_id: str
    children: Optional[List[Document]] = None
    parent: Optional[Document] = None 

    # def __init__(self, page_content: str, embedding: List[float], docstore_id: str, metadata: Optional[dict] = None, children: Optional[List[Document]] = None, parent: Optional[Document] = None) -> None:
    #     super().__init__(page_content, metadata=metadata)
    #     self.embedding = embedding
    #     self.docstore_id = docstore_id
    #     self.children = children
    #     self.parent = parent

    def __repr__(self) -> str:
        page_content = self.page_content[:50] + "..." if len(self.page_content) > 50 else self.page_content
        embedding = self.embedding[:2] + ["..."] if len(self.embedding) > 2 else self.embedding
        metadata = "{...}" if self.metadata else 'None'
        return f"EmbeddingDocument(page_content={page_content}, embedding={embedding}, metadata={metadata})"
