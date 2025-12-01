from agent_card_server.Environment.faiss import FAISSE
from agent_card_server.utils.decorators import test_only
from agent_card_server.Memory.embeddingdoc import EmbeddingDocument
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.faiss import dependable_faiss_import
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import math
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
import numpy as np

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance


class MemoryBank(FAISSE):

    def __init__(self, embedding: Embeddings, index: FAISSE, docstore: Docstore, index_to_id: Dict[int, str], normalize_L2: bool, working_memory=None, inspiration_conversation_history=None):
        super().__init__(embedding, index, docstore, index_to_id, normalize_L2)
        self.latest_summary: EmbeddingDocument = None
        self.overall_report: EmbeddingDocument = None
        self.working_memory: str = working_memory
        self.inspiration_conversation_history: str = working_memory

    # def get_new_memory_from_ids(self, ids: List[str], embedding: Embeddings = None, dim: int = 1536):
    #     print(123123123)
    #     print(123123123)
    #     print(123123123)
    #     print(123123123)
    #     print(123123123)
    #     faiss = dependable_faiss_import()
    #     index = faiss.IndexFlatL2(dim)
    #     index_to_id = dict(enumerate(ids))
    #     documents = self.get_document_by_ids(ids)
    #     embeddings = [doc.embedding for doc in documents]
    #     vector = np.array(embeddings, dtype=np.float32)
    #     index.add(vector)
    #     # process documents, if the document has parent, by parent id not in ids, remove the parent attribute
    #     print("check final things~~~~~~")
    #     print(index_to_id.values())
    #     docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
    #     if embedding is None:
    #         embedding = OpenAIEmbeddings()
    #     normalize_L2 = False
    #     return MemoryBank(
    #         embedding,
    #         index,
    #         docstore,
    #         index_to_id,
    #         normalize_L2
    #     )

    @classmethod
    def initialize_memory(cls, embedding: Embeddings = None, dim: int = 1536):
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(dim)  # 1536
        documents = []
        docstore = InMemoryDocstore(dict(zip([], [])))
        index_to_id = {}
        if embedding is None:
            embedding = OpenAIEmbeddings()
        normalize_L2 = False
        return cls(
            embedding,
            index,
            docstore,
            index_to_id,
            normalize_L2
        )

    @classmethod
    def from_documents(cls, documents: List[EmbeddingDocument], ids: List[str], embedding: Embeddings = None, dim: int = 1536):
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(dim)
        index_to_id = dict(enumerate(ids))
        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        if embedding is None:
            embedding = OpenAIEmbeddings()
        normalize_L2 = False
        return cls(
            embedding,
            index,
            docstore,
            index_to_id,
            normalize_L2
        )

    def get_new_memory_from_ids(self, ids: List[str], embedding: Embeddings = None, dim: int = 1536):
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(dim)
        documents = self.get_document_by_ids(ids)
        valid_ids = [doc.docstore_id for doc in documents]
        index_to_id = dict(enumerate(valid_ids))
        embeddings = [doc.embedding for doc in documents]
        if len(embeddings) > 0:
            vector = np.array(embeddings, dtype=np.float32)
            index.add(vector)
        # process documents, if the document has parent, by parent id not in ids, remove the parent attribute
        for doc in documents:
            if doc.parent and doc.parent.docstore_id not in ids:
                doc.parent = None
        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        if embedding is None:
            embedding = OpenAIEmbeddings()
        normalize_L2 = False
        return MemoryBank(
            embedding,
            index,
            docstore,
            index_to_id,
            normalize_L2
        )

    def get_document_by_ids(self, ids: List[str]) -> List[EmbeddingDocument]:
        return [self.docstore._dict[_id] for _id in ids if _id in self.docstore._dict]

    def add_memory_from_commit(self, commit):
        """
        Add memory from a commit
        """
        # TODO: extract commit blob, get result, metadata, and text content, and add to memory
        # TODO: modify the metadata to include the commit id
        # self.psedo_memory_from_commit(commit)

        # texts = [blob.result["content"] for blob in commit.blobs]
        # metadatas = [blob.result["meta"] for blob in commit.blobs]
        # for meta in metadatas:
        #     meta["commit_id"] = commit.id
        # ids = [blobs.id for blobs in commit.blobs] ## Use task id to identify the memory
        # return self.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        texts = [blob.result["content"]
                 for blob in commit.blobs if blob.result["update_longterm_memory"]]
        metadatas = [blob.result["meta"]
                     for blob in commit.blobs if blob.result["update_longterm_memory"]]
        childrenIds = [blob.result["meta"]['children'] for blob in commit.blobs if (
            blob.result["update_longterm_memory"] and blob.result["meta"].get('children'))]
        children = [self.get_document_by_ids(ids) for ids in childrenIds]
        parent = [None for _ in range(len(texts))]

        if commit.blobs[-1].result["meta"].get("thought") and commit.blobs[-1].result["update_longterm_memory"]:
            self.working_memory = commit.blobs[-1].result["meta"]["thought"]

        for meta in metadatas:
            meta["commit_id"] = commit.id
        # Use task id to identify the memory
        ids = [blob.id for blob in commit.blobs if blob.result["update_longterm_memory"]]
        print(f"adding memory ids {ids}")
        if len(texts) > 0:
            return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, parent=parent, children=children)

    def add_working_summary(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None, parent: Optional[List[Document]] = None,
                            children: Optional[List[List[Document]]] = None, **kwargs: Any):
        return self.add_texts(texts, metadatas, ids, update_latest_summary=True, parent=parent, children=children, **kwargs)

    def add_memory(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None, parent: Optional[List[Document]] = None,
                   children: Optional[List[List[Document]]] = None, **kwargs: Any):
        """
        """
        return self.add_texts(texts, metadatas, ids, parent=parent, children=children, **kwargs)

    def associate_in_memory(self, query: str, top_k: int = 4, **kwargs: Any) -> List[Tuple[EmbeddingDocument, float]]:
        """Associate a query with the memory bank."""
        return self.similarity_search_with_score(query, top_k, **kwargs)

    def get_inner_memories(self) -> List[Document]:
        """Get the leaf documents."""
        return [doc for doc in self.docstore._dict.values() if doc.parent is None]

    def get_leaf_memories(self) -> List[Document]:
        """Get the leaf documents."""
        return [doc for doc in self.docstore._dict.values() if doc.children is None]

    def output_memories_hierarchy(self, inner_memories=None):
        if inner_memories is None:
            inner_memories = self.get_inner_memories()
        if len(inner_memories) > 1:
            # create a psedo parent
            res = {
                "content": "pseudo",
                "id": "pseudo-id",
                "children": [
                    self.output_memories_hierarchy([child])
                    for child in inner_memories
                ]
            }
        else:
            res = {
                "content": inner_memories[0].page_content,
                "id": inner_memories[0].docstore_id
            }
            if inner_memories[0].children is not None and len(inner_memories[0].children) > 0:
                res["children"] = [
                    self.output_memories_hierarchy([child])
                    for child in inner_memories[0].children
                ]
            else:
                res["metadata"] = inner_memories[0].metadata
        return res

    @property
    def all_document_ids(self):
        return list(self.docstore._dict.keys())

    