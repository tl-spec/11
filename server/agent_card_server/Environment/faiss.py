from agent_card_server.utils.decorators import test_only
from agent_card_server.Memory.embeddingdoc import EmbeddingDocument


from langchain_community.vectorstores.faiss import FAISS as _FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import math
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from langchain_openai import OpenAIEmbeddings
import numpy as np

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance

# class EmbeddingDocument(Document):
#     """Document with an associated embedding."""
#     embedding: List[float]
#     docstore_id: str
#     children: Optional[List[Document]] = None
#     parent: Optional[Document] = None 

class FAISSE(_FAISS): 
    """FAISS with explicit embeddings saved for visual exploration purpose."""

    
    def get_document_by_ids(self, ids: List[str]) -> List[EmbeddingDocument]:
        return [self.docstore._dict[_id] for _id in ids if _id in self.docstore._dict]
        
        
    # def get_new_memory_from_ids(self, ids: List[str], embedding: Embeddings=None, dim:int=1536):
    #     faiss = dependable_faiss_import()
    #     index = faiss.IndexFlatL2(dim) 
    #     index_to_id = dict(enumerate(ids))
    #     documents = self.get_document_by_ids(ids)
    #     embeddings = [doc.embedding for doc in documents]
    #     vector = np.array(embeddings, dtype=np.float32)
    #     index.add(vector)
    #     ## process documents, if the document has parent, by parent id not in ids, remove the parent attribute 
    #     print("check final things")
    #     print(index_to_id.values())
    #     docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
    #     if embedding is None:
    #         embedding = OpenAIEmbeddings()
    #     normalize_L2 = False
    #     return FAISSE(
    #         embedding,
    #         index,
    #         docstore,
    #         index_to_id,
    #         normalize_L2
    #     )
    
    @classmethod
    def from_documents(cls, documents: List[EmbeddingDocument], ids: List[str], embedding: Embeddings=None, dim:int=1536):
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
    
    @classmethod
    def from_documents_v2(cls, documents: List[EmbeddingDocument], ids: List[str], embedding: Embeddings=None, dim:int=1536):
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(dim) 
        index_to_id = dict(enumerate(ids))
        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        if embedding is None:
            embedding = OpenAIEmbeddings()
        normalize_L2 = False
        vecstore = cls(
            embedding,
            index,
            docstore,
            index_to_docstore_id,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
        )
        
        return vecstore
        
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        normalize_L2: bool = False,
        **kwargs: Any,
    ):  
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(len(embeddings[0]))
        vector = np.array(embeddings, dtype=np.float32)
        if normalize_L2:
            faiss.normalize_L2(vector)
        index.add(vector)
        documents = []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            embedding_i = embeddings[i]
            documents.append(EmbeddingDocument(page_content=text, metadata=metadata, embedding=embedding_i, docstore_id=ids[i]))
        index_to_id = dict(enumerate(ids))
        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        return cls(
            embedding.embed_query,
            index,
            docstore,
            index_to_id,
            normalize_L2=normalize_L2,
            **kwargs,
        )
    
    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by ID. These are the IDs in the vectorstore.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None or len(ids) == 0:
            return 
        missing_ids = set(ids).difference(self.index_to_docstore_id.values())
        if missing_ids:
            raise ValueError(
                f"Some specified ids do not exist in the current store. Ids not found: "
                f"{missing_ids}"
            )

        ## leverage ids to find each document's children, remove their parent 
        for id_ in ids:
            doc = self.docstore.search(id_)
            if doc.children:
                for child in doc.children:
                    child.parent = None
        
        reversed_index = {id_: idx for idx, id_ in self.index_to_docstore_id.items()}
        index_to_delete = {reversed_index[id_] for id_ in ids}

        self.index.remove_ids(np.fromiter(index_to_delete, dtype=np.int64))
        self.docstore.delete(ids)

        remaining_ids = [
            id_
            for i, id_ in sorted(self.index_to_docstore_id.items())
            if i not in index_to_delete
        ]
        self.index_to_docstore_id = {i: id_ for i, id_ in enumerate(remaining_ids)}

        return True
    
    def __add(
            self,
            texts: Iterable[str],
            embeddings: Iterable[List[float]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            update_latest_summary: bool = False,
            parent: Optional[List[Document]] = None,
            children: Optional[List[List[Document]]] = None,
            **kwargs: Any,
        ) -> List[str]:
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        documents = []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            embedding_i = embeddings[i]
            id_i = ids[i]
            embed_doc = EmbeddingDocument(page_content=text, metadata=metadata, embedding=embedding_i, docstore_id=ids[i], parent=parent[i] if parent else None, children=children[i] if children else None)
            documents.append(embed_doc)
            # iterate children, assign parent to the current document
            if children and children[i]:
                for child in children[i]:
                    child.parent = embed_doc
        
        # Update the latest summary
        if update_latest_summary: 
            self.latest_summary = documents[-1]
        
        # Add to the index, the index_to_id mapping, and the docstore.
        starting_len = len(self.index_to_docstore_id)
        faiss = dependable_faiss_import()
        vector = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        self.index.add(vector)
        # Get list of index, id, and docs.
        full_info = [(starting_len + i, ids[i], doc) for i, doc in enumerate(documents)]
        # Add information to docstore and index.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)
        return [_id for _, _id, _ in full_info]
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        update_latest_summary: bool = False,    
        parent: Optional[List[Document]] = None,
        children: Optional[List[List[Document]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        embeddings = self._embed_documents(texts)
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, update_latest_summary=update_latest_summary, parent=parent, children=children, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.

            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of documents most similar to the query text with
            L2 distance in float. Lower score represents more similarity.
        """
        embedding = self._embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs

    
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Union[Callable, Dict[str, Any]]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        docs = []

        if filter is not None:
            filter_func = self._create_filter_func(filter)

        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            if filter is not None:
                if filter_func(doc.metadata):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            cmp = (
                operator.ge
                if self.distance_strategy
                in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                else operator.le
            )
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]

        for_task = kwargs.get("for_task", False)
        if not for_task:
            candidate_docs = {} 
            for doc, score in docs:
                target_doc = self.get_parent_doc(doc)
                if target_doc.docstore_id not in candidate_docs:
                    candidate_docs[target_doc.docstore_id] = (target_doc, score)
                if len(candidate_docs) == k:
                    break
            return list(candidate_docs.values())
        else:
            return docs[:k]

    def get_parent_doc(self, doc: EmbeddingDocument) -> Optional[EmbeddingDocument]:
        """Get the parent document of a given document."""
        if doc.parent is None:
            return doc 
        else: 
            while doc.parent is not None:
                doc = doc.parent
            return doc
        

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return docs_and_scores

    @test_only
    def psedo_embedding_methods(self, texts: Iterable[str]):
        """
        For testing purposes, we will use a random vector as the embedding for the text.
        """
        num_texts = len(texts)
        import numpy as np 
        return np.random.rand(num_texts, 1536).tolist()
    
    @test_only
    def psedo_add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        update_latest_summary: bool = False,    
        parent: Optional[List[Document]] = None,
        children: Optional[List[List[Document]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        embeddings = self.psedo_embedding_methods(texts)
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, update_latest_summary=update_latest_summary, parent=parent, children=children, **kwargs)

# class MemoryBank(FAISSE):
#     @classmethod 
#     def initialize_memory(cls, embedding: Embeddings=None):
#         faiss = dependable_faiss_import()
#         index = faiss.IndexFlatL2(1536)
#         documents = []
#         docstore = InMemoryDocstore(dict(zip([], [])))
#         index_to_id = {}
#         if embedding is None:
#             embedding = OpenAIEmbeddings()
#         normalize_L2 = False
#         return cls(
#             embedding.embed_query,
#             index,
#             docstore,
#             index_to_id,
#             normalize_L2
#         )
    
#     def add_memory(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None,
#             **kwargs: Any):
#         """"""
#         return self.add_texts(texts, metadatas, ids, **kwargs)

#     def associate_in_memory(self, query: str, top_k: int=4, **kwargs: Any) -> List[Tuple[EmbeddingDocument, float]]:
#         """Associate a query with the memory bank."""
#         return self.similarity_search_with_scores(query, top_k, **kwargs)