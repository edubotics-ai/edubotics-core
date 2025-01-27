import numpy as np
import bm25s
import os

from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as BaseRetrieverLangchain
from .base import BaseRetriever as BaseRetrieverEdubotics

from vectorstore.embedding_model_loader import EmbeddingModelLoader
from langchain_community.vectorstores import FAISS
from config.config_manager import config_manager


class FusionRetrieverBase(BaseRetrieverLangchain):
    vectorstore: VectorStore = None
    config: dict = None
    alpha: float = 0.5
    k: int = 10
    bm25: bm25s.BM25 = None

    def __init__(self, vectorstore: VectorStore, config):
        super().__init__()
        self.vectorstore = vectorstore  # FAISS
        self.config = config
        self.alpha = 0.5
        self.k = config["vectorstore"]["search_top_k"]
        self.bm25 = bm25s.BM25.load(
            os.path.join(
                self.config["vectorstore"]["db_path"], "db_fusion", "bm25_index"
            ),
            load_corpus=True,
        )

        print("FusionRetrieverBase initialized")

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        all_docs = self.vectorstore.similarity_search_with_relevance_scores(
            "", k=self.vectorstore.index.ntotal
        )

        bm25_scores = self.bm25.get_scores(query.split())

        vector_results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=len(all_docs)
        )

        vector_scores = np.array([score for _, score in vector_results])
        vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (
            np.max(vector_scores) - np.min(vector_scores)
        )

        bm25_scores = bm25_scores - np.min(bm25_scores)

        # Check if the max and min are the same to avoid division by zero
        bm25_range = np.max(bm25_scores) - np.min(bm25_scores)
        if bm25_range > 0:
            bm25_scores = bm25_scores / bm25_range
        else:
            bm25_scores = np.zeros_like(bm25_scores)  # or handle it as needed

        combined_scores = self.alpha * vector_scores + (1 - self.alpha) * bm25_scores

        sorted_indices = np.argsort(combined_scores)[::-1]

        return [
            Document(
                page_content=all_docs[i][0].page_content,
                metadata={"score": combined_scores[i], **all_docs[i][0].metadata},
            )
            for i in sorted_indices[:10]
        ]

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self._get_relevant_documents(query, **kwargs)


class FusionRetriever(BaseRetrieverEdubotics):

    def __init__(self):
        return

    def return_retriever(self, vectorstore: VectorStore, config):
        retriever = FusionRetrieverBase(vectorstore, config)
        return retriever


if __name__ == "__main__":
    config = config_manager.get_config().dict()

    embedding_model_loader = EmbeddingModelLoader(config)
    embedding_model = embedding_model_loader.load_embedding_model()

    cwd = os.getcwd()

    faiss_path = os.path.join(
        cwd,
        config["vectorstore"]["db_path"],
        "db_FAISS_sentence-transformers",
        "all-MiniLM-L6-v2_semantic",
    )

    vectorstore = FAISS.load_local(
        faiss_path, embeddings=embedding_model, allow_dangerous_deserialization=True
    )
    retriever = FusionRetrieverBase(vectorstore, config)

    docs = retriever._get_relevant_documents("**Question 1**")
    print(f"Number of documents retrieved: {len(docs)}")

    for doc in docs:
        print(doc.page_content[:50])
        print(doc.metadata["score"])
        print(doc.metadata["source"])
        print("----")
