from ragatouille import RAGPretrainedModel
from edubotics_core.vectorstore.base import VectorStoreBase
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import Any, List
import os
import json


class RAGatouilleLangChainRetrieverWithScore(BaseRetriever):
    model: Any
    kwargs: dict = {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa
    ) -> List[Document]:
        """Get documents relevant to a query."""
        docs = self.model.search(query, **self.kwargs)
        return [
            Document(
                page_content=doc["content"],
                metadata={**doc.get("document_metadata", {}), "score": doc["score"]},
            )
            for doc in docs
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa
    ) -> List[Document]:
        """Get documents relevant to a query."""
        docs = self.model.search(query, **self.kwargs)
        return [
            Document(
                page_content=doc["content"],
                metadata={**doc.get("document_metadata", {}), "score": doc["score"]},
            )
            for doc in docs
        ]


class RAGPretrainedModel(RAGPretrainedModel):
    """
    Adding len property to RAGPretrainedModel
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._document_count = 0

    def set_document_count(self, count):
        self._document_count = count

    def __len__(self):
        return self._document_count

    def as_langchain_retriever(self, **kwargs: Any) -> BaseRetriever:
        return RAGatouilleLangChainRetrieverWithScore(model=self, kwargs=kwargs)


class ColbertVectorStore(VectorStoreBase):
    def __init__(self, config):
        self.config = config
        self._init_vector_db()

    def _init_vector_db(self):
        self.colbert = RAGPretrainedModel.from_pretrained(
            "colbert-ir/colbertv2.0",
            index_root=os.path.join(
                self.config["vectorstore"]["db_path"],
                "db_" + self.config["vectorstore"]["db_option"],
            ),
        )

    def create_database(self, documents, document_names, document_metadata):
        index_path = self.colbert.index(
            index_name="new_idx",
            collection=documents,
            document_ids=document_names,
            document_metadatas=document_metadata,
        )
        print(f"Index created at {index_path}")
        self.colbert.set_document_count(len(document_names))

    def load_database(self):
        path = os.path.join(
            os.getcwd(),
            self.config["vectorstore"]["db_path"],
            "db_" + self.config["vectorstore"]["db_option"],
        )
        self.vectorstore = RAGPretrainedModel.from_index(
            f"{path}/colbert/indexes/new_idx"
        )

        index_metadata = json.load(
            open(f"{path}/colbert/indexes/new_idx/0.metadata.json")
        )
        num_documents = index_metadata["num_passages"]
        self.vectorstore.set_document_count(num_documents)

        return self.vectorstore

    def as_retriever(self):
        return self.vectorstore.as_retriever()

    def __len__(self):
        return len(self.vectorstore)
