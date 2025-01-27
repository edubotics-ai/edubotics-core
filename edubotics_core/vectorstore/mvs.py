from langchain_community.vectorstores import FAISS
from edubotics_core.vectorstore.base import VectorStoreBase
from edubotics_core.retriever.faiss_retriever import FaissRetriever
from edubotics_core.vectorstore.embedding_model_loader import EmbeddingModelLoader
from edubotics_core.vectorstore.helpers import determine_content_type

import os


class MultiVectorStore(VectorStoreBase):

    """
        Implementation of the multi-vector approach, where each vector store corresponds to a different content type.
        A parent folder in the db_path called "mvs" is created, and within it, a folder for each vector store (content type) is created.
    """

    def __init__(self, config):
        self.config = config
        self.vectorstores = {}
        self.content_types = config["metadata"]["content_types"]
        self.local_path = os.path.join(
            self.config["vectorstore"]["db_path"],
            "mvs",
            self.config["vectorstore"]["db_option"]
            + "_",
        )

    def _init_vector_db(self):
        """
        Initializes the vector stores for each content type. Simply creates an empty FAISS vector store for each content type.
        """
        for content_type in self.content_types:
            self.vectorstores[content_type] = FAISS(
                embedding_function=None, index=0, index_to_docstore_id={}, docstore={}
            )

    def create_database(self, document_chunks, embedding_model):
        """
        Creates and saves the vector stores for each content type.
        """
        content_map = {}

        for content_type in self.content_types:
            content_map[content_type] = list(
                filter(lambda x: determine_content_type(x) == content_type, document_chunks))

        for content_type in content_map:
            content_chunks = content_map[content_type]
            if len(content_chunks) > 0:
                for chunk in content_chunks:
                    chunk.metadata["content_type"] = content_type

                self.vectorstores[content_type] = FAISS.from_documents(
                    documents=content_chunks, embedding=embedding_model
                )
                self.vectorstores[content_type].save_local(
                    self.local_path + (content_type or ""))
            else:
                print(f"No content chunks found for {content_type}")

    def load_database(self, embedding_model) -> dict:
        """
        Loads the vector stores for each content type.
        """
        for content_type in self.content_types:
            try:
                path = self.local_path + (content_type or "")
                self.vectorstores[content_type] = FAISS.load_local(
                    path, embedding_model, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading vector store for {content_type}: {e}")
                continue
        return self.vectorstores

    def as_retriever(self):
        """
        Returns retrievers for each content type as a dictionary.
        """
        retrievers = {}
        embedding_model = EmbeddingModelLoader(self.config).load_embedding_model()
        vectorstores = self.load_database(embedding_model)
        for content_type in self.content_types:
            if content_type in vectorstores:
                retriever = FaissRetriever().return_retriever(
                    vectorstores[content_type], self.config)
                retrievers[content_type] = retriever
            else:
                print(f"No vector store found for {content_type}")

        return retrievers

    def __len__(self):
        """
        Returns the total number of documents in all vector stores.
        """
        return sum(len(self.vectorstores[content_type]) for content_type in self.content_types)

    def __str__(self):
        """
        Returns the string representation of the MultiVectorStore.
        """
        return f"MultiVectorStore with {len(self)} documents"
