from langchain_community.vectorstores import FAISS
from modules.vectorstore.base import VectorStoreBase
import os


class FaissVectorStore(VectorStoreBase):
    def __init__(self, config):
        self.config = config
        self._init_vector_db()

    def _init_vector_db(self):
        self.faiss = FAISS(
            embedding_function=None, index=0, index_to_docstore_id={}, docstore={}
        )

    def create_database(self, document_chunks, embedding_model):
        self.vectorstore = self.faiss.from_documents(
            documents=document_chunks, embedding=embedding_model
        )
        self.vectorstore.save_local(
            os.path.join(
                self.config["vectorstore"]["db_path"],
                "db_"
                + self.config["vectorstore"]["db_option"]
                + "_"
                + self.config["vectorstore"]["model"],
            )
        )

    def load_database(self, embedding_model):
        self.vectorstore = self.faiss.load_local(
            os.path.join(
                self.config["vectorstore"]["db_path"],
                "db_"
                + self.config["vectorstore"]["db_option"]
                + "_"
                + self.config["vectorstore"]["model"],
            ),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore

    def as_retriever(self):
        return self.vectorstore.as_retriever()
